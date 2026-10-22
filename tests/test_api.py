"""Tests for the FastAPI application."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from revinsight.api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Return a TestClient bound to the application."""
    return TestClient(app)


@pytest.fixture(scope="module")
def analysis_run(client: TestClient) -> dict:
    """Create one analysis run that the other tests reuse."""
    response = client.post("/api/v1/analyses", data={"dataset": "RevI-Test"})
    assert response.status_code == 201, response.text
    return response.json()


class TestMeta:
    """Health and dataset endpoints."""

    def test_health(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"]
        assert body["uptime_seconds"] >= 0

    def test_datasets(self, client: TestClient) -> None:
        response = client.get("/api/v1/datasets")
        assert response.status_code == 200
        datasets = response.json()["datasets"]
        assert datasets
        assert {"id", "label", "rows", "columns", "months"} <= set(datasets[0])

    def test_primary_dataset_is_listed_first(self, client: TestClient) -> None:
        """The bundled sample dataset must sort before its "-Large" sibling.

        Sorting by full path puts "RevI-Test-Large.csv" first because the
        hyphen sorts before the extension dot, which would make the
        dashboard default to the wrong file.
        """
        response = client.get("/api/v1/datasets")
        ids = [entry["id"] for entry in response.json()["datasets"]]
        if "RevI-Test" in ids and "RevI-Test-Large" in ids:
            assert ids.index("RevI-Test") < ids.index("RevI-Test-Large")

    def test_openapi_schema(self, client: TestClient) -> None:
        response = client.get("/api/openapi.json")
        assert response.status_code == 200
        assert "/api/v1/analyses" in response.json()["paths"]


class TestAnalysisLifecycle:
    """Create, fetch and delete analysis runs."""

    def test_create_run(self, client: TestClient) -> None:
        response = client.post("/api/v1/analyses", data={"dataset": "RevI-Test"})
        assert response.status_code == 201
        body = response.json()
        assert body["run_id"]
        assert body["schema_version"] == "2.0"
        assert body["executive"]["recommendation"]
        assert body["kpis"]["current_employees"] == 7

    def test_create_run_with_assumptions(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyses",
            data={
                "dataset": "RevI-Test",
                "salary": 4.0,
                "commission": 0.07,
                "forecast_months": 3,
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["assumptions"]["salary"] == 4.0
        assert body["assumptions"]["commission"] == 0.07
        assert len(body["forecast"]["values"]) == 3

    def test_create_run_with_unknown_dataset(self, client: TestClient) -> None:
        response = client.post("/api/v1/analyses", data={"dataset": "does-not-exist"})
        assert response.status_code == 404
        assert "detail" in response.json()

    def test_create_run_with_invalid_assumptions(self, client: TestClient) -> None:
        response = client.post("/api/v1/analyses", data={"dataset": "RevI-Test", "commission": 5.0})
        assert response.status_code == 422

    def test_create_run_with_upload(self, client: TestClient, raw_df) -> None:
        buffer = io.StringIO()
        raw_df.to_csv(buffer, index=False)
        buffer.seek(0)
        response = client.post(
            "/api/v1/analyses",
            files={"file": ("upload.csv", buffer.getvalue(), "text/csv")},
            data={"salary": 3.0, "commission": 0.05},
        )
        assert response.status_code == 201
        assert response.json()["dataset"]["name"] == "uploaded"

    def test_create_run_with_invalid_upload(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/analyses",
            files={"file": ("bad.csv", "not,a,valid,csv\n1,2,3,4\n", "text/csv")},
        )
        assert response.status_code in (400, 422)

    def test_get_run(self, client: TestClient, analysis_run: dict) -> None:
        response = client.get(f"/api/v1/analyses/{analysis_run['run_id']}")
        assert response.status_code == 200
        assert response.json()["run_id"] == analysis_run["run_id"]

    def test_get_unknown_run(self, client: TestClient) -> None:
        response = client.get("/api/v1/analyses/does-not-exist")
        assert response.status_code == 404
        assert "detail" in response.json()

    def test_list_runs(self, client: TestClient, analysis_run: dict) -> None:
        response = client.get("/api/v1/analyses")
        assert response.status_code == 200
        runs = response.json()["runs"]
        assert any(r["run_id"] == analysis_run["run_id"] for r in runs)

    def test_delete_run(self, client: TestClient) -> None:
        created = client.post("/api/v1/analyses", data={"dataset": "RevI-Test"}).json()
        response = client.delete(f"/api/v1/analyses/{created['run_id']}")
        assert response.status_code == 200
        assert client.get(f"/api/v1/analyses/{created['run_id']}").status_code == 404

    def test_delete_unknown_run(self, client: TestClient) -> None:
        assert client.delete("/api/v1/analyses/nope").status_code == 404


class TestArtefacts:
    """Generated charts and reports."""

    def test_report_is_generated_on_demand(self, client: TestClient, analysis_run: dict) -> None:
        """The analysis is fast; the report is rendered when asked for."""
        run_id = analysis_run["run_id"]
        response = client.post(f"/api/v1/analyses/{run_id}/report")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        assert body["pdf_url"].endswith("/report.pdf")

    def test_pdf_is_served(self, client: TestClient, analysis_run: dict) -> None:
        run_id = analysis_run["run_id"]
        client.post(f"/api/v1/analyses/{run_id}/report")
        response = client.get(f"/api/v1/analyses/{run_id}/report.pdf")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content[:4] == b"%PDF"

    def test_csv_is_served(self, client: TestClient, analysis_run: dict) -> None:
        response = client.get(f"/api/v1/analyses/{analysis_run['run_id']}/results.csv")
        assert response.status_code == 200
        assert "Month" in response.text.splitlines()[0]

    def test_chart_is_served(self, client: TestClient, analysis_run: dict) -> None:
        run_id = analysis_run["run_id"]
        client.post(f"/api/v1/analyses/{run_id}/report")
        response = client.get(f"/api/v1/analyses/{run_id}/charts/monthly_sales_trend.png")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content[:4] == b"\x89PNG"

    def test_unknown_chart_is_404(self, client: TestClient, analysis_run: dict) -> None:
        response = client.get(f"/api/v1/analyses/{analysis_run['run_id']}/charts/not_a_chart.png")
        assert response.status_code == 404

    def test_report_for_unknown_run_is_404(self, client: TestClient) -> None:
        assert client.post("/api/v1/analyses/nope/report").status_code == 404

    def test_recommendation_text(self, client: TestClient, analysis_run: dict) -> None:
        response = client.get(f"/api/v1/analyses/{analysis_run['run_id']}/recommendation.txt")
        assert response.status_code == 200
        assert "EXECUTIVE SUMMARY" in response.json()["text"]


class TestSimulation:
    """The what-if simulator endpoint."""

    def test_simulate(self, client: TestClient, analysis_run: dict) -> None:
        response = client.post(
            "/api/v1/simulate",
            json={"run_id": analysis_run["run_id"], "salary": 4.0, "commission": 0.06},
        )
        assert response.status_code == 200
        body = response.json()
        assert set(body["scenarios"]) == {"A", "B", "C", "D"}
        assert body["assumptions"]["salary"] == 4.0

    def test_simulate_with_shock(self, client: TestClient, analysis_run: dict) -> None:
        response = client.post(
            "/api/v1/simulate",
            json={"run_id": analysis_run["run_id"], "sales_shock_pct": -20.0},
        )
        assert response.status_code == 200
        assert response.json()["assumptions"]["sales_shock_pct"] == -20.0

    def test_simulate_unknown_run(self, client: TestClient) -> None:
        response = client.post("/api/v1/simulate", json={"run_id": "nope"})
        assert response.status_code == 404

    def test_simulate_rejects_invalid_payload(self, client: TestClient) -> None:
        response = client.post("/api/v1/simulate", json={"salary": 3.0})
        assert response.status_code == 422

    def test_sensitivity_endpoint(self, client: TestClient, analysis_run: dict) -> None:
        response = client.get(f"/api/v1/analyses/{analysis_run['run_id']}/sensitivity")
        assert response.status_code == 200
        body = response.json()
        assert "tornado" in body
        assert "monte_carlo" in body
        assert "grid" in body
