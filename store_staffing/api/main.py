"""FastAPI application exposing the Store Staffing Analyzer analytics engine.

The API is versioned under ``/api/v1`` and serves the web dashboard from
the same origin, so the whole product deploys as a single process.

Run it with::

    uvicorn store_staffing.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..config import config_from_env
from ..utils import load_and_validate_data
from .services import AnalysisService, get_service, public_payload

logger = logging.getLogger(__name__)

__all__ = ["app", "create_app"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = PROJECT_ROOT / "web"

MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024
"""Maximum accepted CSV upload size (25 MB)."""


# --------------------------------------------------------------------------- #
# Request / response models                                                    #
# --------------------------------------------------------------------------- #
class SimulationRequest(BaseModel):
    """Payload for the what-if simulator."""

    run_id: str = Field(..., description="Identifier of a completed analysis run.")
    salary: float | None = Field(
        None, ge=0, le=100, description="Monthly salary per employee (Lakhs)."
    )
    commission: float | None = Field(
        None, ge=0, le=0.95, description="Commission rate as a fraction of sales."
    )
    horizon: int | None = Field(None, ge=1, le=60, description="Forecast horizon in months.")
    decision: str = Field("auto", description="auto | hold | hire | conditional")
    sales_shock_pct: float = Field(
        0.0, ge=-100, le=100, description="Percentage shift applied to the forecast."
    )

    def normalised_decision(self) -> str:
        """Return the decision policy, falling back to ``auto``."""
        value = (self.decision or "auto").strip().lower()
        return value if value in {"auto", "hold", "hire", "conditional"} else "auto"


# --------------------------------------------------------------------------- #
# Application factory                                                          #
# --------------------------------------------------------------------------- #
def create_app(project_root: str | Path | None = None) -> FastAPI:
    """Build and configure the FastAPI application.

    Parameters
    ----------
    project_root : str | Path, optional
        Root directory containing ``data/`` and ``web/``.

    Returns
    -------
    FastAPI
        The configured application.
    """
    root = Path(project_root) if project_root else PROJECT_ROOT

    application = FastAPI(
        title="Store Staffing Analyzer Analytics API",
        description=(
            "Data-driven retail staffing optimisation: forecasting, "
            "financial modelling, scenario analysis and workforce analytics."
        ),
        version=__version__,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("SSA_CORS_ORIGINS", "*").split(","),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def add_security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    service = get_service(root)

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _service() -> AnalysisService:
        """Return the shared analysis service."""
        return service

    def _require_run(run_id: str) -> dict[str, Any]:
        """Return a cached run or raise a 404."""
        payload = service.get_run(run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"Unknown analysis run: {run_id}")
        return payload

    # ------------------------------------------------------------------ #
    # Meta routes                                                         #
    # ------------------------------------------------------------------ #
    @application.get("/api/v1/health", tags=["meta"])
    def health() -> dict[str, Any]:
        """Return service health and uptime information."""
        return {
            "status": "ok",
            "version": __version__,
            "uptime_seconds": round(service.uptime_seconds(), 2),
            "runs_cached": service.run_count(),
        }

    @application.get("/api/v1/datasets", tags=["meta"])
    def datasets(svc: AnalysisService = Depends(_service)) -> dict[str, Any]:
        """List the datasets bundled with the project."""
        return {"datasets": svc.list_datasets()}

    @application.get("/api/v1/analyses", tags=["analyses"])
    def list_analyses(svc: AnalysisService = Depends(_service)) -> dict[str, Any]:
        """List every cached analysis run, newest first."""
        return {"runs": svc.list_runs()}

    # ------------------------------------------------------------------ #
    # Analysis routes                                                     #
    # ------------------------------------------------------------------ #
    @application.post("/api/v1/analyses", status_code=201, tags=["analyses"])
    async def create_analysis(
        dataset: str | None = Form(None, description="Bundled dataset id."),
        file: UploadFile | None = File(None, description="Optional CSV upload."),
        salary: float = Form(3.0, gt=0, le=100),
        commission: float = Form(0.05, ge=0, le=0.95),
        forecast_months: int = Form(6, ge=1, le=36),
        start_month: int = Form(1, ge=1, le=12),
        svc: AnalysisService = Depends(_service),
    ) -> dict[str, Any]:
        """Run the analysis pipeline and return the full result payload.

        The pipeline is CPU-bound, so it is executed in a worker thread to
        keep the event loop responsive for concurrent requests.
        """
        try:
            config = config_from_env(
                salary_per_employee=salary,
                commission_rate=commission,
                forecast_months=forecast_months,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if file is not None and file.filename:
            contents = await await_file(file)
            if contents is None:
                raise HTTPException(status_code=422, detail="The uploaded file is empty.")
            try:
                df = load_and_validate_data(_write_temp_upload(contents, file.filename))
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=422, detail=f"Invalid CSV upload: {exc}") from exc
            try:
                payload = await run_in_threadpool(
                    svc.create_run, df=df, config=config, start_month=start_month
                )
                return public_payload(payload)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        path = svc.resolve_dataset(dataset)
        if path is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Dataset '{dataset}' not found. Available datasets: "
                    f"{[d['id'] for d in svc.list_datasets()]}"
                ),
            )
        try:
            payload = await run_in_threadpool(
                svc.create_run, filepath=str(path), config=config, start_month=start_month
            )
            return public_payload(payload)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/api/v1/analyses/{run_id}", tags=["analyses"])
    def get_analysis(run_id: str) -> dict[str, Any]:
        """Return the payload of a completed analysis run."""
        return public_payload(_require_run(run_id))

    @application.delete("/api/v1/analyses/{run_id}", tags=["analyses"])
    def delete_analysis(run_id: str, svc: AnalysisService = Depends(_service)) -> dict[str, Any]:
        """Delete a cached analysis run and its artefacts."""
        if not svc.delete_run(run_id):
            raise HTTPException(status_code=404, detail=f"Unknown analysis run: {run_id}")
        return {"deleted": True, "run_id": run_id}

    @application.get("/api/v1/analyses/{run_id}/charts/{chart_name}", tags=["artefacts"])
    def get_chart(run_id: str, chart_name: str) -> FileResponse:
        """Return a generated chart as a PNG image."""
        _require_run(run_id)
        path = service.chart_path(run_id, chart_name)
        if path is None:
            try:
                service.ensure_artifacts(run_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=f"Unknown run: {run_id}") from exc
            path = service.chart_path(run_id, chart_name)
        if path is None:
            raise HTTPException(status_code=404, detail=f"Chart not found: {chart_name}")
        return FileResponse(path, media_type="image/png")

    @application.get("/api/v1/analyses/{run_id}/report.pdf", tags=["artefacts"])
    def get_pdf(run_id: str) -> FileResponse:
        """Return the generated PDF report, rendering it if necessary."""
        _require_run(run_id)
        path = service.report_path(run_id, "pdf")
        if path is None:
            try:
                service.ensure_artifacts(run_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=f"Unknown run: {run_id}") from exc
            path = service.report_path(run_id, "pdf")
        if path is None:
            raise HTTPException(status_code=503, detail="The report could not be generated yet.")
        return FileResponse(
            path, media_type="application/pdf", filename="Recommendation_Report.pdf"
        )

    @application.post("/api/v1/analyses/{run_id}/report", tags=["artefacts"])
    def build_report(run_id: str) -> dict[str, Any]:
        """Render the report artefacts for a run and return their URLs.

        The dashboard calls this before offering the download, so the
        report is produced exactly once and cached on disk.
        """
        _require_run(run_id)
        try:
            paths = service.ensure_artifacts(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown run: {run_id}") from exc
        return {
            "run_id": run_id,
            "ready": bool(paths),
            "pdf_url": f"/api/v1/analyses/{run_id}/report.pdf",
            "html_url": f"/api/v1/analyses/{run_id}/report.html",
            "csv_url": f"/api/v1/analyses/{run_id}/results.csv",
            "charts_url": f"/api/v1/analyses/{run_id}/charts",
        }

    @application.get("/api/v1/analyses/{run_id}/report.html", tags=["artefacts"])
    def get_html(run_id: str) -> FileResponse:
        """Return the generated HTML report, rendering it if necessary."""
        _require_run(run_id)
        path = service.report_path(run_id, "html")
        if path is None:
            try:
                service.ensure_artifacts(run_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=f"Unknown run: {run_id}") from exc
            path = service.report_path(run_id, "html")
        if path is None:
            raise HTTPException(status_code=503, detail="The report could not be generated yet.")
        return FileResponse(path, media_type="text/html")

    @application.get("/api/v1/analyses/{run_id}/results.csv", tags=["artefacts"])
    def get_csv(run_id: str) -> FileResponse:
        """Return the exported monthly results as CSV."""
        _require_run(run_id)
        path = service.csv_path(run_id)
        if path is None:
            try:
                service.ensure_artifacts(run_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=f"Unknown run: {run_id}") from exc
            path = service.csv_path(run_id)
        if path is None:
            raise HTTPException(
                status_code=503, detail="The CSV export could not be generated yet."
            )
        return FileResponse(path, media_type="text/csv", filename="analysis_results.csv")

    @application.get("/api/v1/analyses/{run_id}/recommendation.txt", tags=["artefacts"])
    def get_recommendation_text(run_id: str) -> dict[str, Any]:
        """Return the plain-text recommendation."""
        payload = _require_run(run_id)
        return {"run_id": run_id, "text": payload.get("recommendation_text", "")}

    # ------------------------------------------------------------------ #
    # Simulation routes                                                   #
    # ------------------------------------------------------------------ #
    @application.post("/api/v1/simulate", tags=["simulation"])
    def simulate(request: SimulationRequest) -> dict[str, Any]:
        """Re-price the staffing scenarios under changed assumptions."""
        result = service.simulate(
            request.run_id,
            salary=request.salary,
            commission=request.commission,
            horizon=request.horizon,
            decision=request.normalised_decision(),
            sales_shock_pct=request.sales_shock_pct,
        )
        if result is None:
            raise HTTPException(status_code=404, detail=f"Unknown analysis run: {request.run_id}")
        return public_payload(result)

    @application.get("/api/v1/analyses/{run_id}/sensitivity", tags=["simulation"])
    def sensitivity(run_id: str) -> dict[str, Any]:
        """Return the tornado, Monte-Carlo and grid sensitivity blocks."""
        payload = _require_run(run_id)
        block = payload.get("sensitivity", {})
        return public_payload(
            {
                "tornado": block.get("tornado", []),
                "tornado_base_margin": block.get("tornado_base_margin"),
                "monte_carlo": block.get("monte_carlo", {}),
                "grid": block.get("grid", []),
            }
        )

    # ------------------------------------------------------------------ #
    # Error handling                                                      #
    # ------------------------------------------------------------------ #
    @application.exception_handler(HTTPException)
    async def http_exception_handler(_request, exc: HTTPException) -> JSONResponse:
        """Return a uniform error body."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": _error_code(exc.status_code)},
        )

    # ------------------------------------------------------------------ #
    # Static frontend                                                     #
    # ------------------------------------------------------------------ #
    if WEB_DIR.exists():
        application.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets")), name="assets")

        @application.get("/", include_in_schema=False)
        def dashboard() -> FileResponse:
            """Serve the single-page dashboard."""
            return FileResponse(str(WEB_DIR / "index.html"))

        @application.get("/favicon.svg", include_in_schema=False)
        def favicon() -> FileResponse:
            """Serve the dashboard favicon."""
            return FileResponse(str(WEB_DIR / "favicon.svg"))

    else:  # pragma: no cover - only when the web assets are missing

        @application.get("/", include_in_schema=False)
        def missing_web() -> RedirectResponse:
            """Redirect to the API docs when the frontend is not bundled."""
            return RedirectResponse("/api/docs")

    return application


# --------------------------------------------------------------------------- #
# Module-level helpers                                                         #
# --------------------------------------------------------------------------- #
def _error_code(status_code: int) -> str:
    """Map an HTTP status code onto a stable machine-readable code."""
    return {
        400: "bad_request",
        404: "not_found",
        413: "payload_too_large",
        422: "invalid_input",
        500: "internal_error",
    }.get(status_code, "error")


async def await_file(upload: UploadFile) -> bytes | None:
    """Read an uploaded file, enforcing the size limit.

    Parameters
    ----------
    upload : UploadFile
        The uploaded file handle.

    Returns
    -------
    bytes | None
        The file contents, or *None* when the file is empty or too large.
    """
    contents = await upload.read()
    if not contents:
        return None
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    return contents


def _write_temp_upload(contents: bytes, filename: str) -> str:
    """Persist an uploaded CSV to a temporary file and return its path."""
    import tempfile

    suffix = ".csv" if str(filename).lower().endswith(".csv") else ".csv"
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        handle.write(contents)
    finally:
        handle.close()
    return handle.name


app = create_app()
"""The ASGI application instance used by ``uvicorn``."""
