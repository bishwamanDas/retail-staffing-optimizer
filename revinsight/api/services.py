"""Service layer for the RevInsight REST API.

Owns the analysis run registry: dataset discovery, run execution, cached
artefact paths and the what-if simulator.  The FastAPI layer in
:mod:`revinsight.api.main` only handles HTTP concerns and delegates all
business logic here.
"""

from __future__ import annotations

import logging
import shutil
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import DEFAULT_CONFIG, AnalysisConfig
from ..pipeline import run_analysis, simulate_scenarios
from ..utils import load_and_validate_data, round_floats, to_jsonable

logger = logging.getLogger(__name__)

__all__ = ["AnalysisService", "get_service", "public_payload"]

MAX_CACHED_RUNS: int = 24
"""Maximum number of analysis runs kept in memory."""


def public_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip internal state and convert a run payload into JSON-safe data.

    The pipeline keeps private state (dataframes, configuration objects)
    under the ``_internal`` key so the simulator can reuse it.  That state
    must never cross the HTTP boundary, and every remaining value has to
    be a JSON primitive.

    Parameters
    ----------
    payload : dict
        A raw analysis payload.

    Returns
    -------
    dict
        The client-facing payload.
    """
    visible = {key: value for key, value in payload.items() if not key.startswith("_")}
    return round_floats(to_jsonable(visible), 4)


class AnalysisService:
    """Registry and executor for analysis runs.

    Parameters
    ----------
    project_root : str or Path
        Root of the project; used to locate ``data/`` and to store runs.
    max_runs : int, optional
        Maximum number of runs retained in memory (oldest evicted first).
    """

    def __init__(self, project_root: str | Path, max_runs: int = MAX_CACHED_RUNS) -> None:
        self.project_root = Path(project_root).resolve()
        self.data_dir = self.project_root / "data"
        self.runs_dir = self.project_root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.max_runs = max_runs
        self._runs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._artifact_lock: set[str] = set()
        self.started_at = datetime.now(timezone.utc)

    # ------------------------------------------------------------------ #
    # Datasets                                                            #
    # ------------------------------------------------------------------ #
    def list_datasets(self) -> list[dict[str, Any]]:
        """Discover the CSV datasets shipped with the project.

        Returns
        -------
        list of dict
            Each entry has ``id``, ``label``, ``filename``, ``rows``,
            ``columns``, ``months``, ``salespersons`` and ``description``.
        """
        datasets: list[dict[str, Any]] = []
        if not self.data_dir.exists():
            return datasets

        # Sort by stem, not by full path: "RevI-Test" must come before
        # "RevI-Test-Large", which a path sort breaks on the extension.
        paths = sorted(self.data_dir.glob("*.csv"), key=lambda p: p.stem.lower())

        for path in paths:
            entry: dict[str, Any] = {
                "id": path.stem,
                "label": path.stem,
                "filename": path.name,
                "path": str(path),
                "rows": 0,
                "columns": 0,
                "months": 0,
                "salespersons": 0,
                "description": "Uploaded or bundled dataset",
            }
            try:
                df = load_and_validate_data(path)
                entry["rows"] = int(df.shape[0])
                entry["columns"] = int(df.shape[1])
                entry["months"] = int(df["Month"].nunique())
                entry["salespersons"] = int(
                    len([c for c in df.columns if str(c).lower().startswith("sale sp")])
                )
                entry["description"] = (
                    f"{entry['months']} months of history across "
                    f"{entry['salespersons']} salespersons"
                )
            except Exception as exc:  # noqa: BLE001
                entry["description"] = f"Could not be parsed: {exc}"
            datasets.append(entry)
        return datasets

    def resolve_dataset(self, dataset_id: str | None) -> Path | None:
        """Return the path of a bundled dataset by id.

        Parameters
        ----------
        dataset_id : str | None
            Dataset identifier (the CSV stem).

        Returns
        -------
        Path | None
            The resolved path, or *None* when no dataset is available.
        """
        if dataset_id:
            candidate = self.data_dir / f"{dataset_id}.csv"
            if candidate.exists():
                return candidate
            # Fall back to a case-insensitive stem match.
            for path in self.data_dir.glob("*.csv"):
                if path.stem.lower() == str(dataset_id).lower():
                    return path
            return None

        # No dataset requested: the server owns the default so the client
        # never has to hard-code a file name.
        datasets = self.list_datasets()
        if not datasets:
            return None
        return self.data_dir / str(datasets[0]["filename"])

    # ------------------------------------------------------------------ #
    # Runs                                                                #
    # ------------------------------------------------------------------ #
    def create_run(
        self,
        filepath: str | None = None,
        df: pd.DataFrame | None = None,
        config: AnalysisConfig | None = None,
        start_month: int = 1,
        generate_artifacts: bool = False,
    ) -> dict[str, Any]:
        """Execute a new analysis and register it.

        Parameters
        ----------
        filepath : str, optional
            Path to a CSV on disk.
        df : pd.DataFrame, optional
            Uploaded dataframe (takes precedence over *filepath*).
        config : AnalysisConfig, optional
            Business assumptions.
        start_month : int, optional
            Calendar month corresponding to dataset month 1.
        generate_artifacts : bool, optional
            Render the charts and reports immediately.  The API leaves
            this off so the response is fast; the report is produced on
            demand by :meth:`ensure_artifacts`.

        Returns
        -------
        dict
            The analysis payload.

        Raises
        ------
        ValueError
            If the dataset is invalid.
        """
        config = config or DEFAULT_CONFIG
        run_id = uuid.uuid4().hex[:12]
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        payload = run_analysis(
            filepath=filepath,
            df=df,
            output_base_dir=str(run_dir),
            config=config,
            start_month=start_month,
            generate_artifacts=generate_artifacts,
            run_id=run_id,
        )

        with self._lock:
            self._runs[run_id] = payload
            while len(self._runs) > self.max_runs:
                oldest, _ = self._runs.popitem(last=False)
                logger.info("Evicting oldest cached run %s", oldest)
                shutil.rmtree(self.runs_dir / oldest, ignore_errors=True)

        return payload

    def ensure_artifacts(self, run_id: str) -> dict[str, Any]:
        """Render the charts, PDF, HTML and CSV for a run on demand.

        The dashboard draws its own charts, so the downloadable artefacts
        are produced only when someone asks for them.  The result is
        cached on disk, so a second request is instant.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        dict
            The artefact paths.

        Raises
        ------
        KeyError
            If the run is unknown.
        """
        payload = self.get_run(run_id)
        if payload is None:
            raise KeyError(run_id)

        run_dir = self._run_dir(run_id)
        with self._lock:
            self._artifact_lock.add(run_id)

        try:
            existing = payload.get("_internal", {}).get("artifact_paths", {})
            if existing.get("pdf") and Path(existing["pdf"]).exists():
                return existing

            logger.info("Rendering artefacts for run %s", run_id)
            from ..pipeline import render_artifacts

            render_artifacts(payload, str(run_dir))
            return payload.get("_internal", {}).get("artifact_paths", {})
        finally:
            with self._lock:
                self._artifact_lock.discard(run_id)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return a cached run payload, or *None* when unknown."""
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        """Return a summary of every cached run, newest first."""
        with self._lock:
            runs = list(self._runs.values())

        summaries = []
        for payload in runs:
            executive = payload.get("executive", {})
            summaries.append(
                {
                    "run_id": payload.get("run_id"),
                    "dataset": payload.get("dataset", {}).get("name"),
                    "created_at": payload.get("created_at"),
                    "recommendation": executive.get("recommendation"),
                    "confidence": executive.get("confidence"),
                    "current_employees": payload.get("kpis", {}).get("current_employees"),
                }
            )
        summaries.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return summaries

    def delete_run(self, run_id: str) -> bool:
        """Remove a cached run and its artefacts.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        bool
            ``True`` when a run was removed.
        """
        with self._lock:
            existed = self._runs.pop(run_id, None) is not None
        shutil.rmtree(self.runs_dir / run_id, ignore_errors=True)
        return existed

    # ------------------------------------------------------------------ #
    # Artefacts                                                           #
    # ------------------------------------------------------------------ #
    def _run_dir(self, run_id: str) -> Path:
        """Return the artefact directory for *run_id*."""
        return self.runs_dir / run_id

    def chart_path(self, run_id: str, chart_name: str) -> Path | None:
        """Return the path of a generated chart, or *None* when missing."""
        safe = Path(str(chart_name)).name
        if not safe.endswith(".png"):
            safe = f"{safe}.png"
        candidate = self._run_dir(run_id) / "charts" / safe
        return candidate if candidate.exists() else None

    def report_path(self, run_id: str, extension: str = "pdf") -> Path | None:
        """Return the path of the generated report, or *None*."""
        candidate = self._run_dir(run_id) / "reports" / f"Recommendation_Report.{extension}"
        return candidate if candidate.exists() else None

    def csv_path(self, run_id: str) -> Path | None:
        """Return the path of the exported results CSV, or *None*."""
        candidate = self._run_dir(run_id) / "analysis_results.csv"
        return candidate if candidate.exists() else None

    # ------------------------------------------------------------------ #
    # Simulation                                                          #
    # ------------------------------------------------------------------ #
    def simulate(
        self,
        run_id: str,
        salary: float | None = None,
        commission: float | None = None,
        horizon: int | None = None,
        decision: str = "auto",
        sales_shock_pct: float = 0.0,
    ) -> dict[str, Any] | None:
        """Run the what-if simulator against a cached run.

        Returns
        -------
        dict | None
            The simulation payload, or *None* when the run is unknown.
        """
        payload = self.get_run(run_id)
        if payload is None:
            return None
        return simulate_scenarios(
            payload,
            salary=salary,
            commission=commission,
            horizon=horizon,
            decision=decision,
            sales_shock_pct=sales_shock_pct,
        )

    # ------------------------------------------------------------------ #
    # Meta                                                                #
    # ------------------------------------------------------------------ #
    def uptime_seconds(self) -> float:
        """Return the number of seconds since the service was created."""
        return (datetime.now(timezone.utc) - self.started_at).total_seconds()

    def run_count(self) -> int:
        """Return the number of cached runs."""
        with self._lock:
            return len(self._runs)


_SERVICE: AnalysisService | None = None
_SERVICE_LOCK = threading.Lock()


def get_service(project_root: str | Path | None = None) -> AnalysisService:
    """Return the process-wide :class:`AnalysisService` singleton.

    Parameters
    ----------
    project_root : str | Path, optional
        Project root override; defaults to the current working directory.

    Returns
    -------
    AnalysisService
        The shared service instance.
    """
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            root = Path(project_root) if project_root else Path.cwd()
            _SERVICE = AnalysisService(root)
        return _SERVICE
