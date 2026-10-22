"""Command-line interface for the Store Staffing Analyzer staffing analyzer.

Two sub-commands are available::

    python -m store_staffing.cli run data/RevI-Test.csv
    python -m store_staffing.cli serve --port 8000

For backwards compatibility the ``run`` sub-command is implied when the
first argument is a path to a dataset::

    python run.py data/RevI-Test.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import config_from_env
from .pipeline import run_analysis

logger = logging.getLogger(__name__)

__all__ = ["main", "build_parser"]

BANNER = r"""
+==================================================================+
|                                                                  |
|      Store Staffing Analyzer  v2.1.0                             |
| ------------------------------------------                       |
| Data-Driven Staffing Optimisation Platform                       |
|                                                                  |
+==================================================================+
"""


def _setup_logging(verbose: bool = False) -> None:
    """Configure root logging for the CLI."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _safe_print(text: str) -> None:
    """Print *text* without ever raising a Unicode encoding error.

    Windows consoles frequently default to a legacy code page, which makes
    the Rupee sign and typographic characters crash ``print``.  Unencodable
    characters are replaced instead of raising.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="store_staffing",
        description=(
            "Store Staffing Analyzer analyses retail store sales data to provide data-driven "
            "staffing recommendations: EDA, forecasting, financial modelling, "
            "scenario comparison and report generation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run.py data/RevI-Test.csv\n"
            "  python run.py data/RevI-Test.csv --salary 3.5 --commission 0.06\n"
            "  python run.py data/RevI-Test.csv --output-dir my_output\n"
            "  python -m store_staffing.cli serve --port 8000\n"
        ),
    )
    parser.add_argument("--version", action="version", version="store_staffing 2.1.0")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose/debug logging."
    )

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the analysis pipeline on a CSV.")
    run_parser.add_argument("csv_file", help="Path to the sales CSV file.")
    run_parser.add_argument(
        "--output-dir", default=None, help="Base directory for generated outputs."
    )
    run_parser.add_argument(
        "--salary", type=float, default=3.0, help="Monthly salary per salesperson (Lakhs INR)."
    )
    run_parser.add_argument(
        "--commission", type=float, default=0.05, help="Commission rate as a decimal (0.05 = 5%%)."
    )
    run_parser.add_argument(
        "--forecast-months", type=int, default=6, help="Number of months to forecast."
    )
    run_parser.add_argument(
        "--start-month",
        type=int,
        default=1,
        help="Calendar month (1-12) that dataset month 1 corresponds to.",
    )
    run_parser.add_argument(
        "--no-charts", action="store_true", help="Skip chart and report generation."
    )
    run_parser.add_argument(
        "--json", dest="as_json", action="store_true", help="Print the result payload as JSON."
    )
    run_parser.add_argument(
        "--reports-dir",
        default=None,
        help="Directory for the PDF/HTML reports (defaults to <output-dir>/reports).",
    )

    serve_parser = subparsers.add_parser("serve", help="Start the web dashboard and REST API.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload.")
    serve_parser.add_argument(
        "--log-level", default="info", help="uvicorn log level (debug|info|warning|error)."
    )

    return parser


def _run_analysis_command(args: argparse.Namespace) -> int:
    """Execute the ``run`` sub-command.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    _safe_print(BANNER)

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        _safe_print(f"\n[ERROR] Input file not found: {csv_path}")
        return 1

    try:
        config = config_from_env(
            salary_per_employee=args.salary,
            commission_rate=args.commission,
            forecast_months=args.forecast_months,
        )
    except ValueError as exc:
        _safe_print(f"\n[ERROR] Invalid configuration: {exc}")
        return 1

    project_root = Path(__file__).resolve().parents[1]
    output_base = Path(args.output_dir) if args.output_dir else project_root / "output"
    reports_base = Path(args.reports_dir) if args.reports_dir else project_root / "reports"

    _safe_print(f"[INPUT]  Input file : {csv_path.resolve()}")
    _safe_print(f"[OUTPUT] Output dir : {output_base.resolve()}")
    _safe_print(f"[SALARY] Salary     : Rs.{config.salary_per_employee} Lakhs/person/month")
    _safe_print(f"[COMM]   Commission : {config.commission_rate * 100:.1f} %")
    _safe_print(f"[FCST]   Horizon   : {config.forecast_months} months")
    _safe_print("-" * 66)

    try:
        results = run_analysis(
            filepath=str(csv_path),
            output_base_dir=str(output_base),
            config=config,
            start_month=args.start_month,
            generate_artifacts=not args.no_charts,
            reports_base_dir=str(reports_base),
        )
    except FileNotFoundError as exc:
        _safe_print(f"\n[ERROR] {exc}")
        return 1
    except ValueError as exc:
        _safe_print(f"\n[ERROR] Data error: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Analysis failed: %s", exc)
        _safe_print(f"\n[ERROR] Unexpected error: {exc}")
        return 1

    elapsed = results.get("_internal", {}).get("elapsed_seconds", 0.0)
    _safe_print("=" * 66)
    _safe_print(f"[OK] Analysis completed in {elapsed} seconds")
    _safe_print("=" * 66)

    if args.as_json:
        printable = {k: v for k, v in results.items() if not k.startswith("_")}
        _safe_print(json.dumps(printable, indent=2, default=str))
        return 0

    _safe_print("\n[SUMMARY] RECOMMENDATION")
    _safe_print("-" * 66)
    executive = results.get("executive", {})
    kpis = results.get("kpis", {})
    _safe_print(f"  Current staff count : {kpis.get('current_employees')}")
    _safe_print(f"  Recommended action  : {executive.get('recommendation')}")
    _safe_print(f"  Confidence          : {executive.get('confidence')}")
    _safe_print(f"  Avg monthly sales   : Rs.{kpis.get('avg_monthly_sales', 0):.2f} L")
    _safe_print(f"  Avg monthly margin  : Rs.{kpis.get('avg_monthly_margin', 0):.2f} L")
    _safe_print(f"  Forecast model      : {kpis.get('best_forecast_model')}")
    _safe_print(f"  Payback period      : {kpis.get('payback_months')} months")
    _safe_print("-" * 66)

    artifacts = results.get("_internal", {}).get("artifact_paths", {})
    if artifacts:
        _safe_print("\n[FILES] Generated artefacts:")
        for label, key in (
            ("Recommendation", "recommendation"),
            ("Analysis results", "csv"),
            ("PDF report", "pdf"),
            ("HTML report", "html"),
            ("Charts", "charts"),
        ):
            path = artifacts.get(key)
            if path:
                if key == "charts":
                    _safe_print(f"   - {label:18s}: {path}\\*.png")
                else:
                    _safe_print(f"   - {label:18s}: {path}")
    _safe_print("")

    return 0


def _serve_command(args: argparse.Namespace) -> int:
    """Execute the ``serve`` sub-command.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    try:
        import uvicorn
    except ImportError:
        _safe_print("[ERROR] uvicorn is not installed. Run: pip install -r requirements-web.txt")
        return 1

    project_root = Path(__file__).resolve().parents[1]
    _safe_print(BANNER)
    _safe_print(f"[SERVE] Dashboard : http://{args.host}:{args.port}/")
    _safe_print(f"[SERVE] API docs  : http://{args.host}:{args.port}/api/docs")
    _safe_print("-" * 66)

    uvicorn.run(
        "store_staffing.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        app_dir=str(project_root),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code.
    """
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    _setup_logging(verbose="--verbose" in raw_args or "-v" in raw_args)

    parser = build_parser()

    # Legacy form: `python run.py data.csv` -- the `run` sub-command is
    # implied when the first positional argument is not a known command.
    known_commands = {"run", "serve"}
    first_positional = next((arg for arg in raw_args if not arg.startswith("-")), None)
    if first_positional is not None and first_positional not in known_commands:
        raw_args = ["run"] + raw_args

    args = parser.parse_args(raw_args)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "run":
        return _run_analysis_command(args)
    if args.command == "serve":
        return _serve_command(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
