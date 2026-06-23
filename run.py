#!/usr/bin/env python3
"""
RevInsight Store Staffing Analyzer - Entry Point
=================================================

A professional data-driven staffing optimization tool for retail stores.
Analyzes sales data, forecasts demand, and recommends optimal staffing levels
to maximize gross margin.

Usage:
    python run.py data/RevI-Test.csv
    python run.py data/RevI-Test.csv --output-dir custom_output
    python run.py data/RevI-Test.csv --salary 3.5 --commission 0.06

Author: Data Science Submission for RevInsight
"""

import matplotlib
matplotlib.use('Agg')  # Must be before any other matplotlib imports

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from analyze_store import run_full_analysis


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog='RevInsight Store Staffing Analyzer',
        description=(
            'Analyzes retail store sales data to provide data-driven staffing '
            'recommendations. Reads CSV data with monthly sales by salesperson, '
            'performs EDA, financial analysis, forecasting, and scenario comparison '
            'to recommend optimal staffing levels.'
        ),
        epilog=(
            'Example:\n'
            '  python run.py data/RevI-Test.csv\n'
            '  python run.py data/RevI-Test.csv --output-dir my_output\n'
            '  python run.py data/RevI-Test.csv --salary 3.5 --commission 0.06\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to the CSV file containing monthly sales data by salesperson.'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=(
            'Base directory for output files (default: ./output). '
            'Charts, reports, and analysis results will be saved here.'
        )
    )

    parser.add_argument(
        '--salary',
        type=float,
        default=3.0,
        help='Monthly salary per salesperson in Lakhs INR (default: 3.0).'
    )

    parser.add_argument(
        '--commission',
        type=float,
        default=0.05,
        help='Commission rate as a decimal (default: 0.05 for 5%%).'
    )

    parser.add_argument(
        '--forecast-months',
        type=int,
        default=6,
        help='Number of months to forecast (default: 6).'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose/debug logging output.'
    )

    return parser.parse_args()


def print_banner() -> None:
    """Print application banner."""
    banner = """
+==================================================================+
|                                                                  |
|           RevInsight Store Staffing Analyzer v1.0                 |
|           ------------------------------------------             |
|           Data-Driven Staffing Optimization Tool                 |
|                                                                  |
+==================================================================+
    """
    print(banner)


def main() -> int:
    """Main entry point for the application.
    
    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    args = parse_arguments()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    print_banner()

    # Validate input file
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        logger.error(f"Input file not found: {csv_path}")
        print(f"\n[ERROR] Error: File '{csv_path}' does not exist.")
        print("   Please provide a valid path to your sales CSV file.")
        return 1

    if not csv_path.suffix.lower() == '.csv':
        logger.warning(f"File does not have .csv extension: {csv_path}")
        print(f"\n[WARN] Warning: File '{csv_path}' does not have a .csv extension.")

    # Determine output directory
    project_root = Path(__file__).parent
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = project_root / 'output'

    print(f"[INPUT] Input file:  {csv_path.resolve()}")
    print(f"[OUTPUT] Output dir:  {output_base.resolve()}")
    print(f"[SALARY] Salary:      Rs.{args.salary} Lakhs/month/person")
    print(f"[COMMISSION] Commission:  {args.commission * 100:.1f}%")
    print(f"[FORECAST] Forecast:    {args.forecast_months} months")
    print(f"{'-' * 66}")

    start_time = datetime.now()
    logger.info(f"Starting analysis at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        results = run_full_analysis(
            filepath=str(csv_path),
            output_base_dir=str(output_base),
            salary=args.salary,
            commission=args.commission
        )

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        print(f"\n{'=' * 66}")
        print(f"[OK] Analysis completed successfully in {elapsed:.1f} seconds!")
        print(f"{'=' * 66}")
        print(f"\n[FILES] Output files:")
        print(f"   - Recommendation:    {output_base / 'recommendation.txt'}")
        print(f"   - Analysis Results:  {output_base / 'analysis_results.csv'}")
        print(f"   - Charts:            {output_base / 'charts' / '*.png'}")
        print(f"   - Report:            {project_root / 'reports' / 'Recommendation_Report.pdf'}")
        print()

        # Print quick recommendation summary
        if 'recommendation_text' in results:
            print("-" * 66)
            print("[SUMMARY] QUICK RECOMMENDATION SUMMARY")
            print("-" * 66)
            rec_text = results['recommendation_text']
            # Print first few lines
            lines = rec_text.strip().split('\n')
            for line in lines[:20]:
                print(f"   {line}")
            if len(lines) > 20:
                print(f"   ... (see full recommendation in output/recommendation.txt)")
            print()

        logger.info(f"Analysis completed in {elapsed:.1f}s")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n[ERROR] Error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        print(f"\n[ERROR] Data Error: {e}")
        print("   Please check that your CSV has the expected format.")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error during analysis: {e}")
        print(f"\n[ERROR] Unexpected Error: {e}")
        print("   Check the logs for more details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
