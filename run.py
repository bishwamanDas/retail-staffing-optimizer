#!/usr/bin/env python3
"""Store Staffing Analyzer - command-line entry point.

This is a thin convenience wrapper around :mod:`store_staffing.cli` so that the
familiar invocation keeps working::

    python run.py data/RevI-Test.csv
    python run.py data/RevI-Test.csv --salary 3.5 --commission 0.06
    python run.py serve --port 8000

For the full CLI use ``python -m store_staffing.cli --help``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running the file directly from a clone without installing the package.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from store_staffing.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
