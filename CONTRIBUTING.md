# Contributing to Store Staffing Analyzer

Thank you for your interest in contributing to the **Store Staffing Analyzer**! This document provides guidelines and instructions for setting up your local development environment, running tests, and submitting contributions.

---

## 🛠️ Local Development Setup

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/bishwamanDas/retail-staffing-optimizer.git
cd retail-staffing-optimizer

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies (runtime + dev)
pip install -r requirements-dev.txt
```

---

## 🧪 Running Tests & Quality Checks

We use `pytest` for unit and integration testing.

```bash
# Run test suite
pytest tests/ -v

# Run tests with test coverage report
pytest tests/ --cov=store_staffing --cov-report=term-missing
```

### Linting & Formatting
```bash
# Run ruff check
ruff check store_staffing/ tests/

# Run black formatting check
black --check store_staffing/ tests/
```

---

## 🏗️ Architecture & Coding Standards

1. **Deterministic Calculations**: Keep analytics pipelines free of hard-coded magic numbers. Use configurable parameters through `AnalysisConfig`.
2. **Type Annotations**: All public functions in `store_staffing` should include complete PEP 484 type annotations.
3. **Monetary Convention**: All monetary quantities in code, API, and models are denominated in **Lakhs INR (L)** unless explicitly labeled.
4. **Defensive API Contracts**: The API layer and frontend client must strictly adhere to `docs/api_contract.md`.

---

## 🚀 Submitting Pull Requests

1. Create a feature branch: `git checkout -b feat/your-feature-name`
2. Commit with conventional commit messages (`feat: ...`, `fix: ...`, `docs: ...`, `test: ...`).
3. Ensure all tests pass (`pytest tests/`).
4. Push to your fork and submit a Pull Request against `main`.
