# =============================================================================
# RevInsight - Store Staffing Analyzer
# Multi-stage production image: build wheels in a throw-away builder, ship a
# slim runtime that runs as a non-root user.
#
# Build:  docker build -t revinsight:latest .
# Run:    docker run --rm -p 8000:8000 revinsight:latest
# =============================================================================

# ---------------------------------------------------------------------------- #
# Stage 1: builder - compile/install the Python dependencies into a virtualenv
# ---------------------------------------------------------------------------- #
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtualenv that is copied into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Install the third-party dependencies first so that the layer is cached and
# is not invalidated by every source change.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Now the project itself.
COPY pyproject.toml README.md ./
COPY revinsight/ ./revinsight/
COPY run.py ./
RUN pip install --no-cache-dir .

# ---------------------------------------------------------------------------- #
# Stage 2: runtime - minimal image, non-root user, no compilers
# ---------------------------------------------------------------------------- #
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    REVINSIGHT_SALARY=3.0 \
    REVINSIGHT_COMMISSION=0.05 \
    REVINSIGHT_COGS=0.50 \
    REVINSIGHT_FORECAST_MONTHS=6 \
    REVINSIGHT_SEED=42 \
    REVINSIGHT_CORS_ORIGINS=*

# Non-root application user (uid/gid 10001).
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

# Virtualenv with every dependency pre-installed.
COPY --from=builder /opt/venv /opt/venv

# Application source: PYTHONPATH=/app makes this copy take precedence over the
# copy installed inside the virtualenv, so the package root stays /app (the
# API resolves the bundled data/ and web/ folders relative to it).
COPY --from=builder /build/revinsight /app/revinsight
COPY --from=builder /build/run.py /app/run.py

# Static dashboard and the bundled sample datasets.
COPY web/ /app/web/
COPY data/ /app/data/

# Writable runtime directories (analysis runs, generated artefacts).
RUN mkdir -p /app/runs /app/output /app/reports /tmp/matplotlib \
    && chown -R appuser:appuser /app /tmp/matplotlib

USER appuser

EXPOSE 8000

# Health check uses python (curl is not installed in the slim image).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4); sys.exit(0 if r.status == 200 else 1)"]

VOLUME ["/app/runs"]

CMD ["uvicorn", "revinsight.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
