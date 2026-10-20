"""Unit tests for FastAPI security headers and middleware."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from store_staffing.api.main import create_app


@pytest.fixture
def client(project_root):
    """Test client bound to application instance."""
    app = create_app(project_root)
    return TestClient(app)


def test_security_headers_present_on_health_endpoint(client: TestClient):
    """Ensure standard security headers are attached to responses."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


def test_cors_headers_configured(client: TestClient):
    """Ensure CORS configuration permits cross-origin requests."""
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
