"""Shared fixtures for the test suite.

Heavy packages (anthropic, ultralytics, cv2) are stubbed when not installed
so that unit tests run on any Python that has just pytest + stdlib.
E2E tests require the full venv and are skipped automatically if
uvicorn/fastapi/playwright are absent.
"""

import importlib.util
import sys
from unittest.mock import MagicMock

# ── Stub missing heavy packages before any src imports ────────────────────────
# Unit tests only exercise pure-Python logic and never call these at runtime.
# The stubs let them import without the full GPU / AI SDK environment.
for _pkg in ("anthropic", "ultralytics", "cv2", "numpy"):
    if importlib.util.find_spec(_pkg) is None:
        sys.modules.setdefault(_pkg, MagicMock())
# numpy sub-modules that ultralytics references
if isinstance(sys.modules.get("numpy"), MagicMock):
    sys.modules.setdefault("numpy.typing", MagicMock())

# ── Server-side imports (needed only for E2E) ─────────────────────────────────
import threading
import time

import pytest

try:
    import httpx
    import uvicorn
    from unittest.mock import patch

    _HAS_SERVER_DEPS = True
except ImportError:
    _HAS_SERVER_DEPS = False


@pytest.fixture(scope="session")
def app_server():
    """Start the FastAPI app on a fixed test port with pipeline mocked out.

    Patches src.pipeline.run so the RTSP/video pipeline never launches —
    tests only exercise the web layer.

    Skipped automatically when uvicorn/fastapi are not installed.
    """
    if not _HAS_SERVER_DEPS:
        pytest.skip("server deps (uvicorn, httpx) not installed — skipping E2E")

    port = 18999
    base_url = f"http://127.0.0.1:{port}"

    with patch("src.pipeline.run", lambda state: None):
        config = uvicorn.Config(
            "src.web.app:app",
            host="127.0.0.1",
            port=port,
            log_level="error",
        )
        server = uvicorn.Server(config)
        t = threading.Thread(target=server.run, daemon=True)
        t.start()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                httpx.get(f"{base_url}/api/stats", timeout=0.5)
                break
            except Exception:
                time.sleep(0.1)

        yield base_url

        server.should_exit = True


@pytest.fixture(scope="session")
def base_url(app_server):
    """Playwright-compatible base_url pointing at the test server."""
    return app_server
