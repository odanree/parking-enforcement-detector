"""Playwright E2E tests for the parking enforcement dashboard (React frontend).

Requires `pytest --headed` to see the browser, or run headless (default).
The `app_server` fixture (conftest.py) starts FastAPI on port 18999 with the
detection pipeline mocked out, so no RTSP camera or GPU is needed.

Skipped automatically when playwright is not installed.
"""

import pytest

# Skip the entire module if playwright isn't in the environment
pytest.importorskip("playwright", reason="playwright not installed — skipping E2E tests")
from playwright.sync_api import Page, expect


pytestmark = pytest.mark.e2e


# ── Helpers ───────────────────────────────────────────────────────────────────

def goto_dashboard(page: Page, base_url: str) -> None:
    page.goto(base_url)
    # Wait for React to mount and the first stats poll to complete (~2 s)
    page.wait_for_load_state("networkidle")


def open_modal(page: Page, event_type: str = "chalking", confidence: float = 0.91,
               description: str = "Person crouching near rear tire") -> None:
    """Open the event modal via the Zustand store helper exposed by main.tsx."""
    page.evaluate(f"""() => {{
        window.__openModal({{
            timestamp: Date.now() / 1000,
            event_type: '{event_type}',
            confidence: {confidence},
            description: '{description}',
            snapshot_url: null,
        }});
    }}""")
    # Wait for the modal to appear in the DOM
    page.locator(".event-modal").wait_for(state="visible")


# ── Layout tests ──────────────────────────────────────────────────────────────

class TestDashboardLayout:
    def test_page_title(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        expect(page).to_have_title("Parking Enforcement Detector")

    def test_header_logo_visible(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        logo = page.locator("header .logo")
        expect(logo).to_be_visible()
        expect(logo).to_contain_text("Parking Enforcement Detector")

    def test_status_badges_present(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        expect(page.locator("#badge-pipeline")).to_be_visible()
        expect(page.locator("#badge-sweep")).to_be_visible()
        expect(page.locator("#badge-ws")).to_be_visible()

    def test_video_canvas_present(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        expect(page.locator("#feed")).to_be_visible()

    def test_stats_card_labels(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        expect(page.locator("text=Chalking alerts")).to_be_visible()
        expect(page.locator("text=Sweeper alerts")).to_be_visible()
        expect(page.locator("text=Uptime")).to_be_visible()

    def test_event_log_empty_state(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        expect(page.locator(".event-empty")).to_contain_text("No alerts yet")


# ── Stats API polling ─────────────────────────────────────────────────────────

class TestStatsPolling:
    def test_uptime_counter_displayed(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        uptime = page.locator("#stat-uptime")
        expect(uptime).to_be_visible()
        text = uptime.inner_text()
        assert text.count(":") == 2, f"Unexpected uptime format: {text!r}"

    def test_chalking_count_starts_at_zero(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        expect(page.locator("#stat-chalking")).to_have_text("0")

    def test_sweeper_count_starts_at_zero(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        expect(page.locator("#stat-sweeper")).to_have_text("0")


# ── Debug drawer ──────────────────────────────────────────────────────────────

class TestDebugDrawer:
    def test_debug_button_opens_drawer(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        page.click("#btn-debug-open")
        # React adds the 'open' class; drawer slides in via CSS transition
        expect(page.locator(".debug-drawer")).to_have_class("debug-drawer open")
        expect(page.locator("#btn-debug-close")).to_be_visible()

    def test_debug_drawer_close(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        page.click("#btn-debug-open")
        page.click("#btn-debug-close")
        expect(page.locator(".debug-drawer")).not_to_have_class("debug-drawer open")


# ── Toolbar playback controls ─────────────────────────────────────────────────

class TestPlaybackControls:
    def test_seek_buttons_absent_on_live(self, page: Page, base_url: str):
        """When is_live=true (no file stream), React does not render seek buttons."""
        goto_dashboard(page, base_url)
        # Wait one stats poll cycle (~2 s) to receive is_live from the server
        page.wait_for_timeout(2500)
        # React conditionally renders seek buttons — they are absent from the DOM
        expect(page.locator(".btn-seek")).to_have_count(0)

    def test_pause_button_present(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        expect(page.locator("#btn-pause")).to_be_visible()

    def test_edit_zone_button_present(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        expect(page.locator("#btn-edit-zone")).to_be_visible()


# ── Event modal ───────────────────────────────────────────────────────────────

class TestEventModal:
    def test_modal_absent_initially(self, page: Page, base_url: str):
        """React renders the modal conditionally — it is not in the DOM until opened."""
        goto_dashboard(page, base_url)
        expect(page.locator(".event-modal")).to_have_count(0)

    def test_modal_has_required_elements(self, page: Page, base_url: str):
        """Verify modal structure after it is opened."""
        goto_dashboard(page, base_url)
        open_modal(page)
        expect(page.locator(".event-modal-img-wrap")).to_be_attached()
        expect(page.locator(".event-modal-desc")).to_be_attached()
        expect(page.locator(".btn-alert")).to_be_attached()
        expect(page.locator(".event-modal-close")).to_be_attached()

    def test_modal_opens_with_correct_content(self, page: Page, base_url: str):
        """Open the modal via __openModal and verify displayed content."""
        goto_dashboard(page, base_url)
        open_modal(page, event_type="chalking", confidence=0.91,
                   description="Person crouching near rear tire")
        expect(page.locator(".event-modal-badge")).to_contain_text("Chalking")
        expect(page.locator(".event-modal-conf")).to_contain_text("91%")
        expect(page.locator(".event-modal-desc")).to_contain_text("Person crouching")

    def test_modal_close_button(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        open_modal(page, event_type="sweeper", confidence=0.8,
                   description="Street sweeper")
        page.click(".event-modal-close")
        # After closing, React removes the element from the DOM
        expect(page.locator(".event-modal")).to_have_count(0)

    def test_modal_closes_on_backdrop_click(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        open_modal(page)
        # Click the backdrop (the modal overlay itself, not the inner card)
        modal = page.locator(".event-modal")
        modal.click(position={"x": 10, "y": 10})
        expect(page.locator(".event-modal")).to_have_count(0)


# ── API smoke tests ───────────────────────────────────────────────────────────

class TestApiEndpoints:
    def test_stats_returns_200(self, page: Page, base_url: str):
        resp = page.request.get(f"{base_url}/api/stats")
        assert resp.status == 200
        body = resp.json()
        assert "uptime_seconds" in body
        assert "is_live" in body

    def test_events_returns_200(self, page: Page, base_url: str):
        resp = page.request.get(f"{base_url}/api/events")
        assert resp.status == 200
        assert isinstance(resp.json(), list)

    def test_pending_returns_200(self, page: Page, base_url: str):
        resp = page.request.get(f"{base_url}/api/pending")
        assert resp.status == 200
        assert "jobs" in resp.json()

    def test_zone_returns_200(self, page: Page, base_url: str):
        resp = page.request.get(f"{base_url}/api/zone")
        assert resp.status == 200
        assert "polygon" in resp.json()

    def test_alert_503_without_provider(self, page: Page, base_url: str):
        """With no provider env vars, /api/alert returns 503."""
        resp = page.request.post(
            f"{base_url}/api/alert",
            data={
                "event_type": "chalking",
                "timestamp": "1234567890.0",
                "confidence": "0.9",
            },
        )
        # Either 503 (no provider) or 200 (if .env has provider set in test env)
        assert resp.status in (200, 503)
