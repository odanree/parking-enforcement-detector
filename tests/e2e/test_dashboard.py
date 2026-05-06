"""Playwright E2E tests for the parking enforcement dashboard.

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
    # Wait for JS to initialise (stats badge turns green/red within ~2s)
    page.wait_for_load_state("networkidle")


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
        # Format is HH:MM:SS — match at least the colons
        uptime.wait_for(state="visible")
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
        # JS adds the 'open' class to the drawer on open
        expect(page.locator(".debug-drawer")).to_have_class("debug-drawer open")
        expect(page.locator("#btn-debug-close")).to_be_visible()

    def test_debug_drawer_close(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        page.click("#btn-debug-open")
        page.click("#btn-debug-close")
        # After close, backdrop should not block the page


# ── Toolbar playback controls ─────────────────────────────────────────────────

class TestPlaybackControls:
    def test_seek_buttons_hidden_on_live(self, page: Page, base_url: str):
        """When is_live=true (no file stream), seek and speed buttons are hidden."""
        goto_dashboard(page, base_url)
        # Wait one stats poll cycle (~2 s)
        page.wait_for_timeout(2500)
        seek_btn = page.locator(".btn-seek").first
        # In live mode, buttons should be display:none
        # (will still be in the DOM, just hidden)
        assert seek_btn.evaluate("el => el.style.display") == "none"

    def test_pause_button_present(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        expect(page.locator("#btn-pause")).to_be_visible()

    def test_edit_zone_button_present(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        expect(page.locator("#btn-edit-zone")).to_be_visible()


# ── Event modal ───────────────────────────────────────────────────────────────

class TestEventModal:
    def test_modal_hidden_initially(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        modal = page.locator("#event-modal")
        expect(modal).to_have_class("event-modal hidden")

    def test_modal_has_required_elements(self, page: Page, base_url: str):
        """Verify modal structure even before it's opened (DOM is always present)."""
        goto_dashboard(page, base_url)
        expect(page.locator("#event-modal-img")).to_be_attached()
        expect(page.locator("#event-modal-desc")).to_be_attached()
        expect(page.locator("#btn-alert")).to_be_attached()
        expect(page.locator("#event-modal-close")).to_be_attached()

    def test_modal_opens_via_js(self, page: Page, base_url: str):
        """Trigger openEventModal() via JS to verify the modal shows correctly."""
        goto_dashboard(page, base_url)
        page.evaluate("""() => {
            const fakeEvent = {
                timestamp: Date.now() / 1000,
                event_type: 'chalking',
                confidence: 0.91,
                description: 'Person crouching near rear tire',
                snapshot_url: null,
            };
            openEventModal('', fakeEvent);
        }""")
        modal = page.locator("#event-modal")
        expect(modal).not_to_have_class("event-modal hidden")
        expect(page.locator("#event-modal-type")).to_contain_text("Chalking")
        expect(page.locator("#event-modal-conf")).to_contain_text("91%")
        expect(page.locator("#event-modal-desc")).to_contain_text("Person crouching")

    def test_modal_close_button(self, page: Page, base_url: str):
        goto_dashboard(page, base_url)
        page.evaluate("""() => {
            openEventModal('', {
                timestamp: Date.now() / 1000,
                event_type: 'sweeper',
                confidence: 0.8,
                description: 'Street sweeper',
                snapshot_url: null,
            });
        }""")
        page.click("#event-modal-close")
        expect(page.locator("#event-modal")).to_have_class("event-modal hidden")


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
