"""Fisheye camera calibration — fully automatic capture.

Usage:
    python scripts/calibrate_camera.py

No keyboard needed.  The script:
  1. Connects to RTSP_URL (from .env)
  2. Auto-captures when checkerboard corners are detected and the board is
     held still for ~1 second
  3. Waits CAPTURE_COOLDOWN seconds so you can reposition
  4. Saves calibration after TARGET_CAPTURES good frames
  5. Writes config/camera_calibration.yaml and exits

Manual overrides (optional, if you ARE at the keyboard):
    SPACE  — force capture now (if corners found)
    S      — save immediately with current data (needs ≥ MIN_CAPTURES)
    Q/ESC  — quit without saving

Checkerboard:
    Expects an 8×6 inner-corner grid (9×7 squares).
    Display it on an iPad/laptop at full brightness.
    Hold it at varied positions and angles across the full camera view.
    Slow deliberate movements work better than fast sweeping.
"""

from __future__ import annotations

import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv
import os

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
BOARD_W = 8          # inner corners wide
BOARD_H = 6          # inner corners tall
SQUARE_MM = 30.0     # physical square size (only affects scale)
MIN_CAPTURES = 6     # minimum before save is allowed
TARGET_CAPTURES = 20 # auto-saves after this many good captures
OUT_W, OUT_H = 1280, 720
OUT_YAML = Path("config/camera_calibration.yaml")

STABLE_FRAMES   = 8    # frames corners must be still before auto-capture
STABLE_PX       = 6.0  # max mean corner movement (px) to count as "still"
CAPTURE_COOLDOWN = 4.0  # seconds between captures (time to reposition)
# ─────────────────────────────────────────────────────────────────────────────

STREAM_PORT = 8765   # phone opens http://<your-ip>:8765

# ── MJPEG stream server ───────────────────────────────────────────────────────
_latest_jpeg: bytes = b""
_jpeg_lock = threading.Lock()


def _set_frame(frame: np.ndarray) -> None:
    global _latest_jpeg
    # Downscale for streaming — faster transfer over ngrok/mobile
    small = cv2.resize(frame, (640, 360))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if ok:
        with _jpeg_lock:
            _latest_jpeg = buf.tobytes()


class _SnapshotHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def handle_error(self, *args):
        pass  # suppress connection reset noise

    def do_GET(self):
        if self.path == "/":
            html = (
                b"<html><head><meta name='viewport' content='width=device-width,initial-scale=1'>"
                b"<style>*{margin:0;padding:0}body{background:#000}"
                b"img{width:100vw;height:100vh;object-fit:contain}</style></head>"
                b"<body><img src='/stream'></body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with _jpeg_lock:
                        jpg = _latest_jpeg
                    if jpg:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + jpg + b"\r\n"
                        )
                    time.sleep(0.05)  # 20 fps
            except (BrokenPipeError, ConnectionResetError):
                pass


class _QuietServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    def handle_error(self, request, client_address):
        pass


def _start_stream_server() -> None:
    server = _QuietServer(("0.0.0.0", STREAM_PORT), _SnapshotHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Calibration stream → http://192.168.1.138:{STREAM_PORT}  (open on phone)")


# ─────────────────────────────────────────────────────────────────────────────
BOARD_SIZE = (BOARD_W, BOARD_H)
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

_objp = np.zeros((BOARD_W * BOARD_H, 1, 3), dtype=np.float64)
_objp[:, 0, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_MM


class _LiveCapture:
    """Continuously drains the RTSP buffer in a background thread.
    get_frame() always returns the most recent frame with no lag."""

    def __init__(self, src: str) -> None:
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            sys.exit(f"Cannot open stream: {src}")
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stopped = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        while not self._stopped:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self) -> None:
        self._stopped = True
        self._cap.release()


def _open_stream() -> _LiveCapture:
    src = os.getenv("VIDEO_PATH") or os.getenv("RTSP_URL", "")
    if not src:
        sys.exit("Set RTSP_URL or VIDEO_PATH in .env")
    cap = _LiveCapture(src)
    print(f"Stream opened: {src}")
    return cap


def _find_corners(gray: np.ndarray):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, flags)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)


def _calibrate(obj_pts, img_pts, size):
    K = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
    )
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in obj_pts]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in obj_pts]
    rms, K, D, _, _ = cv2.fisheye.calibrate(
        obj_pts, img_pts, size, K, D, rvecs, tvecs, flags, CRITERIA
    )
    return K, D, rms


def _save_yaml(K: np.ndarray, D: np.ndarray, rms: float) -> None:
    data = {
        "camera_matrix": {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        },
        "distortion_coefficients": [float(v) for v in D.flatten()],
        "output_width": OUT_W,
        "output_height": OUT_H,
        "alpha": 0.3,
        "calibration_rms": round(rms, 4),
        "board": {"width": BOARD_W, "height": BOARD_H, "square_mm": SQUARE_MM},
    }
    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_YAML, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False)
    print(f"\n✓ Saved → {OUT_YAML}")
    print(f"  RMS reprojection error: {rms:.4f} px  (< 1.5 is good)")
    print("  Set UNDISTORT_FRAME=true in .env to enable undistortion.")


def _draw_hud(
    img: np.ndarray,
    n_captured: int,
    rms: float | None,
    state: str,          # "searching" | "stabilising" | "cooldown" | "done"
    stable_pct: float,   # 0.0–1.0 progress toward stable capture
    cooldown_remaining: float,
) -> None:
    h, w = img.shape[:2]

    # Dark semi-transparent banner at top
    banner_h = 70
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    # Progress bar background
    bar_y, bar_h = banner_h + 6, 8
    cv2.rectangle(img, (0, bar_y), (w, bar_y + bar_h), (40, 40, 40), -1)

    if state == "stabilising":
        bar_w = int(w * stable_pct)
        cv2.rectangle(img, (0, bar_y), (bar_w, bar_y + bar_h), (0, 210, 80), -1)
        status = f"Hold still...  {int(stable_pct * 100)}%"
        color = (0, 210, 80)
    elif state == "cooldown":
        secs_left = cooldown_remaining
        bar_w = int(w * (1.0 - secs_left / CAPTURE_COOLDOWN))
        cv2.rectangle(img, (0, bar_y), (bar_w, bar_y + bar_h), (0, 160, 255), -1)
        status = f"Captured!  Reposition...  {secs_left:.1f}s"
        color = (0, 200, 255)
    elif state == "done":
        cv2.rectangle(img, (0, bar_y), (w, bar_y + bar_h), (0, 210, 80), -1)
        status = "Done! Saving calibration..."
        color = (0, 210, 80)
    else:
        status = "Searching for checkerboard..."
        color = (100, 100, 100)

    # Capture counter (large)
    counter_text = f"{n_captured} / {TARGET_CAPTURES}"
    cv2.putText(img, counter_text, (16, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)

    # Status text
    cv2.putText(img, status, (220, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # RMS
    if rms is not None:
        rms_color = (0, 210, 80) if rms < 1.5 else (0, 100, 255)
        cv2.putText(img, f"RMS {rms:.3f}px", (w - 190, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, rms_color, 1, cv2.LINE_AA)


def main() -> None:
    cap = _open_stream()
    print(f"\nBoard: {BOARD_W}×{BOARD_H}  |  target: {TARGET_CAPTURES} captures")
    print("Hold iPad up to camera, move slowly to different positions.\n")
    print("Optional keys: SPACE=force capture  S=save now  Q=quit\n")

    _start_stream_server()

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", OUT_W, OUT_H)

    obj_pts: list = []
    img_pts: list = []
    last_K = last_D = None
    last_rms: float | None = None

    stable_count = 0
    prev_corners = None
    last_capture_time = -CAPTURE_COOLDOWN  # allow immediate first capture
    in_cooldown = False

    while True:
        frame = cap.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.resize(frame, (OUT_W, OUT_H))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        now   = time.monotonic()
        n     = len(obj_pts)

        corners = _find_corners(gray)
        display = frame.copy()

        cooldown_remaining = max(0.0, last_capture_time + CAPTURE_COOLDOWN - now)
        in_cooldown = cooldown_remaining > 0

        do_capture = False

        if corners is not None and not in_cooldown:
            cv2.drawChessboardCorners(display, BOARD_SIZE, corners, True)

            # Stability check
            if prev_corners is not None:
                movement = float(np.mean(np.abs(corners - prev_corners)))
                if movement < STABLE_PX:
                    stable_count += 1
                else:
                    stable_count = 0
            prev_corners = corners

            if stable_count >= STABLE_FRAMES:
                do_capture = True
                stable_pct = 1.0
                state = "stabilising"
            else:
                stable_pct = stable_count / STABLE_FRAMES
                state = "stabilising"
        elif in_cooldown:
            state = "cooldown"
            stable_pct = 0.0
            stable_count = 0
            prev_corners = None
        else:
            state = "searching"
            stable_pct = 0.0
            stable_count = 0
            prev_corners = None

        # Auto-capture
        if do_capture:
            obj_pts.append(_objp.copy())
            img_pts.append(corners)
            last_capture_time = now
            stable_count = 0
            prev_corners = None
            n = len(obj_pts)
            print(f"  ✓ Captured #{n}  ", end="", flush=True)

            if n >= MIN_CAPTURES:
                try:
                    last_K, last_D, last_rms = _calibrate(obj_pts, img_pts, (OUT_W, OUT_H))
                    print(f"RMS={last_rms:.4f}px")
                except cv2.error as e:
                    print(f"calibration error: {e}")
            else:
                print()

            if n >= TARGET_CAPTURES:
                state = "done"
                _draw_hud(display, n, last_rms, state, 1.0, 0.0)
                cv2.imshow("Calibration", display)
                cv2.waitKey(1500)
                if last_K is not None:
                    _save_yaml(last_K, last_D, last_rms)
                break

        _draw_hud(display, n, last_rms, state, stable_pct, cooldown_remaining)
        _set_frame(display)
        cv2.imshow("Calibration", display)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            print("Quit without saving.")
            break
        elif key == ord('s') and last_K is not None and n >= MIN_CAPTURES:
            _save_yaml(last_K, last_D, last_rms)
            break
        elif key == ord(' ') and corners is not None and not in_cooldown:
            # Force capture now
            do_capture = True

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
