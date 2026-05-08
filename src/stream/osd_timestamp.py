"""Extract the wall-clock timestamp burned into NVR OSD text.

Crops the top-left region of a frame, runs Tesseract OCR, and parses the
datetime string (format: YYYY-MM-DD HH:MM:SS). Returns a Unix timestamp float
or None if extraction fails — callers fall back to time.time().
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from functools import lru_cache

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Amcrest/Dahua OSD format: "2026-05-07 19:35:43"
_TS_RE = re.compile(r'(\d{4}[-/]\d{2}[-/]\d{2})\s+(\d{2}:\d{2}:\d{2})')

# Crop: top-left corner where OSD timestamp lives.
# Expressed as fractions of frame dimensions — works at any resolution.
_CROP_W = 0.30   # left 30% of width
_CROP_H = 0.08   # top 8% of height


@lru_cache(maxsize=1)
def _tesseract_available() -> bool:
    try:
        import pytesseract
        _set_tesseract_cmd(pytesseract)
        pytesseract.get_tesseract_version()
        return True
    except Exception as e:
        logger.warning("Tesseract not available — OSD extraction disabled: %s", e)
        return False


def _set_tesseract_cmd(pytesseract) -> None:
    if os.name == 'nt':
        default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", default)


def extract_osd_timestamp(frame: np.ndarray) -> float | None:
    """Return the OSD timestamp from the frame as Unix epoch, or None."""
    if not _tesseract_available():
        return None

    import pytesseract

    h, w = frame.shape[:2]
    crop = frame[0: int(h * _CROP_H), 0: int(w * _CROP_W)]

    # Preprocess: grayscale + threshold — white NVR text on dark background
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(
        thresh,
        config='--psm 7 -c tessedit_char_whitelist=0123456789-:/ ',
    )

    m = _TS_RE.search(text)
    if not m:
        return None

    date_str = m.group(1).replace('/', '-')
    time_str = m.group(2)
    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        return dt.astimezone().timestamp()
    except ValueError:
        return None
