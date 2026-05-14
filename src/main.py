"""Headless entry point — no web dashboard.

For the live dashboard run:
    uvicorn src.web.app:app --host 0.0.0.0 --port 8000
"""

# ── Suppress FFmpeg AU-header noise on Windows ────────────────────────────────
# Must run before any cv2 import — see src/main_web.py for full explanation.
import sys
if sys.platform == "win32":
    try:
        import ctypes
        import ctypes.wintypes as _wt
        _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _k32.CreateFileW.restype  = _wt.HANDLE
        _k32.CreateFileW.argtypes = [_wt.LPCWSTR, _wt.DWORD, _wt.DWORD, ctypes.c_void_p, _wt.DWORD, _wt.DWORD, _wt.HANDLE]
        _k32.SetStdHandle.restype  = _wt.BOOL
        _k32.SetStdHandle.argtypes = [_wt.DWORD, _wt.HANDLE]
        _nul = _k32.CreateFileW("NUL", _wt.DWORD(0x40000000), _wt.DWORD(3), None, _wt.DWORD(3), _wt.DWORD(0x80), None)
        _k32.SetStdHandle(_wt.DWORD(0xFFFFFFF4), _nul)
        _ucrt = ctypes.CDLL("ucrtbase")
        _ucrt.__acrt_iob_func.restype  = ctypes.c_void_p
        _ucrt.__acrt_iob_func.argtypes = [ctypes.c_uint]
        _ucrt.freopen_s.restype  = ctypes.c_int
        _ucrt.freopen_s.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
        _fp_out = ctypes.c_void_p(0)
        _ucrt.freopen_s(ctypes.byref(_fp_out), b"NUL", b"w", _ucrt.__acrt_iob_func(2))
        _msvcrt_dll = ctypes.CDLL("msvcrt")
        _iob_t = ctypes.c_char * (48 * 3)
        _iob = _iob_t.in_dll(_msvcrt_dll, "_iob")
        _msvcrt_stderr = ctypes.c_void_p(ctypes.addressof(_iob) + 2 * 48)
        _msvcrt_dll.freopen.restype  = ctypes.c_void_p
        _msvcrt_dll.freopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
        _msvcrt_dll.freopen(b"NUL", b"w", _msvcrt_stderr)
    except Exception:
        pass

import os
os.environ.setdefault("TQDM_DISABLE", "1")

import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/detector.log", encoding="utf-8"),
    ],
)

from src import pipeline

if __name__ == "__main__":
    pipeline.run(state=None)
