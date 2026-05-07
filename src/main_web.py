"""Web dashboard entry point.

Run:
    python -m src.main_web
"""

# ── Suppress FFmpeg AU-header noise on Windows ────────────────────────────────
# This MUST run before any cv2 import.  OpenCV's FFmpeg plugin DLL initialises
# its own private static-CRT stderr by calling GetStdHandle(STD_ERROR_HANDLE)
# when the DLL first loads.  If we redirect that Win32 handle to NUL first,
# the plugin's stderr FILE* will point at NUL from birth and never write noise.
# freopen_s also covers Python's own UCRT instance for good measure.
# Python logging / sys.stderr are Python-level objects; they are unaffected.
import sys
if sys.platform == "win32":
    try:
        import ctypes
        import ctypes.wintypes as _wt

        _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _k32.CreateFileW.restype  = _wt.HANDLE
        _k32.CreateFileW.argtypes = [
            _wt.LPCWSTR, _wt.DWORD, _wt.DWORD,
            ctypes.c_void_p, _wt.DWORD, _wt.DWORD, _wt.HANDLE,
        ]
        _k32.SetStdHandle.restype  = _wt.BOOL
        _k32.SetStdHandle.argtypes = [_wt.DWORD, _wt.HANDLE]

        _nul = _k32.CreateFileW(
            "NUL",
            _wt.DWORD(0x40000000),   # GENERIC_WRITE
            _wt.DWORD(3),            # FILE_SHARE_READ | FILE_SHARE_WRITE
            None,
            _wt.DWORD(3),            # OPEN_EXISTING
            _wt.DWORD(0x80),         # FILE_ATTRIBUTE_NORMAL
            None,
        )
        _k32.SetStdHandle(_wt.DWORD(0xFFFFFFF4), _nul)  # STD_ERROR_HANDLE

        _ucrt = ctypes.CDLL("ucrtbase")
        _ucrt.__acrt_iob_func.restype  = ctypes.c_void_p
        _ucrt.__acrt_iob_func.argtypes = [ctypes.c_uint]
        _ucrt.freopen_s.restype  = ctypes.c_int
        _ucrt.freopen_s.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
        ]
        _fp_out = ctypes.c_void_p(0)
        _ucrt.freopen_s(
            ctypes.byref(_fp_out), b"NUL", b"w", _ucrt.__acrt_iob_func(2)
        )

        # OpenCV's FFmpeg plugin (opencv_videoio_ffmpeg*.dll) links against the
        # legacy msvcrt.dll, not ucrtbase.  Its internal stderr FILE* lives at
        # msvcrt!_iob[2] (sizeof(FILE)=48 on 64-bit).  Redirect that too.
        _msvcrt_dll = ctypes.CDLL("msvcrt")
        _iob_t = ctypes.c_char * (48 * 3)
        _iob = _iob_t.in_dll(_msvcrt_dll, "_iob")
        _msvcrt_stderr = ctypes.c_void_p(ctypes.addressof(_iob) + 2 * 48)
        _msvcrt_dll.freopen.restype  = ctypes.c_void_p
        _msvcrt_dll.freopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
        _msvcrt_dll.freopen(b"NUL", b"w", _msvcrt_stderr)
    except Exception:
        pass  # best-effort; noise is cosmetic

# tqdm writes progress bars to stderr.  Since we just redirected the C-level
# stderr to NUL, tqdm's flush() call hits WinError 1 on the broken fd.
# Disable tqdm globally — progress bars have no value in a server process.
import os as _os
_os.environ.setdefault("TQDM_DISABLE", "1")

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.web.app:app",
        host="0.0.0.0",
        port=8000,
        ws="wsproto",            # avoids write-drain race in websockets legacy protocol
    )
