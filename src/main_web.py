"""Web dashboard entry point.

Run:
    python -m src.main_web
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.web.app:app",
        host="0.0.0.0",
        port=8000,
        ws="wsproto",            # avoids write-drain race in websockets legacy protocol
    )
