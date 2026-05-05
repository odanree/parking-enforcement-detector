"""Headless entry point — no web dashboard.

For the live dashboard run:
    uvicorn src.web.app:app --host 0.0.0.0 --port 8000
"""

import logging
import os

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
