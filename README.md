# Parking Enforcement Detector

Real-time detection of parking enforcement activity using YOLO object detection and a vision-language model (Claude or Ollama). Monitors an RTSP camera feed or video file and alerts when it detects:

- **Chalking** — officer marking tires with chalk
- **Street sweeper** — sweeper vehicle passing through
- **PE vehicle** — parking enforcement vehicle in the zone

## How it works

```
RTSP / video file
      │
      ▼
  YOLO (YOLOv8)          ← detects persons and vehicles per frame
      │
      ▼
  Zone filter            ← ignores detections outside the drawn street zone
      │
      ▼
  ByteTrack              ← tracks objects across frames, bridges occlusions
      │
      ▼
  Behavior analyzers     ← chalking / sweeper / PE vehicle heuristics
      │
      ▼
  VLM (Claude / Ollama)  ← confirms ambiguous events with visual reasoning
      │
      ▼
  Alert (email / ntfy / SMS)
```

The web dashboard streams the annotated video feed via WebSocket and shows live stats, an event log with snapshots, and controls for zone editing, privacy redaction, and playback.

## Quick start

```bash
python -m venv .venv && source .venv/Scripts/activate  # Windows
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY and RTSP_URL (or VIDEO_PATH)
python -m src.main_web
# open http://localhost:8000
```

## Docker

```bash
docker compose up -d
```

The Dockerfile uses a two-stage build: Node 20 compiles the React frontend; Python 3.11 runs the FastAPI server.

## Configuration

Key `.env` variables:

| Variable | Description |
|---|---|
| `RTSP_URL` | `rtsp://user:pass@host/stream` — leave blank for demo mode |
| `VIDEO_PATH` | Path to a local video file for replay mode |
| `VLM_BACKEND` | `claude` (default) or `ollama` or `mock` |
| `ANTHROPIC_API_KEY` | Required when `VLM_BACKEND=claude` |
| `ALERT_EMAIL` + `SMTP_USER` + `SMTP_PASS` | Gmail alert destination |
| `NTFY_TOPIC` | ntfy.sh push notification topic |

See [`.env.example`](.env.example) for all options and [`docs/runbooks/operations.md`](docs/runbooks/operations.md) for detailed setup.

## Stack

- **Detection:** YOLOv8 (Ultralytics) + ByteTrack
- **VLM:** Claude Haiku via Anthropic API, or any Ollama multimodal model
- **Backend:** FastAPI + WebSocket
- **Frontend:** React 19 + Vite + Zustand
- **Alerts:** Email (Gmail SMTP), ntfy.sh, TextBelt SMS, Twilio

## Docs

- [`docs/adr/`](docs/adr/) — architecture decision records
- [`docs/runbooks/operations.md`](docs/runbooks/operations.md) — deployment, tuning, alert setup
- [`docs/LEARNINGS.md`](docs/LEARNINGS.md) — debugging notes and lessons learned
