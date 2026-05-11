# Parking Enforcement Detector

Real-time detection of parking enforcement activity using YOLO object detection and a vision-language model (Claude or Ollama). Monitors an RTSP camera feed or video file and alerts when it detects tire chalking by a PE officer.

## How it works

```
RTSP / video file
      │
      ▼
  YOLO (YOLOv8) + ByteTrack     ← detects and tracks persons per frame
      │
      ▼
  Zone filter                    ← ignores detections outside the street zone
      │
      ▼
  After-hours gate               ← drops YOLO detections < 40% confidence
      │  (outside PE window)       outside the 08:00–16:00 enforcement window
      ▼
  Behavior analyzers             ← chalking / sweeper / PE vehicle heuristics
  (entry_frames, cooldown,         only passes tracks that dwell near a vehicle
   vehicle proximity)
      │
      ▼
  1st-pass VLM (people detection)← lenient check: is a person next to a vehicle?
      │  Claude Haiku / Ollama      positives recorded as People Alert in kanban
      ▼
  RAG retrieval                  ← nearest neighbors from labeled event store
      │  ChromaDB                   auto-rejects if FP-close neighbors exceed threshold
      ▼
  Confirm VLM (chalking check)   ← strict: does the person exhibit chalking behavior?
      │  Claude Haiku / Ollama      runs only when 1st-pass positive + RAG allows it
      ▼
  Dedup (vector store)           ← skip alert if visually identical event was recently sent
      │
      ▼
  Alert (email / ntfy / HA webhook)
      │
      ▼
  Dashboard (React + WebSocket)
```

### Detection phases

| Phase | Signal | Kanban column |
|---|---|---|
| After-hours gate fired | YOLO < 40% outside PE window | Off Hours ✗ |
| 1st-pass negative | No person/vehicle detected | Primary ✗ |
| RAG auto-reject | ≥ N FP-close labeled neighbors | RAG Blocked ✗ |
| Confirm negative | Not chalking per strict VLM | Reeval ✗ |
| 1st-pass positive | Person near vehicle confirmed | People Alert ✓ |
| Confirm positive | Chalking behavior confirmed | Chalking ✓ |

## Pipeline Kanban

The **Pipeline Kanban** panel shows every detection that entered the pipeline within the selected time window, grouped by visual similarity (perceptual hash, Hamming distance ≤ 10). Clicking a grouped card (×N badge) opens a detail modal with left/right navigation through all snapshots in the group.

Each card shows the full pipeline trace: YOLO confidence + track ID, 1st-pass result, RAG neighbors, confirm result, and final outcome.

## Quick start

```bash
python -m venv .venv && source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
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

| Variable | Default | Description |
|---|---|---|
| `RTSP_URL` | — | `rtsp://user:pass@host/stream` — leave blank for demo mode |
| `VIDEO_PATH` | — | Local video file for replay mode (disables dedup and off-hours gate) |
| `VLM_BACKEND` | `claude` | `claude`, `ollama`, or `mock` |
| `ANTHROPIC_API_KEY` | — | Required when `VLM_BACKEND=claude` |
| `PE_WINDOW_ENABLED` | `true` | Enable the enforcement-hours gate |
| `PE_WINDOW_START` | `8` | Start hour (local time, 24 h) of PE enforcement window |
| `PE_WINDOW_END` | `16` | End hour of PE enforcement window |
| `RAG_AUTO_REJECT` | `false` | Auto-reject when RAG finds enough FP-close neighbors |
| `RAG_FP_THRESHOLD` | `0.30` | Distance threshold for FP-close RAG neighbors |
| `RAG_FP_MIN_VOTES` | `3` | Minimum FP-close neighbors to trigger auto-reject |
| `DEMO_MODE` | `false` | Hide admin controls (zone editor, alerts, debug) |
| `HA_WEBHOOK_URL` | — | Home Assistant webhook URL for alert triggers |

See [`.env.example`](.env.example) for all options and [`docs/runbooks/operations.md`](docs/runbooks/operations.md) for detailed setup.

## Stack

- **Detection:** YOLOv8 (Ultralytics) + ByteTrack
- **VLM:** Claude Haiku via Anthropic API, or any Ollama multimodal model
- **RAG:** ChromaDB embedding store for labeled-event retrieval and auto-rejection
- **Backend:** FastAPI + WebSocket
- **Frontend:** React 19 + Vite + Zustand
- **Alerts:** Email (Gmail SMTP), ntfy.sh, Home Assistant webhook

## Docs

- [`docs/adr/`](docs/adr/) — architecture decision records
- [`docs/runbooks/operations.md`](docs/runbooks/operations.md) — deployment, tuning, alert setup
- [`docs/LEARNINGS.md`](docs/LEARNINGS.md) — debugging notes and lessons learned
