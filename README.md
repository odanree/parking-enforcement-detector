# Parking Enforcement Detector

Real-time detection of parking enforcement activity using YOLO object detection and a vision-language model (Claude or Ollama). Monitors an RTSP camera feed or video file and alerts when it detects tire chalking by a PE officer.

## How it works

```
RTSP / video file
      │
      ▼
  YOLO (YOLOv8) + ByteTrack     ← tracks persons; post-track gate drops sub-threshold hits
      │
      ▼
  Zone filter + vehicle proximity← only persons dwelling near a vehicle continue
      │
      ▼
  Person classifier  (opt-in)    ← skips pedestrians/occupants/delivery before the VLM
  Pose priors        (opt-in)    ← crouch / wrist-near-wheel signals
  Wand gate          (opt-in)    ← motion-gated chalk-wand detector (classical CV)
      │
      ▼
  After-hours gate               ← drops low-confidence hits outside the PE window
      │
      ▼
  1st-pass VLM (people detection)← lenient: is a person next to a vehicle?
      │  Claude / Ollama
      ▼
  RAG retrieval (ChromaDB)       ← nearest neighbors (text + opt-in CLIP image embeddings);
      │                            auto-rejects if FP-close neighbors exceed threshold
      ▼
  Confirm VLM (chalking check)   ← strict prompt, or structured-evidence + Python policy (opt-in)
      │
      ▼
  Dedup (vector store)           ← skip alert if a visually identical event was recently sent
      │
      ▼
  Alert (email / ntfy / HA webhook)  →  Dashboard (React + WebSocket)
```

Stages marked **opt-in** default to off (no behaviour change) and are enabled via env vars below.

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

The Dockerfile uses a two-stage build: Node compiles the React frontend; Python 3.11 runs the FastAPI server. The vector store and dataset images persist via the `./data` and `./dataset` bind mounts. Maintenance/ingestion CLIs run in-container, e.g.:

```bash
docker compose run --rm detector python -m scripts.dataset_maintenance stats
```

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

Opt-in accuracy/volume features (default off):

| Variable | Default | Description |
|---|---|---|
| `PERSON_CLASSIFIER_ENABLED` | `false` | Classify each track once; skip pedestrians/occupants/delivery before the VLM |
| `POSE_ESTIMATION_ENABLED` | `false` | YOLOv8-pose crouch / wrist-near-wheel priors fed to the VLM |
| `WAND_GATE` | `off` | `soft` (detect + annotate) or `hard` (escalate to VLM only on a confirmed chalk wand) |
| `VLM_STRUCTURED_PROMPT` | `false` | VLM returns observation flags; a Python policy makes the chalking decision |
| `CLIP_EMBEDDINGS_ENABLED` | `false` | Add CLIP image embeddings for visual RAG (needs `open-clip-torch`) |

See [`.env.example`](.env.example) for all options and [`docs/runbooks/operations.md`](docs/runbooks/operations.md) for detailed setup.

## Stack

- **Detection:** YOLOv8 (Ultralytics) + ByteTrack; opt-in YOLOv8-pose and a classical-CV chalk-wand detector
- **VLM:** Claude via Anthropic API, or any Ollama multimodal model
- **RAG:** ChromaDB store (description text + opt-in CLIP image embeddings) for retrieval and auto-rejection
- **Backend:** FastAPI + WebSocket
- **Frontend:** React 19 + Vite + Zustand
- **Alerts:** Email (Gmail SMTP), ntfy.sh, Home Assistant webhook

## Maintenance

- `scripts/dataset_maintenance.py` — vector-store stats, purge errors/shadows, fix confidence, backfill `person_type`, phash clustering, CLIP backfill, merge
- `scripts/ingest_nvr_positives.py` — ingest ground-truth chalking frames as labeled `chalker` positives (host extract → in-container ingest)
- `scripts/recover_vector_store.py` — rebuild a collection from raw SQLite if its vector segment is damaged

## Docs

- [`docs/adr/`](docs/adr/) — architecture decision records
- [`docs/runbooks/operations.md`](docs/runbooks/operations.md) — deployment, tuning, alert setup
- [`docs/LEARNINGS.md`](docs/LEARNINGS.md) — debugging notes and lessons learned
