# Parking Enforcement Detector

Real-time detection of parking enforcement activity using YOLO object detection and a vision-language model (Claude or Ollama). Monitors an RTSP camera feed or video file and alerts on tire chalking by a PE officer, stopped PE vehicles ([ADR-006](docs/adr/006-pe-vehicle-stopped-detection.md)), and street sweepers.

**Rodent mode** (new): the same pipeline can be flipped into rodent-detection with `DETECTION_MODE=rodent`. Motion-gated VLM classification (no YOLO fine-tune) plus a slew-to-zone dispatcher that pans a secondary Amcrest/Dahua PTZ camera to the zone where a rat/mouse was seen. See [Rodent mode](#rodent-mode) below.

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

**Required for first run** — the app fails to start without an auth key
(see [ADR-031](docs/adr/031-phase-0-api-key-auth.md)):

```bash
# In .env
PED_API_KEY=$(openssl rand -hex 32)          # any high-entropy string, min 16 chars
PED_ALLOWED_ORIGINS=                         # comma-separated; blank = same-origin only
```

The dashboard prompts for this key on first load (stored in `localStorage`).
All `/api/*`, `/ws/*`, `/snapshots/*`, `/dataset/*` requests require it.

Key `.env` variables:

| Variable | Default | Description |
|---|---|---|
| `PED_API_KEY` | — | **Required.** Bearer token for all authenticated routes. |
| `PED_ALLOWED_ORIGINS` | — | Comma-separated cross-origin allowlist (blank = same-origin only). |
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

## Rodent mode

The pipeline can be repurposed to detect rats/mice using the same motion-gated VLM classification pattern. Set `DETECTION_MODE=rodent` in `.env` and the pipeline loads [`config/rodent.yaml`](config/rodent.yaml) instead of `config/detection.yaml`, swaps in a rodent-specific VLM prompt, and skips the parking-only stages (wand gate, pose priors, chalking analyzer, person classifier, RAG neighbours). Deployment is a **strategy pattern** — parking and rodent are separate `DetectionStrategy` implementations under [`src/detection/strategy.py`](src/detection/strategy.py).

**Dual-camera slew-to-zone**: on a rodent positive, [`src/stream/slew.py`](src/stream/slew.py) looks up which zone in `config/rodent.yaml` contains the detection center and issues a `GotoPreset` to the secondary Amcrest/Dahua PTZ camera (`SECONDARY_CAMERA_ID`). Presets are saved in the NVR UI; the mapping is a **zone → preset lookup** (no homography calibration needed for the MVP). Per-event lockout (`SLEW_LOCKOUT_SECONDS`, default 10s) prevents PTZ thrashing while a rat lingers.

```bash
# .env
DETECTION_MODE=rodent
SECONDARY_CAMERA_ID=1
RODENT_SLEW_ENABLED=true
AMCREST_HOST=192.168.1.100
PTZ_CHANNEL_1=2
```

Edit `config/rodent.yaml` → `slew_presets` to map primary-FOV polygons to the preset numbers you saved on the secondary camera.

## Stack

- **Detection:** YOLOv8 (Ultralytics) + ByteTrack; opt-in YOLOv8-pose and a classical-CV chalk-wand detector; person action classifier covers `standing`, `walking`, `running`, `crouching`, `bending`
- **VLM:** Claude via Anthropic API, or any Ollama multimodal model (see [ADR-027](docs/adr/027-gemma3-4b-for-classifier-footprint.md) for the local-model choice)
- **RAG:** ChromaDB store (description text + opt-in CLIP image embeddings, perceptual-hash dedup) for retrieval and auto-rejection
- **Backend:** FastAPI + WebSocket
- **Frontend:** React 19 + Vite + Zustand
- **Alerts:** Email (Gmail SMTP), ntfy.sh, Home Assistant webhook
- **GPU / Compute:** NVIDIA RTX 3090 / CUDA — inference migrated from CPU-torch per [ADR-026](docs/adr/026-gpu-inference-for-yolo.md), superseding the earlier lean-resource profile in [ADR-024](docs/adr/024-lean-resource-profile-and-vlm-model-selection.md)
- **Observability:** Langfuse traces on every VLM call for cost + latency visibility ([ADR-029](docs/adr/029-langfuse-vlm-observability.md))

## Tuning history

Recent accuracy work is captured in [ADR-023 (detection accuracy rework)](docs/adr/023-detection-accuracy-rework.md): YOLO inference threshold raised from `0.30` → `0.55`, position-grid FP suppression added to drop repeated hits at fixed pixel regions, and the RAG auto-reject gate (`RAG_FP_THRESHOLD` / `RAG_FP_MIN_VOTES`) wired up to short-circuit VLM calls on labeled-negative-adjacent detections.

## Maintenance

- `scripts/dataset_maintenance.py` — vector-store stats, purge errors/shadows, fix confidence, backfill `person_type`, phash clustering, CLIP backfill, merge
- `scripts/ingest_nvr_positives.py` — ingest ground-truth chalking frames as labeled `chalker` positives (host extract → in-container ingest)
- `scripts/recover_vector_store.py` — rebuild a collection from raw SQLite if its vector segment is damaged

## Docs

- [`docs/adr/`](docs/adr/) — architecture decision records
- [`docs/runbooks/operations.md`](docs/runbooks/operations.md) — deployment, tuning, alert setup
- [`docs/LEARNINGS.md`](docs/LEARNINGS.md) — debugging notes and lessons learned
