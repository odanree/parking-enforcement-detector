# Operations Runbook

## Starting the dashboard

```bash
source .venv/Scripts/activate   # Windows Git Bash
python -m src.main_web
```

Browse to `http://localhost:8000`. Using bare `uvicorn` instead of `main_web` will cause WebSocket ping crashes on slow/mobile connections (see ADR 009).

## Exposing to mobile / external via ngrok

```bash
ngrok http 8000
```

Copy the `https://` URL from ngrok output. The dashboard is fully functional over ngrok including WebSocket video stream.

## Switching VLM backend

Set in `.env`:

```
VLM_BACKEND=ollama          # local model (default)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma4         # or llava:7b-v1.6-mistral-q4_K_M, etc.

VLM_BACKEND=claude          # Anthropic API
# ANTHROPIC_API_KEY must be set

VLM_BACKEND=mock            # forces all detections true — for pipeline testing only
```

Restart the server after changing `.env`.

## Diagnosing missed detections

1. Open the **Debug drawer** (header → Debug button).
2. Look at the rejected crops and VLM descriptions.
3. Common causes:
   - **"No tool visible"** → prompt is too strict; update `_USER_PROMPT` in `src/vlm/analyzer.py` to flag on posture/proximity instead.
   - **Blurry/dark crops** → camera angle or lighting issue; check that `INPUT_WIDTH`/`INPUT_HEIGHT` match camera resolution.
   - **Wrong zone** → person is outside the street zone; use Edit Zone to adjust.
   - **Model too conservative** → switch to Claude backend (`VLM_BACKEND=claude`) for better instruction-following.

## Diagnosing false positives

1. Check Event Log snapshot — is the person actually near a vehicle's tire?
2. If the zone is too large (includes driveways, sidewalks), use **Edit Zone** to tighten it.
3. If a specific vehicle (e.g. Tesla in driveway) keeps triggering:
   - Adjust zone to exclude the driveway.
   - Raise `entry_frames` or `entry_min_px` in `config/detection.yaml` under `pe_vehicle`.
4. For chalking false positives (person walking past a car): raise `cooldown_seconds` to reduce repeat alerts.

## Adjusting the detection zone

1. Click **Edit Zone** in the dashboard toolbar.
2. Click to add vertices, drag to move them, right-click to remove, Ctrl-Z to undo.
3. Click **Save** — the zone is hot-reloaded in the pipeline within one frame and persisted to `config/detection.yaml`.

## Privacy redaction

1. Click **Edit Regions** to enter draw mode.
2. Drag to draw a black-out box over any license plate or sensitive area.
3. Right-click a box to delete it.
4. Click **Save** — regions are persisted to `config/privacy.json` and survive restarts.
5. Toggle **Privacy** button to enable/disable redaction on the live stream and snapshots.

## Checking VLM processing time

The **VLM Processing** card appears in the side panel whenever a VLM job is in flight. It shows:
- The exact crop sent to the model (not the full frame).
- Which type of event is being analyzed (Chalking / Sweeper / PE Vehicle).
- Sample # — how many times this track has been sent to the VLM.
- Elapsed time since submission, ticking every 100 ms.
- On completion: ✓ (green) if detected, ✗ (grey) if not — visible for 4 seconds before clearing.

## Sending alerts from the event modal

Tap any snapshot in the event log to open the detail modal. The **Send Alert** button posts to `POST /api/alert`. Configure exactly one provider in `.env`:

### Email (recommended for personal use)

```
ALERT_EMAIL=you@gmail.com
SMTP_USER=you@gmail.com
SMTP_PASS=xxxx xxxx xxxx xxxx   # Gmail App Password — NOT your regular password
```

To generate a Gmail App Password: Google Account → Security → 2-Step Verification → App passwords.

### ntfy.sh push notification (free, requires ntfy app)

```
NTFY_TOPIC=ped-alerts-yourname   # any unique string
```

Install the ntfy app on your phone and subscribe to the same topic.

### TextBelt SMS (paid, no A2P registration)

```
TEXTBELT_KEY=<key from textbelt.com>
ALERT_PHONE=+1xxxxxxxxxx
```

### Twilio (requires A2P 10DLC registration for US numbers)

```
TWILIO_ACCOUNT_SID=ACxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxx
TWILIO_FROM_NUMBER=+1xxxxxxxxxx
ALERT_PHONE=+1xxxxxxxxxx
```

> **Note:** Twilio 10DLC long-code numbers are blocked for US SMS without A2P registration (takes weeks). Toll-free numbers require org verification. Use email or ntfy for personal deployments.

## Common `.env` variables

| Variable | Default | Notes |
|---|---|---|
| `VIDEO_PATH` | *(unset)* | Path to video file or directory; unset = RTSP live |
| `RTSP_URL` | *(required if no VIDEO_PATH)* | `rtsp://user:pass@host/stream` |
| `VLM_BACKEND` | `claude` | `claude` / `ollama` / `mock` |
| `OLLAMA_MODEL` | `llava:7b-v1.6-mistral-q4_K_M` | Any multimodal Ollama model |
| `INPUT_WIDTH` | `1280` | Detection resolution |
| `INPUT_HEIGHT` | `720` | Detection resolution |
| `INFERENCE_THRESHOLD` | `0.20` | YOLO confidence threshold |
| `LOG_LEVEL` | `INFO` | `DEBUG` for verbose frame logs |
| `ALERT_EMAIL` | *(unset)* | Destination email for Send Alert |
| `SMTP_USER` | *(unset)* | Gmail address used to send alerts |
| `SMTP_PASS` | *(unset)* | Gmail App Password |
| `NTFY_TOPIC` | *(unset)* | ntfy.sh topic name |
| `TEXTBELT_KEY` | *(unset)* | TextBelt API key |
| `ALERT_PHONE` | `+17145671107` | Destination phone for SMS providers |

## Log file

Logs are written to `logs/detector.log` and stdout simultaneously. To tail:

```bash
tail -f logs/detector.log
```
