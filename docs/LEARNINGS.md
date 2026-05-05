# Learnings — parking-enforcement-detector

---

## L1 — OpenCV RTSP defaults to UDP, causing frame smearing that fools YOLO

**Date:** 2026-05-04

**What happened:** On first run over WiFi, YOLO was producing intermittent
`person` detections in the street zone even when the street was empty. The
confidence scores were low (0.66–0.70, just above the 0.65 threshold) and the
bounding boxes were wide, flat shapes rather than upright person silhouettes.
Reviewing the raw frames revealed horizontal smearing artifacts — bands from
the previous frame blended into the current one — caused by UDP packet loss.

**Fix:** Set `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp` as an
environment variable before any `cv2.VideoCapture` is opened. Done at module
import time in `rtsp_handler.py` so it applies process-wide:

```python
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
```

**Why it matters:** Smearing artifacts are specifically called out in the
original spec (Section 4) as a known source of VLM false positives. The same
corruption that fools YOLO at stage 1 will produce worse results at stage 2
because the VLM receives a corrupted crop. Fix it at the source.

---

## L2 — YOLOv8 `model.track()` silently requires `lap` for ByteTrack

**Date:** 2026-05-04

**What happened:** On first run, ultralytics auto-installed `lap>=0.5.12`
mid-execution with a warning: *"Restart runtime or rerun command for updates
to take effect."* In the same process, `model.track()` continued running but
all `boxes.id` values were `None` — ByteTrack was not active. The chalking and
sweeper analyzers both key on `track_id`, so with `None` IDs every detection
was effectively untracked and no behavioral signatures could accumulate.

**Fix:** Add `lap>=0.5.12` explicitly to `requirements.txt` so it installs
during `pip install -r requirements.txt` rather than being auto-fetched at
runtime. Always restart the process after any auto-install warning from
ultralytics.

**Why it matters:** Silent `None` track IDs cause the behavioral analyzers to
silently do nothing. The pipeline appears to run correctly (YOLO detects
objects, logs show detections) but no alerts ever fire. Without knowing to look
for `None` IDs, this is very hard to diagnose.

---

## L3 — Local LLaVA models wrap JSON in markdown fences despite explicit instruction

**Date:** 2026-05-04

**What happened:** The VLM prompt explicitly states "Output only a JSON object"
and gives the exact schema. Claude API follows this reliably. Ollama with
`llava:7b-v1.6-mistral-q4_K_M` consistently wraps the response in a markdown
code fence:

```
```json
{ "chalking_detected": false, ... }
```
```

`json.loads()` raises `JSONDecodeError` on the raw response, causing every VLM
call to return the silent `_FALLBACK` dict (`chalking_detected: false,
confidence: 0.0`). Alerts stop firing entirely with no logged error because the
exception is caught broadly.

**Fix:** Strip code fences in `_parse_json()` before calling `json.loads()`:

```python
text = re.sub(r"```(?:json)?", "", text).strip("`").strip()
```

Log a warning (not an exception) when JSON parsing fails so the failure is
visible without crashing the pipeline.

**Why it matters:** Local models are inconsistent about instruction following on
output format constraints. Never assume a model that says "output only JSON"
will actually output only JSON — always defensively strip common wrappers.

---

## L4 — Bounding-box height decrease must be relative to per-track baseline

**Date:** 2026-05-04

**What happened:** The first chalking analyzer implementation compared bbox
height against a fixed constant (half of frame height). A person who entered
the zone close to the camera (large bbox, height ~350 px) never triggered
because their "standing" height was already well above the constant. A person
at the far end of the zone (height ~80 px) triggered immediately on any minor
variation because small absolute changes exceeded 30% of a small baseline.

**Fix:** Compute the baseline as the mean of the first `history_frames // 3`
heights recorded after the track enters the zone — relative to the track's own
initial size, not any global constant. A 30% decrease from *that individual's*
recent standing height is the correct signal regardless of distance from camera.

**Why it matters:** Camera perspective means the same physical event (person
crouching) produces very different pixel-space measurements depending on where
in the frame it occurs. Any threshold on absolute pixel values will be wrong for
some portion of the camera's field of view.

---

## L5 — Stationary masking needs spread over a window, not per-frame velocity

**Date:** 2026-05-04

**What happened:** The first stationary mask implementation flagged a track as
moving if displacement from the previous frame exceeded `stationary_px` (15 px).
A parked car with a tree branch casting moving shadows across it was tagged as
"moving" by YOLO every few frames as the bounding box slightly shifted. It
passed the stationary mask and entered the zone filter, producing repeated
sweeper false positives on non-sweep days.

**Fix:** Replace the per-frame velocity check with a sliding-window spread
check: collect the last `stationary_frames` (30) center positions per track,
compute `max(spread_x, spread_y)`, and only consider a track "moving" if this
spread exceeds the threshold. Small oscillations in the bounding box (±5–10 px
from YOLO jitter) average out over 30 frames; a genuinely moving object
accumulates real spread.

**Why it matters:** YOLO bounding boxes are noisy — the same stationary object
produces slightly different boxes on each frame due to model non-determinism and
compression artifacts. Any single-frame velocity measure will misclassify
stationary objects during low-confidence detection frames.
