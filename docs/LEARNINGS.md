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

## L6 — VLM "confidence" field reflects certainty in the answer, not probability of the event

**Date:** 2026-05-05

**What happened:** The debug drawer showed a frame where a person was standing directly beside a vehicle's rear wheel. The VLM returned `chalking_detected: false, confidence: 1.0`. The intuitive read was "100% confident it's not chalking" — which seemed wrong. But a low-confidence negative (e.g. `confidence: 0.1`) was being interpreted as "90% chance of chalking" which is equally wrong.

**Clarification:** The `confidence` field in the VLM response is the model's self-assessed certainty in whatever answer it gave — positive or negative. `{chalking_detected: false, confidence: 1.0}` means "I'm very sure there's no chalking." `{chalking_detected: false, confidence: 0.1}` means "I said no, but I'm not sure." Neither implies anything about the probability of the opposite.

**Fix:** Do not invert confidence to derive detection probability. Use confidence only to rank the strength of a positive detection for the event log (e.g. "95%" displayed alongside a chalking alert). For negatives, use the debug description to understand why.

**Why it matters:** Misreading the confidence field as an inverse probability leads to incorrect threshold logic and bad prompt tuning hypotheses.

---

## L7 — VLM tool-visibility requirement fails at overhead camera distance

**Date:** 2026-05-05

**What happened:** The chalking prompt required a "visible stick or chalk tool." From 5–10 m overhead, a chalk stick held near the ground is 2–4 px wide — reliably invisible to any 7B VLM. The model returned high-confidence negatives on frames that clearly showed a person adjacent to a rear wheel.

**Fix:** Shift the detection criterion from tool-visibility to behavioral indicators: crouching near a tire, bending toward a wheel, close physical contact with the wheel area. A tool being visible is a sufficient condition but not a necessary one. Updated both the user prompt and the system prompt to communicate this explicitly, and instructed the model to prefer false positives over false negatives.

**Why it matters:** Tool-visible prompts are designed for close-up footage. For overhead surveillance footage, behavior and proximity are the only reliable signals. Always calibrate the VLM prompt against actual camera distance and angle before deployment.

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

---

## L8 — `overflow-y: auto` on `body` with `height: auto` silently disables Android page scroll

**Date:** 2026-05-05

**What happened:** The mobile CSS breakpoint set `html, body { height: auto !important; overflow-y: auto }` to unlock page scroll on Android. On a desktop browser narrowed to mobile width, scroll worked. On a Pixel running Chrome for Android, the page did not scroll at all.

**Root cause:** When `body { height: auto; overflow-y: auto }`, the body element becomes its own scroll container and sizes to exactly fit its children. Because body never overflows itself, there is no scrollable content and no scrollbar appears. The browser's native viewport scroll never activates.

**Fix:** Remove `overflow-y` from `html, body` in the mobile breakpoint. Use `overflow: visible !important` instead, which lets the viewport (not body) be the scroll container. The viewport scrolls automatically when body content is taller than the viewport.

**Why it matters:** Setting `overflow-y: auto` on `body` is a common suggestion for enabling mobile scroll but it produces the opposite effect when combined with `height: auto`. The distinction between "body is the scroller" and "viewport is the scroller" is invisible on desktop (mouse wheel bypasses it) but critical on Android (touch scroll goes to whichever element is the scroll container).

---

## L9 — `overflow: hidden` on a parent card swallows touch events on Android, preventing inner element scroll

**Date:** 2026-05-05

**What happened:** The `.events-card` had `overflow: hidden` (to clip content to its border-radius on desktop). Inside it, `.event-list` had `max-height: 60vh; overflow-y: scroll`. On desktop, the event list scrolled correctly. On Android Chrome, touching the event list did nothing — the list did not scroll.

**Root cause:** On Android Chrome, when a user touches a child element whose ancestor has `overflow: hidden`, Chrome locks the touch gesture to the ancestor's scroll context. Since the ancestor (`events-card`) had no scroll (it was a fixed-height clipping container), the touch was consumed and discarded.

**Fix:** On mobile, change `.events-card` to `overflow: visible` and `.event-list` to `max-height: none; overflow-y: visible`. Remove the nested scroll entirely and let the page scroll show all events. Nested scroll containers inside `overflow: hidden` parents are unreliable on Android.

**Why it matters:** Nested scroll is a common desktop pattern that breaks on Android. The reliable mobile pattern is a single scroll container (the page) with content that grows to its natural height.

---

## L10 — Twilio US SMS requires A2P 10DLC registration; toll-free requires org verification

**Date:** 2026-05-05

**What happened:** A Twilio 10DLC long-code number was purchased to send SMS alerts. The first send returned error 30034: message blocked because the number was not associated with an approved A2P 10DLC Campaign. Switching to a toll-free number prompted organization verification in the Twilio console.

**Fix:** For personal/small deployments, bypass carrier-registered SMS entirely. Email via Gmail SMTP (using an App Password) requires no registration and delivers as a push notification on any phone. ntfy.sh is a free push notification alternative. TextBelt paid tier works without A2P registration.

**Why it matters:** US SMS regulations tightened significantly in 2023–2024. Any new Twilio number requires A2P registration (weeks) or toll-free verification before sending to US numbers. Personal projects should default to email or push notifications and treat SMS as a paid/registered-deployment option.

---

## L11 — `fetchEvents` filter set `lastEventTs` before filtering, showing only the newest event on page refresh

**Date:** 2026-05-05

**What happened:** On page refresh, the event log showed only one event (the most recent) even though the server had many stored. During an active session, events accumulated correctly one by one.

**Root cause:** `fetchEvents` updated `lastEventTs = newest` before computing `newOnes = events.filter(e => e.timestamp > (lastEventTs - 0.001))`. After the assignment, `lastEventTs` equaled `newest`, so the filter only passed events within 1 ms of the newest — effectively just one event. On first load (`lastEventTs` starting at 0), all previous events were silently dropped.

**Fix:** Capture `prevLastTs = lastEventTs` before updating it. On first load (`prevLastTs === 0`), render all events. On subsequent polls, filter `e.timestamp > prevLastTs` to add only genuinely new events.

**Why it matters:** A filter that references a variable it just modified is a classic off-by-one class of bug. The symptom (works during session, breaks on refresh) is a strong signal that initial state differs from steady-state — look for variables initialized to 0 or null that are mutated before use.

---

## L12 — `width: 100%; aspect-ratio: 16/9` causes oversized canvas on ultrawide monitors

**Date:** 2026-05-05

**What happened:** After fixing the canvas to maintain 16:9 with `width: 100%; aspect-ratio: 16/9`, the video was correct on standard monitors but dominated ultrawide screens. The `1fr` grid column on a 3440px monitor was ~3000px wide, making the canvas 1688px tall — taller than the viewport and squishing the side panel into an unusable strip.

**Fix:** Replace `width: 100%` with `width: min(100%, calc((100vh - 110px) * 16 / 9))`. The second argument caps the canvas width at the value that produces exactly the available viewport height at 16:9. `min()` picks the smaller of the two, so narrow screens behave normally while ultrawide screens cap the canvas to a viewport-filling size.

**Why it matters:** `1fr` in a grid column grows without bound on wide screens. Any element that derives its height from its width via `aspect-ratio` will eventually exceed the viewport height as the screen gets wider. Cap width by both the container width AND the viewport-height-derived max width.

---

## L13 — CSS zoom-to-cursor formula: `panX * ratio` drifts; correct is `panX += cursorX * (1 - ratio)`

**Date:** 2026-05-05

**What happened:** A scroll-to-zoom feature on the event modal image used `panX2 = cursorX * (1 - ratio) + panX1 * ratio`. After the first scroll the zoom appeared correct, but on subsequent scrolls the focal point drifted back toward the original cursor position rather than tracking the current cursor.

**Root cause:** The `panX1 * ratio` term accumulates error. Each zoom step multiplies the accumulated pan by the zoom ratio, causing the pan to grow proportionally rather than staying fixed. The correct derivation is:

Screen position of image point P: `screenX = P.x * zoom + panX`.  
To keep P fixed under cursor at `cursorX` (relative to visual center): `panX2 = panX1 + cursorX * (1 - ratio)`.

This is simply `panX += cursorX * (1 - ratio)` — add a delta, never multiply the accumulated pan.

**Why it matters:** The `panX * ratio` form appears intuitive ("scale the existing offset too") but is mathematically wrong. Always derive zoom-to-point formulas from the fixed-point constraint: the image coordinate under the cursor must be identical before and after the zoom step.

---

## L14 — VLM prompt gate phrasing inverted: "no trunk → not chalking" instead of "no trunk → proceed"

**Date:** 2026-05-06

**What happened:** The trunk-exclusion gate was written as "If ANY vehicle has an open trunk, STOP and return false." The VLM correctly applied the gate when a trunk was open but then described negative results as "No open trunks detected, so chalking was not flagged" — inverting the logic. The absence of trunks was treated as a reason *against* chalking rather than clearance to check for it.

**Fix:** Restructured the prompt to make the two outcomes explicit: "If YES (open trunk) → set false, skip STEP 2. If NO open trunk → the exclusion does NOT apply, PROCEED to STEP 2." The positive path ("proceed") was previously implicit and the VLM filled the gap incorrectly.

**Why it matters:** LLMs complete patterns. A gate written as "if X, stop" implies "the absence of X is relevant to the answer" without specifying how. Always spell out both branches of a conditional explicitly, especially for safety gates that must not be inverted.

---

## L15 — `entry_frames` and `buffer_sample_every_n` must be co-designed; mismatched values cause single-frame VLM calls

**Date:** 2026-05-06

**What happened:** `entry_frames` was set to 2 (fire VLM early) and `buffer_sample_every_n` to 5 (store every 5th frame). By the time the VLM fired at count=2, the buffer had only stored one frame (at count=0; count=5 not yet reached). The multi-frame context feature was silently degraded to single-frame on every first call.

**Fix:** Set `entry_frames = (frame_buffer_size - 1) × buffer_sample_every_n + 1`. This ensures the buffer is exactly full at the first VLM trigger. The buffer stores at N=0, B, 2B, …, (F-1)B; the VLM fires when count reaches (F-1)B+1.

**Why it matters:** Two independent config values controlling the same feature create silent misconfiguration. The buffer fill rate and the VLM trigger rate are coupled; deriving one from the other via a documented formula prevents the frame-count mismatch.

---

## L16 — Walking-cane exclusion in CONDITION A was too easy to trigger, suppressing chalk-rod detections

**Date:** 2026-05-06

**What happened:** The prompt excluded a long thin object if it was "unambiguously a vertical walking cane used for body support." At overhead camera distance, a chalk rod held at the side and pointing downward is visually indistinguishable from a walking stick. The VLM regularly applied the walking-cane exclusion to PE officers carrying chalk rods, returning CONDITION A as false.

**Fix:** Raised the exclusion bar significantly: "Do NOT exclude an object as a walking cane unless the person is clearly elderly AND visibly leaning all their body weight on it for balance with each step." A person walking normally while carrying any long thin object near parked vehicles must be flagged.

**Why it matters:** Any exclusion clause in a detection prompt is an escape hatch. The harder it is to trigger, the fewer false negatives it causes. Exclusions should require multiple independent observable features (age + gait + weight-bearing), not a single ambiguous one.

---

## L17 — Removing CONDITION B's posture requirement trades false negatives for false positives

**Date:** 2026-05-07

**What happened:** CONDITION B originally required crouching or bending at a rear wheel. PE officers who chalked while standing were missed. The posture requirement was removed entirely ("any posture at rear wheel → flag"). This immediately caused false positives on car owners standing beside their own vehicles.

**Fix:** Restored a discriminating requirement but shifted it from posture to time: CONDITION B now requires the person to be *stationary* at the rear wheel position across the *majority of frames* in the evaluation window. A PE officer dwells at the wheel for several seconds; a car owner briefly passes it. The multi-frame context window (ADR 020) makes this temporal check possible.

**Why it matters:** Posture and proximity are each individually ambiguous at overhead camera distance. The reliable discriminator is temporal: chalking requires sustained presence at a specific location. Any single-frame criterion (posture, tool, proximity alone) will have unacceptable false-positive or false-negative rates in real street footage.

---

## L18 — A new enum value needs its CSS `.active` rule, or a toggle "won't stick" while actually saving fine

**Date:** 2026-05-23

**What happened:** Added a new `resident` person-type. After selecting it in the kanban modal the button appeared not to "stick." Hours could have gone into the data flow: verified the backend container restarted, that `resident` was in the validation set, that the `POST /person-type` returned 200, that the value persisted in ChromaDB, and that `/api/pipeline/trace` returned `person_type: 'resident'` on the card. All correct. The button even carried the `active` class in the DOM. The real cause: `index.css` had a per-type highlight rule for every category (`.btn-pt-pedestrian.active`, `.btn-pt-occupant.active`, …) **except** the newly added `.btn-pt-resident.active`. So the selected button got the class but no visual style — it looked unselected.

**Fix:** Add the matching `.btn-pt-resident.active` rule. More generally: when adding an enum value that has per-value styling, grep the CSS for the sibling values and add the parallel rule in the same commit.

**Why it matters:** "It saves but the UI doesn't reflect it" has two very different root causes — state/data flow vs. presentation. When the data demonstrably persists (DB + API both show the value) and the element even has the right class, **check the stylesheet before the data flow.** A missing per-variant CSS rule is invisible in logic and easy to overlook when adding a new enum option. Also: adding a frontend enum touches several layers (prompt/validation set, skip/routing logic, the button list, AND the CSS) — miss any one and it silently half-works.

---

## L19 — A model's disk size is not its VRAM footprint, and capacity you can't use is just footprint

**Date:** 2026-05-24

**What happened:** Asked to lower the VLM's "7 GB" GPU usage, the first surprise was that `qwen2.5vl:7b` — listed as 6 GB by `ollama list` — actually occupied **~14 GB** resident (`ollama ps`, 100% GPU, context 4096). For a vision model the disk artifact is just the quantized language weights; loaded, it also carries the vision encoder and the KV cache, often 2x+ the disk size. The probe also varied with context window (a default-context `ollama run` reported 22 GB vs 14 GB at the production 4096). Second surprise: scoring three sizes (moondream 1.3 GB, gemma3:4b 4.4 GB, qwen 14 GB) on 199 labeled person-type events, **all three landed at ~30–35% accuracy** — gemma3:4b (35.2%) even edged the 3x-larger qwen (34.2%), while moondream collapsed to labeling nearly everything "pedestrian."

**Fix:** Picked gemma3:4b (equal accuracy, lower `unknown` rate, ~⅓ the VRAM — ADR-027). Measure loaded footprint with `ollama ps` at the *production context*, never `ollama list`.

**Why it matters:** Two transferable points. (1) **Footprint is the loaded size at your real context, not the download size** — sizing a GPU or comparing models off `ollama list` is wrong by ~2x for VLMs. (2) **When accuracy is flat across model sizes, the task taxonomy is the ceiling, not model capacity** — so the only honest lever is footprint, and the bigger model was paying ~10 GB of VRAM for nothing. This is ADR-024's "bigger isn't better" in the cost-cutting direction. Corollary worth chasing: a classifier stuck at ~35% may not be earning its VLM cost at all.
