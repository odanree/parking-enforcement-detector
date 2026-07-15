# ADR 025 — Re-localize the person on the hi-res snapshot

**Status:** Accepted
**Date:** 2026-05-22

## Context

The dashboard renders a green YOLO bbox on a 4K "HI-RES" frame. On cam1 the box
visibly trailed a walking person — the operator flagged it as "the bbox lagging
behind the person." It is.

The displayed/analysed image and the bbox come from **two independent captures
taken at different instants**:

- The YOLO bbox is computed on the **RTSP frame** (`detector.detect`, run every
  frame), then resized to the detection resolution `1280×720`.
- The **HI-RES image** is a **separate HTTP fetch** from the camera's
  `snapshot.cgi` endpoint (`fetch_hires_jpeg`, `src/stream/hires_snapshot.py`),
  issued only when a person becomes a chalking candidate.

`_annotate_hires` simply scales the RTSP bbox up onto the snapshot. Scaling
fixes resolution, **not time** — the snapshot lands a few hundred ms after the
RTSP frame (network + camera capture latency), so a moving person is drawn where
they *were*.

Measured resolutions (live cameras, 2026-05-22):

| Camera | RTSP `stream1` | snapshot.cgi main | Aspect match? |
|---|---|---|---|
| cam1 | 1024×480 (2.13:1) | 4096×1944 (2.11:1) | yes — same FOV, 4× |
| cam0 | 704×576 (4:3) | 3840×2160 (16:9) | **no** — different crop |

For **cam1** (the reported case) the FOV matches and the scale composes exactly
(`resize 1024→1280 ×1.25` then `annotate 1280→4096 ×3.2` = ×4.0; vertical
×1.5·×2.7 = ×4.05 = `480→1944`). So the geometry is correct and the residual
offset is **purely temporal lag**. (cam0 additionally has a genuine FOV/crop
mismatch — out of scope here, noted below.)

Worse, the lag isn't only cosmetic: the 1280×720 VLM window and the classifier
crop (`_crop_vlm_from_hires`, `_native_person_crop`) are **centered on the same
lagged bbox**, so the VLM was analysing a crop centered where the person used to
be — degrading classification, not just the overlay.

Drawing the box on the RTSP frame instead would fix alignment but throw away the
4K image the operator wants for evidence and for the VLM.

## Decision

**Re-run YOLO on the hi-res snapshot itself** and use that box, taken at the
snapshot's own instant, for everything downstream. Keep the 4K image.

- New `ObjectDetector.localize_person(frame, search_bbox)` — a **stateless**
  single-frame `predict()` (no `.track()`, no stationary mask, no grid
  suppression) that returns the person bbox in snapshot-pixel coords nearest the
  scaled RTSP center.
- **ROI search, not full-frame.** A distant person is only tens of px tall in a
  4K frame; a full-frame `predict` (default `imgsz=640`, ~6× downscale) erases
  them — measured **0 detections** on a real 3840×2160 snapshot. So the seed
  (scaled RTSP bbox) defines an ROI padded 2× its size; predict runs on that
  native-res crop at `imgsz=1280` and the box is offset back to full-frame
  coords. Same snapshot via the ROI path: people **are** recovered. Without
  this, re-localization would silently fall back to the lagged bbox for exactly
  the small/moving people it's meant to fix.
- It runs on a **separate, lazily-loaded YOLO instance** (`self._loc_model`),
  *not* `self._model`. A `predict()` on the tracking model would fire the
  persistent ByteTrack/BoT-SORT callbacks and corrupt track-ID continuity that
  the whole pipeline keys on (chalking counters, classifier cache, wand
  tracker). The second instance shares the same weights file; lazy load means
  the extra memory is paid only when the feature actually runs.
- In `pipeline.py`, after the snapshot is fetched, all hi-res consumers (green
  box, VLM crop, classifier crop, wand crop + motion mask) are routed through a
  single `(_hb, _hb_w, _hb_h)` triple — the snapshot-native bbox and dims, so
  the existing scale-by-`w/det_w` helpers become identity. Passing the same
  triple to both `_native_person_crop` and `_motion_mask_for_crop` keeps the
  det-res MOG2 mask aligned to the native crop.
- **Fallback**: if no person is found on the snapshot (they left frame during Δ,
  or YOLO misses at the new scale), keep the scaled RTSP bbox and log it — never
  worse than before.
- Gated by `HIRES_RELOCALIZE` (default **true**); set false to revert to
  scale-only behaviour.

## Consequences

**Positive**
- Box, 4K image, VLM crop and classifier crop all align to one instant — fixes
  both the overlay lag and the off-center analysis crops.
- 4K evidence frame is preserved.
- Tracker state untouched (separate model instance).

**Negative / watch out**
- One extra YOLO inference per chalking candidate event (not per frame) —
  negligible against the VLM call that follows.
- A second YOLO model resident in RAM once the feature fires; counter to ADR-024
  leanness, but it's the small `yolov8n`-class weights, lazy-loaded, and only
  while events occur. Disable via `HIRES_RELOCALIZE=false` if footprint matters
  more than alignment.
- Nearest-center matching can pick the wrong person if two are very close in the
  snapshot; acceptable, and no worse than the scaled bbox would be.
- **cam0's FOV/crop mismatch is not addressed here** — its 4:3 RTSP vs 16:9
  snapshot needs a separate fix (matched substreams, or a calibrated crop
  transform). Re-localization happens to sidestep it for the *box* (the snapshot
  box is now self-consistent), but the RTSP→snapshot center seed is coarse on
  that camera.

## Principle reinforced

When an overlay is drawn on a frame other than the one it was computed from,
suspect a **two-capture timing gap** before a coordinate-math bug. Here the math
was correct; the captures simply weren't the same moment. Confirmed by probing
both streams' actual resolutions rather than guessing.
