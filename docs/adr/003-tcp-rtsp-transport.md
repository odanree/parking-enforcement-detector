# ADR 003 — Force TCP for RTSP stream transport

**Status**: Accepted
**Date**: 2026-05-04

## Context

OpenCV's VideoCapture with FFmpeg defaults to **UDP** for RTSP streams. UDP is
lower-latency but unreliable: dropped packets are not retransmitted and result
in partial frames being delivered to the decoder.

On a busy WiFi network (2.4 GHz, multiple devices, microwave interference)
these partial frames manifest as horizontal smearing artifacts — bands of the
previous frame blended into the current one. This is dangerous for this
pipeline:

1. YOLO false positives — smearing in the street zone can create bounding boxes
   around artifacts that look like people
2. VLM false positives — the spec explicitly notes that frame "smearing" can
   look like chalk marks to the AI (Section 4 of the original spec)
3. Silent failure — the pipeline continues running, processing corrupt frames,
   with no error logged

## Decision

Force TCP transport by setting the FFmpeg environment variable before any
`cv2.VideoCapture` is opened:

```python
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
```

This is done at module import time in `rtsp_handler.py` so it applies
process-wide, regardless of call order.

TCP RTSP retransmits lost packets and delivers complete frames at the cost of
slightly higher latency (~50–150 ms additional buffering). For a parking
enforcement use case (events unfold over seconds, not milliseconds) this
trade-off is unambiguously correct.

## Consequences

**Positive:**
- Eliminates frame smearing and the false positives it causes
- Works transparently — camera firmware needs no configuration change
- The env var approach means camera URL strings stay clean (no query params)

**Negative:**
- ~100 ms additional end-to-end latency vs. UDP — irrelevant for this use case
- If the camera only supports UDP (rare, mostly older models), the stream will
  fail to open rather than falling back to UDP; the reconnect loop in
  `RTSPHandler` will retry indefinitely
- Must be set before the first `VideoCapture` call — late imports could
  theoretically open a UDP stream before the env var is set
