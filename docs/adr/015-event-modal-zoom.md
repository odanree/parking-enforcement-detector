# ADR 015 — Event Modal: Cursor-Based Zoom with Pan

## Context

The event detail modal shows a snapshot that operators need to inspect closely — license plates, chalk marks, and posture details are often small. A static image at `max-height: 60vh` was insufficient for this analysis.

## Decision

Add scroll-to-zoom and drag-to-pan on the modal image:

- **Scroll wheel** zooms 1× to 8× toward the cursor position.
- **Drag** pans when zoom > 1.
- Zoom and pan reset automatically when the modal opens or closes.

Modal dimensions increased to `max-width: 1300px / max-height: 95vh` and image to `max-height: 78vh` to give more pixel real estate before zooming is needed.

### Zoom-to-cursor math

The transform applied is `scale(zoom) translate(panX/zoom, panY/zoom)`. The screen position of an image point `P` relative to the element's layout center is `screenX = P.x * zoom + panX`.

To keep the image point under the cursor fixed when zoom changes from `s1` to `s2` (ratio `r = s2/s1`):

```
panX2 = panX1 + cursorX * (1 - r)
```

where `cursorX` is the cursor's distance from the visual center of the image, obtained from `getBoundingClientRect()` which returns post-transform bounds.

A wrong earlier implementation used `panX1 * ratio` instead of `panX1`, causing the accumulated pan to drift with each scroll step — the zoom appeared to re-center toward the original cursor position rather than tracking the current one.

## Consequences

- The image container needs `overflow: hidden` to clip the zoomed image to the modal bounds.
- `getBoundingClientRect()` is called on every wheel event; this is fine at human scroll speeds.
- Touch pinch-to-zoom is not implemented — mobile users can use the browser's native zoom on the full-screen image.
