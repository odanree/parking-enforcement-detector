# ADR 014 — Video Canvas Aspect Ratio with `min()` for Ultrawide Monitors

## Context

After fixing the canvas to maintain 16:9 (`aspect-ratio: 16/9; width: 100%`), the video panel on ultrawide monitors (3440px+) was receiving a `1fr` grid column of 3000+ px. At 16:9 that computed to a canvas taller than the viewport, clipping the bottom of the feed and leaving almost no room for the side panel.

## Decision

Cap the canvas width using `width: min(100%, calc((100vh - 110px) * 16 / 9))`:

- `100%` — full panel width on narrow screens (normal behavior)
- `calc((100vh - 110px) * 16 / 9)` — the width that produces exactly the available viewport height at 16:9, accounting for the header (49px) and toolbar (~60px)

`min()` picks whichever is smaller, so the canvas never grows wider than what the screen height can show. On ultrawide the canvas caps at ~1700px, leaving the side panel fully usable. On standard 1080p it fills naturally.

`align-items: center` on `.video-panel` centers the canvas horizontally when it is narrower than the panel. All toolbars get `width: 100%; box-sizing: border-box` so they span the full panel regardless.

## Consequences

- The video panel may show black padding on either side of the canvas on very wide screens — intentional, matching standard video player behavior.
- The `110px` offset is an approximation of header + toolbar height. If the toolbar wraps to two rows, the canvas may slightly exceed viewport height; this is a tolerable edge case.
- Zone editing coordinate mapping (`toFrame`) is unaffected because it reads `getBoundingClientRect()` on the overlay canvas directly.
