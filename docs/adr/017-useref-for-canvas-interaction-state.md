# ADR 017 — useRef for canvas interaction state; useState only for JSX branching

## Context

`ZoneOverlay` and `PrivacyOverlay` manage a significant amount of mutable state: polygon points, drag indices, history stacks, pan origin, etc. Each mouse event during an edit session can change multiple of these values. If these were stored in `useState`, every mouse-move event during a drag would trigger a React re-render, and the canvas would have to be redrawn indirectly through React's reconciliation cycle rather than directly in the event handler.

## Decision

All canvas interaction state lives in `useRef`:

- `pointsRef`, `historyRef`, `savedRef`, `dragIdx`, `dragEdge`, `dragPrev`, `didDrag` in `ZoneOverlay`
- `regionsRef`, `draftRef`, `dragRef` in `PrivacyOverlay`
- `zoom`, `panX`, `panY`, `panning`, `panOriginX`, `panOriginY` in the `ZoomableImage` subcomponent of `EventModal`

Canvas redraws are triggered imperatively (`drawZone(canvas, ...)`) inside event handlers and a bare `useEffect(() => { redraw(); })` (no deps — runs after every render that does happen).

`useState` is reserved for values that gate JSX branching:

- `editing: boolean` in `VideoPanel` — controls whether the toolbar row is rendered and whether the canvas class/pointer-events are active
- `modalEvent` in the Zustand store — controls whether `EventModal` renders at all
- The 10 fps tick counter in `VlmQueue` — only exists to force re-renders for the elapsed timer

## Consequences

- Mouse-move and wheel events during canvas interactions produce **zero React re-renders**. Canvas frames are drawn directly by the event handler.
- The `useEffect(() => { redraw(); })` pattern with no deps array is intentional and not a mistake — it syncs the canvas to whatever React state did cause a re-render (e.g. `editing` toggling), without adding those values to a deps array that would otherwise re-attach all event listeners.
- `editing` state is lifted to `VideoPanel` rather than living inside the overlay components. This is necessary so the zone-toolbar and privacy-toolbar DOM nodes render as siblings of `.video-toolbar` in `VideoPanel`, not nested inside `toolbar-right`. Overlays accept `editing: boolean` and `onDone: () => void` as props and use a `useEffect` on `editing` to run their enter-edit initialization logic.
