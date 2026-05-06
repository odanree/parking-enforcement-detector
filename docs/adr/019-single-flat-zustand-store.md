# ADR 019 — Single flat Zustand store for shared UI state

## Context

The React dashboard has several pieces of state that are read by multiple components simultaneously:

- `stats` — read by `Header` (pipeline/sweep badges), `VideoToolbar` (pause/speed/motion/privacy buttons), `VideoPanel` (fps badge), `VlmQueue` (hidden when no active jobs)
- `events` — read by `EventLog`; written by `useEvents`
- `debugItems` / `debugOpen` — read by `DebugDrawer` and `Header` (badge count)
- `modalEvent` — read by `EventModal`; written by `EventLog` item clicks
- `wsStatus` — read by `Header` (WS badge) and `VideoPanel` (no-signal overlay)

A prop-drilling approach would require passing `stats` through `App → main → VideoPanel → VideoToolbar` and `App → main → SidePanel → StatsCard`, which is 3–4 levels for a value that changes every 2 s.

React Context was considered but rejected: Context re-renders all consumers on every change. With `stats` updating at 2 s and `pending` at 1 s, every component subscribed to a single context would re-render on every poll, including the canvas-heavy `VideoPanel`.

## Decision

Use a single flat [Zustand](https://zustand-demo.pmnd.rs/) store with selector-based subscriptions.

```ts
const stats = useAppStore((s) => s.stats);         // only re-renders on stats change
const wsStatus = useAppStore((s) => s.wsStatus);   // only re-renders on wsStatus change
```

Each component subscribes to exactly the fields it reads. Zustand only re-renders a component when the value returned by its selector changes (by reference equality). Components that read disjoint fields are never woken up by each other's updates.

The store is flat — no slices, no nested objects beyond the data shapes themselves (`Stats`, `AppEvent[]`, etc.). All setters are named actions on the same store object.

## Consequences

- A component that subscribes to `stats` re-renders every 2 s regardless of whether it uses all fields. This is acceptable; `stats` is a small object and React reconciliation of a single component is sub-millisecond.
- There is no devtools middleware. Adding Zustand devtools is a one-line change if debugging store transitions becomes necessary.
- The store is a module singleton. Tests that import components must reset store state between tests or mock the store module. The current unit tests do not mount React components, so this is not yet an issue.
- `useAppStore.getState()` is used inside async event handlers (fetch callbacks) where the hook call would be invalid. This is the standard Zustand pattern for accessing store state outside of render.
