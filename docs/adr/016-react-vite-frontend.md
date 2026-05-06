# ADR 016 — React + Vite frontend replacing Jinja2 template

## Context

The original dashboard was a single Jinja2 template (`src/web/templates/index.html`) with vanilla JS — polling, DOM manipulation, and canvas drawing all inline. The file had grown to ~1 200 lines of mixed HTML/CSS/JS with no component boundaries, making it hard to isolate and test individual behaviours. The debug drawer, event modal, zone overlay, and privacy overlay each had their own ad-hoc state management patterns that were inconsistent with one another.

## Decision

Replace the Jinja2 template with a React 19 + Vite + TypeScript frontend in `frontend/`. Build output goes to `frontend-dist/` and is served by FastAPI via `FileResponse` at `GET /` and a `StaticFiles` mount at `/assets`.

Key choices within this decision:

- **Vite** for dev server (HMR) and production bundling; `vite.config.ts` proxies `/api`, `/snapshots`, and `/ws` to `localhost:8000` so the React dev server and FastAPI coexist without CORS issues.
- **TypeScript** throughout; `tsc -b` runs as part of `npm run build` so type errors block the build.
- **Big-bang replacement** rather than incremental embedding. The old template is fully replaced in one PR. Incremental embedding (React islands inside a Jinja2 page) would have required a complex shared-state bridge between vanilla JS and React for the canvas overlays.

## Consequences

- The `src/web/static/` directory (old dashboard CSS and icons) is no longer served. The `/static` FastAPI mount was removed.
- `src/web/templates/` is no longer used. `Jinja2Templates` and `Request` imports were removed from `app.py`.
- A `frontend/` directory must be built before deploying (`npm run build`). The `frontend-dist/` output is committed or built in CI.
- The Playwright E2E test suite must be updated whenever the React component structure changes class names or DOM hierarchy that tests depend on.
