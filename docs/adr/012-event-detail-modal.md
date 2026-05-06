# ADR 012 — Event Detail Modal with LLM Response and Alert Button

## Context

The original snapshot lightbox showed only a full-screen image when a user tapped a thumbnail in the event log. It provided no context: the LLM description that caused the alert, the confidence score, or the exact timestamp were only visible in the event log row itself (truncated). There was also no way to escalate an event to an external alert from the UI.

## Decision

Replace the single-image lightbox with a two-panel event detail modal:

- **Left panel**: full snapshot image
- **Right panel**: event type badge (color-coded by type), confidence %, full ISO timestamp, the complete LLM response text, and a **Send Alert** button

The modal closes on backdrop click, the × button, or Escape. On mobile it slides up from the bottom as a sheet (image stacked above info panel).

Event data (type, confidence, timestamp, description, snapshot_url) is stored in a `_eventDataMap` (JS `Map` keyed by snapshot URL) when each event item is built, so the modal can populate without an extra API call.

## Consequences

- The LLM response is now always one tap away from the event thumbnail — no need to read truncated text in the event row.
- The alert button gives operators a one-tap escalation path from the detection directly to a phone notification.
- Memory overhead is negligible: the map holds at most 30 event objects (the event list cap).
