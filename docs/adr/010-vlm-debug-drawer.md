# ADR 010 — VLM Debug Drawer for Rejected Frames

## Context

During testing it was not possible to tell why the VLM was returning negative results. The pipeline was submitting crops to the VLM and silently discarding non-detections. Without visibility into what was being sent and what the VLM said, tuning the prompt was guesswork.

## Decision

Every VLM call that returns a negative result (all three flags false) is stored in a capped in-memory deque (`AppState._debug_rejected`, maxlen=100) along with the crop thumbnail, confidence score, description, and timestamp.

A "Debug" button in the dashboard header opens a slide-out drawer (right side on desktop, bottom sheet on mobile) showing all rejected crops with full VLM descriptions. A red badge on the button counts accumulated rejections. A "Clear" button resets the list via `DELETE /api/debug/rejected`.

The pending-jobs card was also extended: jobs stay visible after completion for 4 seconds showing a ✓/✗ overlay and "Detected / Not detected" label so the user can see the VLM result in context.

## Consequences

- Rejected crops are held in memory only — they disappear on server restart. This is intentional; the drawer is a live debugging tool, not a persistent log.
- At maxlen=100 with thumbnails at ~5 KB each, memory overhead is ~500 KB worst case.
- The drawer directly exposed the llava:7b prompt sensitivity problem (ADR 011) within one test run — confirming the value of in-app observability over log-file debugging.
