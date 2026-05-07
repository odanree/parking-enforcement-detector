# ADR 021 — ChromaDB vector store for VLM evaluation labeling

## Context

Every VLM call (positive and negative) produces a structured result: description text, confidence, detected flag, camera ID, and frame images. As the system accumulates evaluations across hours of footage, manually reviewing individual events to build a training dataset becomes impractical. The requirement is to:

1. Persist all evaluations (not just confirmed alerts) across container restarts.
2. Surface semantically similar events so they can be batch-labeled rather than reviewed one by one.
3. Export labeled data in a format usable for fine-tuning or few-shot example selection.

## Decision

Use [ChromaDB](https://docs.trychroma.com/) as an embedded persistent vector store.

- **Collection**: `chalking_evals` — one entry per chalking VLM evaluation (positive and negative).
- **Document**: the VLM description string, embedded using `DefaultEmbeddingFunction` (ONNX-based `all-MiniLM-L6-v2`). Semantic similarity over descriptions clusters visually similar scenarios (person with stick, person crouching at wheel, person exiting trunk) without requiring image embeddings.
- **Metadata**: `detected` (int 0/1), `confidence` (float), `camera_id`, `timestamp`, `label` (empty string until labeled), `thumb_file`, `frame_files` (comma-separated filenames).
- **Frame images**: saved as JPEG files under `dataset/` and served at `/dataset/` via FastAPI `StaticFiles`. Filenames are stored in metadata; base64 blobs are not stored in ChromaDB to avoid large SQLite rows.

Four REST endpoints exposed:
- `GET /api/dataset` — paginated list
- `POST /api/dataset/{id}/label` — set label (`true_positive`, `false_positive`, `true_negative`, `false_negative`)
- `GET /api/dataset/similar/{id}` — semantic nearest-neighbours for batch labeling
- `GET /api/dataset/export` — JSON export of all labeled entries

Both pipeline threads share a single `EventVectorStore` instance (passed as a parameter to `pipeline.run()`), consistent with how `AppState` is shared.

Two named Docker volumes persist storage across restarts: `ped_vectors` (`/app/data/vectors`) and `ped_dataset` (`/app/dataset`).

## Consequences

- **Embedding model download on first start**: `DefaultEmbeddingFunction` downloads `all-MiniLM-L6-v2` (~80 MB) on first use. This requires outbound HTTPS from the container. The model is cached in the ChromaDB client directory and not re-downloaded.
- **Thread safety**: ChromaDB's `PersistentClient` uses SQLite with WAL mode and is safe for concurrent writes from multiple pipeline threads.
- **Label vocabulary is open-ended**: the `label` field accepts any of four strings or empty. Future work may add a dedicated labeling UI or stricter schema enforcement.
- **No image embeddings yet**: semantic search operates on description text, not visual content. Two visually identical frames that produce different VLM descriptions will not cluster together. CLIP-based image embeddings would improve recall but add significant dependency weight.
