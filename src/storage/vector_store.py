"""Persistent vector store for VLM evaluation events.

Each chalking evaluation (positive and negative) is stored here for labeling
and training-data purposes.  ChromaDB embeds the VLM description text so
similar events can be retrieved by semantic search.

Frame images are saved as JPEG files in the dataset directory; only filenames
are kept in ChromaDB metadata to avoid storing large base64 blobs in SQLite.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import time
import uuid
from pathlib import Path

import chromadb
import numpy as np
import cv2

logger = logging.getLogger(__name__)

_LABELS = {"true_positive", "false_positive", "true_negative", "false_negative"}
PERSON_TYPES = {"pedestrian", "occupant", "worker_landscape", "worker_delivery", "chalker", ""}


def _try_default_ef():
    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        ef = DefaultEmbeddingFunction()
        ef(["warmup"])  # trigger model download now so the first add() doesn't fail
        return ef
    except Exception:
        logger.warning("ChromaDB DefaultEmbeddingFunction unavailable — semantic search disabled")
        return None


def _try_clip_embedder():
    """Return a persistent ClipEmbedder or None.

    Loads the open_clip model ONCE (unlike ChromaDB's OpenCLIPEmbeddingFunction,
    which reloaded weights per call). open_clip is an optional dependency; if it
    isn't installed we run without image embeddings.
    """
    if os.getenv("CLIP_EMBEDDINGS_ENABLED", "false").lower() != "true":
        return None
    try:
        from src.storage.clip_embedder import ClipEmbedder
        return ClipEmbedder()
    except Exception as exc:
        logger.warning(
            "CLIP embedder unavailable (%s) — install open-clip-torch and pillow "
            "to enable image-similarity search.", exc.__class__.__name__,
        )
        return None


class EventVectorStore:
    def __init__(
        self,
        db_path: str = "data/vectors",
        dataset_path: str = "dataset",
    ) -> None:
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self._dataset = Path(dataset_path)
        self._dataset.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=db_path)
        ef = _try_default_ef()
        self._col = self._client.get_or_create_collection(
            name="chalking_evals",
            embedding_function=ef,
        )

        # Parallel CLIP image-embedding collection (opt-in). We compute vectors
        # ourselves with a persistent model and write them via embeddings=, so
        # the collection has NO embedding_function (ChromaDB never re-embeds).
        self._clip_col = None
        self._clip_embedder = _try_clip_embedder()
        if self._clip_embedder is not None:
            try:
                # v2: a fresh collection with NO embedding function. The original
                # "chalking_evals_clip" had OpenCLIPEmbeddingFunction baked into its
                # persisted config, which ChromaDB re-instantiated (reloading the
                # model) on every operation. We supply vectors explicitly instead.
                self._clip_col = self._client.get_or_create_collection(
                    name="chalking_evals_clipv2",
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("CLIP image-embedding collection ready (%d entries)", self._clip_col.count())
            except Exception:
                logger.exception("CLIP collection init failed — continuing text-only")
                self._clip_col = None

        # Persistent dedup: load all known hashes so restarts don't re-add the
        # same frame if the pipeline replays the same footage.
        all_meta = self._col.get(include=["metadatas"], limit=100_000).get("metadatas", [])
        self._thumb_hashes: set[str] = {
            m["thumb_hash"] for m in all_meta if m.get("thumb_hash")
        }
        count = self._col.count()
        logger.info("Vector store ready (%d events, %d unique thumbs)", count, len(self._thumb_hashes))

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        description: str,
        detected: bool,
        confidence: float,
        camera_id: int,
        thumbnail_b64: str = "",
        frames_b64: list[str] | None = None,
        model_primary: str = "",
        model_confirm: str = "",
        yolo_confidence: float | None = None,
        person_type: str = "",
        capture_source: str = "chalking",
        hires_b64: str = "",
        timestamp: float | None = None,
    ) -> str:
        thumb_hash = _hash_b64(thumbnail_b64) if thumbnail_b64 else ""
        if thumb_hash and thumb_hash in self._thumb_hashes:
            logger.warning("Skipping duplicate frame (hash %s, source=%s)", thumb_hash, capture_source)
            return ""

        ts = timestamp if timestamp is not None else time.time()
        event_id = f"{int(ts * 1000)}_{uuid.uuid4().hex[:6]}"

        thumb_file = self._save_image(event_id, thumbnail_b64, "thumb") if thumbnail_b64 else ""
        hires_file = self._save_image(event_id, hires_b64, "hires") if hires_b64 else ""
        logger.info("vector_store.add: source=%s hires=%s thumb=%s", capture_source, bool(hires_file), bool(thumb_file))
        frame_files: list[str] = []
        for i, f in enumerate(frames_b64 or []):
            fname = self._save_image(event_id, f, f"f{i}")
            if fname:
                frame_files.append(fname)

        thumb_phash = _phash_b64(thumbnail_b64) if thumbnail_b64 else ""
        metadata: dict = {
            "detected":        int(detected),
            "confidence":      float(confidence),
            "camera_id":       int(camera_id),
            "timestamp":       float(ts),
            "label":           "",
            "person_type":     person_type,
            "capture_source":  capture_source,
            "thumb_file":      thumb_file,
            "hires_file":      hires_file,
            "frame_files":     ",".join(frame_files),
            "thumb_hash":      thumb_hash,
            "thumb_phash":     thumb_phash,
            "model_primary":   model_primary,
            "model_confirm":   model_confirm,
            "yolo_confidence": float(yolo_confidence) if yolo_confidence is not None else -1.0,
            "model_evals":     "",
        }
        try:
            self._col.add(
                documents=[description or "no description"],
                metadatas=[metadata],
                ids=[event_id],
            )
            if thumb_hash:
                self._thumb_hashes.add(thumb_hash)
        except Exception:
            logger.exception("Failed to add event %s to vector store", event_id)

        # CLIP image embedding (mirrors event_id). We embed with the persistent
        # model and store the vector directly. Best-effort; never blocks the
        # main write.
        if self._clip_col is not None and self._clip_embedder is not None and thumbnail_b64:
            try:
                bgr = _decode_b64_to_bgr(thumbnail_b64)
                vec = self._clip_embedder.embed_image(bgr) if bgr is not None else None
                if vec is not None:
                    self._clip_col.add(
                        ids=[event_id],
                        embeddings=[vec],
                        metadatas=[{
                            "detected":       int(detected),
                            "label":          "",
                            "capture_source": capture_source,
                            "camera_id":      int(camera_id),
                        }],
                    )
            except Exception:
                logger.debug("CLIP add failed for %s", event_id, exc_info=True)
        return event_id

    def update_label(self, event_id: str, label: str) -> None:
        if label not in _LABELS and label != "":
            raise ValueError(f"label must be one of {_LABELS} or empty string")
        existing = self._col.get(ids=[event_id], include=["metadatas"])
        if not existing["ids"]:
            raise KeyError(event_id)
        meta = existing["metadatas"][0]
        meta["label"] = label
        self._col.update(ids=[event_id], metadatas=[meta])

    def update_person_type(self, event_id: str, person_type: str) -> None:
        if person_type not in PERSON_TYPES:
            raise ValueError(f"person_type must be one of {PERSON_TYPES}")
        existing = self._col.get(ids=[event_id], include=["metadatas"])
        if not existing["ids"]:
            raise KeyError(event_id)
        meta = existing["metadatas"][0]
        meta["person_type"] = person_type
        self._col.update(ids=[event_id], metadatas=[meta])

    # ── Read ──────────────────────────────────────────────────────────────────

    def delete(self, event_id: str) -> None:
        """Remove an event from the collection and delete its image files."""
        existing = self._col.get(ids=[event_id], include=["metadatas"])
        if not existing["ids"]:
            raise KeyError(event_id)
        meta = existing["metadatas"][0]
        for key in ("thumb_file", "hires_file"):
            fname = meta.get(key, "")
            if fname:
                (self._dataset / fname).unlink(missing_ok=True)
        for fname in meta.get("frame_files", "").split(","):
            if fname:
                (self._dataset / fname).unlink(missing_ok=True)
        self._col.delete(ids=[event_id])
        self._thumb_hashes.discard(meta.get("thumb_hash", ""))
        if self._clip_col is not None:
            try:
                self._clip_col.delete(ids=[event_id])
            except Exception:
                logger.debug("CLIP delete failed for %s", event_id, exc_info=True)

    def get_filtered(
        self,
        since_ts: float | None = None,
        limit: int = 500,
        offset: int = 0,
        label: str | None = None,
        camera_id: int | None = None,
        capture_source: str | None = None,
        detected: int | None = None,
    ) -> list[dict] | dict:
        """Fetch events with optional server-side filters.

        since_ts       — only return events with timestamp >= this value
        label          — '' = unlabeled, 'true_positive' etc., None = any label
        camera_id      — filter to a specific camera, None = all cameras
        capture_source — 'chalking', 'zone_pedestrian', None = all
        detected       — 1 = detections only, 0 = rejections only, None = all

        When offset is non-zero or paginated results are needed, returns a
        dict {total, offset, limit, items}.  Otherwise returns list[dict] for
        backward compatibility with the kanban endpoint.
        """
        conditions: list[dict] = []
        if since_ts is not None:
            conditions.append({"timestamp": {"$gte": since_ts}})
        if label is not None:
            conditions.append({"label": {"$eq": label}})
        if camera_id is not None:
            conditions.append({"camera_id": {"$eq": camera_id}})
        if capture_source is not None:
            conditions.append({"capture_source": {"$eq": capture_source}})
        if detected is not None:
            conditions.append({"detected": {"$eq": detected}})

        where: dict | None = None
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        # Paginated path — return dict with total
        if offset > 0 or limit <= 200:
            try:
                count_kw: dict = {"include": ["metadatas"], "limit": 100_000}
                if where:
                    count_kw["where"] = where
                count_result = self._col.get(**count_kw)
                total = len(count_result["ids"])
            except Exception:
                total = self._col.count()
            kwargs: dict = {"include": ["documents", "metadatas"], "limit": limit, "offset": offset}
            if where:
                kwargs["where"] = where
            try:
                result = self._col.get(**kwargs)
            except Exception:
                logger.exception("get_filtered paginated failed")
                result = {"ids": [], "documents": [], "metadatas": []}
            return {"total": total, "offset": offset, "limit": limit, "items": self._flatten(result)}

        # Legacy path (kanban) — return plain list.
        # ChromaDB get() has no ORDER BY so a bare limit=500 returns arbitrary
        # records, not the most recent 500.  Fetch all matching IDs first (cheap —
        # metadata only), sort them chronologically (IDs are "{ms_ts}_{uuid}"),
        # then retrieve only the most recent `limit` documents.
        try:
            id_kw: dict = {"include": [], "limit": 100_000}
            if where:
                id_kw["where"] = where
            id_result = self._col.get(**id_kw)
            all_ids = sorted(id_result.get("ids", []))          # lexicographic = chronological
            recent_ids = all_ids[-limit:] if len(all_ids) > limit else all_ids
            if not recent_ids:
                return []
            result = self._col.get(ids=recent_ids, include=["documents", "metadatas"])
        except Exception:
            logger.exception("get_filtered failed — falling back to get_all")
            result = self._col.get(include=["documents", "metadatas"], limit=limit)
        return self._flatten(result)

    def get_all(self, offset: int = 0, limit: int = 50) -> dict:
        total = self._col.count()
        result = self._col.get(
            include=["documents", "metadatas"],
            limit=limit,
            offset=offset,
        )
        items = self._flatten(result)
        return {"total": total, "offset": offset, "limit": limit, "items": items}

    def query_similar_by_text(self, text: str, n: int = 5) -> list[dict]:
        """Find the n most similar events to the given description text.

        Returns all neighbors (labeled and unlabeled) sorted by distance.
        Callers should filter on ``label`` if they need only labeled events.
        """
        total = self._col.count()
        if total < 1:
            return []
        results = self._col.query(
            query_texts=[text],
            n_results=min(n, total),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"id": eid, "description": doc, "distance": round(dist, 4), **meta}
            for eid, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def query_similar_by_image(self, image_bytes: bytes, n: int = 5) -> list[dict]:
        """Find n events whose thumbnail is visually closest to the given image.

        Returns rich metadata by cross-referencing back to the main collection.
        No-op (returns []) when the CLIP collection is disabled or empty.
        """
        if self._clip_col is None or self._clip_embedder is None or self._clip_col.count() < 1:
            return []
        bgr = _decode_jpeg_bytes_to_bgr(image_bytes)
        vec = self._clip_embedder.embed_image(bgr) if bgr is not None else None
        if vec is None:
            return []
        try:
            results = self._clip_col.query(
                query_embeddings=[vec],
                n_results=min(n, self._clip_col.count()),
                include=["distances", "metadatas"],
            )
        except Exception:
            logger.exception("CLIP query failed")
            return []
        ids = results.get("ids", [[]])[0]
        dists = results.get("distances", [[]])[0]
        if not ids:
            return []
        # Cross-reference into the main collection for full metadata + description.
        main = self._col.get(ids=ids, include=["documents", "metadatas"])
        by_id = {
            eid: {"id": eid, "description": doc, **meta}
            for eid, doc, meta in zip(main["ids"], main["documents"], main["metadatas"])
        }
        out: list[dict] = []
        for eid, dist in zip(ids, dists):
            row = by_id.get(eid)
            if row is None:
                continue
            row["distance"] = round(float(dist), 4)
            out.append(row)
        return out

    def query_similar_image(self, event_id: str, n: int = 10) -> list[dict]:
        """Image-similarity counterpart to ``query_similar``: nearest by CLIP."""
        if self._clip_col is None:
            return []
        meta = self._col.get(ids=[event_id], include=["metadatas"])
        if not meta["ids"]:
            return []
        thumb_file = meta["metadatas"][0].get("thumb_file", "")
        if not thumb_file:
            return []
        path = self._dataset / thumb_file
        if not path.exists():
            return []
        return [
            r for r in self.query_similar_by_image(path.read_bytes(), n=n + 1)
            if r["id"] != event_id
        ][:n]

    def query_similar(self, event_id: str, n: int = 10) -> list[dict]:
        src = self._col.get(ids=[event_id], include=["documents"])
        if not src["documents"]:
            return []
        total = self._col.count()
        if total < 2:
            return []
        results = self._col.query(
            query_texts=[src["documents"][0]],
            n_results=min(n + 1, total),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for eid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if eid == event_id:
                continue
            out.append({"id": eid, "description": doc, "distance": round(dist, 4), **meta})
        return out[:n]

    def export_labeled(self) -> list[dict]:
        result = self._col.get(
            where={"label": {"$ne": ""}},
            include=["documents", "metadatas"],
        )
        return self._flatten(result)

    def get_frame_bytes(self, event_id: str) -> list[bytes]:
        """Load the stored JPEG frames for an event from disk. Returns [] if missing."""
        result = self._col.get(ids=[event_id], include=["metadatas"])
        if not result["ids"]:
            return []
        frame_files = [
            f for f in result["metadatas"][0].get("frame_files", "").split(",") if f
        ]
        out = []
        for fname in frame_files:
            path = self._dataset / fname
            if path.exists():
                out.append(path.read_bytes())
        return out

    def update_reeval(
        self,
        event_id: str,
        backend: str,
        detected: bool,
        confidence: float,
        description: str,
    ) -> None:
        """Store a second-opinion VLM result alongside the original evaluation."""
        existing = self._col.get(ids=[event_id], include=["metadatas"])
        if not existing["ids"]:
            raise KeyError(event_id)
        meta = existing["metadatas"][0]
        meta["reeval_backend"]     = backend
        meta["reeval_detected"]    = int(detected)
        meta["reeval_confidence"]  = float(confidence)
        meta["reeval_description"] = description
        self._col.update(ids=[event_id], metadatas=[meta])

    def add_model_eval(
        self,
        event_id: str,
        model_name: str,
        backend: str,
        detected: bool,
        confidence: float,
        description: str,
    ) -> None:
        """Append a named model evaluation result to an event's model_evals list.

        Idempotent per model — re-running the same model overwrites its entry.
        """
        import json as _json
        existing = self._col.get(ids=[event_id], include=["metadatas"])
        if not existing["ids"]:
            raise KeyError(event_id)
        meta = existing["metadatas"][0]
        try:
            evals: list[dict] = _json.loads(meta.get("model_evals") or "[]")
        except Exception:
            evals = []
        # Overwrite existing entry for this model
        evals = [e for e in evals if e.get("model") != model_name]
        evals.append({
            "model":       model_name,
            "backend":     backend,
            "detected":    int(detected),
            "confidence":  round(float(confidence), 4),
            "description": description,
            "ts":          float(time.time()),
        })
        meta["model_evals"] = _json.dumps(evals)
        self._col.update(ids=[event_id], metadatas=[meta])

    def clear_pipeline_events(self) -> int:
        """Delete only pipeline detection events (capture_source='chalking').

        Preserves zone_pedestrian captures and any manually labeled dataset records
        so the training dataset survives a kanban history clear.
        Returns the number of records removed.
        """
        all_result = self._col.get(include=["metadatas"], limit=100_000)
        to_delete: list[str] = []
        files_to_remove: list[str] = []

        for eid, meta in zip(all_result["ids"], all_result["metadatas"]):
            # Treat missing/empty capture_source as "chalking" (pre-feature events)
            src = meta.get("capture_source") or "chalking"
            if src == "chalking":
                to_delete.append(eid)
                for fkey in ("thumb_file", "hires_file"):
                    if meta.get(fkey):
                        files_to_remove.append(meta[fkey])
                for f in (meta.get("frame_files") or "").split(","):
                    if f:
                        files_to_remove.append(f)
                if meta.get("thumb_hash"):
                    self._thumb_hashes.discard(meta["thumb_hash"])

        if to_delete:
            self._col.delete(ids=to_delete)
            if self._clip_col is not None:
                try:
                    self._clip_col.delete(ids=to_delete)
                except Exception:
                    logger.debug("CLIP delete-batch failed", exc_info=True)

        removed_files = 0
        for fname in files_to_remove:
            try:
                (self._dataset / fname).unlink(missing_ok=True)
                removed_files += 1
            except Exception:
                pass

        logger.info(
            "Pipeline history cleared: %d events, %d files removed (%d dataset events preserved)",
            len(to_delete), removed_files, self._col.count(),
        )
        return len(to_delete)

    def clear_all(self) -> int:
        """Delete every event from the collection and all dataset images. Returns count removed."""
        count = self._col.count()
        if count:
            all_ids = self._col.get(include=[], limit=100_000)["ids"]
            if all_ids:
                self._col.delete(ids=all_ids)
                if self._clip_col is not None:
                    try:
                        self._clip_col.delete(ids=all_ids)
                    except Exception:
                        logger.debug("CLIP clear_all delete failed", exc_info=True)
        self._thumb_hashes.clear()
        removed_files = 0
        for f in self._dataset.iterdir():
            try:
                f.unlink()
                removed_files += 1
            except Exception:
                pass
        logger.info("Vector store cleared: %d events, %d dataset files removed", count, removed_files)
        return count

    def count(self) -> int:
        return self._col.count()

    def deduplicate(self) -> int:
        """Remove duplicate entries that share the same thumb_hash. Keeps the
        earliest (lowest ID) entry. Returns the number of records removed."""
        result = self._col.get(include=["metadatas"], limit=100_000)
        seen: dict[str, str] = {}   # hash → first event_id
        to_delete: list[str] = []
        for eid, meta in zip(result["ids"], result["metadatas"]):
            h = meta.get("thumb_hash", "")
            if not h:
                continue
            if h in seen:
                to_delete.append(eid)
            else:
                seen[h] = eid
        if to_delete:
            self._col.delete(ids=to_delete)
            logger.info("Deduplicated vector store: removed %d duplicate entries", len(to_delete))
        return len(to_delete)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _save_image(self, event_id: str, b64: str, suffix: str) -> str:
        filename = f"{event_id}_{suffix}.jpg"
        try:
            (self._dataset / filename).write_bytes(base64.b64decode(b64))
            return filename
        except Exception:
            logger.warning("Could not save image %s", filename, exc_info=True)
            return ""

    @staticmethod
    def _flatten(result: dict) -> list[dict]:
        return [
            {"id": eid, "description": doc, **meta}
            for eid, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]


def _decode_b64_to_bgr(b64: str):
    """Decode a base64 JPEG into an (H, W, 3) BGR numpy array (ClipEmbedder converts to RGB)."""
    try:
        return _decode_jpeg_bytes_to_bgr(base64.b64decode(b64))
    except Exception:
        return None


def _decode_jpeg_bytes_to_bgr(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)   # BGR or None


def _hash_b64(b64: str) -> str:
    """MD5 of the raw image bytes (decoded from base64). Used for frame dedup."""
    try:
        return hashlib.md5(base64.b64decode(b64)).hexdigest()
    except Exception:
        return ""


def _phash_b64(b64: str) -> str:
    """8×8 average perceptual hash → 16-char hex. Used for visual grouping."""
    try:
        data  = base64.b64decode(b64)
        arr   = np.frombuffer(data, np.uint8)
        img   = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ""
        small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA).flatten()
        bits  = small > small.mean()
        return f'{int("".join("1" if b else "0" for b in bits), 2):016x}'
    except Exception:
        return ""
