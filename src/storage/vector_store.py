"""Persistent vector store for VLM evaluation events.

Each chalking evaluation (positive and negative) is stored here for labeling
and training-data purposes.  ChromaDB embeds the VLM description text so
similar events can be retrieved by semantic search.

Frame images are saved as JPEG files in the dataset directory; only filenames
are kept in ChromaDB metadata to avoid storing large base64 blobs in SQLite.
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from pathlib import Path

import chromadb

logger = logging.getLogger(__name__)

_LABELS = {"true_positive", "false_positive", "true_negative", "false_negative"}


def _try_default_ef():
    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        return DefaultEmbeddingFunction()
    except Exception:
        logger.warning("ChromaDB DefaultEmbeddingFunction unavailable — semantic search disabled")
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
        logger.info("Vector store ready (%d events)", self._col.count())

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        description: str,
        detected: bool,
        confidence: float,
        camera_id: int,
        thumbnail_b64: str = "",
        frames_b64: list[str] | None = None,
    ) -> str:
        event_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"

        thumb_file = self._save_image(event_id, thumbnail_b64, "thumb") if thumbnail_b64 else ""
        frame_files: list[str] = []
        for i, f in enumerate(frames_b64 or []):
            fname = self._save_image(event_id, f, f"f{i}")
            if fname:
                frame_files.append(fname)

        metadata: dict = {
            "detected":    int(detected),
            "confidence":  float(confidence),
            "camera_id":   int(camera_id),
            "timestamp":   float(time.time()),
            "label":       "",
            "thumb_file":  thumb_file,
            "frame_files": ",".join(frame_files),
        }
        try:
            self._col.add(
                documents=[description or "no description"],
                metadatas=[metadata],
                ids=[event_id],
            )
        except Exception:
            logger.exception("Failed to add event %s to vector store", event_id)
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

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_all(self, offset: int = 0, limit: int = 50) -> dict:
        total = self._col.count()
        result = self._col.get(
            include=["documents", "metadatas"],
            limit=limit,
            offset=offset,
        )
        items = self._flatten(result)
        return {"total": total, "offset": offset, "limit": limit, "items": items}

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

    def count(self) -> int:
        return self._col.count()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _save_image(self, event_id: str, b64: str, suffix: str) -> str:
        filename = f"{event_id}_{suffix}.jpg"
        try:
            (self._dataset / filename).write_bytes(base64.b64decode(b64))
            return filename
        except Exception:
            logger.debug("Could not save image %s", filename)
            return ""

    @staticmethod
    def _flatten(result: dict) -> list[dict]:
        return [
            {"id": eid, "description": doc, **meta}
            for eid, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]
