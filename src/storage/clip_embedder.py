"""Persistent CLIP image embedder.

ChromaDB's built-in OpenCLIPEmbeddingFunction re-instantiates the model on
every call in our usage, which makes both backfill and the live pipeline reload
~350 MB of weights per event. This wraps a single open_clip model loaded once
and exposes batched image embedding, so the model stays resident.

Embeddings are L2-normalised, so cosine distance in ChromaDB is meaningful.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "ViT-B-32"
_PRETRAINED = "laion2b_s34b_b79k"


class ClipEmbedder:
    """Loads open_clip once; embeds BGR frames to normalised vectors."""

    def __init__(self, model_name: str = _MODEL_NAME, pretrained: str = _PRETRAINED) -> None:
        import torch
        import open_clip
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._model.eval().to(self._device)
        self.dim = int(self._model.visual.output_dim)
        logger.info("ClipEmbedder ready (%s/%s, device=%s, dim=%d)",
                    model_name, pretrained, self._device, self.dim)

    def embed_images(self, bgr_images: list[np.ndarray]) -> list[list[float]]:
        from PIL import Image
        tensors = []
        for bgr in bgr_images:
            if bgr is None or bgr.size == 0:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensors.append(self._preprocess(Image.fromarray(rgb)))
        if not tensors:
            return []
        batch = self._torch.stack(tensors).to(self._device)
        with self._torch.no_grad():
            feats = self._model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32").tolist()

    def embed_image(self, bgr: np.ndarray) -> list[float] | None:
        out = self.embed_images([bgr])
        return out[0] if out else None
