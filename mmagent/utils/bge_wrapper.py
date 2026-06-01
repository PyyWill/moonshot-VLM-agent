"""BGE-M3 text embedding via sentence-transformers (CLS pool + L2 norm).

Used for embedding episodic/semantic text nodes at write time and user queries
at retrieval time. Cosine similarity between BGE-M3 vectors is the text-side
analogue of DINOv2 similarity on the object side.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

_MODEL = None
_DEVICE = None
_DIM = 1024


def build_text_embedder(model_dir: str, device: str = "cuda:2") -> None:
    global _MODEL, _DEVICE, _DIM
    if _MODEL is not None:
        return
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading BGE-M3 from {model_dir} on {device}")
    _MODEL = SentenceTransformer(model_dir, device=device)
    _MODEL.eval()
    _DEVICE = device
    get_dim = getattr(_MODEL, "get_embedding_dimension", None) or _MODEL.get_sentence_embedding_dimension
    _DIM = int(get_dim())


def embed(
    texts: Iterable[str],
    batch_size: int = 16,
    max_length: int | None = 8192,
) -> np.ndarray:
    """Return (N, D) unit-normalized float32 embeddings. D=1024 for BGE-M3."""
    if _MODEL is None:
        raise RuntimeError("call build_text_embedder(...) first")

    texts = list(texts)
    if not texts:
        return np.zeros((0, _DIM), dtype=np.float32)

    if max_length is not None:
        _MODEL.max_seq_length = int(max_length)

    vecs = _MODEL.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32)


def embed_one(text: str) -> np.ndarray:
    """Convenience: embed a single string, return (D,) float32."""
    return embed([text])[0]
