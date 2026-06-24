"""Vector similarity helpers (cosine), used by topic matching, KB ranking, and seed dedup."""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def best_match(query: list[float], candidates: list[list[float]]) -> tuple[int, float]:
    """Return (index, score) of the most similar candidate. (-1, 0.0) if none."""
    best_idx, best_score = -1, -1.0
    for i, c in enumerate(candidates):
        s = cosine_similarity(query, c)
        if s > best_score:
            best_idx, best_score = i, s
    if best_idx == -1:
        return -1, 0.0
    return best_idx, best_score
