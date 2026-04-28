"""Farthest-point sampling for the subject reference-embedding cap.

The plan calls for at most N=20 reference embeddings per subject, picked
to cover pose/lighting/angle variation rather than 20 near-duplicates.
"""
from __future__ import annotations

import numpy as np


def farthest_point_sample(embeddings: list[list[float]], k: int) -> list[int]:
    """Return indices of `k` embeddings selected by farthest-point sampling
    (cosine distance). If `len(embeddings) <= k`, returns all indices."""
    if not embeddings:
        return []
    if len(embeddings) <= k:
        return list(range(len(embeddings)))

    arr = np.asarray(embeddings, dtype="float32")
    # Normalize for cosine distance
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)

    # Seed from the medoid (closest to centroid) for stability across calls
    centroid = arr.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    seed = int(np.argmax(arr @ centroid))

    selected = [seed]
    # Cosine distance = 1 - cos similarity
    min_dist = 1.0 - arr @ arr[seed]
    while len(selected) < k:
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        d = 1.0 - arr @ arr[nxt]
        min_dist = np.minimum(min_dist, d)
    return selected


def cap_reference_embeddings(embeddings: list[list[float]], cap: int) -> list[list[float]]:
    idx = farthest_point_sample(embeddings, cap)
    return [embeddings[i] for i in idx]
