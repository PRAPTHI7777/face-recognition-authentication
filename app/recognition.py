import os
from typing import Dict, Any

import numpy as np

from .embedder import Embedder
from .enrollment import load_enrolled
from .matcher import cosine_similarity


def recognize(image_path: str, threshold: float = 0.5, ctx_id: int = 0) -> Dict[str, Any]:
    """Recognize a single face in an image against enrolled identities.

    Returns a dict with keys:
      - status: one of 'no_face', 'multiple_faces', 'no_enrollments', 'unknown', 'matched'
      - message: human-readable explanation
      - name: matched name when status == 'matched'
      - similarity: float similarity of the best match (if applicable)

    The function will not throw for expected conditions (no faces, multiple faces, no enrollments);
    it raises FileNotFoundError if the image is missing or other IO errors if the model fails.
    """
    embedder = Embedder.get_instance(ctx_id=ctx_id)

    # get embeddings for all detected faces
    try:
        embeddings = embedder.get_embeddings(image_path)
    except FileNotFoundError:
        raise

    if len(embeddings) == 0:
        return {"status": "no_face", "message": "No face detected in the image"}
    if len(embeddings) > 1:
        return {"status": "multiple_faces", "message": "Multiple faces detected; please provide image with exactly one face"}

    emb = embeddings[0]
    if emb is None:
        return {"status": "no_face", "message": "Detected face had invalid embedding"}

    enrolled = load_enrolled()
    if len(enrolled) == 0:
        return {"status": "no_enrollments", "message": "No enrolled identities found"}

    best = None
    best_sim = -1.0
    for record in enrolled:
        sim = float(cosine_similarity(emb, record["embedding"]))
        if sim > best_sim:
            best_sim = sim
            best = record

    # ensure best exists
    if best is None:
        return {"status": "no_enrollments", "message": "No valid enrolled identities found"}

    if best_sim >= threshold:
        return {"status": "matched", "message": f"Matched {best['name']}", "name": best["name"], "similarity": float(best_sim)}
    else:
        return {"status": "unknown", "message": "No enrolled identity passed the threshold", "similarity": float(best_sim)}
