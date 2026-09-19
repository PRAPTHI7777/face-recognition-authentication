import os
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .embedder import Embedder


ENROLL_DIR = Path(__file__).resolve().parents[1] / "data" / "enrolled"
ENROLL_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_name(name: str) -> str:
    # simple sanitization for filename use
    return "_".join(name.strip().split()).replace("/", "_")


def _enrollment_path_for(name: str) -> Path:
    safe = _sanitize_name(name)
    return ENROLL_DIR / f"{safe}.json"


def enroll(image_path: str, person_name: str, overwrite: bool = False, ctx_id: int = 0) -> Path:
    """Enroll a person from an image.

    Args:
        image_path: path to the photo (must contain exactly one face)
        person_name: canonical name for the enrolled identity
        overwrite: if False and a record exists for the name, raise FileExistsError
        ctx_id: InsightFace context id (0 for GPU, -1 for CPU) — passed to embedder singleton on first init

    Returns:
        Path to the created enrollment JSON file.

    Raises:
        FileNotFoundError, ValueError, FileExistsError
    """
    embedder = Embedder.get_instance(ctx_id=ctx_id)

    # get single embedding; this validates face count
    emb = embedder.get_embedding_for_single_face(image_path)

    out_path = _enrollment_path_for(person_name)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Enrollment for '{person_name}' already exists at {out_path}. Use overwrite=True to replace.")

    record = {
        "name": person_name,
        "embedding": emb.astype(float).tolist(),
        "source_image": str(image_path),
        "enrolled_at": int(time.time()),
    }

    # write atomically
    tmp = out_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    tmp.replace(out_path)

    return out_path


def load_enrolled() -> list:
    """Load all enrolled identities from disk.

    Returns a list of dicts: {"name": str, "embedding": np.ndarray, "path": Path}
    """
    enrolled = []
    for p in sorted(ENROLL_DIR.glob("*.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            emb = np.array(data.get("embedding", []), dtype=np.float32)
            if emb.size == 0:
                continue
            # ensure normalized
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            emb = emb / norm
            enrolled.append({"name": data.get("name", p.stem), "embedding": emb, "path": p})
        except Exception:
            # skip malformed files but do not raise
            continue
    return enrolled
