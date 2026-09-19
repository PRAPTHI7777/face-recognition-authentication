import os
import numpy as np
from insightface.app import FaceAnalysis


class Embedder:
    """Wrapper around InsightFace FaceAnalysis that prepares the model once and
    provides utility methods to get embeddings from image paths.

    Usage:
        emb = Embedder(ctx_id=0)
        embeddings = emb.get_embeddings("photo.jpg")
    """

    _instance = None

    def __init__(self, name: str = "buffalo_l", ctx_id: int = 0, det_size=(640, 640)):
        # initialize FaceAnalysis and prepare the model
        self.name = name
        self.ctx_id = ctx_id
        self.det_size = det_size
        self.app = FaceAnalysis(name=name)
        # prepare may be somewhat slow; do it once per process
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    @classmethod
    def get_instance(cls, name: str = "buffalo_l", ctx_id: int = 0, det_size=(640, 640)):
        """Return a singleton Embedder instance (simple reuse to avoid repeated loads)."""
        if cls._instance is None:
            cls._instance = cls(name=name, ctx_id=ctx_id, det_size=det_size)
        return cls._instance

    def get_faces(self, image_path: str):
        """Load an image and run face detection/analysis.

        Returns the list of detected face objects from insightface.app.FaceAnalysis.get()
        """
        import cv2

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image or unsupported format: {image_path}")

        faces = self.app.get(img)
        return faces

    def get_embedding_for_single_face(self, image_path: str) -> np.ndarray:
        """Detect faces in image and return a single 512-d L2-normalized embedding.

        Raises ValueError for 0 or multiple faces.
        """
        faces = self.get_faces(image_path)
        if len(faces) == 0:
            raise ValueError("No face detected in image")
        if len(faces) > 1:
            raise ValueError("Multiple faces detected in image; expected exactly one")

        emb = np.array(faces[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm == 0:
            raise ValueError("Extracted embedding has zero norm")
        emb = emb / norm
        return emb

    def get_embeddings(self, image_path: str):
        """Return L2-normalized embeddings for all detected faces (list of numpy arrays)."""
        faces = self.get_faces(image_path)
        embeddings = []
        for f in faces:
            emb = np.array(f.embedding, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm == 0:
                # skip zero embeddings but include a placeholder
                embeddings.append(None)
            else:
                embeddings.append(emb / norm)
        return embeddings
