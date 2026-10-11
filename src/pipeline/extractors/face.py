import numpy as np
import cv2
import torch
from insightface.app import FaceAnalysis
from src.utils.gpu_guard import get_device


class FaceExtractor:
    def __init__(self, strict_gpu=True):
        # strict_gpu=True  → require GPU (ingestion phase on Colab), ctx_id=0
        # strict_gpu=False → allow CPU fallback (query server on local machine), ctx_id=-1
        device = get_device(strict=strict_gpu)

        # ctx_id=0  → GPU 0
        # ctx_id=-1 → CPU
        ctx_id = 0 if device.type == 'cuda' else -1

        # Step 6: Face Extraction (RetinaFace & ArcFace via insightface)
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=ctx_id, det_size=(320, 320))  # 320x320 is sufficient for person crops (not full frames)

        mode = "GPU" if ctx_id == 0 else "CPU"
        print(f"Initialized Face Extractor (RetinaFace/ArcFace via InsightFace) on {mode}")

    def extract(self, person_crop):
        """
        Returns a 512-dimensional vector describing the face.
        Returns None if no face is visible in the crop.
        """
        if person_crop.size == 0 or person_crop.shape[0] < 10 or person_crop.shape[1] < 10:
            return None

        # InsightFace expects BGR images (which person_crop already is from OpenCV)
        faces = self.app.get(person_crop)

        if len(faces) == 0:
            return None  # No face found

        # If multiple faces are found in the crop, take the largest one
        best_face = faces[0]
        if len(faces) > 1:
            max_area = 0
            for face in faces:
                x1, y1, x2, y2 = face.bbox
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_face = face

        # Quality gate: 0.75 threshold rejects blurry/partial faces to keep Qdrant embeddings clean
        if hasattr(best_face, 'det_score') and best_face.det_score < 0.75:
            return None

        embedding = best_face.embedding  # (512,) numpy array, already L2-normalized by buffalo_l internally

        return embedding.tolist()

    def extract_batch(self, crops):
        """
        Returns a list of 512-dimensional face vectors for multiple crops (pseudo-batch).
        If no valid face is found across all crops, returns None.
        """
        if not crops:
            return None

        embeddings = []
        for person_crop in crops:
            if person_crop.size == 0 or person_crop.shape[0] < 10 or person_crop.shape[1] < 10:
                continue

            faces = self.app.get(person_crop)
            if not faces:
                continue

            best_face = faces[0]
            if len(faces) > 1:
                max_area = 0
                for face in faces:
                    x1, y1, x2, y2 = face.bbox
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        best_face = face

            if hasattr(best_face, 'det_score') and best_face.det_score < 0.75:
                continue

            embeddings.append(best_face.embedding.tolist())

        if not embeddings:
            return None

        return embeddings
