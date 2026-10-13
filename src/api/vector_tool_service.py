"""
vector_tool_service.py
----------------------
Service for extracting OSNet appearance vectors from images.
Pure extraction and Base64 encoding utility — does NOT read or modify any files on disk.

Handles:
  1. Decodes uploaded image (or base64 image data).
  2. Optional YOLO person detection to crop the primary person.
  3. OSNet 512-dim feature vector extraction with L2 normalization.
  4. Base64 encoding of:
     - Vector binary bytes (for ultra-compact, 1-click clipboard copy)
     - Image crop & full frame preview (for visual verification)
"""

import base64
import numpy as np
import cv2
from typing import Dict, Any, Optional

from src.pipeline.detector import PersonDetector
from src.pipeline.extractors.appearance import AppearanceExtractor


class VectorToolService:
    def __init__(self, detector: Optional[PersonDetector] = None, extractor: Optional[AppearanceExtractor] = None):
        # Share models if passed from server to conserve VRAM
        self.detector = detector if detector is not None else PersonDetector(strict_gpu=False)
        self.extractor = extractor if extractor is not None else AppearanceExtractor(strict_gpu=False)

    def extract_from_image(self, image_bytes: bytes, auto_crop: bool = True) -> Dict[str, Any]:
        """
        Extracts 512-dimensional OSNet appearance vector from an image.
        Returns:
          - vector: list of 512 floats (rounded to 6 decimal places)
          - vector_base64: base64-encoded float32 bytes of the vector (easy single-line copy)
          - preview_crop: base64 jpeg data URI of the cropped person
          - preview_orig: base64 jpeg data URI of original image with bounding box
          - stats: vector length, L2 norm, min, max, mean
        """
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image file. Could not decode image.")

        img_h, img_w = frame.shape[:2]
        crop = frame
        detected = False
        box_data = None

        if auto_crop:
            boxes, confidences = self.detector.detect(frame)
            if boxes:
                # Pick the largest bounding box (assumed primary subject)
                best_idx = max(range(len(boxes)), key=lambda i: float(boxes[i][2]) * float(boxes[i][3]))
                x, y, w, h = boxes[best_idx]
                conf = confidences[best_idx] if confidences else 1.0

                x1 = max(0, int(round(x)))
                y1 = max(0, int(round(y)))
                x2 = min(img_w, int(round(x + w)))
                y2 = min(img_h, int(round(y + h)))

                if (x2 > x1) and (y2 > y1):
                    crop = frame[y1:y2, x1:x2]
                    detected = True
                    box_data = {
                        "x": x1,
                        "y": y1,
                        "w": x2 - x1,
                        "h": y2 - y1,
                        "confidence": round(float(conf), 4),
                    }

        # Extract 512-dim vector from crop
        raw_vector = self.extractor.extract(crop)
        if raw_vector is None:
            raise ValueError("Failed to extract appearance vector from the image crop.")

        vec = np.array(raw_vector, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm

        vector_list = [round(float(v), 6) for v in vec]

        # Base64 encode the vector as raw float32 bytes (compact, portable, easy to copy)
        vector_bytes = vec.astype(np.float32).tobytes()
        vector_base64 = base64.b64encode(vector_bytes).decode("ascii")

        # Base64 encode image previews
        _, crop_buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        crop_base64 = "data:image/jpeg;base64," + base64.b64encode(crop_buf).decode("utf-8")

        preview_frame = frame.copy()
        if detected and box_data:
            bx, by, bw, bh = box_data["x"], box_data["y"], box_data["w"], box_data["h"]
            cv2.rectangle(preview_frame, (bx, by), (bx + bw, by + bh), (0, 255, 128), 2)
            label_text = f"Person ({box_data['confidence']*100:.1f}%)"
            cv2.putText(preview_frame, label_text, (bx, max(15, by - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2, cv2.LINE_AA)

        max_dim = 800
        if max(img_h, img_w) > max_dim:
            scale = max_dim / max(img_h, img_w)
            preview_frame = cv2.resize(preview_frame, (int(img_w * scale), int(img_h * scale)))

        _, orig_buf = cv2.imencode(".jpg", preview_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        orig_base64 = "data:image/jpeg;base64," + base64.b64encode(orig_buf).decode("utf-8")

        return {
            "success": True,
            "detected_person": detected,
            "crop_dimensions": {"width": crop.shape[1], "height": crop.shape[0]},
            "original_dimensions": {"width": img_w, "height": img_h},
            "box": box_data,
            "vector_length": len(vector_list),
            "l2_norm": round(norm, 5),
            "stats": {
                "min": round(float(np.min(vec)), 5),
                "max": round(float(np.max(vec)), 5),
                "mean": round(float(np.mean(vec)), 5),
            },
            "vector": vector_list,
            "vector_base64": vector_base64,
            "preview_crop": crop_base64,
            "preview_orig": orig_base64,
        }
