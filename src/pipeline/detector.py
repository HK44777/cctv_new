import sys
from typing import Tuple, List
import numpy as np
from ultralytics import YOLO

from src.utils.gpu_guard import get_device


class PersonDetector:
    def __init__(self, model_path: str = "yolov8n.pt", strict_gpu: bool = True):
        """
        Initialize the YOLOv8 Person Detector.
        
        Args:
            model_path: Path to the YOLOv8 weights file.
            strict_gpu: If True, requires a GPU. If False, allows CPU fallback.
        """
        self.device = get_device(strict=strict_gpu)
        
        # Determine the correct device string for YOLO
        self.yolo_device = self.device.type
        if self.yolo_device == 'cuda' and self.device.index is not None:
            self.yolo_device = f"cuda:{self.device.index}"
        elif self.yolo_device == 'cuda':
            self.yolo_device = 'cuda:0'  # Default to 0 if PyTorch didn't assign an index

        try:
            # Load the YOLO model
            self.model = YOLO(model_path)
            print(f"[PersonDetector] ✅ Initialized YOLOv8 on {self.yolo_device}")
        except Exception as e:
            print(f"[PersonDetector] ❌ Failed to load YOLO model: {e}", file=sys.stderr)
            raise

    def detect(self, frame: np.ndarray) -> Tuple[List[List[float]], List[float]]:
        """
        Detects people in a given frame.

        Args:
            frame: Enhanced BGR frame (numpy array).
            
        Returns:
            Tuple containing:
            - List of bounding boxes in [x1, y1, width, height] format.
            - List of confidences (floats) for the 'person' class.
        """
        # Run inference. classes=[0] ensures we only detect 'person'
        results = self.model(frame, classes=[0], device=self.yolo_device, verbose=False)

        boxes = []
        confidences = []

        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                # YOLO returns bounding box in xyxy format (x1, y1, x2, y2)
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].item())

                # DeepSORT tracking algorithm expects [left, top, width, height]
                x1, y1, x2, y2 = xyxy
                w = x2 - x1
                h = y2 - y1

                # Append as standard Python floats
                boxes.append([float(x1), float(y1), float(w), float(h)])
                confidences.append(conf)

        return boxes, confidences
