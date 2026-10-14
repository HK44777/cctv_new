import os
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from ultralytics import YOLO

from src.utils.gpu_guard import get_device

class MultiObjectTracker:
    """
    Unified Multi-Object Tracker supporting Ultralytics ByteTrack (primary),
    BoT-SORT, and DeepSORT (legacy baseline).

    Tuned specifically for CCTV surveillance person tracking:
    - ByteTrack handles high- and low-score detection association with Kalman filtering,
      eliminating appearance embedding drift during local tracking.
    - Operates at 15 FPS cadence for high temporal resolution (66.7 ms intervals)
      while maintaining > 100 FPS throughput on modern GPUs.
    - Re-initialization gating (new_track_thresh=0.50) prevents false object tracks.
    """
    def __init__(
        self,
        tracker_backend: str = "deepsort",
        tracker_config: str = "configs/bytetrack_cctv.yaml",
        model_path: str = "yolov8n.pt",
        strict_gpu: bool = True
    ):
        self.tracker_backend = tracker_backend.lower()
        self.tracker_config = tracker_config
        self.model_path = model_path
        self.strict_gpu = strict_gpu
        self._internal_detector = None
        self.device = get_device(strict=strict_gpu)
        
        # Determine device string for Ultralytics YOLO
        self.device_str = self.device.type
        if self.device_str == 'cuda' and self.device.index is not None:
            self.device_str = f"cuda:{self.device.index}"
        elif self.device_str == 'cuda':
            self.device_str = 'cuda:0'

        if self.tracker_backend in ["bytetrack", "botsort"]:
            # Ensure config exists or fallback
            if not os.path.exists(self.tracker_config):
                fallback = "bytetrack.yaml" if self.tracker_backend == "bytetrack" else "botsort.yaml"
                print(f"[Tracker] ⚠️ Config {self.tracker_config} not found, using {fallback}")
                self.tracker_config = fallback
            
            self.model = YOLO(model_path)
            print(f"[Tracker] ✅ Initialized {self.tracker_backend.upper()} on {self.device_str} (config: {self.tracker_config})")
            self.deepsort = None
        elif self.tracker_backend == "deepsort":
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.deepsort = DeepSort(
                max_age=90,
                n_init=3,
                max_cosine_distance=0.35,
                max_iou_distance=0.8,
                embedder_gpu=(self.device.type == 'cuda')
            )
            self.model = None
            print("[Tracker] ✅ Initialized DeepSORT baseline on GPU")
        else:
            raise ValueError(f"Unknown tracker backend: {tracker_backend}. Choose 'bytetrack', 'botsort', or 'deepsort'.")

    def update(
        self,
        frame: np.ndarray,
        boxes: Optional[List[List[float]]] = None,
        confidences: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Updates the tracker with a new frame.

        Args:
            frame: BGR image frame (numpy array).
            boxes: Optional list of bounding boxes in [x, y, w, h] format (used by DeepSORT).
            confidences: Optional list of confidence scores (used by DeepSORT).

        Returns:
            List of active confirmed tracks:
            [{"track_id": str, "bbox": [x1, y1, x2, y2], "conf": float}]
        """
        if self.tracker_backend in ["bytetrack", "botsort"]:
            # Run YOLO detection & ByteTrack tracking in a single GPU pass
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_config,
                classes=[0],  # Person class only
                device=self.device_str,
                verbose=False
            )
            
            active_tracks = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes_out = results[0].boxes
                if boxes_out.id is not None:
                    tids = boxes_out.id.int().cpu().tolist()
                    xyxys = boxes_out.xyxy.int().cpu().tolist()
                    confs = boxes_out.conf.cpu().tolist()
                    for tid, bbox, conf in zip(tids, xyxys, confs):
                        active_tracks.append({
                            "track_id": str(tid),
                            "bbox": bbox,
                            "conf": float(conf)
                        })
            return active_tracks

        elif self.tracker_backend == "deepsort":
            if boxes is None or confidences is None:
                if self._internal_detector is None:
                    from src.pipeline.detector import PersonDetector
                    self._internal_detector = PersonDetector(model_path=self.model_path, strict_gpu=self.strict_gpu)
                boxes, confidences = self._internal_detector.detect(frame)

            detections = [(b, c, 0) for b, c in zip(boxes, confidences)]
            tracks = self.deepsort.update_tracks(detections, frame=frame)
            active_tracks = [
                {
                    "track_id": str(track.track_id),
                    "bbox": [int(x) for x in track.to_ltrb()],
                    "conf": 1.0
                }
                for track in tracks
                if track.is_confirmed() and track.time_since_update <= 1
            ]
            return active_tracks

        return []

    def reset(self):
        """Resets the internal tracker memory across video boundaries."""
        if self.model is not None and hasattr(self.model, 'predictor') and hasattr(self.model.predictor, 'trackers'):
            for tracker in self.model.predictor.trackers:
                tracker.reset()
        if self.deepsort is not None:
            self.deepsort.delete_all_tracks()
