from typing import List, Dict, Any
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class MultiObjectTracker:
    def __init__(self):
        # Step 3: Multi-Object Tracking (DeepSORT)
        # Using a maximum age of 60 frames (2.0s) to bridge occlusions and prevent track fragmentation
        # Enforce GPU for the internal MobileNetV2 embedder
        self.tracker = DeepSort(max_age=60, n_init=3, embedder_gpu=True)
        print("[Tracker] ✅ Initialized Multi-Object Tracker (DeepSORT) on GPU (max_age=60)")

    def update(self, frame: np.ndarray, boxes: List[List[float]], confidences: List[float]) -> List[Dict[str, Any]]:
        """
        Updates the tracker with new detections from the current frame.

        Args:
            frame: Enhanced BGR frame (numpy array).
            boxes: List of YOLO bounding boxes in [left, top, width, height] format.
            confidences: List of confidence scores corresponding to each box.
            
        Returns:
            List of active, confirmed tracks. Each track is a dictionary containing:
            - "track_id": Unique integer ID for the person.
            - "bbox": Smoothed bounding box in [x1, y1, x2, y2] format (integers).
        """
        # Format detections for DeepSORT: [([x, y, w, h], confidence, class_id)]
        # Using pythonic zip instead of range(len())
        detections = [
            (box, conf, 0) for box, conf in zip(boxes, confidences)
        ]
            
        # Update tracker memory
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Only return tracks that are confirmed AND were updated in the *current* frame
        # (time_since_update == 1). Otherwise, we risk extracting features from the
        # background when DeepSORT is just "coasting" a lost track.
        active_tracks = [
            {
                "track_id": track.track_id,
                "bbox": [int(x) for x in track.to_ltrb()]
            }
            for track in tracks
            if track.is_confirmed() and track.time_since_update == 1
        ]
            
        return active_tracks
