"""
demo_service.py
---------------
Handles the controlled-demo query flow for both image and video-clip uploads.

How it works:
  1. Extract an OSNet appearance vector from the query (image or video clip).
  2. Match that vector against precomputed reference vectors in `resources.json`.
  3. Find the best-matching person and retrieve their (cam_id, track_id) pairs.
  4. Query Qdrant by payload filter to get the exact Qdrant points for those tracks.
  5. Build and return a chronological timeline for the frontend.

This approach bypasses the live Re-ID pipeline entirely and uses a pre-verified
manifest, making demo results reliable and consistent.
"""

import json
import numpy as np
import cv2

from typing import List, Dict, Any, Optional, Tuple

from src.pipeline.detector import PersonDetector
from src.pipeline.extractors.appearance import AppearanceExtractor
from src.database.qdrant_client import DatabaseClient


# ── Configuration ──────────────────────────────────────────────────────────────

RESOURCES_PATH  = "data/resources.json"
MATCH_THRESHOLD = 0.55   # Minimum cosine similarity required to confirm a match


# ── DemoService ────────────────────────────────────────────────────────────────

class DemoService:
    """
    Orchestrates the controlled-demo query flow.

    On startup:
      - Loads the reference manifest (resources.json) into memory.
      - Initialises YOLO detector and OSNet appearance extractor.

    At query time:
      - Extracts a vector from the uploaded image / selected video track.
      - Finds the closest person in the manifest (cosine similarity).
      - Fetches their Qdrant track segments and returns a timeline.
    """

    def __init__(
        self,
        db: Optional[DatabaseClient] = None,
        detector: Optional[PersonDetector] = None,
        app_ext: Optional[AppearanceExtractor] = None,
    ):
        print("[DemoService] Initializing...")
        self.detector = detector if detector is not None else PersonDetector(strict_gpu=True)
        self.app_ext  = app_ext  if app_ext  is not None else AppearanceExtractor(strict_gpu=True)
        self.db       = db       if db       is not None else DatabaseClient()
        self.manifest = self._load_manifest()
        print(f"[DemoService] Ready. {len(self.manifest)} person(s) in manifest.")


    # ── Manifest Loading ────────────────────────────────────────────────────────

    def _load_manifest(self) -> List[Dict]:
        """
        Reads resources.json and pre-converts each person's reference vector
        to a numpy array so cosine similarity is fast at query time.
        """
        try:
            with open(RESOURCES_PATH, "r") as f:
                entries = json.load(f)

            valid = []
            for entry in entries:
                vec = entry.get("vector", [])
                if not vec:
                    # Skip entries where the vector has not been filled in yet
                    print(f"[DemoService] Skipping '{entry.get('label', '?')}' — vector is empty.")
                    continue
                entry["_vec_np"] = np.array(vec, dtype=np.float32)
                valid.append(entry)

            print(f"[DemoService] Loaded {len(valid)} valid entries from {RESOURCES_PATH}.")
            return valid

        except FileNotFoundError:
            print(f"[DemoService] WARNING: {RESOURCES_PATH} not found. No matches will be possible.")
            return []
        except Exception as e:
            print(f"[DemoService] ERROR loading manifest: {e}")
            return []


    # ── Vector Extraction ───────────────────────────────────────────────────────

    def _extract_vector_from_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Detects the largest person in the image, crops them, and extracts
        an L2-normalised OSNet appearance vector.

        Returns None if detection or extraction fails.
        """
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            print("[DemoService] Could not decode image bytes.")
            return None

        boxes, _ = self.detector.detect(frame)

        if boxes:
            # Pick the largest bounding box — assumed to be the target person
            largest_box = max(boxes, key=lambda b: float(b[2]) * float(b[3]))
            x, y, w, h  = largest_box
            img_h, img_w = frame.shape[:2]
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(img_w, int(round(x + w)))
            y2 = min(img_h, int(round(y + h)))
            crop = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else frame
        else:
            # No person detected — fall back to the full frame
            print("[DemoService] No person detected in image. Using full frame.")
            crop = frame

        return self._to_unit_vector(crop)

    def _extract_best_vector_from_crops(
        self, crops: List[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[Dict], float]:
        """
        Best-frame selection for video clip queries.

        For every crop, extracts an OSNet vector and scores it against every
        manifest entry. Tracks the single (frame, entry) pair with the highest
        cosine similarity across the entire loop.

        Returns:
          (best_vec, best_entry, best_sim)
          — best_vec   : L2-normalised vector of the winning frame
          — best_entry : the manifest entry it matched against (already known here)
          — best_sim   : the cosine similarity score
          All three are None / -1.0 if no valid crop was processed.
        """
        best_vec   = None
        best_entry = None
        best_sim   = -1.0

        for crop in crops:
            if crop is None or crop.size == 0:
                continue

            vec = self._to_unit_vector(crop)
            if vec is None:
                continue

            # Score this frame against every manifest entry
            for entry in self.manifest:
                sim = float(np.dot(vec, entry["_vec_np"]))
                if sim > best_sim:
                    best_sim   = sim
                    best_vec   = vec
                    best_entry = entry   # save the winner — no need to re-search later

        return best_vec, best_entry, best_sim

    def _to_unit_vector(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Runs OSNet on the given crop and returns an L2-normalised vector.
        Returns None if OSNet extraction fails.
        """
        raw_vec, _ = self.app_ext.extract_with_parts(crop)
        if raw_vec is None:
            return None

        vec  = np.array(raw_vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


    # ── Manifest Matching ───────────────────────────────────────────────────────

    def _match_manifest(self, query_vec: np.ndarray) -> Tuple[Optional[Dict], float]:
        """
        Iterates over all manifest entries and finds the one with the
        highest cosine similarity to the query vector.

        Returns:
          (matched_entry, sim_score)  — entry is None if below MATCH_THRESHOLD.
        """
        best_entry = None
        best_sim   = -1.0

        for entry in self.manifest:
            sim = float(np.dot(query_vec, entry["_vec_np"]))
            if sim > best_sim:
                best_sim  = sim
                best_entry = entry

        if best_sim < MATCH_THRESHOLD:
            print(f"[DemoService] No match — best sim was {best_sim:.3f} (threshold: {MATCH_THRESHOLD}).")
            return None, best_sim

        print(f"[DemoService] Matched '{best_entry.get('label', '?')}' with sim={best_sim:.3f}.")
        return best_entry, best_sim


    # ── Timeline Builder ────────────────────────────────────────────────────────

    def _build_timeline(self, tracks: List[Dict]) -> List[Dict]:
        """
        For each (cam_id, track_id) in the matched person's track list:
          1. Queries Qdrant using a payload filter (no vector search needed).
          2. Extracts timestamp, end_time, bounding_boxes, and source video.
          3. Returns a sorted list of timeline entries for the frontend.
        """
        timeline = []

        for track in tracks:
            cam_id   = track["cam_id"]
            track_id = track["track_id"]

            points = self.db.get_by_camera_and_track(cam_id, track_id)

            if not points:
                print(f"[DemoService] No Qdrant points found for {cam_id} / {track_id}.")
                continue

            for point in points:
                payload = point.payload or {}
                timeline.append({
                    "uuid":           point.id,
                    "camera_id":      cam_id,
                    "track_id":       track_id,
                    "start_time":     payload.get("timestamp",       0.0),
                    "end_time":       payload.get("end_time",        0.0),
                    "start_time_str": payload.get("start_time_str",  "08:00:00"),
                    "end_time_str":   payload.get("end_time_str",    "08:00:00"),
                    "source_video":   payload.get("source_video_path", ""),
                })

        # Sort all sightings chronologically for clean timeline display
        timeline.sort(key=lambda x: x["start_time"])
        return timeline


    # ── Public Entry Points ─────────────────────────────────────────────────────

    def handle_image_query(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Full demo flow for an uploaded image:
          Extract vector → Match manifest → Build timeline → Return result.
        """
        # Step 1: Extract appearance vector from the uploaded image
        query_vec = self._extract_vector_from_image(image_bytes)
        if query_vec is None:
            return {
                "status":   "error",
                "message":  "Could not extract features from the uploaded image.",
                "timeline": []
            }

        # Step 2: Find the closest person in the manifest
        matched_entry, sim_score = self._match_manifest(query_vec)
        if matched_entry is None:
            return {
                "status":   "no_match",
                "message":  f"No person matched (best similarity: {sim_score:.3f}).",
                "timeline": []
            }

        # Step 3: Fetch Qdrant data and build the chronological timeline
        timeline = self._build_timeline(matched_entry["tracks"])

        return {
            "status":     "matched",
            "label":      matched_entry.get("label", "Unknown"),
            "confidence": round(sim_score, 4),
            "timeline":   timeline,
        }

    def handle_video_clip_query(self, crops: List[np.ndarray]) -> Dict[str, Any]:
        """
        Full demo flow for a video clip query (called after the user selects a track).

        'crops' is the list of person crops for the selected track,
        extracted from the tracking cache by the caller.

        Uses best-frame selection: the single frame most similar to any manifest
        entry is chosen. The matching entry and score are already known from that
        search, so no second manifest pass is needed.
        """
        # Step 1: Find the best frame — matching entry and sim score come for free
        best_vec, matched_entry, sim_score = self._extract_best_vector_from_crops(crops)

        if best_vec is None:
            return {
                "status":   "error",
                "message":  "Could not extract features from the selected track.",
                "timeline": []
            }

        if matched_entry is None or sim_score < MATCH_THRESHOLD:
            return {
                "status":   "no_match",
                "message":  f"No person matched (best similarity: {sim_score:.3f}).",
                "timeline": []
            }

        print(f"[DemoService] Matched '{matched_entry.get('label', '?')}' with sim={sim_score:.3f}.")

        # Step 2: Fetch Qdrant data and build the chronological timeline
        timeline = self._build_timeline(matched_entry["tracks"])

        return {
            "status":     "matched",
            "label":      matched_entry.get("label", "Unknown"),
            "confidence": round(sim_score, 4),
            "timeline":   timeline,
        }
