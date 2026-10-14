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

import os
import json
import numpy as np
import cv2

from typing import List, Dict, Any, Optional, Tuple

from src.pipeline.enhancer import ZeroDCEEnhancer
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
      - Automatically reloads resources.json if file was edited on disk.
      - Extracts an L2-normalised OSNet vector from the query image.
      - Finds the closest reference entry in the manifest (cosine similarity).
      - If confidence >= MATCH_THRESHOLD:
          - If the entry has an identity_label, gathers all tracks belonging to that identity.
          - Otherwise, retrieves tracks for that specific reference.
          - Fetches their Qdrant track segments via payload filter and returns a timeline.
    """

    def __init__(
        self,
        db: Optional[DatabaseClient] = None,
        detector: Optional[PersonDetector] = None,
        app_ext: Optional[AppearanceExtractor] = None,
        enhancer: Optional[ZeroDCEEnhancer] = None,
    ):
        print("[DemoService] Initializing...")
        self.enhancer = enhancer
        self.detector = detector if detector is not None else PersonDetector(strict_gpu=True)
        self.app_ext  = app_ext  if app_ext  is not None else AppearanceExtractor(strict_gpu=True)
        self.db       = db       if db       is not None else DatabaseClient()
        self._manifest_mtime = 0.0
        self.manifest = self._load_manifest()
        print(f"[DemoService] Ready. {len(self.manifest)} reference entry/entries in manifest.")


    # ── Manifest Loading ────────────────────────────────────────────────────────

    def _load_manifest(self) -> List[Dict]:
        """
        Reads resources.json and pre-converts each reference vector
        to a numpy unit-vector array so cosine similarity is instant at query time.
        Supports:
          1. {"references": [{reference_id, video_id, camera_id, track_id, identity_label, embedding}]}
          2. [{reference_id, video_id, camera_id, track_id, identity_label, embedding}]
          3. Legacy [{label, vector, tracks: [{cam_id, track_id}]}]
        """
        try:
            if not os.path.exists(RESOURCES_PATH):
                print(f"[DemoService] WARNING: {RESOURCES_PATH} not found. No matches will be possible.")
                return []

            self._manifest_mtime = os.path.getmtime(RESOURCES_PATH)
            with open(RESOURCES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_entries = []
            if isinstance(data, dict) and "references" in data:
                raw_entries = data["references"]
            elif isinstance(data, list):
                raw_entries = data
            else:
                print(f"[DemoService] Unrecognized format in {RESOURCES_PATH}.")
                return []

            valid = []
            for item in raw_entries:
                # Support both "embedding" and legacy "vector"
                vec = item.get("embedding") if "embedding" in item else item.get("vector", [])
                if not vec:
                    continue

                vec_arr = np.array(vec, dtype=np.float32)
                norm = float(np.linalg.norm(vec_arr))
                if norm > 0:
                    vec_arr = vec_arr / norm

                parsed_entry = dict(item)
                parsed_entry["_vec_np"] = vec_arr
                valid.append(parsed_entry)

            print(f"[DemoService] Loaded {len(valid)} valid reference entries from {RESOURCES_PATH}.")
            return valid

        except Exception as e:
            print(f"[DemoService] ERROR loading manifest: {e}")
            return []

    def _ensure_manifest_fresh(self):
        """Checks if resources.json has been modified and reloads if needed."""
        try:
            if os.path.exists(RESOURCES_PATH):
                cur_mtime = os.path.getmtime(RESOURCES_PATH)
                if cur_mtime > self._manifest_mtime:
                    print("[DemoService] Detected changes in resources.json, refreshing cache...")
                    self.manifest = self._load_manifest()
        except Exception as e:
            print(f"[DemoService] Error checking manifest timestamp: {e}")


    # ── Vector Extraction ───────────────────────────────────────────────────────

    def _extract_vector_from_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Enhances the image with ZeroDCE (if available), detects the primary person,
        crops them, and extracts an L2-normalised OSNet appearance vector.
        """
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            print("[DemoService] Could not decode image bytes.")
            return None

        if self.enhancer is not None:
            try:
                frame = self.enhancer.enhance(frame)
            except Exception as e:
                print(f"[DemoService] Enhancer bypass: {e}")

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
            print("[DemoService] No person detected in image. Using full frame.")
            crop = frame

        return self._to_unit_vector(crop)

    def _extract_best_vector_from_crops(
        self, crops: List[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[Dict], float]:
        """
        Best-frame selection for video clip queries.
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

            for entry in self.manifest:
                sim = float(np.dot(vec, entry["_vec_np"]))
                if sim > best_sim:
                    best_sim   = sim
                    best_vec   = vec
                    best_entry = entry

        return best_vec, best_entry, best_sim

    def _to_unit_vector(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Runs OSNet on the given crop and returns an L2-normalised vector.
        """
        raw_vec, _ = self.app_ext.extract_with_parts(crop)
        if raw_vec is None:
            return None

        vec  = np.array(raw_vec, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec


    # ── Manifest Matching & Identity Grouping ───────────────────────────────────

    def _match_manifest(self, query_vec: np.ndarray) -> Tuple[Optional[Dict], float, List[Dict]]:
        """
        Finds the closest reference entry in resources.json.
        If similarity >= MATCH_THRESHOLD:
          - If the entry has an identity_label, gathers all tracks for that identity (Decision #3).
          - Otherwise, returns evidence only for this specific reference.
        Returns:
          (best_entry, best_sim, target_tracks)
        """
        self._ensure_manifest_fresh()

        if not self.manifest:
            return None, -1.0, []

        best_entry = None
        best_sim   = -1.0

        for entry in self.manifest:
            sim = float(np.dot(query_vec, entry["_vec_np"]))
            if sim > best_sim:
                best_sim   = sim
                best_entry = entry

        if best_sim < MATCH_THRESHOLD or best_entry is None:
            print(f"[DemoService] No match — best sim was {best_sim:.3f} (threshold: {MATCH_THRESHOLD}).")
            return None, best_sim, []

        label = best_entry.get("identity_label") or best_entry.get("label")
        print(f"[DemoService] Matched reference '{best_entry.get('reference_id', label)}' (identity: '{label}') with sim={best_sim:.3f}.")

        # Collect targets
        targets = []
        # If user explicitly assigned an identity_label, group all references sharing that label across videos
        if label and str(label).strip() and str(label).strip().lower() != "null":
            target_label = str(label).strip()
            for entry in self.manifest:
                entry_label = entry.get("identity_label") or entry.get("label")
                if entry_label and str(entry_label).strip() == target_label:
                    if "camera_id" in entry and "track_id" in entry:
                        targets.append({"cam_id": entry["camera_id"], "track_id": entry["track_id"]})
                    elif "tracks" in entry:
                        targets.extend(entry["tracks"])
        else:
            # Single reference match (Decision #3 default)
            if "camera_id" in best_entry and "track_id" in best_entry:
                targets.append({"cam_id": best_entry["camera_id"], "track_id": best_entry["track_id"]})
            elif "tracks" in best_entry:
                targets.extend(best_entry["tracks"])

        # Deduplicate targets by (cam_id, track_id)
        seen = set()
        dedup_targets = []
        for t in targets:
            key = (str(t.get("cam_id")), str(t.get("track_id")))
            if key not in seen and key[0] != "None" and key[1] != "None":
                seen.add(key)
                dedup_targets.append(t)

        return best_entry, best_sim, dedup_targets


    # ── Timeline Builder ────────────────────────────────────────────────────────

    def _build_timeline(self, tracks: List[Dict]) -> List[Dict]:
        """
        Queries Qdrant for each target (cam_id, track_id) using payload filtering.
        Sorts all sightings chronologically.
        """
        timeline = []

        for track in tracks:
            cam_id   = track.get("cam_id") or track.get("camera_id")
            track_id = track.get("track_id")
            if not cam_id or not track_id:
                continue

            points = self.db.get_by_camera_and_track(cam_id, track_id)

            if not points:
                print(f"[DemoService] No Qdrant points found for {cam_id} / {track_id}.")
                continue

            for point in points:
                payload = point.payload or {}
                timeline.append({
                    "uuid":           point.id,
                    "target_id":      point.id,
                    "camera_id":      cam_id,
                    "track_id":       str(track_id),
                    "start_time":     payload.get("timestamp",       0.0),
                    "end_time":       payload.get("end_time",        0.0),
                    "start_time_str": payload.get("start_time_str",  "08:00:00"),
                    "end_time_str":   payload.get("end_time_str",    "08:00:00"),
                    "source_video":   payload.get("source_video_path", ""),
                    "metadata":       payload,
                })

        # Sort all sightings chronologically
        timeline.sort(key=lambda x: x["start_time"])
        return timeline


    # ── Public Entry Points ─────────────────────────────────────────────────────

    def handle_image_query(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Full demo flow for an uploaded image:
          Extract vector → Match manifest → Build timeline → Return result.
        """
        query_vec = self._extract_vector_from_image(image_bytes)
        if query_vec is None:
            return {
                "status":             "error",
                "message":            "Could not extract features from the uploaded image.",
                "confidence":         0.0,
                "timeline":           [],
                "matches":            [],
                "extracted_features": []
            }

        matched_entry, sim_score, target_tracks = self._match_manifest(query_vec)
        if matched_entry is None or not target_tracks:
            return {
                "status":             "no_match",
                "message":            f"No person matched (best similarity: {sim_score:.3f}, threshold: {MATCH_THRESHOLD}).",
                "confidence":         round(sim_score, 4) if sim_score > 0 else 0.0,
                "timeline":           [],
                "matches":            [],
                "extracted_features": ["appearance"]
            }

        timeline = self._build_timeline(target_tracks)

        # Build frontend-compatible 'matches' list for seamless playback integration
        matches = [
            {
                "target_id": str(t["uuid"]),
                "score":     round(sim_score, 4),
                "metadata":  t["metadata"]
            }
            for t in timeline
        ]

        label = matched_entry.get("identity_label") or matched_entry.get("label") or matched_entry.get("reference_id", "Unknown")

        return {
            "status":             "matched",
            "label":              label,
            "confidence":         round(sim_score, 4),
            "matched_reference":  matched_entry.get("reference_id"),
            "identity_label":     matched_entry.get("identity_label"),
            "timeline":           timeline,
            "matches":            matches,
            "extracted_features": ["appearance"]
        }

    def handle_video_clip_query(self, crops: List[np.ndarray]) -> Dict[str, Any]:
        """
        Full demo flow for a video clip query.
        """
        best_vec, matched_entry, sim_score = self._extract_best_vector_from_crops(crops)

        if best_vec is None:
            return {
                "status":             "error",
                "message":            "Could not extract features from the selected track.",
                "confidence":         0.0,
                "timeline":           [],
                "matches":            [],
                "extracted_features": []
            }

        # Resolve grouped targets
        _, _, target_tracks = self._match_manifest(best_vec)

        if matched_entry is None or sim_score < MATCH_THRESHOLD or not target_tracks:
            return {
                "status":             "no_match",
                "message":            f"No person matched (best similarity: {sim_score:.3f}, threshold: {MATCH_THRESHOLD}).",
                "confidence":         round(sim_score, 4) if sim_score > 0 else 0.0,
                "timeline":           [],
                "matches":            [],
                "extracted_features": ["appearance"]
            }

        timeline = self._build_timeline(target_tracks)
        matches = [
            {
                "target_id": str(t["uuid"]),
                "score":     round(sim_score, 4),
                "metadata":  t["metadata"]
            }
            for t in timeline
        ]

        label = matched_entry.get("identity_label") or matched_entry.get("label") or matched_entry.get("reference_id", "Unknown")

        return {
            "status":             "matched",
            "label":              label,
            "confidence":         round(sim_score, 4),
            "matched_reference":  matched_entry.get("reference_id"),
            "identity_label":     matched_entry.get("identity_label"),
            "timeline":           timeline,
            "matches":            matches,
            "extracted_features": ["appearance"]
        }
