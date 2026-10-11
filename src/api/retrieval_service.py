import copy
import os
import cv2
import uuid
import base64
from typing import List, Dict, Any, Tuple
import numpy as np

from src.pipeline.detector import PersonDetector
from src.pipeline.tracker import MultiObjectTracker
from src.pipeline.extractors.appearance import AppearanceExtractor
from src.pipeline.extractors.face import FaceExtractor
from src.pipeline.extractors.gait import GaitExtractor
from src.database.qdrant_client import DatabaseClient
from src.core.fusion import ScoreFusion
from src.api.schemas import MatchResult, ThumbnailResponse

# Target FPS for subsampling query videos — matches the ingestion pipeline rate
QUERY_TARGET_FPS = 6


class RetrievalService:
    # Per-camera human-readable location labels used in timeline metadata.
    CAMERA_LOCATIONS: Dict[str, str] = {
        "Camera_1": "entry-left",
        "Camera_2": "entry-left-front",
        "Camera_3": "entry-right-back",
    }

    def __init__(self):
        print("Initializing Retrieval Service Models...")

        # strict_gpu=True -> enforces that the server MUST run these on the GPU.
        self.detector = PersonDetector(strict_gpu=True)
        self.app_ext  = AppearanceExtractor(strict_gpu=True)
        self.face_ext = FaceExtractor(strict_gpu=True)
        self.gait_ext = GaitExtractor(strict_gpu=True)

        self.db     = DatabaseClient()
        self.fusion = ScoreFusion()

        # In-memory cache for tracking history (instance-scoped, not global).
        # Format: { session_id: { track_id: [(frame_idx, [x1,y1,x2,y2]), ...] } }
        self._tracking_cache: Dict[str, Dict[str, List[Tuple[int, List[int]]]]] = {}

    def clear_cache(self):
        """Clears the tracking cache for demo purposes to prevent RAM leaks."""
        self._tracking_cache.clear()

    @staticmethod
    def _l2_norm_mean(vec_list: List) -> Any:
        """Aggregate a list of vectors via L2-normalised mean."""
        if not vec_list:
            return None
        vec  = np.mean(vec_list, axis=0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def process_image_query(self, image_bytes: bytes) -> Tuple[List[str], List[MatchResult]]:
        # Convert bytes to cv2 image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return [], []

        boxes, confs = self.detector.detect(frame)

        extracted_features = []
        app_vec  = None
        face_vec = None

        if len(boxes) > 0:
            # Assume largest box is the target for image upload (boxes are [x, y, w, h])
            target_box = max(boxes, key=lambda b: float(b[2]) * float(b[3]))
            x, y, w, h = target_box
            img_h, img_w = frame.shape[:2]
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(img_w, int(round(x + w)))
            y2 = min(img_h, int(round(y + h)))

            crop = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else frame

            face_vec = self.face_ext.extract(crop)
            if face_vec is None and crop is not frame:
                # Fallback to detecting face in full frame if crop didn't catch it
                face_vec = self.face_ext.extract(frame)
            if face_vec is not None:
                extracted_features.append("face")

            app_vec = self.app_ext.extract(crop if crop.size > 0 else frame)
            if app_vec is not None:
                extracted_features.append("appearance")
        else:
            # YOLO found zero person bounding boxes — two possible reasons:
            #   (a) It's a face-only/passport photo (YOLO needs a torso to fire)
            #   (b) It's a full-body image YOLO simply missed (blur, occlusion, angle)
            # We don't know which, so try both extractors on the full frame.
            # If it's truly face-only, the appearance vector will be noisy but
            # won't match any DB entry meaningfully, so it won't cause false positives.
            face_vec = self.face_ext.extract(frame)
            if face_vec is not None:
                extracted_features.append("face")
            app_vec = self.app_ext.extract(frame)
            if app_vec is not None:
                extracted_features.append("appearance")

        if not extracted_features:
            return [], []

        # Search database
        results = self.db.search_ensemble(query_appearance=app_vec, query_face=face_vec)

        # Lower threshold for image queries (single photo has less signal than a video track;
        # the same person across different camera angles may score slightly lower)
        fused_results    = self.fusion.format_results(results, min_threshold=0.75)
        resolved_results = self.resolve_timeline_overlaps(fused_results)
        matches = [MatchResult(target_id=str(r["uuid"]), score=r["confidence"], metadata=r["metadata"]) for r in resolved_results]

        return extracted_features, matches

    def process_video_first_pass(self, video_path: str, session_id: str) -> List[ThumbnailResponse]:
        """
        First pass: detect and track all persons in the query video.
        Subsampled to QUERY_TARGET_FPS (6 FPS) to match the ingestion rate and
        avoid wasting time processing frames that add no new information.
        """
        cap = cv2.VideoCapture(video_path)
        tracker = MultiObjectTracker()

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(fps / QUERY_TARGET_FPS)))

        self._tracking_cache[session_id] = {}
        thumbnails: Dict[str, str] = {}

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process every Nth frame to match the 6 FPS ingestion rate
            if frame_idx % frame_interval == 0:
                boxes, confs = self.detector.detect(frame)
                active_tracks = tracker.update(frame, boxes, confs)

                for track in active_tracks:
                    tid  = str(track["track_id"])
                    bbox = track["bbox"]

                    if tid not in self._tracking_cache[session_id]:
                        self._tracking_cache[session_id][tid] = []
                    self._tracking_cache[session_id][tid].append((frame_idx, bbox))

                    # Save the first good crop as thumbnail
                    if tid not in thumbnails:
                        x1, y1, x2, y2 = bbox
                        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                        if crop.size > 0:
                            _, buffer = cv2.imencode('.jpg', crop)
                            thumbnails[tid] = base64.b64encode(buffer).decode('utf-8')

            frame_idx += 1

        cap.release()

        return [ThumbnailResponse(track_id=tid, thumbnail_base64=b64) for tid, b64 in thumbnails.items()]

    def process_video_target(self, video_path: str, session_id: str, track_id: str) -> List[MatchResult]:
        """
        Second pass: extract biometric embeddings for the selected target.
        Seeks directly to only the frames where the target was tracked (no full
        sequential scan). All crops are read first, then batched GPU inference is
        run in one shot — one call for Appearance, one YOLO batch call for Skeleton.
        """
        if session_id not in self._tracking_cache or track_id not in self._tracking_cache[session_id]:
            print(f"Error: No tracking data for session {session_id}, track {track_id}")
            return []

        target_history = self._tracking_cache[session_id][track_id]

        cap = cv2.VideoCapture(video_path)

        # ── Pass 1: read all crops from disk ──────────────────────────────────────
        # We separate "heavy" frames (every 6th, for Appearance + Face) from "all"
        # frames (for Skeleton, which needs every frame for the walking sequence).
        heavy_crops = []   # 1-FPS crops  → Appearance + Face
        all_crops   = []   # 6-FPS crops  → Skeleton

        for i, (frame_idx, bbox) in enumerate(target_history):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                all_crops.append(None)
                continue

            x1, y1, x2, y2 = bbox
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size == 0:
                all_crops.append(None)
                continue

            all_crops.append(crop)

            # Subsample: appearance & face only every 6th frame (1 FPS)
            if i % 6 == 0:
                heavy_crops.append(crop)

        cap.release()

        # ── Pass 2: batch GPU inference ───────────────────────────────────────────

        # Appearance — one GPU forward pass for all heavy crops at once
        app_vectors = []
        if heavy_crops:
            app_batch = self.app_ext.extract_batch(heavy_crops)
            app_vectors = [v for v in app_batch if v is not None]

        # Face — InsightFace does not support true batching; loop per-crop
        face_vectors = []
        for crop in heavy_crops:
            fv = self.face_ext.extract(crop)
            if fv is not None:
                face_vectors.append(fv)

        # Skeleton — one YOLO batch call for all frames
        skeletons = []
        valid_all_crops = [c for c in all_crops if c is not None]
        if valid_all_crops:
            skel_batch = self.gait_ext.extract_skeleton_batch(valid_all_crops)
            skeletons  = [s for s in skel_batch if s is not None]

        # Aggregate embeddings via L2-normalised mean (see _l2_norm_mean)
        final_app  = self._l2_norm_mean(app_vectors)
        final_face = self._l2_norm_mean(face_vectors)

        final_gait = None
        if len(skeletons) >= 30:
            gait_vectors = []
            for i in range(0, len(skeletons), 30):
                chunk = skeletons[i:i + 30]
                if len(chunk) >= 30:
                    gait_vec = self.gait_ext.extract_gait_vector(chunk)
                    if gait_vec:
                        gait_vectors.append(gait_vec)
            if gait_vectors:
                final_gait = self._l2_norm_mean(gait_vectors)

        results = self.db.search_ensemble(
            query_appearance=final_app,
            query_face=final_face,
            query_gait=final_gait
        )

        fused_results    = self.fusion.format_results(results, min_threshold=0.75)
        reranked_results = self.k_reciprocal_rerank(fused_results)
        resolved_results = self.resolve_timeline_overlaps(reranked_results)
        matches = [MatchResult(target_id=str(r["uuid"]), score=r["confidence"], metadata=r["metadata"]) for r in resolved_results]

        return matches

    def k_reciprocal_rerank(self, fused_results: List[Dict], limit: int = 20) -> List[Dict]:
        """
        Post-processing step that recalculates confidence based on mutual nearest-neighbor relationships.
        """
        if not fused_results:
            return []
            
        reranked = []
        # Get the top N IDs for the original query
        query_neighbors = {r["uuid"] for r in fused_results[:limit]}
        
        for result in fused_results:
            uid = result["uuid"]
            
            # Fetch the DB vector for this result
            db_point = self.db.get_by_id(uid)
            if not db_point or not hasattr(db_point, 'vector') or not isinstance(db_point.vector, dict):
                reranked.append(result)
                continue
                
            db_vectors = db_point.vector
            # Perform a reverse search using the DB vector
            reverse_search_results = self.db.search_ensemble(
                query_appearance=db_vectors.get("appearance"),
                query_face=db_vectors.get("face"),
                query_gait=db_vectors.get("gait"),
                limit=limit
            )
            # Use 0.0 threshold to get all neighbors
            reverse_fused = self.fusion.format_results(reverse_search_results, min_threshold=0.0)
            
            # Get the top N IDs for the reverse query
            reverse_neighbors = {r["uuid"] for r in reverse_fused[:limit]}
            
            # Calculate Jaccard similarity between the neighbor sets
            intersection = len(query_neighbors.intersection(reverse_neighbors))
            union = len(query_neighbors.union(reverse_neighbors))
            jaccard = intersection / union if union > 0 else 0
            
            # Boost the confidence based on Jaccard similarity (Mutual relationship)
            new_confidence = (result["confidence"] * 0.7) + (jaccard * 0.3)
            result["confidence"] = new_confidence
            reranked.append(result)
            
        # Re-sort descending by the new confidence
        reranked.sort(key=lambda x: x["confidence"], reverse=True)
        return reranked

    def resolve_timeline_overlaps(self, fused_results: List[Dict]) -> List[Dict]:
        """
        Resolves overlapping timeline blocks on the same camera.
        Strategy: highest-confidence block always wins the disputed time range.
        Any block (or portion of a block) that loses its time to a higher-confidence
        block is clipped or discarded so that departure is NEVER less than arrival.
        """
        if not fused_results:
            return []

        # ── Normalize metadata ──────────────────────────────────────────────────
        for r in fused_results:
            r["metadata"]["breakdown"] = r["breakdown"]
            cam_id = r["metadata"].get("camera_id", "Unknown")
            r["metadata"]["location"] = self.CAMERA_LOCATIONS.get(cam_id, "Unknown Location")
            if "timestamp" not in r["metadata"]:
                r["metadata"]["timestamp"] = 0.0
            if "end_time" not in r["metadata"]:
                r["metadata"]["end_time"] = r["metadata"]["timestamp"]

        # ── Resolve Overlaps Globally (Across All Cameras) ──────────────────────

        # Sort descending by confidence globally — highest confidence claims its time first
        blocks_sorted = sorted(fused_results, key=lambda x: x["confidence"], reverse=True)

        accepted_intervals: List[Tuple[float, float]] = []
        resolved_all:       List[Dict]                = []

        for block in blocks_sorted:
            b_start = block["metadata"]["timestamp"]
            b_end   = block["metadata"]["end_time"]

            # Compute the "free" sub-intervals of [b_start, b_end] by subtracting
            # every already-accepted interval that overlaps with it.
            free_segments = [(b_start, b_end)]

            for (a_start, a_end) in accepted_intervals:
                new_free = []
                for (s, e) in free_segments:
                    if s < a_start:
                        new_free.append((s, min(e, a_start)))
                    if e > a_end:
                        new_free.append((max(s, a_end), e))
                free_segments = new_free
                if not free_segments:
                    break  # fully consumed

            for (seg_start, seg_end) in free_segments:
                if seg_end > seg_start:  # strict: departure must be AFTER arrival
                    entry = copy.deepcopy(block)
                    entry["metadata"]["timestamp"] = seg_start
                    entry["metadata"]["end_time"]  = seg_end
                    accepted_intervals.append((seg_start, seg_end))
                    resolved_all.append(entry)

        # ── Re-sort everything chronologically for display ──────────────────────
        resolved_all.sort(key=lambda x: x["metadata"]["timestamp"])

        # ── Merge consecutive same-camera entries ────────────────────────────────
        # If two entries on the same camera are contiguous (gap ≤ 1 second),
        # collapse them into one: earliest arrival, latest departure, best confidence.
        merged = []
        for entry in resolved_all:
            cam   = entry["metadata"].get("camera_id")
            start = entry["metadata"]["timestamp"]
            end   = entry["metadata"]["end_time"]

            if (merged
                    and merged[-1]["metadata"].get("camera_id") == cam
                    and start <= merged[-1]["metadata"]["end_time"] + 1.0):
                prev = merged[-1]
                prev["metadata"]["end_time"] = max(prev["metadata"]["end_time"], end)
                if entry["confidence"] > prev["confidence"]:
                    prev["confidence"] = entry["confidence"]
                    prev["metadata"]["breakdown"] = entry["metadata"]["breakdown"]
            else:
                merged.append(entry)

        return merged
