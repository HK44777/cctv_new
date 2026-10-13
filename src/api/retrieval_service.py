import copy
import os
import cv2
import uuid
import base64
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import numpy as np

from src.pipeline.enhancer import ZeroDCEEnhancer
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

    # Spatio-temporal distance matrix (metres) between camera viewpoints
    CAMERA_TOPOLOGY: Dict[Tuple[str, str], float] = {
        ("Camera_1", "Camera_1"): 0.0,
        ("Camera_2", "Camera_2"): 0.0,
        ("Camera_3", "Camera_3"): 0.0,
        ("Camera_1", "Camera_2"): 15.0,
        ("Camera_2", "Camera_1"): 15.0,
        ("Camera_1", "Camera_3"): 30.0,
        ("Camera_3", "Camera_1"): 30.0,
        ("Camera_2", "Camera_3"): 20.0,
        ("Camera_3", "Camera_2"): 20.0,
    }

    def __init__(self):
        print("Initializing Retrieval Service Models...")

        # strict_gpu=True -> enforces that the server MUST run these on the GPU.
        self.enhancer = ZeroDCEEnhancer(strict_gpu=True)
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

    @staticmethod
    def compute_part_histograms(crop: np.ndarray) -> Tuple[Any, Any]:
        """
        Splits person crop into Upper Torso (12-52% H) and Lower Legs/Pants (52-92% H)
        and computes normalized 2D Hue-Saturation color histograms with achromatic masking.
        """
        if crop is None or crop.size == 0:
            return None, None
        h, w = crop.shape[:2]
        if h < 20 or w < 10:
            return None, None
        y1_u, y2_u = int(0.12 * h), int(0.52 * h)
        y1_l, y2_l = int(0.52 * h), int(0.92 * h)
        x1, x2 = int(0.12 * w), int(0.88 * w)
        upper = crop[y1_u:y2_u, x1:x2]
        lower = crop[y1_l:y2_l, x1:x2]
        if upper.size == 0 or lower.size == 0:
            return None, None
        hsv_u = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
        hsv_l = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)

        # Achromatic mask: separate low-saturation (S < 30) or dark (V < 30) pixels
        sat_u, val_u = hsv_u[:, :, 1], hsv_u[:, :, 2]
        mask_u = ((sat_u >= 30) & (val_u >= 30)).astype(np.uint8)
        mask_u = mask_u if np.sum(mask_u) > 10 else None

        sat_l, val_l = hsv_l[:, :, 1], hsv_l[:, :, 2]
        mask_l = ((sat_l >= 30) & (val_l >= 30)).astype(np.uint8)
        mask_l = mask_l if np.sum(mask_l) > 10 else None

        hist_u = cv2.calcHist([hsv_u], [0, 1], mask_u, [16, 16], [0, 180, 0, 256])
        hist_l = cv2.calcHist([hsv_l], [0, 1], mask_l, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist_u, hist_u, 1, 0, cv2.NORM_L1)
        cv2.normalize(hist_l, hist_l, 1, 0, cv2.NORM_L1)
        return hist_u, hist_l

    @staticmethod
    def compare_spatial_color(hist_q_u, hist_q_l, hist_c_u, hist_c_l) -> Tuple[float, float]:
        """
        Calculates Upper and Lower color similarities using Bhattacharyya distance.
        Similarity = max(0.0, 1.0 - d_bhattacharyya). Returns (sim_upper, sim_lower).
        """
        if hist_q_u is None or hist_c_u is None or hist_q_l is None or hist_c_l is None:
            return 1.0, 1.0
        d_u = cv2.compareHist(hist_q_u, hist_c_u, cv2.HISTCMP_BHATTACHARYYA)
        d_l = cv2.compareHist(hist_q_l, hist_c_l, cv2.HISTCMP_BHATTACHARYYA)
        return max(0.0, 1.0 - d_u), max(0.0, 1.0 - d_l)

    @staticmethod
    def _sample_crop_from_video(vpath: str, frame_idx: int, bbox: List[int]) -> Any:
        try:
            if not os.path.exists(vpath):
                return None
            cap = cv2.VideoCapture(vpath)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return None
            frame = cv2.resize(frame, (640, 480))
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame.shape[:2]
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            return crop if crop.size > 0 else None
        except Exception:
            return None

    def process_image_query(self, image_bytes: bytes) -> Tuple[List[str], List[MatchResult]]:
        # Convert bytes to cv2 image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return [], []

        # Enhance query frame with Zero-DCE to match video ingestion pipeline
        frame = self.enhancer.enhance(frame)

        boxes, confs = self.detector.detect(frame)

        extracted_features = []
        app_vec  = None
        face_vec = None
        crop = frame

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

            app_vec, q_part_vecs = self.app_ext.extract_with_parts(crop if crop.size > 0 else frame)
            if app_vec is not None:
                extracted_features.append("appearance")
        else:
            face_vec = self.face_ext.extract(frame)
            if face_vec is not None:
                extracted_features.append("face")
            app_vec, q_part_vecs = self.app_ext.extract_with_parts(frame)
            if app_vec is not None:
                extracted_features.append("appearance")

        if not extracted_features:
            return [], []

        # Search database with generous candidate limit
        results = self.db.search_ensemble(query_appearance=app_vec, query_face=face_vec, limit=50)

        # 1. Fuse scores with Biometric Guarding
        query_has_face = (face_vec is not None)
        # Calibrated cross-camera Re-ID threshold
        threshold = 0.50
        fused_results = self.fusion.format_results(results, min_threshold=threshold, query_has_face=query_has_face)

        # 2. Aspect-Ratio Aware Spatial Chromatic & OSNet Horizontal Stripe Dissection
        target_crop = crop if (len(boxes) > 0 and crop.size > 0) else frame
        h_t, w_t = target_crop.shape[:2]
        aspect_ratio = h_t / max(1, w_t)
        # Full-body crops have H/W >= 1.8. Bust/torso crops have H/W < 1.8.
        is_full_body = (aspect_ratio >= 1.8)

        q_hist_u, q_hist_l = self.compute_part_histograms(target_crop)

        for r in fused_results:
            vpath = r["metadata"].get("source_video_path")
            bboxes = r["metadata"].get("bounding_boxes", {})
            if vpath and bboxes:
                frame_keys = list(bboxes.keys())
                mid_frame_str = frame_keys[len(frame_keys) // 2]
                cand_box = bboxes[mid_frame_str]
                cand_crop = self._sample_crop_from_video(vpath, int(mid_frame_str), cand_box)
                if cand_crop is not None and cand_crop.size > 0:
                    # A. Spatial Chromatic HSV Comparison
                    if q_hist_u is not None:
                        c_hist_u, c_hist_l = self.compute_part_histograms(cand_crop)
                        sim_u, sim_l = self.compare_spatial_color(q_hist_u, q_hist_l, c_hist_u, c_hist_l)
                        
                        if sim_u < 0.20:
                            r["confidence"] *= 0.90
                        elif sim_u >= 0.45:
                            r["confidence"] = min(1.0, r["confidence"] * 1.05)

                        if is_full_body and q_hist_l is not None:
                            if sim_l < 0.20:
                                r["confidence"] *= 0.90
                            elif sim_l >= 0.45:
                                r["confidence"] = min(1.0, r["confidence"] * 1.05)

                    # B. OSNet Part-based Horizontal Stripe Dissection (Torso, Mid, Legs)
                    if q_part_vecs is not None:
                        _, c_part_vecs = self.app_ext.extract_with_parts(cand_crop)
                        if c_part_vecs is not None:
                            sim_p0 = float(np.dot(q_part_vecs[0], c_part_vecs[0]))  # Upper torso
                            sim_p1 = float(np.dot(q_part_vecs[1], c_part_vecs[1]))  # Mid torso / waist
                            sim_p2 = float(np.dot(q_part_vecs[2], c_part_vecs[2]))  # Lower legs / trousers

                            if is_full_body:
                                # Top garment matches but trousers contradict (eradicating green-shirt impostors)
                                if sim_p0 > 0.55 and sim_p2 < 0.25:
                                    r["confidence"] *= 0.80  # Decisive stripe penalty
                                elif sim_p0 > 0.50 and sim_p1 > 0.50 and sim_p2 > 0.50:
                                    r["confidence"] = min(1.0, r["confidence"] * 1.08)
                            else:
                                if sim_p0 > 0.55 and sim_p1 > 0.55:
                                    r["confidence"] = min(1.0, r["confidence"] * 1.05)

        # 3. Apply Mutual Nearest-Neighbor (k-reciprocal) verification
        reranked_results = self.k_reciprocal_rerank(fused_results)

        # 4. Resolve timeline overlaps
        resolved_results = self.resolve_timeline_overlaps(reranked_results)

        # 5. Spatio-Temporal Feasibility Chain with Hard Velocity Floor (v_max = 3.0 m/s) and Tiering
        final_selected = self.filter_and_tier_matches(resolved_results, min_thresh=threshold)

        matches = [MatchResult(target_id=str(r["uuid"]), score=r["confidence"], metadata=r["metadata"]) for r in final_selected]

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

                    # Save the first good frame as thumbnail (full frame with bbox + Track ID drawn)
                    if tid not in thumbnails:
                        x1, y1, x2, y2 = bbox
                        display_frame = frame.copy()
                        # Draw bounding box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Draw Track ID label
                        label = f"Track ID: {tid}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.65
                        thickness = 2
                        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        badge_y1 = max(0, y1 - th - 8)
                        badge_y2 = y1
                        cv2.rectangle(display_frame, (x1, badge_y1), (x1 + tw + 10, badge_y2), (0, 255, 0), -1)
                        cv2.putText(display_frame, label, (x1 + 5, y1 - 5),
                                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                        _, buffer = cv2.imencode('.jpg', display_frame)
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
        if len(skeletons) >= 15:
            gait_vectors = []
            for i in range(0, len(skeletons), 15):
                chunk = skeletons[i:i + 30]
                if len(chunk) >= 15:
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

        min_thresh = 0.55 if (final_face or final_gait) else 0.62
        fused_results    = self.fusion.format_results(results, min_threshold=min_thresh)
        reranked_results = self.k_reciprocal_rerank(fused_results)
        resolved_results = self.resolve_timeline_overlaps(reranked_results)

        # Spatio-Temporal Feasibility Chain with Hard Velocity Floor (v_max = 3.0 m/s) and Tiering
        final_selected = self.filter_and_tier_matches(resolved_results, min_thresh=min_thresh)
        matches = [MatchResult(target_id=str(r["uuid"]), score=r["confidence"], metadata=r["metadata"]) for r in final_selected]

        return matches

    def k_reciprocal_rerank(self, fused_results: List[Dict], limit: int = 20) -> List[Dict]:
        """
        Post-processing step that recalculates confidence based on mutual nearest-neighbor relationships.
        Asymmetric candidate hubs that fail mutual linking with the query cluster are penalized.
        """
        if not fused_results:
            return []
            
        reranked = []
        query_neighbors = {r["uuid"] for r in fused_results[:limit]}
        
        for result in fused_results:
            uid = result["uuid"]
            
            db_point = self.db.get_by_id(uid)
            if not db_point or not hasattr(db_point, 'vector') or not isinstance(db_point.vector, dict):
                reranked.append(result)
                continue
                
            db_vectors = db_point.vector
            reverse_search_results = self.db.search_ensemble(
                query_appearance=db_vectors.get("appearance"),
                query_face=db_vectors.get("face"),
                query_gait=db_vectors.get("gait"),
                limit=limit
            )
            reverse_fused = self.fusion.format_results(reverse_search_results, min_threshold=0.0)
            reverse_neighbors = {r["uuid"] for r in reverse_fused[:limit]}
            
            intersection = len(query_neighbors.intersection(reverse_neighbors))
            min_set_size = max(1, min(len(query_neighbors), len(reverse_neighbors)))
            overlap_ratio = intersection / min_set_size
            
            # Mutual nearest-neighbor verification:
            # Reward reciprocal mutual matches (+5% to +10%)
            if overlap_ratio >= 0.30:
                new_confidence = min(1.0, result["confidence"] * (1.0 + 0.10 * overlap_ratio))
            else:
                new_confidence = result["confidence"] * (0.92 + 0.08 * overlap_ratio)
                
            result["confidence"] = round(new_confidence, 4)
            reranked.append(result)
            
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
        def _to_clock(sec: float) -> str:
            total_sec = 8 * 3600 + max(0.0, float(sec or 0.0))
            h = int((total_sec // 3600) % 24)
            m = int((total_sec % 3600) // 60)
            s = int(total_sec % 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        for r in fused_results:
            r["metadata"]["breakdown"] = r["breakdown"]
            cam_id = r["metadata"].get("camera_id", "Unknown")
            r["metadata"]["location"] = self.CAMERA_LOCATIONS.get(cam_id, "Unknown Location")
            
            # Default to 0.0 if timestamp is missing or None
            if "timestamp" not in r["metadata"] or r["metadata"].get("timestamp") is None:
                r["metadata"]["timestamp"] = 0.0
            if "end_time" not in r["metadata"] or r["metadata"].get("end_time") is None:
                r["metadata"]["end_time"] = r["metadata"]["timestamp"]
                
            r["metadata"]["start_time_str"] = _to_clock(r["metadata"]["timestamp"])
            r["metadata"]["end_time_str"] = _to_clock(r["metadata"]["end_time"])

        # ── Resolve Overlaps Per-Camera (Allowing Multi-Camera Concurrent Sightings) ──
        # Each camera independently resolves overlapping timeline blocks so that
        # valid sightings across Camera 1, Camera 2, and Camera 3 can all be surfaced!
        by_camera = defaultdict(list)
        for block in fused_results:
            cam_id = block["metadata"].get("camera_id", "Unknown")
            by_camera[cam_id].append(block)

        resolved_all: List[Dict] = []

        for cam_id, cam_blocks in by_camera.items():
            cam_blocks_sorted = sorted(cam_blocks, key=lambda x: x["confidence"], reverse=True)
            accepted_intervals: List[Tuple[float, float]] = []

            for block in cam_blocks_sorted:
                b_start = block["metadata"]["timestamp"]
                b_end   = block["metadata"]["end_time"]

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
                        break  # fully consumed on this camera

                for (seg_start, seg_end) in free_segments:
                    if seg_end >= seg_start:
                        entry = copy.deepcopy(block)
                        entry["metadata"]["timestamp"] = seg_start
                        entry["metadata"]["end_time"]  = max(seg_end, seg_start + 0.1)
                        accepted_intervals.append((seg_start, entry["metadata"]["end_time"]))
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

    def filter_and_tier_matches(self, resolved_results: List[Dict], min_thresh: float) -> List[Dict]:
        """
        Applies:
        1. Distinctiveness Margin Analysis (Generic Clothing / Green Shirt Detector)
        2. Spatio-Temporal Feasibility Chain with Hard Velocity Floor (v_max = 3.0 m/s)
        3. Multi-tier Output Classification ('confirmed' vs 'possible')
        """
        if not resolved_results:
            return []

        # 1. Distinctiveness Margin
        valid_candidates = [r for r in resolved_results if r["confidence"] >= (min_thresh - 0.05)]
        if not valid_candidates:
            return []

        valid_candidates.sort(key=lambda x: x["confidence"], reverse=True)
        is_ambiguous = False
        if len(valid_candidates) >= 3:
            top_score = valid_candidates[0]["confidence"]
            third_score = valid_candidates[2]["confidence"]
            has_face_in_top = any(r.get("breakdown", {}).get("face", 0.0) >= 0.35 for r in valid_candidates[:3])
            # If top scores are clustered tightly without facial biometric confirmation
            if (top_score - third_score) < 0.035 and not has_face_in_top:
                is_ambiguous = True

        effective_thresh = (min_thresh + 0.05) if is_ambiguous else min_thresh

        # Filter candidates passing effective threshold
        qualified = [r for r in valid_candidates if r["confidence"] >= effective_thresh]
        if not qualified:
            return []

        # 2. Spatio-Temporal Feasibility Chain
        # Group candidates by camera
        by_cam = defaultdict(list)
        for r in qualified:
            cam = r["metadata"].get("camera_id", "Unknown")
            by_cam[cam].append(r)

        # Pick the strongest global sighting as the spatio-temporal Anchor
        anchor = qualified[0]
        selected_by_cam = {anchor["metadata"].get("camera_id", "Unknown"): anchor}

        # For remaining cameras, pick candidate that does NOT violate physical velocity constraint
        v_max = 3.0  # m/s (maximum running velocity)
        for cam, candidates in by_cam.items():
            if cam in selected_by_cam:
                continue
            best_valid = None
            best_adj_score = -1.0

            for cand in candidates:
                cand_ok = True
                total_prior = 0.0

                # Validate against all currently accepted sightings in the chain
                for accepted_cam, accepted in selected_by_cam.items():
                    dist = self.CAMERA_TOPOLOGY.get(
                        (cand["metadata"].get("camera_id"), accepted["metadata"].get("camera_id")), 20.0
                    )
                    t_c_start = cand["metadata"].get("timestamp", 0.0)
                    t_c_end   = cand["metadata"].get("end_time", t_c_start)
                    t_a_start = accepted["metadata"].get("timestamp", 0.0)
                    t_a_end   = accepted["metadata"].get("end_time", t_a_start)

                    # Check temporal overlap across distinct cameras (impossible clone)
                    if max(t_c_start, t_a_start) < min(t_c_end, t_a_end):
                        cand_ok = False
                        break

                    # Non-overlapping: check transit time
                    if t_c_start >= t_a_end:
                        dt = t_c_start - t_a_end
                    else:
                        dt = t_a_start - t_c_end

                    min_t = dist / v_max
                    if dt < min_t:
                        # Impossible teleportation!
                        cand_ok = False
                        break

                    # Soft prior: lognormal distribution
                    exp_t = max(1.0, dist / 1.2)
                    mu = np.log(exp_t)
                    sigma = 0.6
                    lp = -((np.log(max(1e-3, dt)) - mu) ** 2) / (2.0 * (sigma ** 2))
                    total_prior += lp

                if cand_ok:
                    adj_score = cand["confidence"] * (1.0 + 0.05 * max(-1.0, min(1.0, total_prior)))
                    if adj_score > best_adj_score:
                        best_adj_score = adj_score
                        best_valid = cand

            if best_valid is not None:
                selected_by_cam[cam] = best_valid

        # 3. Multi-tier Tagging & Sorting
        final_list = list(selected_by_cam.values())
        has_multi_cam_chain = (len(final_list) >= 2)

        for r in final_list:
            face_score = r.get("breakdown", {}).get("face", 0.0)
            if face_score >= 0.35:
                r["metadata"]["tier"] = "confirmed"
                r["metadata"]["verification_reason"] = "Biometric Face Match"
            elif has_multi_cam_chain:
                r["metadata"]["tier"] = "confirmed"
                r["metadata"]["verification_reason"] = "Spatio-Temporal Feasible Trajectory"
            else:
                r["metadata"]["tier"] = "possible"
                r["metadata"]["verification_reason"] = "Visual Appearance Match"

            r["metadata"]["distinctiveness"] = "ambiguous" if is_ambiguous else "distinct"
            r["metadata"]["st_verified"] = has_multi_cam_chain

        final_list.sort(key=lambda x: x["metadata"].get("timestamp", 0.0))
        return final_list

