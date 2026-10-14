import copy
import os
import cv2
import uuid
import base64
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
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

# Target FPS for subsampling query videos — 15 FPS provides optimal balance
# between temporal continuity (66.7ms step) and real-time GPU throughput (>100 FPS)
QUERY_TARGET_FPS = 15


class RetrievalService:
    # Per-camera human-readable location labels used in timeline metadata.
    CAMERA_LOCATIONS: Dict[str, str] = {
        "Camera_1": "entry-left",
        "Camera_2": "entry-left-front",
        "Camera_3": "entry-right-back",
        "Camera_4": "hallway-corridor",
    }

    # Spatio-temporal distance matrix (metres) between camera viewpoints
    CAMERA_TOPOLOGY: Dict[Tuple[str, str], float] = {
        ("Camera_1", "Camera_1"): 0.0,
        ("Camera_2", "Camera_2"): 0.0,
        ("Camera_3", "Camera_3"): 0.0,
        ("Camera_4", "Camera_4"): 0.0,
        ("Camera_1", "Camera_2"): 15.0,
        ("Camera_2", "Camera_1"): 15.0,
        ("Camera_1", "Camera_3"): 30.0,
        ("Camera_3", "Camera_1"): 30.0,
        ("Camera_2", "Camera_3"): 20.0,
        ("Camera_3", "Camera_2"): 20.0,
        ("Camera_1", "Camera_4"): 25.0,
        ("Camera_4", "Camera_1"): 25.0,
        ("Camera_2", "Camera_4"): 20.0,
        ("Camera_4", "Camera_2"): 20.0,
        ("Camera_3", "Camera_4"): 15.0,
        ("Camera_4", "Camera_3"): 15.0,
    }

    def __init__(self, db_client: Optional[DatabaseClient] = None):
        print("Initializing Retrieval Service Models...")

        # strict_gpu=True -> enforces that the server MUST run these on the GPU.
        self.enhancer = ZeroDCEEnhancer(strict_gpu=True)
        self.detector = PersonDetector(strict_gpu=True)
        self.app_ext  = AppearanceExtractor(strict_gpu=True)
        self.face_ext = FaceExtractor(strict_gpu=True)
        self.gait_ext = GaitExtractor(strict_gpu=True)

        self.db     = db_client if db_client is not None else DatabaseClient()
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

    @staticmethod
    def _create_person_thumbnail(frame: np.ndarray, bbox: List[int], tid: str) -> str:
        """
        Creates a high-clarity, square-padded crop centered on the target person.
        Ensures the person is front-and-center, clearly visible, and not shrunk into a wide camera shot.
        """
        img_h, img_w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        # Center of the person
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Add 15% margin so the whole person (head, torso, legs) is comfortably inside
        target_size = int(max(bw, bh) * 1.30)
        half = target_size // 2

        crop_x1 = max(0, cx - half)
        crop_y1 = max(0, cy - half)
        crop_x2 = min(img_w, cx + half)
        crop_y2 = min(img_h, cy + half)

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        if crop.size == 0:
            crop = frame.copy()

        # Draw green bounding box around person relative to the crop
        bx1 = max(1, x1 - crop_x1)
        by1 = max(1, y1 - crop_y1)
        bx2 = min(crop.shape[1] - 2, x2 - crop_x1)
        by2 = min(crop.shape[0] - 2, y2 - crop_y1)

        cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 255, 128), 2)

        # Draw neat Track ID badge
        label = f"Track #{tid}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        badge_y1 = max(0, by1 - th - 6)
        badge_y2 = by1
        cv2.rectangle(crop, (bx1, badge_y1), (bx1 + tw + 8, badge_y2), (0, 255, 128), -1)
        cv2.putText(crop, label, (bx1 + 4, by1 - 4), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # Resize to standard crisp 256x256 square thumbnail for the frontend card grid
        crop_resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
        _, buffer = cv2.imencode('.jpg', crop_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return base64.b64encode(buffer).decode('utf-8')

    def process_video_first_pass(self, video_path: str, session_id: str) -> List[ThumbnailResponse]:
        """
        First pass: detect and track all persons in the query video using GPU-accelerated ByteTrack.
        Operates at QUERY_TARGET_FPS (15 FPS) cadence for optimal Kalman motion continuity and track stability.
        
        Applies multi-signal tracklet validation (conf >= 0.45, aspect >= 1.15, height >= 70px, duration >= 1.0s)
        to eliminate false-positive object tracks (backpacks on benches) while preserving stationary seated students.
        
        Extracts multi-crop quality-filtered OSNet appearance representations and applies constrained
        global tracklet association (enforcing temporal non-overlap, spatial continuity, and competitive margin)
        to prevent erroneous cross-student identity merges.
        """
        cap = cv2.VideoCapture(video_path)
        tracker = MultiObjectTracker(tracker_backend="deepsort")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(fps / QUERY_TARGET_FPS)))

        self._tracking_cache[session_id] = {}
        best_candidates: Dict[str, Dict[str, Any]] = {}

        # Multi-signal diagnostic tracking accumulators
        track_confs: Dict[str, List[float]] = defaultdict(list)
        track_bboxes: Dict[str, List[List[int]]] = defaultdict(list)
        track_crops: Dict[str, List[Tuple[float, int, np.ndarray]]] = defaultdict(list)
        track_centers: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                active_tracks = tracker.update(frame)

                for track in active_tracks:
                    tid  = str(track["track_id"])
                    bbox = track["bbox"]
                    conf = track.get("conf", 0.8)

                    if tid not in self._tracking_cache[session_id]:
                        self._tracking_cache[session_id][tid] = []
                    self._tracking_cache[session_id][tid].append((frame_idx, bbox))

                    track_confs[tid].append(conf)
                    track_bboxes[tid].append(bbox)

                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    track_centers[tid].append((cx, cy))

                    bw = max(1, bbox[2] - bbox[0])
                    bh = max(1, bbox[3] - bbox[1])
                    area = bw * bh

                    h_frame, w_frame = frame.shape[:2]
                    crop = frame[max(0, y1):min(h_frame, y2), max(0, x1):min(w_frame, x2)]

                    # Track best clear appearance
                    if tid not in best_candidates or (conf >= 0.50 and area > best_candidates[tid]["area"]):
                        best_candidates[tid] = {
                            "frame": frame.copy(),
                            "bbox": bbox,
                            "area": area,
                            "conf": conf
                        }

                    # Collect quality crops for multi-crop Re-ID
                    if conf >= 0.45 and bh >= 70 and crop.size > 0:
                        track_crops[tid].append((conf, area, crop.copy()))

            frame_idx += 1

        cap.release()

        # ── 1. Multi-signal Validation Filter (replaces naive MIN_TRACK_FRAMES=6) ──
        # Eliminates backpacks, chairs, and fleeting glitches without discarding seated students
        MIN_DETECTIONS = max(6, int(round(QUERY_TARGET_FPS * 1.0)))  # >= 1.0s of observations
        valid_tids = []
        rejected_tracks = {}

        for tid in list(best_candidates.keys()):
            hits = len(self._tracking_cache[session_id].get(tid, []))
            confs = track_confs.get(tid, [0.0])
            avg_conf = sum(confs) / max(1, len(confs))

            bboxes = track_bboxes.get(tid, [])
            avg_h = sum((b[3] - b[1]) for b in bboxes) / max(1, len(bboxes))
            avg_w = sum((b[2] - b[0]) for b in bboxes) / max(1, len(bboxes))
            aspect = avg_h / max(1.0, avg_w)

            reasons = []
            if hits < MIN_DETECTIONS:
                reasons.append(f"insufficient duration ({hits} hits < {MIN_DETECTIONS})")
            if avg_conf < 0.45:
                reasons.append(f"low detection confidence ({avg_conf:.2f} < 0.45)")
            if avg_h < 70:
                reasons.append(f"bounding box too small ({avg_h:.1f}px < 70px)")
            if aspect < 1.15:
                reasons.append(f"non-human aspect ratio ({aspect:.2f} < 1.15)")
            if len(track_crops.get(tid, [])) < 2:
                reasons.append("insufficient clear crops for biometric identification")

            if not reasons:
                valid_tids.append(tid)
            else:
                rejected_tracks[tid] = ", ".join(reasons)

        print(f"[Tracker] Validated {len(valid_tids)} legitimate persons. Filtered {len(rejected_tracks)} noise/object tracks:")
        for r_tid, reason in rejected_tracks.items():
            print(f"  |-- Excluded Track #{r_tid}: {reason}")

        # ── 2. Robust Multi-Crop OSNet Appearance Embeddings ──
        track_embeddings: Dict[str, np.ndarray] = {}
        track_metadata: Dict[str, Dict[str, Any]] = {}

        for tid in valid_tids:
            crops_list = track_crops.get(tid, [])
            if not crops_list:
                continue

            # Sample up to 5 quality crops distributed across the track duration
            n_crops = len(crops_list)
            step = max(1, n_crops // 5)
            selected_crops = [crops_list[i][2] for i in range(0, n_crops, step)[:5]]

            vecs = self.app_ext.extract_batch(selected_crops)
            valid_vecs = [np.array(v, dtype=np.float32) for v in vecs if v is not None]
            if valid_vecs:
                mean_vec = np.mean(valid_vecs, axis=0)
                norm = np.linalg.norm(mean_vec)
                if norm > 0:
                    track_embeddings[tid] = mean_vec / norm

            # Record temporal and spatial bounds for association
            history = self._tracking_cache[session_id].get(tid, [])
            centers = track_centers.get(tid, [])
            track_metadata[tid] = {
                "start_frame": history[0][0] if history else 0,
                "end_frame": history[-1][0] if history else 0,
                "start_center": centers[0] if centers else (0.0, 0.0),
                "end_center": centers[-1] if centers else (0.0, 0.0),
                "frame_set": set(h[0] for h in history)
            }

        # ── 3. Constrained Tracklet Association (Replaces Unsafe 0.72 Single-Crop Merge) ──
        # Keeps local_track_id separate from global identity hypotheses
        # Enforces:
        #  a) Temporal non-overlap (distinct simultaneous persons rejected)
        #  b) Spatial continuity and velocity feasibility
        #  c) High cosine similarity (>= 0.82) with competitive margin (>= 0.04)
        active_representatives = []
        tracklet_to_global: Dict[str, str] = {tid: tid for tid in valid_tids}

        for tid in valid_tids:
            if tid not in track_embeddings:
                active_representatives.append(tid)
                continue

            vec = track_embeddings[tid]
            meta = track_metadata[tid]

            candidate_matches = []

            for rep in active_representatives:
                if rep not in track_embeddings:
                    continue

                rep_meta = track_metadata[rep]

                # Check 1: Temporal Overlap Constraint
                overlap_frames = meta["frame_set"].intersection(rep_meta["frame_set"])
                n_overlap = len(overlap_frames)
                if n_overlap > 3:
                    # Overlapping in time: distinct people simultaneously in the scene -> HARD REJECTION
                    continue

                # Check 2: Temporal Order and Gap
                if meta["end_frame"] <= rep_meta["start_frame"]:
                    earlier_meta, later_meta = meta, rep_meta
                elif rep_meta["end_frame"] <= meta["start_frame"]:
                    earlier_meta, later_meta = rep_meta, meta
                else:
                    continue

                dt_seconds = (later_meta["start_frame"] - earlier_meta["end_frame"]) / fps
                if dt_seconds > 45.0:  # Time gap too long for reliable local association
                    continue

                # Check 3: Physical Velocity Constraint
                dx = later_meta["start_center"][0] - earlier_meta["end_center"][0]
                dy = later_meta["start_center"][1] - earlier_meta["end_center"][1]
                dist_px = np.sqrt(dx * dx + dy * dy)
                velocity_px_s = dist_px / max(0.1, dt_seconds)

                if velocity_px_s > 600.0 and dt_seconds < 1.0:
                    continue  # Teleportation across camera frame

                sim = float(np.dot(vec, track_embeddings[rep]))
                if sim >= 0.82:
                    candidate_matches.append((sim, rep))

            # Select best match with competitive margin
            matched_rep = None
            if candidate_matches:
                candidate_matches.sort(key=lambda x: x[0], reverse=True)
                top_sim, top_rep = candidate_matches[0]
                second_sim = candidate_matches[1][0] if len(candidate_matches) > 1 else 0.0

                # Enforce margin over second-best candidate to prevent merging similar-clothed students
                if (top_sim - second_sim) >= 0.04 or len(candidate_matches) == 1:
                    matched_rep = top_rep

            if matched_rep is not None:
                print(f"[Association] Globally linked fragmented Tracklet #{tid} -> Representative Person #{matched_rep} (Sim={top_sim:.3f})")
                tracklet_to_global[tid] = matched_rep

                # Extend representative's cached frames for comprehensive evidence search
                self._tracking_cache[session_id][matched_rep].extend(
                    self._tracking_cache[session_id][tid]
                )

                # Keep the clearer/larger crop if this fragment had better camera proximity
                if best_candidates[tid]["area"] > best_candidates[matched_rep]["area"]:
                    best_candidates[matched_rep] = best_candidates[tid]
            else:
                active_representatives.append(tid)

        # ── 4. Build clean thumbnails for the consolidated people ──
        thumbnails_list = []
        sorted_tids = sorted(active_representatives, key=lambda t: int(t) if t.isdigit() else t)
        for tid in sorted_tids:
            cand = best_candidates[tid]
            thumb_b64 = self._create_person_thumbnail(cand["frame"], cand["bbox"], tid)
            thumbnails_list.append(ThumbnailResponse(track_id=tid, thumbnail_base64=thumb_b64))

        return thumbnails_list

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
        if not target_history:
            return []

        # Sort history chronologically by frame index
        sorted_history = sorted(target_history, key=lambda x: x[0])

        # If the track has more than 45 observations, pick the highest-resolution continuous 45-frame window
        # (Where the person is closest and clearest to the camera for biometric fidelity)
        MAX_FRAMES = 45
        if len(sorted_history) > MAX_FRAMES:
            best_start = 0
            max_area_sum = 0
            for s in range(0, len(sorted_history) - MAX_FRAMES + 1, 5):
                area_sum = sum((b[2] - b[0]) * (b[3] - b[1]) for _, b in sorted_history[s:s + MAX_FRAMES])
                if area_sum > max_area_sum:
                    max_area_sum = area_sum
                    best_start = s
            active_history = sorted_history[best_start:best_start + MAX_FRAMES]
        else:
            active_history = sorted_history

        # ── Pass 1: read crops using fast sequential grab instead of repeated seeking ──
        cap = cv2.VideoCapture(video_path)
        all_crops = []
        current_pos = -1

        for frame_idx, bbox in active_history:
            # If current position is close to target frame, grab forward (0.1ms per frame vs 80ms cap.set)
            if current_pos == -1 or frame_idx < current_pos or (frame_idx - current_pos) > 25:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                current_pos = frame_idx
            else:
                while current_pos < frame_idx:
                    cap.grab()
                    current_pos += 1

            ret, frame = cap.read()
            current_pos += 1
            if not ret:
                all_crops.append(None)
                continue

            x1, y1, x2, y2 = bbox
            h_f, w_f = frame.shape[:2]
            crop = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
            if crop.size == 0:
                all_crops.append(None)
                continue

            all_crops.append(crop)

        cap.release()

        # ── Pass 2: batch GPU inference ──
        valid_crops = [c for c in all_crops if c is not None and c.size > 0]
        if not valid_crops:
            return []

        # Select top 5 largest/clearest crops for Appearance and Face
        heavy_crops = sorted(valid_crops, key=lambda c: c.shape[0] * c.shape[1], reverse=True)[:5]

        # Appearance — GPU forward pass for the top 5 crops
        app_vectors = []
        if heavy_crops:
            app_batch = self.app_ext.extract_batch(heavy_crops)
            app_vectors = [v for v in app_batch if v is not None]

        # Face — InsightFace per-crop on top 5 crops
        face_vectors = []
        for crop in heavy_crops:
            fv = self.face_ext.extract(crop)
            if fv is not None:
                face_vectors.append(fv)

        # Skeleton — YOLO batch pose on valid crops
        skeletons = []
        if valid_crops:
            skel_batch = self.gait_ext.extract_skeleton_batch(valid_crops)
            skeletons  = [s for s in skel_batch if s is not None]

        # Aggregate embeddings via L2-normalised mean
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

    def k_reciprocal_rerank(self, fused_results: List[Dict], limit: int = 10) -> List[Dict]:
        """
        Post-processing step that recalculates confidence based on mutual nearest-neighbor relationships.
        Asymmetric candidate hubs that fail mutual linking with the query cluster are penalized.
        """
        if not fused_results:
            return []

        reranked = []
        query_neighbors = {r["uuid"] for r in fused_results[:limit]}

        for i, result in enumerate(fused_results):
            # Only rerank the top candidates to keep interactive query latency sub-second
            if i >= 8:
                reranked.append(result)
                continue

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

