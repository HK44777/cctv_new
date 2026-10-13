import cv2
import time
from src.pipeline.enhancer import ZeroDCEEnhancer
from src.pipeline.detector import PersonDetector
from src.pipeline.tracker import MultiObjectTracker
from src.pipeline.extractors.appearance import AppearanceExtractor
from src.pipeline.extractors.face import FaceExtractor
from src.pipeline.extractors.gait import GaitExtractor
from src.core.track_manager import TrackManager
from src.database.qdrant_client import DatabaseClient

class VideoProcessor:
    def __init__(self):
        print("Initializing Pipeline...")
        # strict_gpu=True -> enforces that the ingestion pipeline MUST run these on the GPU.
        self.enhancer = ZeroDCEEnhancer(strict_gpu=True)
        self.detector = PersonDetector(strict_gpu=True)
        
        # We need a dictionary to hold a separate tracker instance for each camera
        self.trackers = {}

        # Per-camera counter of received (processed) frames - used to subsample heavy extractors
        # Since queue sends at 6 FPS, every 6th received frame = 1 FPS for face & appearance
        self.cam_frame_counts = {}
        
        self.app_ext  = AppearanceExtractor(strict_gpu=True)
        self.face_ext = FaceExtractor(strict_gpu=True)
        self.gait_ext = GaitExtractor(strict_gpu=True)
        
        self.db = DatabaseClient()
        self.track_manager = TrackManager(self.db)

    def process_frame(self, frame, camera_id, timestamp=None, video_path=None, frame_idx=None):
        t_start = time.perf_counter()
        
        # Step 1: Enhance
        t0 = time.perf_counter()
        enhanced = self.enhancer.enhance(frame)
        t_enhance = time.perf_counter() - t0
        
        # Step 2: Detect
        t0 = time.perf_counter()
        boxes, confs = self.detector.detect(enhanced)
        t_detect = time.perf_counter() - t0
        
        # Step 3: Track (Per-Camera Tracker)
        t0 = time.perf_counter()
        if camera_id not in self.trackers:
            self.trackers[camera_id] = MultiObjectTracker()
            
        active_tracks = self.trackers[camera_id].update(enhanced, boxes, confs)
        t_track = time.perf_counter() - t0
        
        # Timestamp (use provided video timestamp or fallback to wall-clock for live streams)
        current_time = timestamp if timestamp is not None else time.time()
        
        num_tracks = len(active_tracks)
        img_h, img_w = enhanced.shape[:2]

        # Increment per-camera received-frame counter ONCE per frame (outside the per-person loop)
        self.cam_frame_counts[camera_id] = self.cam_frame_counts.get(camera_id, 0) + 1
        extract_heavy = (self.cam_frame_counts[camera_id] % 6 == 0)  # True at 1 FPS

        # Step 4: Batch-crop all active persons
        # Collect per-track info and crops first, then run GPU inference in one batch call.
        global_tids = []
        crops       = []
        metadatas   = []
        bboxes      = []

        for track in active_tracks:
            local_tid  = track["track_id"]
            global_tid = f"{camera_id}_{local_tid}"
            x1, y1, x2, y2 = track["bbox"]

            # 5% padding margin to prevent cropping heads/feet
            w = x2 - x1
            h = y2 - y1
            pad_x = int(w * 0.05)
            pad_y = int(h * 0.05)

            crop = enhanced[
                max(0, y1 - pad_y) : min(img_h, y2 + pad_y),
                max(0, x1 - pad_x) : min(img_w, x2 + pad_x)
            ]
            if crop.size == 0:
                continue

            global_tids.append(global_tid)
            crops.append(crop)
            metadatas.append({"camera_id": camera_id, "timestamp": current_time})
            bboxes.append([int(x1), int(y1), int(x2), int(y2)])

        # Step 5 & 6: Batch Appearance + Face (at 1 FPS)
        t_extract_app  = 0.0
        t_extract_face = 0.0
        t_extract_gait = 0.0
        t_register     = 0.0

        app_vecs  = [None] * len(crops)
        face_vecs = [None] * len(crops)

        if crops and extract_heavy:
            # Appearance batch - one GPU forward pass for ALL persons in this frame
            t0 = time.perf_counter()
            app_batch = self.app_ext.extract_batch(crops)
            t_extract_app = time.perf_counter() - t0
            for i, vec in enumerate(app_batch):
                if i < len(app_vecs):
                    app_vecs[i] = vec

            # Face - InsightFace does not support true batching; loop per-crop
            t0 = time.perf_counter()
            for i, crop in enumerate(crops):
                face_vecs[i] = self.face_ext.extract(crop)
            t_extract_face = time.perf_counter() - t0

        # Step 7: Skeleton batch (every frame - needed for 30-frame gait window)
        if crops:
            t0 = time.perf_counter()
            skeleton_maps = self.gait_ext.extract_skeleton_batch(crops)
            t_extract_gait = time.perf_counter() - t0
        else:
            skeleton_maps = []

        # Step 8: Occlusion Gating & Register per track
        # Prevent ID-switch contamination: If bounding boxes overlap (IoU > 0.25),
        # do not accumulate appearance or face embeddings into the track's running average.
        def _calc_iou(b1, b2):
            ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
            a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
            union = a1 + a2 - inter
            return (inter / union) if union > 0 else 0.0

        is_occluded = [False] * len(bboxes)
        for i in range(len(bboxes)):
            for j in range(len(bboxes)):
                if i != j and _calc_iou(bboxes[i], bboxes[j]) > 0.25:
                    is_occluded[i] = True
                    break

        t0 = time.perf_counter()
        for i, global_tid in enumerate(global_tids):
            skeleton = skeleton_maps[i] if i < len(skeleton_maps) else None
            # Gate appearance & face when occluded to prevent chimeric embeddings
            app_to_reg = None if is_occluded[i] else app_vecs[i]
            face_to_reg = None if is_occluded[i] else face_vecs[i]

            self.track_manager.register_frame_data(
                track_id=global_tid,
                appearance=app_to_reg,
                face=face_to_reg,
                skeleton=skeleton,
                metadata=metadatas[i],
                video_path=video_path,
                frame_idx=frame_idx,
                bbox=bboxes[i]
            )
            self.track_manager.process_gait_buffer(global_tid, self.gait_ext)
        t_register = time.perf_counter() - t0

        # Step 9: Clean up RAM and save to Qdrant for any tracks that have timed out
        t0 = time.perf_counter()
        self.track_manager.cleanup_stale_tracks(current_time, gait_extractor=self.gait_ext)
        t_cleanup = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start
        
        # Logging
        print(f"\n[{camera_id}] Frame Processing Time: {t_total*1000:.1f}ms")
        print(f"  |-- Enhance: {t_enhance*1000:.1f}ms")
        print(f"  |-- Detect:  {t_detect*1000:.1f}ms")
        print(f"  |-- Track:   {t_track*1000:.1f}ms")
        if num_tracks > 0:
            print(f"  |-- Extractions ({num_tracks} tracks, batched):")
            print(f"  |   |-- Appearance (batch): {t_extract_app*1000:.1f}ms  avg {t_extract_app*1000/num_tracks:.1f}ms/track")
            print(f"  |   |-- Face       (loop):  {t_extract_face*1000:.1f}ms  avg {t_extract_face*1000/num_tracks:.1f}ms/track")
            print(f"  |   -- Skeleton   (batch): {t_extract_gait*1000:.1f}ms  avg {t_extract_gait*1000/num_tracks:.1f}ms/track")
            print(f"  |-- Register/RAM: {t_register*1000:.1f}ms  avg {t_register*1000/num_tracks:.1f}ms/track")
        else:
            print(f"  |-- Extractions: 0ms (0 tracks)")
            print(f"  |-- Register/RAM:0ms (0 tracks)")
        print(f"  -- Cleanup: {t_cleanup*1000:.1f}ms")

    def finalize_all(self):
        """Called by main.py at the very end to flush remaining tracks"""
        # Pass gait_ext so final skeleton buffers (15-29 frames) produce embeddings instead of being discarded
        self.track_manager.close_all_tracks(gait_extractor=self.gait_ext)
        self.cam_frame_counts.clear()
        print("Finalized all remaining tracks.")
