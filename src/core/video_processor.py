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

        # Per-camera counter of received (processed) frames — used to subsample heavy extractors
        # Since queue sends at 6 FPS, every 6th received frame = 1 FPS for face & appearance
        self.cam_frame_counts = {}
        
        self.app_ext = AppearanceExtractor(strict_gpu=True)
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
        
        t_extract_app = 0.0
        t_extract_face = 0.0
        t_extract_gait = 0.0
        t_register = 0.0
        
        num_tracks = len(active_tracks)
        img_h, img_w = enhanced.shape[:2]

        # Increment per-camera received-frame counter ONCE per frame (outside the per-person loop)
        self.cam_frame_counts[camera_id] = self.cam_frame_counts.get(camera_id, 0) + 1
        extract_heavy = (self.cam_frame_counts[camera_id] % 6 == 0)  # True at 1 FPS

        for track in active_tracks:
            local_tid = track["track_id"]

            # CRITICAL: DeepSORT will output ID=1 for camera A, and ID=1 for camera B.
            # We must namespace the ID to prevent TrackManager from mixing them up.
            global_tid = f"{camera_id}_{local_tid}"
            
            x1, y1, x2, y2 = track["bbox"]
            
            # Add a 5% padding margin to the bounding box to prevent cropping edges (heads/feet)
            w = x2 - x1
            h = y2 - y1
            pad_x = int(w * 0.05)
            pad_y = int(h * 0.05)
            
            # Step 4: Precision Crop (Safely clamped to image dimensions with margin)
            crop = enhanced[max(0, y1 - pad_y):min(img_h, y2 + pad_y), max(0, x1 - pad_x):min(img_w, x2 + pad_x)]
            
            if crop.size == 0:
                continue
                
            app_vec = None
            face_vec = None
            
            # Subsample: extract heavy features (Face & Appearance) at 1 FPS (every 6th received frame per camera)
            if extract_heavy:
                # Step 5: Appearance
                t_e = time.perf_counter()
                app_vec = self.app_ext.extract(crop)
                t_extract_app += time.perf_counter() - t_e
                
                # Step 6: Face
                t_e = time.perf_counter()
                face_vec = self.face_ext.extract(crop)
                t_extract_face += time.perf_counter() - t_e
            
            # Step 7: Skeleton Map (SkeletonGait) - Extract on ALL frames to build 30-frame sequence
            t_e = time.perf_counter()
            skeleton = self.gait_ext.extract_skeleton(crop)
            t_extract_gait += time.perf_counter() - t_e
            
            # Register to RAM
            t_r = time.perf_counter()
            self.track_manager.register_frame_data(
                track_id=global_tid, 
                appearance=app_vec, 
                face=face_vec, 
                skeleton=skeleton,
                metadata={"camera_id": camera_id, "timestamp": current_time},
                video_path=video_path,
                frame_idx=frame_idx,
                bbox=[int(x1), int(y1), int(x2), int(y2)]
            )
            
            # Process Gait buffer if full
            self.track_manager.process_gait_buffer(global_tid, self.gait_ext)
            t_register += time.perf_counter() - t_r
                
        # Step 8: Clean up RAM and save to Qdrant for any tracks that have timed out
        t0 = time.perf_counter()
        # Pass gait_ext so we can squeeze out final gait embeddings for quick walkers
        self.track_manager.cleanup_stale_tracks(current_time, gait_extractor=self.gait_ext)
        t_cleanup = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start
        
        # Logging
        print(f"\n[{camera_id}] Frame Processing Time: {t_total*1000:.1f}ms")
        print(f"  ├─ Enhance: {t_enhance*1000:.1f}ms")
        print(f"  ├─ Detect:  {t_detect*1000:.1f}ms")
        print(f"  ├─ Track:   {t_track*1000:.1f}ms")
        if num_tracks > 0:
            print(f"  ├─ Extractions ({num_tracks} tracks):")
            print(f"  │  ├─ Appearance: {t_extract_app*1000:.1f}ms (avg {t_extract_app*1000/num_tracks:.1f}ms/track)")
            print(f"  │  ├─ Face:       {t_extract_face*1000:.1f}ms (avg {t_extract_face*1000/num_tracks:.1f}ms/track)")
            print(f"  │  └─ Skeleton:   {t_extract_gait*1000:.1f}ms (avg {t_extract_gait*1000/num_tracks:.1f}ms/track)")
            print(f"  ├─ Register/RAM:{t_register*1000:.1f}ms (avg {t_register*1000/num_tracks:.1f}ms/track)")
        else:
            print(f"  ├─ Extractions: 0ms (0 tracks)")
            print(f"  ├─ Register/RAM:0ms (0 tracks)")
        print(f"  └─ Cleanup: {t_cleanup*1000:.1f}ms")

    def finalize_all(self):
        """Called by main.py at the very end to flush remaining tracks"""
        # Pass gait_ext so final skeleton buffers (15-29 frames) produce embeddings instead of being discarded
        self.track_manager.close_all_tracks(gait_extractor=self.gait_ext)
        self.cam_frame_counts.clear()
        print("Finalized all remaining tracks.")
