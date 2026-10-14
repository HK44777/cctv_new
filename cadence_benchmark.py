import cv2
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

video_path = 'data/temp_uploads/2a88a209-1d6f-4b1a-ac77-00f09a7f377d.mp4'

cadences = [
    ("ByteTrack", "configs/bytetrack_cctv.yaml", 30, 1),
    ("ByteTrack", "configs/bytetrack_cctv.yaml", 15, 2),
    ("ByteTrack", "configs/bytetrack_cctv.yaml", 10, 3),
    ("ByteTrack", "configs/bytetrack_cctv.yaml", 6, 5),
    ("DeepSORT", None, 6, 5),
]

NUM_FRAMES = 600

print(f"{'Tracker':<12} | {'Cadence':<10} | {'Frames Proc':<12} | {'Time (s)':<10} | {'FPS Proc':<10} | {'Total Tracks':<12} | {'Long Tracks (>=1s)':<18}")
print("-" * 95)

for tracker_type, cfg, target_fps, interval in cadences:
    cap = cv2.VideoCapture(video_path)
    t0 = time.time()
    tracks = {}
    frames_proc = 0
    
    if tracker_type == "ByteTrack":
        model = YOLO("yolov8n.pt")
        for f_idx in range(NUM_FRAMES):
            ret, frame = cap.read()
            if not ret: break
            if f_idx % interval == 0:
                frames_proc += 1
                res = model.track(frame, persist=True, tracker=cfg, classes=[0], verbose=False)
                boxes = res[0].boxes
                if boxes is not None and boxes.id is not None:
                    tids = boxes.id.int().cpu().tolist()
                    for tid in tids:
                        if tid not in tracks:
                            tracks[tid] = []
                        tracks[tid].append(f_idx)
    else: # DeepSORT
        from src.pipeline.detector import PersonDetector
        detector = PersonDetector(strict_gpu=True)
        deepsort = DeepSort(max_age=90, n_init=3, max_cosine_distance=0.35, max_iou_distance=0.8, embedder_gpu=True)
        for f_idx in range(NUM_FRAMES):
            ret, frame = cap.read()
            if not ret: break
            if f_idx % interval == 0:
                frames_proc += 1
                boxes, confs = detector.detect(frame)
                dets = [(b, c, 0) for b, c in zip(boxes, confs)]
                active = deepsort.update_tracks(dets, frame=frame)
                for tr in active:
                    if tr.is_confirmed() and tr.time_since_update <= 1:
                        tid = str(tr.track_id)
                        if tid not in tracks:
                            tracks[tid] = []
                        tracks[tid].append(f_idx)
                        
    cap.release()
    elapsed = time.time() - t0
    proc_fps = frames_proc / elapsed
    long_tracks = [t for t, f in tracks.items() if len(f) >= target_fps]
    
    print(f"{tracker_type:<12} | {target_fps} FPS{'':<5} | {frames_proc:<12} | {elapsed:<10.2f} | {proc_fps:<10.1f} | {len(tracks):<12} | {len(long_tracks):<18}")
