import cv2
import time
import numpy as np
from ultralytics import YOLO

video_path = 'data/temp_uploads/2a88a209-1d6f-4b1a-ac77-00f09a7f377d.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO('yolov8n.pt')

tracks = {}
t0 = time.time()
for f_idx in range(600):
    ret, frame = cap.read()
    if not ret: 
        break
    if f_idx % 2 == 0:  # 15 FPS
        res = model.track(frame, persist=True, tracker='configs/bytetrack_cctv.yaml', classes=[0], verbose=False)
        boxes = res[0].boxes
        if boxes is not None and boxes.id is not None:
            tids = boxes.id.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            xyxys = boxes.xyxy.cpu().tolist()
            for tid, conf, xyxy in zip(tids, confs, xyxys):
                if tid not in tracks:
                    tracks[tid] = {'hits': 0, 'start': f_idx, 'end': f_idx, 'confs': []}
                tracks[tid]['hits'] += 1
                tracks[tid]['end'] = f_idx
                tracks[tid]['confs'].append(conf)

cap.release()
elapsed = time.time() - t0
print(f'Tuned ByteTrack: {len(tracks)} total tracks in 600 frames ({elapsed:.2f}s, {600/elapsed:.1f} FPS):')
for tid, d in sorted(tracks.items()):
    hits = d['hits']
    avg_c = sum(d['confs']) / hits
    start_f = d['start']
    end_f = d['end']
    print(f'  Track {tid:2d}: {hits:3d} hits, span [{start_f:3d}-{end_f:3d}], avg_conf={avg_c:.2f}')
