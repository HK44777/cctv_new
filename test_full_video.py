import cv2
import time
import numpy as np
from ultralytics import YOLO

video_path = 'data/temp_uploads/2a88a209-1d6f-4b1a-ac77-00f09a7f377d.mp4'
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
model = YOLO('yolov8n.pt')

tracks = {}
t0 = time.time()
f_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if f_idx % 2 == 0:  # 15 FPS cadence
        res = model.track(frame, persist=True, tracker='configs/bytetrack_cctv.yaml', classes=[0], verbose=False)
        boxes = res[0].boxes
        if boxes is not None and boxes.id is not None:
            tids = boxes.id.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            xyxys = boxes.xyxy.cpu().tolist()
            for tid, conf, xyxy in zip(tids, confs, xyxys):
                if tid not in tracks:
                    tracks[tid] = {
                        'hits': 0, 
                        'start': f_idx, 
                        'end': f_idx, 
                        'confs': [],
                        'box_hw': []
                    }
                tracks[tid]['hits'] += 1
                tracks[tid]['end'] = f_idx
                tracks[tid]['confs'].append(conf)
                h = xyxy[3] - xyxy[1]
                w = xyxy[2] - xyxy[0]
                tracks[tid]['box_hw'].append((h, w))
    f_idx += 1

cap.release()
elapsed = time.time() - t0
print(f"Processed {f_idx} frames in {elapsed:.2f}s ({f_idx/elapsed:.1f} FPS)")
print(f"Total raw tracks discovered: {len(tracks)}")

# Apply validation filter (Section 8)
valid_tracks = {}
filtered_reasons = {}

for tid, d in tracks.items():
    hits = d['hits']
    avg_conf = sum(d['confs']) / hits
    avg_h = sum(hw[0] for hw in d['box_hw']) / hits
    avg_w = sum(hw[1] for hw in d['box_hw']) / hits
    aspect = avg_h / max(1.0, avg_w)
    
    # Validation criteria:
    # 1. Minimum duration: at least 15 hits (1.0s at 15 FPS)
    # 2. Average confidence: >= 0.45
    # 3. Minimum human height: >= 70 pixels
    # 4. Aspect ratio: >= 1.15 (filters square backpacks)
    rejection = []
    if hits < 15:
        rejection.append(f"Too short ({hits} hits < 15)")
    if avg_conf < 0.45:
        rejection.append(f"Low avg conf ({avg_conf:.2f} < 0.45)")
    if avg_h < 70:
        rejection.append(f"Too small height ({avg_h:.1f}px < 70px)")
    if aspect < 1.15:
        rejection.append(f"Non-human aspect ratio ({aspect:.2f} < 1.15)")
        
    if not rejection:
        valid_tracks[tid] = {
            'hits': hits,
            'span': [d['start'], d['end']],
            'avg_conf': avg_conf,
            'avg_h': avg_h,
            'aspect': aspect
        }
    else:
        filtered_reasons[tid] = ", ".join(rejection)

print(f"\n--- VALIDATED HUMAN TRACKS ({len(valid_tracks)}) ---")
for tid, v in sorted(valid_tracks.items()):
    span_s = (v['span'][1] - v['span'][0]) / 30.01
    print(f"  Track {tid:2d}: {v['hits']:4d} hits | span [{v['span'][0]:4d}-{v['span'][1]:4d}] ({span_s:5.1f}s) | conf={v['avg_conf']:.2f} | H={v['avg_h']:.0f}px | aspect={v['aspect']:.2f}")

print(f"\n--- FILTERED FALSE TRACKS / NOISE ({len(filtered_reasons)}) ---")
for tid, r in sorted(filtered_reasons.items()):
    print(f"  Track {tid:2d}: REJECTED -> {r}")
