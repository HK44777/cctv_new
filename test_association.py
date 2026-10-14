import cv2
import numpy as np
import math
from ultralytics import YOLO
from src.pipeline.extractors.appearance import AppearanceExtractor

video_path = 'data/temp_uploads/2a88a209-1d6f-4b1a-ac77-00f09a7f377d.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO('yolov8n.pt')
app_ext = AppearanceExtractor(strict_gpu=True)

# 1. Run tracking & collect crops and metadata
tracks_data = {}
f_idx = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    if f_idx % 2 == 0:  # 15 FPS
        res = model.track(frame, persist=True, tracker='configs/bytetrack_cctv.yaml', classes=[0], verbose=False)
        boxes = res[0].boxes
        if boxes is not None and boxes.id is not None:
            tids = boxes.id.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            xyxys = boxes.xyxy.cpu().tolist()
            for tid, conf, xyxy in zip(tids, confs, xyxys):
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                h, w = frame.shape[:2]
                box_h = y2 - y1
                box_w = x2 - x1
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                
                if tid not in tracks_data:
                    tracks_data[tid] = {
                        'frames': [],
                        'bboxes': [],
                        'confs': [],
                        'crops': [],
                        'centers': []
                    }
                tracks_data[tid]['frames'].append(f_idx)
                tracks_data[tid]['bboxes'].append([x1, y1, x2, y2])
                tracks_data[tid]['confs'].append(conf)
                tracks_data[tid]['centers'].append(((x1+x2)/2.0, (y1+y2)/2.0))
                
                # Keep high-quality crops for appearance
                if conf >= 0.50 and box_h >= 80 and crop.size > 0:
                    tracks_data[tid]['crops'].append((conf, box_h * box_w, crop.copy()))
    f_idx += 1

cap.release()

# 2. Filter false tracks
valid_tracks = {}
for tid, d in tracks_data.items():
    hits = len(d['frames'])
    avg_conf = sum(d['confs']) / hits
    avg_h = sum(b[3]-b[1] for b in d['bboxes']) / hits
    avg_w = sum(b[2]-b[0] for b in d['bboxes']) / hits
    aspect = avg_h / max(1.0, avg_w)
    
    if hits >= 15 and avg_conf >= 0.45 and avg_h >= 70 and aspect >= 1.15 and len(d['crops']) >= 3:
        valid_tracks[tid] = d

print(f"Validated tracks for association: {len(valid_tracks)} tracks ({list(valid_tracks.keys())})")

# 3. Compute multi-crop L2-normalized mean OSNet embeddings
track_embeds = {}
for tid, d in valid_tracks.items():
    # Pick up to 5 quality crops (spread across time)
    n_crops = len(d['crops'])
    step = max(1, n_crops // 5)
    selected_crops = [d['crops'][i][2] for i in range(0, n_crops, step)[:5]]
    
    vecs = app_ext.extract_batch(selected_crops)
    vecs = [np.array(v, dtype=np.float32) for v in vecs if v is not None]
    if vecs:
        mean_vec = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            track_embeds[tid] = {
                'embedding': mean_vec / norm,
                'start_f': d['frames'][0],
                'end_f': d['frames'][-1],
                'start_center': d['centers'][0],
                'end_center': d['centers'][-1],
                'frame_set': set(d['frames'])
            }

# 4. Association logic with Temporal & Spatial Constraints
# Compare all pairs (t1, t2)
associations = []
rejections = []

tids = sorted(track_embeds.keys())
for i in range(len(tids)):
    for j in range(i+1, len(tids)):
        t1, t2 = tids[i], tids[j]
        e1, e2 = track_embeds[t1], track_embeds[t2]
        
        sim = float(np.dot(e1['embedding'], e2['embedding']))
        
        # Check temporal overlap
        overlap_frames = e1['frame_set'].intersection(e2['frame_set'])
        n_overlap = len(overlap_frames)
        
        if n_overlap > 3:
            # Overlapping in time! Are they physically separated?
            rejections.append((t1, t2, sim, f"Simultaneous in scene ({n_overlap} overlapping frames)"))
            continue
            
        # Order temporally: earlier track, later track
        if e1['end_f'] <= e2['start_f']:
            earlier, later = e1, e2
            pair = (t1, t2)
        elif e2['end_f'] <= e1['start_f']:
            earlier, later = e2, e1
            pair = (t2, t1)
        else:
            rejections.append((t1, t2, sim, "Ambiguous temporal overlap"))
            continue
            
        dt_frames = later['start_f'] - earlier['end_f']
        dt_seconds = dt_frames / 30.01
        
        # Spatial displacement
        dx = later['start_center'][0] - earlier['end_center'][0]
        dy = later['start_center'][1] - earlier['end_center'][1]
        dist_px = math.sqrt(dx*dx + dy*dy)
        velocity_px_s = dist_px / max(0.1, dt_seconds)
        
        # Spatial-temporal plausibility
        if dt_seconds > 45.0:
            rejections.append((pair[0], pair[1], sim, f"Time gap too large ({dt_seconds:.1f}s > 45s)"))
            continue
            
        if velocity_px_s > 600.0 and dt_seconds < 1.0:
            rejections.append((pair[0], pair[1], sim, f"Velocity impossible ({velocity_px_s:.0f} px/s)"))
            continue
            
        # Cosine similarity threshold
        if sim >= 0.82:
            associations.append((pair[0], pair[1], sim, dt_seconds, dist_px))
        else:
            rejections.append((pair[0], pair[1], sim, f"Cosine similarity {sim:.3f} < 0.82"))

print("\n--- ASSOCIATED TRACKLETS (GLOBAL PERSON LINKAGES) ---")
if not associations:
    print("  No tracklets merged (all tracklets represent distinct people or separate visits).")
for a in associations:
    print(f"  Track {a[0]} -> Track {a[1]}: Cosine Sim={a[2]:.3f}, Time Gap={a[3]:.1f}s, Distance={a[4]:.0f}px")

print("\n--- REPRESENTATIVE REJECTIONS ---")
for r in rejections[:15]:
    print(f"  Track {r[0]} vs Track {r[1]}: Cosine Sim={r[2]:.3f} -> REJECTED: {r[3]}")
