import os
import cv2
import glob
import uuid
import numpy as np
import re
from collections import defaultdict
from tqdm import tqdm

from src.pipeline.extractors.appearance import AppearanceExtractor
from src.pipeline.extractors.face import FaceExtractor
from src.pipeline.extractors.gait import GaitExtractor
from src.database.qdrant_client import DatabaseClient
from src.core.fusion import ScoreFusion

print("Initializing Extractors (This will download weights if missing)...")
app_ext = AppearanceExtractor(strict_gpu=True)
face_ext = FaceExtractor(strict_gpu=True)
gait_ext = GaitExtractor(strict_gpu=True)

print("Initializing Local Qdrant Database...")
db = DatabaseClient()
fusion = ScoreFusion()

def parse_mars_filename(filename):
    if filename.startswith('._'):
        return None, None, None
    if '_' in filename:
        parts = filename.split('_')
        if len(parts) >= 3:
            return parts[0], parts[1][:2], parts[1][2:]
    match = re.search(r'^([a-zA-Z0-9\-]+)C(\d)T(\d+)', filename)
    if match:
        return match.group(1), 'c' + match.group(2), match.group(3)
    return None, None, None

def group_tracklets(split_name):
    tracklets = defaultdict(list)
    search_pattern = f"C:/Users/PC1/Downloads/mars_dataset/OneDrive_1_2022-4-2/{split_name}/{split_name}/**/*.jpg"

    image_paths = glob.glob(search_pattern, recursive=True)

    for path in image_paths:
        filename = os.path.basename(path)
        pid, cid, tid = parse_mars_filename(filename)
        if pid is None or pid == '0000' or pid == '-1':
            continue
        tracklets[(pid, cid, tid)].append(path)

    for key in tracklets:
        tracklets[key].sort()
    return tracklets

def extract_tracklet_features(image_paths):
    app_vectors = []
    face_vectors = []
    skeleton_maps = []

    # READ ONCE
    tracklet_images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            tracklet_images.append(img)
            
    if not tracklet_images:
        return None, None, None

    # BATCH GAIT
    skeletons = gait_ext.extract_skeleton_batch(tracklet_images)
    for skel in skeletons:
        if skel is not None:
            skeleton_maps.append(skel)

    # FRAME SAMPLING
    step = max(1, len(tracklet_images) // 5)
    sampled_images = tracklet_images[::step]

    # BATCH APPEARANCE
    app_vectors = app_ext.extract_batch(sampled_images)

    # SEQUENTIAL FACE
    for img in sampled_images:
        face = face_ext.extract(img)
        if face is not None:
            face_vectors.append(face)

    final_app, final_face, final_gait = None, None, None
    if app_vectors:
        mean_app = np.mean(app_vectors, axis=0)
        final_app = (mean_app / np.linalg.norm(mean_app)).tolist()
    if face_vectors:
        mean_face = np.mean(face_vectors, axis=0)
        final_face = (mean_face / np.linalg.norm(mean_face)).tolist()
    if len(skeleton_maps) >= 15:
        final_gait = gait_ext.extract_gait_vector(skeleton_maps[:30])
    return final_app, final_face, final_gait

def main():
    gallery_tracklets = group_tracklets('bbox_test')
    if not gallery_tracklets:
        print(f"Error: Could not find any images for gallery (bbox_test). Please make sure the 'bbox_test' folder is in your workspace.")
        return
        
    print(f"Found {len(gallery_tracklets)} tracklets in Gallery. Starting Ingestion...")
    for (pid, cid, tid), img_paths in tqdm(gallery_tracklets.items()):
        app, face, gait = extract_tracklet_features(img_paths)
        if app or face or gait:
            metadata = {"person_id": pid, "camera_id": cid, "tracklet_id": tid, "is_gallery": True}
            db.insert(uuid=str(uuid.uuid4()), appearance=app, face=face, gait=gait, metadata=metadata)
    print("✅ Gallery Ingestion Complete!\n")

    query_tracklets = group_tracklets('query')
    print(f"Found {len(query_tracklets)} tracklets in Query set. Starting Evaluation...")
    rank1_hits, rank5_hits, rank10_hits, total_queries = 0, 0, 0, 0
    ap_scores = []

    for (q_pid, q_cid, q_tid), img_paths in tqdm(query_tracklets.items()):
        app, face, gait = extract_tracklet_features(img_paths)
        if not app and not face and not gait:
            continue

        search_results = db.search_ensemble(query_appearance=app, query_face=face, query_gait=gait, limit=50)
        formatted_results = fusion.format_results(search_results, min_threshold=0.0)

        filtered_results = []
        for res in formatted_results:
            g_pid = res["metadata"]["person_id"]
            g_cid = res["metadata"]["camera_id"]
            if g_pid == q_pid and g_cid == q_cid:
                continue
            filtered_results.append(res)

        if not filtered_results:
            continue

        total_queries += 1
        matches = [res["metadata"]["person_id"] == q_pid for res in filtered_results]

        if matches:
            if any(matches[:1]): rank1_hits += 1
            if any(matches[:5]): rank5_hits += 1
            if any(matches[:10]): rank10_hits += 1

            correct_count, precision_sum = 0, 0.0
            for k, is_match in enumerate(matches):
                if is_match:
                    correct_count += 1
                    precision_sum += correct_count / (k + 1)
            if correct_count > 0:
                ap_scores.append(precision_sum / correct_count)
            else:
                ap_scores.append(0.0)

    print("\n" + "="*45)
    print(" 🏆 FINAL MARS EVALUATION METRICS (YOUR PROJECT) 🏆")
    print("="*45)
    print(f"Total Valid Queries: {total_queries}")
    print(f"Rank-1 Accuracy:  {(rank1_hits / total_queries) * 100:.2f}%")
    print(f"Rank-5 Accuracy:  {(rank5_hits / total_queries) * 100:.2f}%")
    print(f"Rank-10 Accuracy: {(rank10_hits / total_queries) * 100:.2f}%")
    if ap_scores:
        print(f"mAP (Mean AP):    {np.mean(ap_scores) * 100:.2f}%")
    print("="*45)

if __name__ == "__main__":
    main()
