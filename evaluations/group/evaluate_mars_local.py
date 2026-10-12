import os
import sys
import concurrent.futures
import cv2
import glob
import uuid
import numpy as np
import re
import gc
import torch
from collections import defaultdict
from tqdm import tqdm

# Persistent worker pool to avoid recreating threads 8,000+ times
EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=3)

# Ensure workspace root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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

def find_mars_dir():
    """Finds MARS dataset directory from standard locations."""
    candidates = [
        "C:/Users/PC1/Downloads/mars_dataset/OneDrive_1_2022-4-2",
        os.path.join(ROOT_DIR, "data", "mars"),
        os.path.join(ROOT_DIR, "mars_dataset"),
        os.path.expanduser("~/Downloads/mars_dataset"),
    ]
    for path in candidates:
        if path and (os.path.exists(os.path.join(path, "query"))
                     or os.path.exists(os.path.join(path, "bbox_test"))):
            return path
    return candidates[0]  # Will fail with clear error if none found

MARS_DIR = find_mars_dir()

def group_tracklets(split_name):
    tracklets = defaultdict(list)
    # Try direct path first, then nested (some MARS downloads have split_name/split_name)
    search_pattern = os.path.join(MARS_DIR, split_name, "**", "*.jpg")
    image_paths = glob.glob(search_pattern, recursive=True)
    if not image_paths:
        search_pattern = os.path.join(MARS_DIR, split_name, split_name, "**", "*.jpg")
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
    # READ ONCE
    tracklet_images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            tracklet_images.append(img)
            
    if not tracklet_images:
        return None, None, None

    # FRAME SAMPLING
    step = max(1, len(tracklet_images) // 5)
    sampled_images = tracklet_images[::step]

    # --- DEFINE PARALLEL TASKS ---
    def process_gait():
        skeleton_maps = []
        skeletons = gait_ext.extract_skeleton_batch(tracklet_images)
        for skel in skeletons:
            if skel is not None:
                skeleton_maps.append(skel)
        if len(skeleton_maps) >= 15:
            return gait_ext.extract_gait_vector(skeleton_maps[:30])
        return None

    def process_appearance():
        app_vectors = app_ext.extract_batch(sampled_images)
        if app_vectors is not None and len(app_vectors) > 0:
            mean_app = np.mean(app_vectors, axis=0)
            return (mean_app / np.linalg.norm(mean_app)).tolist()
        return None

    def process_face():
        face_vectors = face_ext.extract_batch(sampled_images)
        if face_vectors is not None and len(face_vectors) > 0:
            mean_face = np.mean(face_vectors, axis=0)
            return (mean_face / np.linalg.norm(mean_face)).tolist()
        return None

    # --- EXECUTE TASKS IN PARALLEL (reusing persistent thread pool) ---
    future_gait = EXECUTOR.submit(process_gait)
    future_app = EXECUTOR.submit(process_appearance)
    future_face = EXECUTOR.submit(process_face)

    final_gait = future_gait.result()
    final_app = future_app.result()
    final_face = future_face.result()

    return final_app, final_face, final_gait

def main():
    print(f"MARS dataset root: {MARS_DIR}")
    gallery_tracklets = group_tracklets('bbox_test')
    if not gallery_tracklets:
        print(f"Error: Could not find any images for gallery (bbox_test) at '{MARS_DIR}'.")
        return

    # CRITICAL: Clear DB before each run to prevent stale/duplicate gallery entries
    print("Clearing previous Qdrant collection data...")
    db.client.delete_collection(db.collection_name)
    db._ensure_collection()
    print("✅ Collection reset. Starting fresh ingestion.")

    BATCH_SIZE = 100
    batch_points = []
    print(f"Found {len(gallery_tracklets)} tracklets in Gallery. Starting Ingestion (Batched)...")
    for idx, ((pid, cid, tid), img_paths) in enumerate(tqdm(gallery_tracklets.items(), desc="Ingesting Gallery"), start=1):
        app, face, gait = extract_tracklet_features(img_paths)
        if app or face or gait:
            metadata = {"person_id": pid, "camera_id": cid, "tracklet_id": tid, "is_gallery": True}
            batch_points.append({
                "uuid": str(uuid.uuid4()),
                "appearance": app,
                "face": face,
                "gait": gait,
                "metadata": metadata
            })
            if len(batch_points) >= BATCH_SIZE:
                db.insert_batch(batch_points)
                batch_points.clear()

        # Periodic VRAM & memory flush every 100 tracklets
        if idx % 100 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    if batch_points:
        db.insert_batch(batch_points)
        batch_points.clear()
    # Ground truth: Count valid gallery tracklets per person per camera for official Re-ID mAP
    gallery_rel_count = defaultdict(lambda: defaultdict(int))
    for (g_pid, g_cid, _) in gallery_tracklets.keys():
        gallery_rel_count[g_pid][g_cid] += 1

    query_tracklets = group_tracklets('query')
    print(f"Found {len(query_tracklets)} tracklets in Query set. Starting Evaluation...")
    rank1_hits, rank5_hits, rank10_hits, total_queries = 0, 0, 0, 0
    ap_scores = []

    for idx, ((q_pid, q_cid, q_tid), img_paths) in enumerate(tqdm(query_tracklets.items(), desc="Evaluating Queries"), start=1):
        if idx % 100 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        # Exclude queries where the person does not exist in any other camera in gallery
        total_rel_in_gallery = sum(
            count for g_cid, count in gallery_rel_count[q_pid].items() if g_cid != q_cid
        )
        if total_rel_in_gallery == 0:
            continue

        app, face, gait = extract_tracklet_features(img_paths)
        if not app and not face and not gait:
            continue

        # Retrieve all gallery tracklets to evaluate full ranking without truncation
        search_results = db.search_ensemble(query_appearance=app, query_face=face, query_gait=gait, limit=len(gallery_tracklets))
        formatted_results = fusion.format_results(search_results, min_threshold=0.0)

        # Filter out same-camera junk
        filtered_results = []
        for res in formatted_results:
            g_pid = res["metadata"]["person_id"]
            g_cid = res["metadata"]["camera_id"]
            if g_pid == q_pid and g_cid == q_cid:
                continue
            filtered_results.append(res)

        total_queries += 1
        matches = [res["metadata"]["person_id"] == q_pid for res in filtered_results]

        if matches:
            if any(matches[:1]): rank1_hits += 1
            if any(matches[:5]): rank5_hits += 1
            if any(matches[:10]): rank10_hits += 1

            # Standard AP: sum of (precision@k * relevance@k) / total_rel_in_gallery
            correct_count, precision_sum = 0, 0.0
            for k, is_match in enumerate(matches):
                if is_match:
                    correct_count += 1
                    precision_sum += correct_count / (k + 1)
            ap_scores.append(precision_sum / total_rel_in_gallery)
        else:
            ap_scores.append(0.0)

    print("\n" + "="*45)
    print(" 🏆 FINAL MARS EVALUATION METRICS (YOUR PROJECT) 🏆")
    print("=" * 45)
    print(f"Total Valid Queries: {total_queries}")
    print(f"Rank-1 Accuracy:  {(rank1_hits / max(1, total_queries)) * 100:.2f}%")
    print(f"Rank-5 Accuracy:  {(rank5_hits / max(1, total_queries)) * 100:.2f}%")
    print(f"Rank-10 Accuracy: {(rank10_hits / max(1, total_queries)) * 100:.2f}%")
    if ap_scores:
        print(f"mAP (Mean AP):    {np.mean(ap_scores) * 100:.2f}%")
    print("=" * 45)

if __name__ == "__main__":
    main()
