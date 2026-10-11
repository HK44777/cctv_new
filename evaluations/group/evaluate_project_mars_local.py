import os
import sys
import concurrent.futures
import cv2
import glob
import uuid
import yaml
import numpy as np
import re
from collections import defaultdict
from tqdm import tqdm

# Ensure workspace root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Safely import the Project modules by prepending Project to sys.path
PROJECT_DIR = os.path.join(ROOT_DIR, 'Project')
sys.path.insert(0, PROJECT_DIR)

from src.features import FaceEmbedder, BodyEmbedder, GaitEmbedder

# Restore sys.path
sys.path.pop(0)

# Import root modules
from src.database.qdrant_client import DatabaseClient
from src.core.fusion import ScoreFusion

print("Loading Project Config...")
config_path = os.path.join(PROJECT_DIR, 'config', 'default.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Enforce CUDA GPU execution
config['device'] = 'cuda'

print("Initializing Extractors (This will download weights if missing)...")
face_ext = FaceEmbedder(config)
body_ext = BodyEmbedder(config)
gait_ext = GaitEmbedder(config)

print("Initializing Local Qdrant Database...")
db = DatabaseClient()
# Create a ScoreFusion module with the same matching weights as Project
fusion_weights = config.get("matching", {}).get("weights", {"face": 0.5, "body": 0.35, "gait": 0.15})
fusion = ScoreFusion()
fusion.W_FACE = fusion_weights.get("face", 0.5)
fusion.W_APP = fusion_weights.get("body", 0.35)
fusion.W_GAIT = fusion_weights.get("gait", 0.15)

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
    return candidates[0]

MARS_DIR = find_mars_dir()

def group_tracklets(split_name):
    tracklets = defaultdict(list)
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
    # Load all images in tracklet once
    tracklet_images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            tracklet_images.append(img)
            
    if not tracklet_images:
        return None, None, None

    # Step sampling for appearance and face (to speed up processing)
    step = max(1, len(tracklet_images) // 5)
    sampled_images = tracklet_images[::step]

    # --- DEFINE PARALLEL TASKS ---
    def process_gait():
        gait_feat = gait_ext.extract(tracklet_images)
        if gait_feat is not None:
            return gait_feat.tolist()
        return None

    def process_appearance():
        app_vectors = body_ext.extract_batch(sampled_images)
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

    # --- EXECUTE TASKS IN PARALLEL ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_gait = executor.submit(process_gait)
        future_app = executor.submit(process_appearance)
        future_face = executor.submit(process_face)

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

        # Count total relevant gallery items for this query person (excluding same-cam junk)
        # Denominator for correct mAP: AP = sum(P@k * rel_k) / num_relevant_in_gallery
        num_relevant = sum(
            1 for res in formatted_results
            if res["metadata"]["person_id"] == q_pid
            and res["metadata"]["camera_id"] != q_cid  # exclude same-cam junk
        )
        if num_relevant == 0:
            continue

        total_queries += 1
        matches = [res["metadata"]["person_id"] == q_pid for res in filtered_results]

        if matches:
            if any(matches[:1]): rank1_hits += 1
            if any(matches[:5]): rank5_hits += 1
            if any(matches[:10]): rank10_hits += 1

            # Standard AP: sum of (precision@k * relevance@k) / total_relevant_in_gallery
            correct_count, precision_sum = 0, 0.0
            for k, is_match in enumerate(matches):
                if is_match:
                    correct_count += 1
                    precision_sum += correct_count / (k + 1)
            ap_scores.append(precision_sum / num_relevant)

    print("\n" + "="*45)
    print(" 🏆 FINAL MARS EVALUATION METRICS (BASE PROJECT) 🏆")
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
