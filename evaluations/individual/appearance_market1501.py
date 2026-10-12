import os
import sys
import glob
import cv2
import numpy as np
from tqdm import tqdm

# Ensure workspace root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pipeline.extractors.appearance import AppearanceExtractor
from evaluations.common import parse_market1501_filename, compute_reid_metrics, print_metrics_table


def find_market_dir() -> str:
    """
    Finds Market-1501 dataset directory across standard locations, workspace paths,
    and dynamic directory searches (including zip extraction variations).
    """
    # 1. Direct standard candidates
    candidates = [
        "C:/Users/PC1/Downloads/Market-1501-v15.09.15",
        "C:/Users/PC1/Downloads/Market-1501",
        "C:/Users/PC1/Downloads/mars_dataset/OneDrive_1_2022-4-2/Market-1501-v15.09.15",
        "C:/Users/PC1/Downloads/mars_dataset/Market-1501-v15.09.15",
        os.path.join(ROOT_DIR, "Market-1501-v15.09.15"),
        os.path.join(ROOT_DIR, "data", "Market-1501-v15.09.15"),
        os.path.join(ROOT_DIR, "data", "Market-1501"),
        os.path.expanduser("~/Downloads/Market-1501-v15.09.15"),
        os.path.expanduser("~/Downloads/Market-1501"),
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            if os.path.exists(os.path.join(path, "query")) and os.path.exists(os.path.join(path, "bounding_box_test")):
                return path

    # 2. Dynamic search across common download & workspace roots
    search_roots = [
        "C:/Users/PC1/Downloads",
        "C:/Users/PC1/Downloads/mars_dataset",
        "C:/Users/PC1/Downloads/mars_dataset/OneDrive_1_2022-4-2",
        os.path.abspath(os.path.join(ROOT_DIR, "..")),
        ROOT_DIR,
    ]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        try:
            for entry in os.listdir(root):
                if "market" in entry.lower():
                    full_p = os.path.join(root, entry)
                    if os.path.isdir(full_p):
                        # Direct match
                        if os.path.exists(os.path.join(full_p, "query")) and os.path.exists(os.path.join(full_p, "bounding_box_test")):
                            return full_p
                        # Subdirectory match (e.g. nested folder from unzipping)
                        for sub in os.listdir(full_p):
                            sub_p = os.path.join(full_p, sub)
                            if os.path.isdir(sub_p) and os.path.exists(os.path.join(sub_p, "query")) and os.path.exists(os.path.join(sub_p, "bounding_box_test")):
                                return sub_p
        except Exception:
            pass

    return candidates[0]


def load_dataset(folder_path: str):
    """Loads valid image paths, person IDs, and camera IDs from a Market-1501 folder."""
    if not os.path.exists(folder_path):
        return [], np.array([]), np.array([])

    filenames = sorted(os.listdir(folder_path))
    valid_paths, pids, cams = [], [], []
    for fname in filenames:
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')) or fname.startswith('._'):
            continue
        pid, cam = parse_market1501_filename(fname)
        if pid is not None and cam is not None:
            valid_paths.append(os.path.join(folder_path, fname))
            pids.append(pid)
            cams.append(cam)
    return valid_paths, np.array(pids), np.array(cams)


def extract_features(extractor: AppearanceExtractor, image_paths: list, batch_size: int = 64):
    """Batched extraction of OSNet appearance vectors on GPU with length alignment guarantee."""
    features = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting OSNet features (GPU)"):
        batch_paths = image_paths[i : i + batch_size]
        crops = []
        for p in batch_paths:
            img = cv2.imread(p)
            if img is None:
                # Fallback blank crop if file is corrupted/unreadable to keep array sizes 1:1
                img = np.zeros((256, 128, 3), dtype=np.uint8)
            crops.append(img)

        vecs = extractor.extract_batch(crops)
        features.extend(vecs)
    return np.array(features)


def main():
    market_dir = find_market_dir()
    query_dir = os.path.join(market_dir, "query")
    gallery_dir = os.path.join(market_dir, "bounding_box_test")

    print("=" * 65)
    print(" 👔 EVALUATION: STANDALONE APPEARANCE (OSNet) ON MARKET-1501")
    print("=" * 65)
    print(f"Dataset root : {market_dir}")
    print(f"Query dir    : {query_dir}")
    print(f"Gallery dir  : {gallery_dir}")
    print(f"Hardware     : GPU STRICT (CUDA)")
    print(f"Sampling     : Single-frame crops (Market-1501 static images)")

    if not os.path.exists(query_dir) or not os.path.exists(gallery_dir):
        print(f"\n❌ Error: Could not find Market-1501 folders at '{market_dir}'.")
        print("Please ensure Market-1501 is placed in your downloads or data folder.")
        sys.exit(1)

    print("\nLoading dataset lists...")
    q_paths, q_pids, q_cams = load_dataset(query_dir)
    g_paths, g_pids, g_cams = load_dataset(gallery_dir)

    # Filter distractors out of queries if any exist
    valid_q_mask = (q_pids != -1) & (q_pids != 0)
    q_paths = [q_paths[i] for i in range(len(q_paths)) if valid_q_mask[i]]
    q_pids = q_pids[valid_q_mask]
    q_cams = q_cams[valid_q_mask]

    num_g_distractors = int(np.sum((g_pids == -1) | (g_pids == 0)))
    num_g_valid = len(g_pids) - num_g_distractors

    print(f"-> Gallery images (bounding_box_test): {len(g_paths)} ({len(np.unique(g_pids[g_pids > 0]))} persons, {num_g_distractors} distractors)")
    print(f"-> Query images (query)              : {len(q_paths)} ({len(np.unique(q_pids))} unique persons)")

    # Strict GPU execution
    print("\nInitializing OSNet Appearance Extractor on GPU...")
    extractor = AppearanceExtractor(strict_gpu=True)

    print("\nExtracting features for Gallery set (bounding_box_test)...")
    g_feats = extract_features(extractor, g_paths, batch_size=64)

    print("\nExtracting features for Query set (query)...")
    q_feats = extract_features(extractor, q_paths, batch_size=64)

    print("\nComputing Re-ID retrieval metrics (Market-1501 camera filtering protocol)...")
    results = compute_reid_metrics(q_feats, g_feats, q_pids, g_pids, q_cams, g_cams)
    print_metrics_table("APPEARANCE (OSNet) ON MARKET-1501 RESULTS", results)


if __name__ == "__main__":
    main()
