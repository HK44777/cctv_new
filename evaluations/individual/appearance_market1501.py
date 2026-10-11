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
    """Finds Market-1501 dataset directory from standard locations."""
    candidates = [
        "C:/Users/PC1/Downloads/Market-1501-v15.09.15",
        os.path.join(ROOT_DIR, "data", "Market-1501-v15.09.15"),
        os.path.join(ROOT_DIR, "data", "Market-1501"),
        os.path.join(ROOT_DIR, "Market-1501-v15.09.15"),
        os.path.expanduser("~/Downloads/Market-1501-v15.09.15"),
    ]
    for path in candidates:
        if path and os.path.exists(os.path.join(path, "query")) and os.path.exists(os.path.join(path, "bounding_box_test")):
            return path
    return candidates[0]


def load_dataset(folder_path: str):
    """Loads valid image paths, person IDs, and camera IDs from a Market-1501 folder."""
    images = glob.glob(os.path.join(folder_path, "*.jpg"))
    images.sort()

    valid_paths, pids, cams = [], [], []
    for p in images:
        pid, cam = parse_market1501_filename(p)
        if pid is not None and cam is not None:
            valid_paths.append(p)
            pids.append(pid)
            cams.append(cam)
    return valid_paths, np.array(pids), np.array(cams)


def extract_features(extractor: AppearanceExtractor, image_paths: list, batch_size: int = 64):
    """Batched extraction of OSNet appearance vectors on GPU."""
    features = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting OSNet features (GPU)"):
        batch_paths = image_paths[i : i + batch_size]
        crops = [cv2.imread(p) for p in batch_paths]
        valid_crops = [c for c in crops if c is not None]
        if not valid_crops:
            continue
        vecs = extractor.extract_batch(valid_crops)
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
    print(f"Hardware     : GPU STRICT (CUDA)")
    print(f"Sampling     : Single-frame crops (Market-1501 static images)")

    if not os.path.exists(query_dir) or not os.path.exists(gallery_dir):
        print(f"\n❌ Error: Could not find Market-1501 folders at '{market_dir}'.")
        print("Please ensure Market-1501 is placed in your downloads or data folder.")
        sys.exit(1)

    print("\nLoading dataset lists...")
    q_paths, q_pids, q_cams = load_dataset(query_dir)
    g_paths, g_pids, g_cams = load_dataset(gallery_dir)
    print(f"-> Gallery images (bounding_box_test): {len(g_paths)} ({len(np.unique(g_pids))} unique persons)")
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
