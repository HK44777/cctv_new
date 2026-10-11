import os
import sys
import glob
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pipeline.extractors.appearance import AppearanceExtractor
from evaluations.common import parse_mars_filename, compute_reid_metrics, print_metrics_table


def find_mars_dir() -> str:
    """Finds MARS dataset directory from standard locations."""
    candidates = [
        "C:/Users/PC1/Downloads/mars_dataset/OneDrive_1_2022-4-2",
        os.path.join(ROOT_DIR, "data", "mars"),
        os.path.join(ROOT_DIR, "mars_dataset"),
        os.path.expanduser("~/Downloads/mars_dataset"),
    ]
    for path in candidates:
        if path and (
            os.path.exists(os.path.join(path, "query"))
            or os.path.exists(os.path.join(path, "bbox_test"))
        ):
            return path
    return candidates[0]


def group_tracklets(base_dir: str, split_name: str):
    tracklets = defaultdict(list)
    search_pattern = os.path.join(base_dir, split_name, "**", "*.jpg")
    img_paths = glob.glob(search_pattern, recursive=True)
    if not img_paths:
        search_pattern = os.path.join(base_dir, split_name, split_name, "**", "*.jpg")
        img_paths = glob.glob(search_pattern, recursive=True)

    img_paths.sort()
    for p in img_paths:
        fname = os.path.basename(p)
        pid, cid, tid = parse_mars_filename(fname)
        if pid is None or pid == 0 or pid == -1:
            continue
        tracklets[(pid, cid, tid)].append(p)

    return tracklets


def extract_tracklet_appearance(extractor: AppearanceExtractor, image_paths: list):
    """
    Frame Sampling: Stride-based sampling of 5 evenly spaced frames per tracklet,
    matching evaluate_mars_local.py.
    """
    crops = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is not None:
            crops.append(img)
    if not crops:
        return None

    # Sample 5 evenly spaced frames across the tracklet
    step = max(1, len(crops) // 5)
    sampled = crops[::step]

    vecs = extractor.extract_batch(sampled)
    if vecs and len(vecs) > 0:
        mean_vec = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            return mean_vec / norm
    return None


def main():
    mars_dir = find_mars_dir()
    print("=" * 65)
    print(" 👔 EVALUATION: STANDALONE APPEARANCE (OSNet) ON MARS TRACKLETS")
    print("=" * 65)
    print(f"Dataset root : {mars_dir}")
    print(f"Hardware     : GPU STRICT (CUDA)")
    print(f"Sampling     : 5 evenly spaced frames per tracklet (stride = len // 5)")

    gallery_tracklets = group_tracklets(mars_dir, "bbox_test")
    query_tracklets = group_tracklets(mars_dir, "query")

    if not gallery_tracklets or not query_tracklets:
        print(f"\n❌ Error: Could not find MARS tracklets at '{mars_dir}'.")
        print("Please ensure bbox_test and query directories exist.")
        sys.exit(1)

    print(f"-> Gallery tracklets (bbox_test): {len(gallery_tracklets)}")
    print(f"-> Query tracklets (query)      : {len(query_tracklets)}")

    # Strict GPU execution
    print("\nInitializing OSNet Appearance Extractor on GPU...")
    extractor = AppearanceExtractor(strict_gpu=True)

    print("\nIngesting Gallery tracklets (bbox_test)...")
    g_feats, g_pids, g_cams = [], [], []
    for (pid, cid, tid), img_paths in tqdm(gallery_tracklets.items(), desc="Gallery Ingestion"):
        vec = extract_tracklet_appearance(extractor, img_paths)
        if vec is not None:
            g_feats.append(vec)
            g_pids.append(pid)
            g_cams.append(cid)

    print("\nEvaluating Query tracklets (query)...")
    q_feats, q_pids, q_cams = [], [], []
    for (pid, cid, tid), img_paths in tqdm(query_tracklets.items(), desc="Query Evaluation"):
        vec = extract_tracklet_appearance(extractor, img_paths)
        if vec is not None:
            q_feats.append(vec)
            q_pids.append(pid)
            q_cams.append(cid)

    print("\nComputing Re-ID metrics (MARS camera-filtering protocol)...")
    results = compute_reid_metrics(
        np.array(q_feats), np.array(g_feats),
        np.array(q_pids), np.array(g_pids),
        np.array(q_cams), np.array(g_cams)
    )
    print_metrics_table("APPEARANCE (OSNet) ON MARS RESULTS", results)


if __name__ == "__main__":
    main()
