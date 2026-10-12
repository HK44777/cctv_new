import os
import sys
import glob
import cv2
import numpy as np
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pipeline.extractors.face import FaceExtractor
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


def extract_faces_from_folder(extractor: FaceExtractor, folder_path: str, desc: str = "Extracting Faces"):
    if not os.path.exists(folder_path):
        return np.array([]), np.array([]), np.array([]), 0.0, 0, 0

    filenames = sorted(os.listdir(folder_path))
    feats, pids, cams = [], [], []
    total_samples = 0
    detected_faces = 0

    for fname in tqdm(filenames, desc=desc):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')) or fname.startswith('._'):
            continue
        pid, cam = parse_market1501_filename(fname)
        if pid is None or cam is None:
            continue
        total_samples += 1

        p = os.path.join(folder_path, fname)
        img = cv2.imread(p)
        if img is None:
            continue

        vec = extractor.extract(img)
        if vec is not None:
            feats.append(vec)
            pids.append(pid)
            cams.append(cam)
            detected_faces += 1

    det_rate = (detected_faces / max(1, total_samples)) * 100.0
    return np.array(feats), np.array(pids), np.array(cams), det_rate, total_samples, detected_faces


def main():
    market_dir = find_market_dir()
    query_dir = os.path.join(market_dir, "query")
    gallery_dir = os.path.join(market_dir, "bounding_box_test")

    print("=" * 65)
    print(" 👤 EVALUATION: STANDALONE FACE (ARCFACE) ON MARKET-1501")
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

    # Strict GPU execution
    print("\nInitializing InsightFace (RetinaFace + ArcFace) on GPU...")
    extractor = FaceExtractor(strict_gpu=True)

    print("\nExtracting face embeddings for Gallery set (bounding_box_test)...")
    g_feats, g_pids, g_cams, g_det_rate, g_tot, g_det = extract_faces_from_folder(
        extractor, gallery_dir, desc="Gallery Ingestion (bounding_box_test)"
    )

    print("\nExtracting face embeddings for Query set (query)...")
    q_feats, q_pids, q_cams, q_det_rate, q_tot, q_det = extract_faces_from_folder(
        extractor, query_dir, desc="Query Evaluation (query)"
    )

    # Filter out distractor queries if any
    valid_q_mask = (q_pids != -1) & (q_pids != 0)
    q_feats = q_feats[valid_q_mask]
    q_pids = q_pids[valid_q_mask]
    q_cams = q_cams[valid_q_mask]

    print("\n" + "-" * 55)
    print(" 👤 FACE DETECTION COVERAGE STATS (MARKET-1501)")
    print("-" * 55)
    print(f"Gallery Samples : {g_tot} | Faces Detected: {g_det} ({g_det_rate:.1f}%)")
    print(f"Query Samples   : {q_tot} | Faces Detected: {q_det} ({q_det_rate:.1f}%)")
    print("-" * 55)

    if len(q_feats) == 0 or len(g_feats) == 0:
        print("\n❌ Error: No valid face embeddings extracted to perform ranking.")
        sys.exit(1)

    print("\nComputing Re-ID metrics on valid face queries...")
    results = compute_reid_metrics(q_feats, g_feats, q_pids, g_pids, q_cams, g_cams)
    results["face_detection_rate_query"] = q_det_rate
    print_metrics_table("FACE (ARCFACE) ON MARKET-1501 RESULTS", results)


if __name__ == "__main__":
    main()
