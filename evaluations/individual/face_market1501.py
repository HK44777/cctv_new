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


def extract_faces_from_folder(extractor: FaceExtractor, folder_path: str, desc: str = "Extracting Faces"):
    images = glob.glob(os.path.join(folder_path, "*.jpg"))
    images.sort()

    feats, pids, cams = [], [], []
    total_samples = 0
    detected_faces = 0

    for p in tqdm(images, desc=desc):
        pid, cam = parse_market1501_filename(p)
        if pid is None or cam is None:
            continue
        total_samples += 1

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
    print(f"Hardware     : GPU STRICT (CUDA)")
    print(f"Sampling     : Single-frame crops (Market-1501 static images)")

    if not os.path.exists(query_dir) or not os.path.exists(gallery_dir):
        print(f"\n❌ Error: Could not find Market-1501 folders at '{market_dir}'.")
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
