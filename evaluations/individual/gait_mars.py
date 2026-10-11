import os
import sys
import glob
import cv2
import numpy as np
import yaml
from collections import defaultdict
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from evaluations.common import parse_mars_filename, compute_reid_metrics, print_metrics_table


def find_mars_dir() -> str:
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


def extract_ours_gait(extractor, img_paths: list, min_skeletons: int = 15):
    """
    Frame Sampling for SkeletonGait:
    Uses the full tracklet sequence. YOLOv8-pose extracts 17 COCO keypoints per frame.
    Requires at least 15 valid skeleton heatmaps; the first 30 frames are passed to the 3D CNN.
    """
    images = [cv2.imread(p) for p in img_paths]
    images = [img for img in images if img is not None]
    if len(images) < min_skeletons:
        return None

    skeletons = extractor.extract_skeleton_batch(images)
    valid_skels = [s for s in skeletons if s is not None]
    if len(valid_skels) >= min_skeletons:
        vec = extractor.extract_gait_vector(valid_skels[:30])
        if vec is not None:
            vec = np.array(vec)
            norm = np.linalg.norm(vec)
            if norm > 0:
                return vec / norm
    return None


def extract_paper_gait(embedder, img_paths: list):
    """
    Frame Sampling for Base Paper:
    Uses all frames in the tracklet, computes Otsu binary silhouettes, averages into a GEI,
    and reduces via PCA.
    """
    images = [cv2.imread(p) for p in img_paths]
    images = [img for img in images if img is not None]
    if not images:
        return None
    vec = embedder.extract(images)
    if vec is not None:
        vec = np.array(vec)
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
    return None


def evaluate_gait_extractor(extract_fn, tracklets_dict, desc: str):
    feats, pids, cams = [], [], []
    for (pid, cid, tid), img_paths in tqdm(tracklets_dict.items(), desc=desc):
        vec = extract_fn(img_paths)
        if vec is not None:
            feats.append(vec)
            pids.append(pid)
            cams.append(cid)
    return np.array(feats), np.array(pids), np.array(cams)


def run_model_evaluation(model_type: str, gallery_tracklets, query_tracklets):
    if model_type == "ours":
        from src.pipeline.extractors.gait import GaitExtractor
        print("\nLoading YOUR SkeletonGait Extractor (YOLOv8-pose + 3D CNN) on GPU...")
        ext = GaitExtractor(strict_gpu=True)
        extract_fn = lambda paths: extract_ours_gait(ext, paths, min_skeletons=15)
        label = "YOUR SKELETONGAIT PIPELINE (src/)"
    else:
        project_dir = os.path.join(ROOT_DIR, "Project")
        sys.path.insert(0, project_dir)
        from src.features.gait_embedder import GaitEmbedder
        sys.path.pop(0)

        cfg_file = os.path.join(project_dir, "config", "default.yaml")
        with open(cfg_file, "r") as f:
            cfg = yaml.safe_load(f)
        cfg["device"] = "cuda"
        print("\nLoading BASE PAPER GaitEmbedder (Otsu Silhouette + GEI + PCA) on GPU...")
        embedder = GaitEmbedder(cfg)
        extract_fn = lambda paths: extract_paper_gait(embedder, paths)
        label = "BASE PAPER GAIT (GEI + PCA) PIPELINE (Project/)"

    print(f"\nIngesting Gallery tracklets (bbox_test) for {label}...")
    g_feats, g_pids, g_cams = evaluate_gait_extractor(extract_fn, gallery_tracklets, f"Gallery Ingestion ({model_type})")

    print(f"\nEvaluating Query tracklets (query) for {label}...")
    q_feats, q_pids, q_cams = evaluate_gait_extractor(extract_fn, query_tracklets, f"Query Evaluation ({model_type})")

    if len(q_feats) == 0 or len(g_feats) == 0:
        print(f"❌ Error: Could not extract enough gait samples for {label}.")
        return None

    results = compute_reid_metrics(q_feats, g_feats, q_pids, g_pids, q_cams, g_cams)
    print_metrics_table(f"GAIT ON MARS ({label})", results)
    return results


def main():
    mars_dir = find_mars_dir()
    print("=" * 65)
    print(" 🚶 EVALUATION: GAIT COMPARISON ON MARS (PAPER GEI vs YOUR SKELETONGAIT)")
    print("=" * 65)
    print(f"Dataset root : {mars_dir}")
    print(f"Hardware     : GPU STRICT (CUDA)")
    print(f"Sampling     : Full tracklet video sequences (min 15 frames, up to 30)")

    gallery_tracklets = group_tracklets(mars_dir, "bbox_test")
    query_tracklets = group_tracklets(mars_dir, "query")

    if not gallery_tracklets or not query_tracklets:
        print(f"\n❌ Error: Could not find MARS tracklets at '{mars_dir}'.")
        sys.exit(1)

    print(f"-> Gallery tracklets (bbox_test): {len(gallery_tracklets)}")
    print(f"-> Query tracklets (query)      : {len(query_tracklets)}")

    # 1. Run Base Paper Gait Model
    print("\n" + "=" * 60)
    print(" [1/2] RUNNING BASE PAPER GAIT MODEL (GEI + PCA)")
    print("=" * 60)
    res_paper = run_model_evaluation("paper", gallery_tracklets, query_tracklets)

    # 2. Run Your SkeletonGait Model
    print("\n" + "=" * 60)
    print(" [2/2] RUNNING YOUR SKELETONGAIT MODEL (YOLOv8-POSE + 3D CNN)")
    print("=" * 60)
    res_ours = run_model_evaluation("ours", gallery_tracklets, query_tracklets)

    # 3. Print Side-by-Side Comparison
    if res_paper and res_ours:
        print("\n" + "=" * 70)
        print(" 📊 HEAD-TO-HEAD GAIT COMPARISON: BASE PAPER (GEI) VS YOUR SKELETONGAIT")
        print("=" * 70)
        print(f"{'Metric':<22} | {'Base Paper (GEI)':<18} | {'Your SkeletonGait':<18} | {'Improvement':<12}")
        print("-" * 70)
        for metric_key, label in [("rank1", "Rank-1 Accuracy"), ("rank5", "Rank-5 Accuracy"), ("rank10", "Rank-10 Accuracy"), ("mAP", "mAP")]:
            p_val = res_paper.get(metric_key, 0.0)
            o_val = res_ours.get(metric_key, 0.0)
            diff = o_val - p_val
            sign = "+" if diff >= 0 else ""
            print(f"{label:<22} | {p_val:>16.2f}% | {o_val:>16.2f}% | {sign}{diff:>10.2f}%")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
