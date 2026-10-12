import os
import re
import numpy as np
import torch
import torch.nn.functional as F


def parse_market1501_filename(filename: str):
    """
    Parses Market-1501 filename pattern: [pid]_c[camid]s[seq]_[frame]_[idx].jpg
    Returns: (pid, camid) or (None, None) if invalid junk file (e.g., ._).
    Note: Distractor IDs (-1, 0000) are retained for gallery distractor filtering.
    """
    base = os.path.basename(filename)
    if base.startswith("._"):
        return None, None
    pattern = re.compile(r"([-\d]+)_c(\d)", re.IGNORECASE)
    match = pattern.search(base)
    if match:
        pid = int(match.group(1))
        camid = int(match.group(2))
        return pid, camid
    return None, None


def parse_mars_filename(filename: str):
    """
    Parses MARS filename patterns:
      1) 0001C1T0001F001.jpg -> pid=1, camid=1, tid=1
      2) 0001_c1t0001_0001.jpg -> pid=1, camid=1, tid=1
    Returns: (pid, camid, tid) or (None, None, None) if junk.
    """
    base = os.path.basename(filename)
    if base.startswith("._") or base.startswith("-1") or base.startswith("0000"):
        return None, None, None

    # Format 1: 0001_c1t0001...
    if "_" in base:
        parts = base.split("_")
        if len(parts) >= 2:
            try:
                pid = int(parts[0])
                cid_str = parts[1]
                cid_match = re.search(r"c(\d+)", cid_str.lower())
                tid_match = re.search(r"t(\d+)", cid_str.lower())
                cid = int(cid_match.group(1)) if cid_match else 0
                tid = int(tid_match.group(1)) if tid_match else 0
                return pid, cid, tid
            except Exception:
                pass

    # Format 2: 0001C1T0001F001
    match = re.search(r"^([a-zA-Z0-9\-]+)C(\d)T(\d+)", base, re.IGNORECASE)
    if match:
        try:
            pid = int(match.group(1))
        except ValueError:
            pid = match.group(1)
        cid = int(match.group(2))
        tid = int(match.group(3))
        return pid, cid, tid

    return None, None, None


def compute_reid_metrics(
    q_feats: np.ndarray,
    g_feats: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_cams: np.ndarray,
    g_cams: np.ndarray,
    max_rank: int = 50,
):
    """
    Computes CMC (Rank-1, 5, 10, 20) and mAP following standard Re-ID benchmark rules:
    - Exclude gallery images from the same camera with the same person ID (camera-filtering junk).
    - Exclude distractor IDs (-1, 0).
    """
    q_tensor = torch.from_numpy(q_feats).float()
    g_tensor = torch.from_numpy(g_feats).float()

    # L2 normalize just to ensure unit sphere for cosine distance
    q_tensor = F.normalize(q_tensor, p=2, dim=1)
    g_tensor = F.normalize(g_tensor, p=2, dim=1)

    # Cosine distance = 1 - cosine similarity = 2 - 2 * (q @ g.T) / 2
    # Efficient matrix multiplication
    sim_mat = torch.mm(q_tensor, g_tensor.t()).numpy()
    # Distance matrix: lower is closer
    dist_mat = 1.0 - sim_mat

    num_q = len(q_pids)
    cmc_scores = np.zeros(max_rank)
    ap_scores = []
    valid_queries = 0

    for i in range(num_q):
        q_pid = q_pids[i]
        q_cam = q_cams[i]

        # Ignore distractor queries
        if q_pid == -1 or q_pid == 0:
            continue

        q_dist = dist_mat[i]

        sorted_indices = np.argsort(q_dist)
        sorted_pids = g_pids[sorted_indices]
        sorted_cams = g_cams[sorted_indices]

        # Standard Re-ID evaluation: junk if same person in same camera, or distractor IDs (-1, 0)
        junk_mask = ((sorted_pids == q_pid) & (sorted_cams == q_cam)) | (sorted_pids == -1) | (sorted_pids == 0)
        keep_mask = ~junk_mask

        matches = (sorted_pids[keep_mask] == q_pid).astype(np.int32)
        if len(matches) == 0:
            continue

        num_rel = matches.sum()
        if num_rel == 0:
            # Query person does not exist in any other camera in the gallery
            continue

        valid_queries += 1

        # CMC
        raw_cmc = matches.cumsum()
        raw_cmc[raw_cmc > 1] = 1
        cmc_padded = (
            np.ones(max_rank) * raw_cmc[-1]
            if len(raw_cmc) > 0
            else np.zeros(max_rank)
        )
        cmc_padded[: min(len(raw_cmc), max_rank)] = raw_cmc[:max_rank]
        cmc_scores += cmc_padded

        # AP (Average Precision)
        cum_matches = matches.cumsum()
        precisions = [cum_matches[k] / (k + 1.0) for k in range(len(matches))]
        ap = np.sum(np.array(precisions) * matches) / num_rel
        ap_scores.append(ap)

    if valid_queries > 0:
        cmc_scores /= valid_queries
        mean_ap = np.mean(ap_scores) * 100.0
    else:
        mean_ap = 0.0

    return {
        "valid_queries": valid_queries,
        "mAP": mean_ap,
        "rank1": cmc_scores[0] * 100.0 if len(cmc_scores) > 0 else 0.0,
        "rank5": cmc_scores[4] * 100.0 if len(cmc_scores) > 4 else 0.0,
        "rank10": cmc_scores[9] * 100.0 if len(cmc_scores) > 9 else 0.0,
        "rank20": cmc_scores[19] * 100.0 if len(cmc_scores) > 19 else 0.0,
    }


def print_metrics_table(title: str, results: dict):
    """Pretty prints evaluation metrics to console."""
    print("\n" + "=" * 55)
    print(f" 🏆 {title} 🏆")
    print("=" * 55)
    print(f"{'Metric':<25} | {'Score (%)':<15}")
    print("-" * 55)
    print(f"{'Valid Queries Evaluated':<25} | {results.get('valid_queries', 0)}")
    print(f"{'Rank-1 Accuracy':<25} | {results.get('rank1', 0.0):.2f}%")
    print(f"{'Rank-5 Accuracy':<25} | {results.get('rank5', 0.0):.2f}%")
    print(f"{'Rank-10 Accuracy':<25} | {results.get('rank10', 0.0):.2f}%")
    if "rank20" in results:
        print(f"{'Rank-20 Accuracy':<25} | {results.get('rank20', 0.0):.2f}%")
    print(f"{'mAP (Mean Average Prec.)':<25} | {results.get('mAP', 0.0):.2f}%")
    print("=" * 55 + "\n")
