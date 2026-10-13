# from collections import defaultdict


# class ScoreFusion:
#     def __init__(self):
#         # Baseline weights based on the research paper
#         self.W_FACE = 0.50
#         self.W_APP = 0.35
#         self.W_GAIT = 0.15

#     def format_results(self, search_results, min_threshold=0.55, query_has_face=False):
#         """
#         Formats results using Late Fusion with Biometric Guarding:
#         - When a query person has a verified face, clothing-only matches (no face in DB)
#           cannot be treated as 100% ground-truth and have their confidence scaled realistically.
#         - Prioritizes true multi-modal matches (Face + Appearance) over mere color matches.
#         """
#         scores_by_uuid = defaultdict(dict)
#         metadata_by_uuid = {}
        
#         for modality in ["appearance", "face", "gait"]:
#             for scored_point in search_results.get(modality, []):
#                 uid = scored_point.id
#                 scores_by_uuid[uid][modality] = scored_point.score
#                 if uid not in metadata_by_uuid:
#                     metadata_by_uuid[uid] = scored_point.payload
                    
#         final_results = []
        
#         for uid, scores in scores_by_uuid.items():
#             has_face = "face" in scores
#             has_app  = "appearance" in scores
#             has_gait = "gait" in scores
            
#             face_score = scores.get("face", 0.0)
#             app_score  = scores.get("appearance", 0.0)
#             gait_score = scores.get("gait", 0.0)
            
#             final_score = 0.0
            
#             if has_face and has_app and has_gait:
#                 # Full Multi-Modal Trifecta
#                 final_score = (face_score * self.W_FACE) + (app_score * self.W_APP) + (gait_score * self.W_GAIT)
#             elif has_face and has_app and not has_gait:
#                 # Strong Biometric Pair: Face + Clothing
#                 # If face actively contradicts query (face_score < 0.25), penalize imposter
#                 if face_score < 0.25:
#                     final_score = (face_score * 0.5) + (app_score * 0.5) * 0.70
#                 else:
#                     total_w = self.W_FACE + self.W_APP
#                     final_score = (face_score * (self.W_FACE / total_w)) + (app_score * (self.W_APP / total_w))
#             elif has_app and has_gait and not has_face:
#                 # Motion + Appearance
#                 total_w = self.W_APP + self.W_GAIT
#                 final_score = (app_score * (self.W_APP / total_w)) + (gait_score * (self.W_GAIT / total_w))
#             elif has_face and not has_app and not has_gait:
#                 # Pure Facial Match
#                 final_score = face_score
#             elif has_app and not has_face and not has_gait:
#                 # Appearance only: In real CCTV, distant/overhead cameras rarely capture frontal faces.
#                 # Passive Guarding: Do not penalize candidates simply because camera pitch/distance precluded facial detection.
#                 final_score = app_score
#             else:
#                 total_w = sum([self.W_FACE if has_face else 0, 
#                                self.W_APP if has_app else 0, 
#                                self.W_GAIT if has_gait else 0])
#                 if total_w > 0:
#                     final_score = (face_score * (self.W_FACE / total_w) if has_face else 0) + \
#                                   (app_score * (self.W_APP / total_w) if has_app else 0) + \
#                                   (gait_score * (self.W_GAIT / total_w) if has_gait else 0)

#             if final_score >= min_threshold:
#                 final_results.append({
#                     "uuid":       uid,
#                     "confidence": final_score,
#                     "metadata":   metadata_by_uuid[uid],
#                     "breakdown":  scores
#                 })
                
#         # Sort by confidence (descending) to ensure top matches are returned first
#         final_results.sort(key=lambda x: x["confidence"], reverse=True)
#         return final_results



"""
fusion.py - tiered score fusion with face veto and true per-modality rescoring.

Key differences from the weighted-average version:
  1. Candidates are rescored against their STORED vectors, so "face missing" means
     the candidate really has no face vector, not "it fell outside the top-50 face search".
  2. Face is a veto: a clearly different face rejects the candidate whatever the clothes say.
  3. Appearance alone can never produce a "confirmed" result, only "possible".
  4. Gait is ignored unless gait_trusted=True (validate it first).
"""
from collections import defaultdict

import numpy as np

MODALITIES = ("appearance", "face", "gait")
TIER_RANK = {"confirmed": 0, "possible": 1}


def _cos(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float(a @ b / (na * nb))


class ScoreFusion:
    def __init__(
        self,
        client=None,
        collection="persons",
        face_veto=0.25,        # face below this => reject
        face_strong=0.45,      # face at/above this => confirmed
        app_only_min=0.50,     # appearance-only calibrated threshold
        grey_app_min=0.50,     # grey-zone face needs appearance at least this
        grey_score_min=0.50,
        margin=0.035,          # top-1 vs top-2 gap below this => ambiguous
        gait_trusted=True,     # incorporate gait when available
        camera_key="camera_id",
    ):
        if client is None:
            try:
                from src.database.qdrant_client import DatabaseClient
                self._db = DatabaseClient()
                self.client = self._db.client
            except Exception:
                self.client = None
        else:
            self.client = client

        self.collection = collection
        self.face_veto = face_veto
        self.face_strong = face_strong
        self.app_only_min = app_only_min
        self.grey_app_min = grey_app_min
        self.grey_score_min = grey_score_min
        self.margin = margin
        self.gait_trusted = gait_trusted
        self.camera_key = camera_key

    # ------------------------------------------------------------------ rescoring
    def _rescore(self, uids, query_vectors):
        """Cosine of the query vs each candidate's stored vectors."""
        if not self.client:
            return {}
        try:
            records = self.client.retrieve(
                collection_name=self.collection,
                ids=list(uids),
                with_payload=True,
                with_vectors=True,
            )
        except Exception:
            return {}

        out = {}
        for r in records:
            vecs = r.vector if isinstance(r.vector, dict) else {}
            scores = {}
            for m in MODALITIES:
                q, v = query_vectors.get(m), vecs.get(m)
                if q is not None and v is not None:
                    s = _cos(q, v)
                    if s is not None:
                        scores[m] = s
            out[str(r.id)] = (scores, r.payload)
        return out

    # ------------------------------------------------------------------ fusion rule
    def fuse(self, s_face, s_app, s_gait):
        """Returns (score, tier, branch); tier in {'confirmed','possible','reject'}."""
        if s_face is not None and s_app is not None:
            if s_face < self.face_veto:
                return 0.5 * s_face, "reject", "face_veto"
            if s_face >= self.face_strong:
                return 0.7 * s_face + 0.3 * s_app, "confirmed", "face_strong+app"
            score = 0.6 * s_face + 0.4 * s_app
            ok = s_app >= self.grey_app_min and score >= self.grey_score_min
            return score, ("confirmed" if ok else "possible"), "face_grey+app"

        if s_face is not None:
            if s_face < self.face_veto:
                return s_face, "reject", "face_only_veto"
            tier = "confirmed" if s_face >= self.face_strong else "possible"
            return s_face, tier, "face_only"

        if s_app is not None:
            score, branch = s_app, "app_only"
            if s_gait is not None and self.gait_trusted:
                score, branch = 0.8 * s_app + 0.2 * s_gait, "app+gait"
            tier = "possible" if score >= self.app_only_min else "reject"
            return score, tier, branch

        return 0.0, "reject", "no_evidence"

    # ------------------------------------------------------------------ public API
    def format_results(self, search_results, query_vectors=None, min_threshold=0.50, query_has_face=False, include_possible=True):
        """
        Accepts either:
          format_results(search_results, query_vectors={...})
          or
          format_results(search_results, min_threshold=0.50, query_has_face=True)
        """
        if isinstance(query_vectors, (float, int)):
            min_threshold = float(query_vectors)
            query_vectors = None

        payload_fallback = {}
        for m in MODALITIES:
            for p in search_results.get(m, []):
                payload_fallback.setdefault(str(p.id), p.payload)
        uids = list(payload_fallback.keys())
        if not uids:
            return []

        if query_vectors and isinstance(query_vectors, dict) and self.client:
            rescored = self._rescore(uids, query_vectors)
        else:
            # Extract scores directly from search_results
            rescored = {}
            for m in MODALITIES:
                for p in search_results.get(m, []):
                    uid = str(p.id)
                    if uid not in rescored:
                        rescored[uid] = ({}, p.payload)
                    rescored[uid][0][m] = p.score

        results = []
        for uid, (scores, payload) in rescored.items():
            score, tier, branch = self.fuse(
                scores.get("face"), scores.get("appearance"), scores.get("gait")
            )
            if score < min_threshold or tier == "reject" or (tier == "possible" and not include_possible):
                continue
            results.append(
                {
                    "uuid": uid,
                    "confidence": score,
                    "tier": tier,
                    "branch": branch,
                    "ambiguous": False,
                    "metadata": payload or payload_fallback.get(uid, {}),
                    "breakdown": scores,
                }
            )

        # Margin check: lookalike clusters on one camera are flagged, never "confirmed".
        by_cam = defaultdict(list)
        for r in results:
            by_cam[(r["metadata"] or {}).get(self.camera_key)].append(r)
        for group in by_cam.values():
            group.sort(key=lambda r: r["confidence"], reverse=True)
            if len(group) > 1 and group[0]["tier"] == "possible":
                if group[0]["confidence"] - group[1]["confidence"] < self.margin:
                    for r in group[:2]:
                        r["ambiguous"] = True

        results.sort(key=lambda r: (TIER_RANK[r["tier"]], -r["confidence"]))
        return results
