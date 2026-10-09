from collections import defaultdict


class ScoreFusion:
    def __init__(self):
        # Baseline weights based on the research paper
        self.W_FACE = 0.50
        self.W_APP = 0.35
        self.W_GAIT = 0.15

    def format_results(self, search_results, min_threshold=0.75):
        """
        Formats the results from the Ensemble Search using Late Fusion
        and Dynamic Weighting.
        """
        # Group scores by user ID
        scores_by_uuid = defaultdict(dict)
        metadata_by_uuid = {}
        
        for modality in ["appearance", "face", "gait"]:
            for scored_point in search_results.get(modality, []):
                uid = scored_point.id
                scores_by_uuid[uid][modality] = scored_point.score
                if uid not in metadata_by_uuid:
                    metadata_by_uuid[uid] = scored_point.payload
                    
        final_results = []
        
        for uid, scores in scores_by_uuid.items():
            has_face = "face" in scores
            has_app = "appearance" in scores
            has_gait = "gait" in scores
            
            face_score = scores.get("face", 0.0)
            app_score = scores.get("appearance", 0.0)
            gait_score = scores.get("gait", 0.0)
            
            final_score = 0.0
            
            if has_face and has_app and has_gait:
                final_score = (face_score * self.W_FACE) + (app_score * self.W_APP) + (gait_score * self.W_GAIT)
            elif has_app and has_gait and not has_face:
                total_w = self.W_APP + self.W_GAIT
                final_score = (app_score * (self.W_APP/total_w)) + (gait_score * (self.W_GAIT/total_w))
            elif has_face and has_app and not has_gait:
                total_w = self.W_FACE + self.W_APP
                final_score = (face_score * (self.W_FACE/total_w)) + (app_score * (self.W_APP/total_w))
            elif has_face and not has_app and not has_gait:
                final_score = face_score
            elif has_app and not has_face and not has_gait:
                # Appearance-only penalty (e.g., 0.85 multiplier) to prevent confusers
                final_score = app_score * 0.85
            else:
                # Fallback for any other combination
                total_w = sum([self.W_FACE if has_face else 0, 
                               self.W_APP if has_app else 0, 
                               self.W_GAIT if has_gait else 0])
                if total_w > 0:
                    final_score = (face_score * (self.W_FACE/total_w) if has_face else 0) + \
                                  (app_score * (self.W_APP/total_w) if has_app else 0) + \
                                  (gait_score * (self.W_GAIT/total_w) if has_gait else 0)

            if final_score >= min_threshold:
                final_results.append({
                    "uuid":       uid,
                    "confidence": final_score,
                    "metadata":   metadata_by_uuid[uid],
                    "breakdown":  scores
                })
                
        # Sort by confidence (descending) to ensure top matches are returned first
        final_results.sort(key=lambda x: x["confidence"], reverse=True)
        return final_results
