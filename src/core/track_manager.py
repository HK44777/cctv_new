import uuid
import numpy as np

class TrackManager:
    def __init__(self, db_client):
        # Dictionary to hold RAM buffers for active tracks
        # Format: {track_id: {"appearances": [], "faces": [], "skeletons": [], "gaits": [], "metadata": {}}}
        self.active_tracks = {}
        self.db_client = db_client
        print("Initialized Track Manager")

    @staticmethod
    def _update_running_average(dict_ref, new_vec):
        """Helper to compute running average of feature vectors."""
        if new_vec is None:
            return
            
        if dict_ref["count"] == 0:
            dict_ref["mean"] = np.array(new_vec)
        else:
            n = dict_ref["count"]
            dict_ref["mean"] = (dict_ref["mean"] * n + np.array(new_vec)) / (n + 1)
        dict_ref["count"] += 1

    @staticmethod
    def _l2_norm_array(arr):
        """Helper to L2 normalize a numpy array for cosine similarity."""
        if arr is None: 
            return None
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def register_frame_data(self, track_id, appearance, face, skeleton, metadata, video_path=None, frame_idx=None, bbox=None):
        """
        Step 5, 6, 7: Append per-frame data to RAM buffers (Optimized to Running Average)
        """
        if track_id not in self.active_tracks:
            meta = metadata.copy()
            if video_path:
                meta["source_video_path"] = video_path
            meta["bounding_boxes"] = {}
            self.active_tracks[track_id] = {
                "appearances": {"mean": None, "count": 0},
                "faces": {"mean": None, "count": 0},
                "skeletons": [],   # Max 30
                "gaits": {"mean": None, "count": 0},
                "metadata": meta # start_time, camera_id
            }
            
        track = self.active_tracks[track_id]
        
        # Safely update running averages (handles None automatically)
        self._update_running_average(track["appearances"], appearance)
        self._update_running_average(track["faces"], face)
            
        if skeleton is not None:
            track["skeletons"].append(skeleton)
            
        # Update end time
        track["metadata"]["end_time"] = metadata.get("timestamp")
        
        if frame_idx is not None and bbox is not None:
            track["metadata"]["bounding_boxes"][str(frame_idx)] = bbox

    def process_gait_buffer(self, track_id, gait_extractor):
        """
        Triggered when silhouette buffer reaches 30 frames
        """
        if track_id not in self.active_tracks:
            return
            
        track = self.active_tracks[track_id]
        if len(track["skeletons"]) >= 30:
            # Send 30 skeleton maps to SkeletonGait
            gait_vector = gait_extractor.extract_gait_vector(track["skeletons"][:30])
            self._update_running_average(track["gaits"], gait_vector)

            # Drain the oldest 15 frames to create a sliding window
            track["skeletons"] = track["skeletons"][15:]

    def cleanup_stale_tracks(self, current_time, gait_extractor=None, timeout=10.0):
        """
        Check for tracks that haven't been seen in `timeout` seconds and close them.
        """
        # We must use list() to avoid RuntimeError: dictionary changed size during iteration
        for track_id in list(self.active_tracks.keys()):
            track = self.active_tracks[track_id]
            end_time = track["metadata"].get("end_time")
            
            # If the track hasn't been updated recently, close it and push to Qdrant
            if end_time is not None and (current_time - end_time) > timeout:
                self.close_track(track_id, gait_extractor)

    def close_track(self, track_id, gait_extractor=None):
        """
        Step 8: Track Termination
        Step 9: Vector Aggregation & Normalization (Using Running Average)
        Step 10: Global Identity Assignment
        Step 11: Database Insertion
        """
        if track_id not in self.active_tracks:
            return
            
        track = self.active_tracks[track_id]
        
        # Squeeze out final gait if they walked out of frame quickly (or had <30 frames left in buffer)
        if gait_extractor and len(track["skeletons"]) >= 15:
            gait_vector = gait_extractor.extract_gait_vector(track["skeletons"])
            self._update_running_average(track["gaits"], gait_vector)
        
        # Extract the running averages and L2 normalize
        final_appearance = self._l2_norm_array(track["appearances"]["mean"]) if track["appearances"]["count"] > 0 else None
        final_face = self._l2_norm_array(track["faces"]["mean"]) if track["faces"]["count"] > 0 else None
        final_gait = self._l2_norm_array(track["gaits"]["mean"]) if track["gaits"]["count"] > 0 else None
        
        # We only save if we have at least one vector
        if final_appearance or final_face or final_gait:
            # Step 10: Generate UUID
            global_id = str(uuid.uuid4())
            
            # Step 11: Insert to Qdrant
            self.db_client.insert(
                uuid=global_id,
                appearance=final_appearance,
                face=final_face,
                gait=final_gait,
                metadata=track["metadata"]
            )
            print(f"Track closed and saved to DB. TrackID: {track_id} -> UUID: {global_id}")
            
        # Cleanup RAM
        del self.active_tracks[track_id]

    def close_all_tracks(self, gait_extractor=None):
        """Force close all tracks at end of video. Passes gait_extractor to squeeze final buffers."""
        track_ids = list(self.active_tracks.keys())
        for tid in track_ids:
            self.close_track(tid, gait_extractor=gait_extractor)
