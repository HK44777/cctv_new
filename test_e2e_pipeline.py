import os
import sys
import cv2
import base64
import time

from src.api.retrieval_service import RetrievalService

class MockDatabaseClient:
    """Mock DatabaseClient to allow testing retrieval without competing for the local Qdrant file lock."""
    def search_ensemble(self, query_appearance=None, query_face=None, query_gait=None, limit=20):
        return {"appearance": [], "face": [], "gait": []}
    def get_by_id(self, uuid):
        return None

def test_pipeline():
    print("=" * 70)
    print("STARTING END-TO-END RETRIEVAL PIPELINE TEST")
    print("=" * 70)
    
    import glob
    candidates = glob.glob("data/temp_uploads/*.mp4") + ["data/new_videos/cam_1.mp4"]
    video_path = candidates[0] if candidates else None
    if not video_path or not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return False
    print(f"Testing with video: {video_path}")

    service = RetrievalService(db_client=MockDatabaseClient())
    test_session_id = "test_e2e_session_001"
    
    # ── First Pass: Video Processing & Thumbnails ──
    print("\n--- Pass 1: process_video_first_pass ---")
    t0 = time.time()
    thumbnails = service.process_video_first_pass(video_path, test_session_id)
    t_pass1 = time.time() - t0
    
    print(f"First pass completed in {t_pass1:.2f} seconds.")
    print(f"Total candidate person thumbnails returned: {len(thumbnails)}")
    
    assert len(thumbnails) > 0, "No thumbnails returned from first pass!"
    for i, thumb in enumerate(thumbnails):
        assert thumb.track_id, f"Thumbnail {i} missing track_id!"
        assert len(thumb.thumbnail_base64) > 100, f"Thumbnail {i} has empty base64 data!"
        print(f"  Person Thumbnail #{thumb.track_id}: Base64 size = {len(thumb.thumbnail_base64)} bytes")

    # ── Second Pass: Target Selection & Feature Search ──
    target_tid = thumbnails[0].track_id
    print(f"\n--- Pass 2: process_video_target (Target Person #{target_tid}) ---")
    t0 = time.time()
    results = service.process_video_target(video_path, test_session_id, target_tid)
    t_pass2 = time.time() - t0
    
    print(f"Second pass completed in {t_pass2:.2f} seconds.")
    print(f"Match results returned: {len(results)}")

    print("\n" + "=" * 70)
    print("✅ ALL END-TO-END PIPELINE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
