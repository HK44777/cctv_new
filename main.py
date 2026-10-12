import os
import cv2
import threading
import queue
from src.core.video_processor import VideoProcessor

def format_cctv_time(seconds_offset: float, base_hour: int = 8) -> str:
    """Formats an offset in seconds from base_hour (08:00:00 AM) to HH:MM:SS."""
    total_sec = base_hour * 3600 + max(0.0, float(seconds_offset or 0.0))
    h = int((total_sec // 3600) % 24)
    m = int((total_sec % 3600) // 60)
    s = int(total_sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def frame_producer(vid_file_path, cam_id, frame_queue):
    cap = cv2.VideoCapture(vid_file_path)
    if not cap.isOpened():
        print(f"Error opening video {vid_file_path}")
        frame_queue.put(("DONE", None, None, None, None))
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30 # Default to 30 if unable to read FPS
        
    target_fps = 6 # Force exactly 6 frames per second to the queue (1 every 6 frames = 1 second)
    frame_interval = max(1, int(round(fps / target_fps)))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Video starts recording at 08:00:00 AM; timestamp is elapsed seconds
            timestamp = frame_count / fps
            # Resize frame to save RAM and speed up CPU processing
            small_frame = cv2.resize(frame, (640, 480))
            
            # Put frame into queue as a plain tuple
            frame_queue.put((cam_id, small_frame, vid_file_path, frame_count, timestamp))
            
            # Periodic logging showing realistic 8:00-based CCTV time
            if frame_count % (target_fps * 30) == 0:
                print(f"[{cam_id}] Reading frame {frame_count} | CCTV Time: {format_cctv_time(timestamp)}")
            
        frame_count += 1
        
    cap.release()
    print(f"Finished reading from {cam_id}")
    frame_queue.put(("DONE", None, None, None, None))

def frame_consumer(frame_queue, processor, total_producers):
    finished_producers = 0
    while True:
        cam_id, frame, vid_file_path, frame_count, timestamp = frame_queue.get()
        
        if cam_id == "DONE":
            finished_producers += 1
            if finished_producers == total_producers:
                break
            continue
            
        processor.process_frame(frame, cam_id, timestamp, vid_file_path, frame_count)

def main():
    print("=" * 65)
    print("Starting Offline Ingestion Phase (All CCTV Videos start: 08:00:00 AM)")
    print("=" * 65)
    
    video_dir = "data/videos"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
        print(f"Please place your video files in '{video_dir}' and run again.")
        return

    videos = [
        ("cam1.mp4", "Camera_1"),
        ("cam2.mp4", "Camera_2"),
        ("cam3.mp4", "Camera_3")
    ]
    
    processor = VideoProcessor()
    
    # Normal FIFO Queue — per-camera ordering is guaranteed by each producer thread
    # Maxsize to prevent loading all frames into memory at once
    frame_queue = queue.Queue(maxsize=150)
    
    valid_videos = []
    for vid_file, cam_id in videos:
        path = os.path.join(video_dir, vid_file)
        if os.path.exists(path):
            valid_videos.append((path, cam_id))
        else:
            print(f"Skipping {vid_file} - Not found in {video_dir}")
            
    if not valid_videos:
        print("No valid videos found.")
        return
        
    # Start a single worker (consumer) thread to process frames from the queue
    consumer_thread = threading.Thread(
        target=frame_consumer, 
        args=(frame_queue, processor, len(valid_videos))
    )
    consumer_thread.start()
    
    # Start producer threads (one for each camera)
    producer_threads = []
    for path, cam_id in valid_videos:
        t = threading.Thread(
            target=frame_producer,
            args=(path, cam_id, frame_queue)
        )
        t.start()
        producer_threads.append(t)
        
    # Wait for all producers to finish reading
    for t in producer_threads:
        t.join()
        
    # Wait for the consumer to finish processing all frames
    consumer_thread.join()
    
    # Force close any remaining active tracks
    processor.finalize_all()
    
    print("Ingestion Complete. All tracks stored in Qdrant.")

if __name__ == "__main__":
    main()
