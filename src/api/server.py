import asyncio
import os
import sys
import uuid
import cv2

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional
import uvicorn

# Add the project root to sys.path to allow running the script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.api.schemas import ImageUploadResponse, VideoUploadResponse, TargetExtractionRequest, TargetExtractionResponse
from src.api.retrieval_service import RetrievalService

app = FastAPI(title="Retrieval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = RetrievalService()

TEMP_DIR = "data/temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


@app.post("/upload/image", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    # Run synchronous extraction in a thread pool so it doesn't block the event loop
    loop = asyncio.get_running_loop()
    features, matches = await loop.run_in_executor(
        None, service.process_image_query, contents
    )
    return ImageUploadResponse(
        message="Image processed",
        extracted_features=features,
        matches=matches
    )


@app.post("/upload/video", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    # Clear old data (prevents RAM/disk leaks between demo runs)
    service.clear_cache()
    for f in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, f)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: could not delete temp file {file_path}: {e}")

    session_id = str(uuid.uuid4())
    video_path = os.path.join(TEMP_DIR, f"{session_id}.mp4")

    # Write uploaded file to disk using async read so we don't block the event loop
    contents = await file.read()
    with open(video_path, "wb") as buffer:
        buffer.write(contents)

    # Run CPU-heavy first pass in a thread pool executor so the event loop stays responsive
    loop = asyncio.get_running_loop()
    thumbnails = await loop.run_in_executor(
        None, service.process_video_first_pass, video_path, session_id
    )

    return VideoUploadResponse(
        message="Video processed for thumbnails",
        session_id=session_id,
        thumbnails=thumbnails
    )


@app.post("/extract/video/target", response_model=TargetExtractionResponse)
async def extract_target(request: TargetExtractionRequest):
    video_path = os.path.join(TEMP_DIR, f"{request.session_id}.mp4")

    if not os.path.exists(video_path):
        return TargetExtractionResponse(message="Session expired or video not found", matches=[])

    # Run CPU-heavy second pass in a thread pool executor
    loop = asyncio.get_running_loop()
    matches = await loop.run_in_executor(
        None, service.process_video_target, video_path, request.session_id, request.track_id
    )

    return TargetExtractionResponse(
        message=f"Target {request.track_id} extracted and searched",
        matches=matches
    )


# Global dictionary to track where a client paused
CLIENT_SESSIONS = {}

async def generate_evidence_stream(video_path: str, boxes: dict, client_id: str = None, cam_id: str = "Camera_1"):
    """
    Async MJPEG generator for the evidence playback endpoint.
    Overlays CCTV time (starting from 08:00:00 AM), camera ID, and tracking bounding boxes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_indices = sorted([int(k) for k in boxes.keys()])
    if not frame_indices:
        cap.release()
        return

    # Start 1 second before, end 1 second after
    start_frame = max(0, frame_indices[0] - int(fps))
    end_frame   = frame_indices[-1] + int(fps)

    # If this client is resuming from a pause, pick up exactly where they left off
    if client_id and client_id in CLIENT_SESSIONS:
        start_frame = CLIENT_SESSIONS[client_id]

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    last_box      = None

    try:
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Save progress before yielding, so if client disconnects (pauses), 
            # we know exactly where to resume
            if client_id:
                CLIENT_SESSIONS[client_id] = current_frame

            # Resize to 640×480 to match the resolution used during ingestion
            frame = cv2.resize(frame, (640, 480))

            str_idx = str(current_frame)
            if str_idx in boxes:
                last_box = boxes[str_idx]

            # ── 1. Calculate CCTV timestamp starting from 08:00:00 AM ──
            time_sec = current_frame / fps
            total_sec = 8 * 3600 + time_sec
            cur_h = int((total_sec // 3600) % 24)
            cur_m = int((total_sec % 3600) // 60)
            cur_s = int(total_sec % 60)
            cur_ms = int((total_sec % 1) * 100)
            time_display = f"{cur_h:02d}:{cur_m:02d}:{cur_s:02d}.{cur_ms:02d}"

            # ── 2. Draw target bounding box and label ──
            if last_box:
                x1, y1, x2, y2 = [int(v) for v in last_box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = "TARGET IDENTIFIED"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                badge_y1 = max(0, y1 - th - 6)
                badge_y2 = y1
                cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 8, badge_y2), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1 + 4, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # ── 3. Draw CCTV OSD Header Banner ──
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 32), (15, 23, 42), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # Red recording indicator dot
            cv2.circle(frame, (16, 16), 4, (0, 0, 255), -1)
            osd_text = f"REC  |  CAM: {cam_id}  |  TIME: {time_display}"
            cv2.putText(frame, osd_text, (28, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            current_frame += 1
            # asyncio.sleep instead of time.sleep — does NOT block the event loop
            await asyncio.sleep(1 / fps)

    finally:
        # Always release the capture, even if the client disconnects mid-stream
        cap.release()


@app.get("/evidence/{uuid}")
async def stream_evidence(uuid: str, client_id: Optional[str] = None):
    point = service.db.get_by_id(uuid)
    if not point:
        raise HTTPException(status_code=404, detail="Match not found")

    metadata   = point.payload or {}
    video_path = metadata.get("source_video_path")
    boxes      = metadata.get("bounding_boxes", {})
    cam_id     = metadata.get("camera_id", "Camera_1")

    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Source video not found on disk")

    return StreamingResponse(
        generate_evidence_stream(video_path, boxes, client_id, cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/health")
async def health():
    return {"status": "ok", "base_time": "08:00:00"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
