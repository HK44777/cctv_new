"""
run_vector_tool.py
------------------
Standalone launcher for the OSNet Vector Studio web application.
Performs image extraction and Base64 encoding. Does NOT touch resources.json on disk.

Usage:
    python run_vector_tool.py
    Then open: http://localhost:8055
"""

import os
import sys
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Ensure root path is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.api.vector_tool_service import VectorToolService

app = FastAPI(title="OSNet Vector Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = VectorToolService()
HTML_PATH = os.path.join(os.path.dirname(__file__), "src", "api", "static", "vector_tool.html")


@app.get("/", response_class=HTMLResponse)
@app.get("/vector-tool", response_class=HTMLResponse)
async def index():
    if os.path.exists(HTML_PATH):
        with open(HTML_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>UI file not found.</h1>", status_code=404)


@app.post("/tools/extract-vector")
async def extract_vector(file: UploadFile = File(...), auto_crop: bool = True):
    contents = await file.read()
    try:
        return service.extract_from_image(contents, auto_crop=auto_crop)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    port = 8055
    print("\n" + "=" * 65)
    print("   OSNet Vector Studio (Extractor & Base64 Encoder)")
    print(f"   Opening Web Application at: http://localhost:{port}")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=port)
