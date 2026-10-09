from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ThumbnailResponse(BaseModel):
    track_id: str
    thumbnail_base64: str

class VideoUploadResponse(BaseModel):
    message: str
    session_id: str
    thumbnails: List[ThumbnailResponse]

class TargetExtractionRequest(BaseModel):
    session_id: str
    track_id: str

class MatchResult(BaseModel):
    target_id: str
    score: float
    metadata: Dict[str, Any]

class TargetExtractionResponse(BaseModel):
    message: str
    matches: List[MatchResult]

class ImageUploadResponse(BaseModel):
    message: str
    extracted_features: List[str]
    matches: List[MatchResult]
