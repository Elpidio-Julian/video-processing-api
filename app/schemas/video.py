from pydantic import BaseModel, HttpUrl
from typing import Optional
from datetime import datetime

class VideoProcessRequest(BaseModel):
    source_url: str

class VideoProcessResponse(BaseModel):
    job_id: str
    status: str = "queued"
    estimated_time: int

class VideoStatusResponse(BaseModel):
    status: str
    progress: float
    processed_url: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    user_id: str

class VideoMetadata(BaseModel):
    filename: str
    original_size: int
    mime_type: str
    duration: Optional[float]
    timestamp: datetime
    user_id: str

class ProcessingOptions(BaseModel):
    target_aspect_ratio: str = "9:16"
    background_blur: bool = False
    background_color: Optional[str] = None
    quality: str = "high"

class User(BaseModel):
    uid: str
    email: Optional[str] = None
    email_verified: bool = False 