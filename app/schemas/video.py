from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime
from enum import Enum

class HUDLocation(str, Enum):
    BOTTOM_RIGHT = "bottom_right"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    TOP_LEFT = "top_left"

class VideoProcessRequest(BaseModel):
    videoUrl: str
    videoId: str
    userId: str
    prompt: Optional[str] = None

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