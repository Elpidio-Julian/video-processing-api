from fastapi import APIRouter, HTTPException, Depends, status
from app.schemas.video import VideoProcessRequest, VideoProcessResponse, VideoStatusResponse, User
from app.services.video_processor import VideoProcessor
from app.core.auth import get_current_user
from typing import Optional

router = APIRouter(prefix="/videos", tags=["videos"])

@router.post(
    "/process",
    response_model=VideoProcessResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {
            "description": "Video processing job accepted",
            "content": {
                "application/json": {
                    "example": {
                        "job_id": "abc123",
                        "status": "queued",
                        "estimated_time": 300
                    }
                }
            }
        },
        401: {
            "description": "Invalid authentication credentials"
        },
        500: {
            "description": "Internal server error during processing"
        }
    }
)
async def process_video(
    request: VideoProcessRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Process a video from Firebase Storage using OpenShot Cloud API.
    
    This endpoint:
    1. Accepts a Firebase Storage URL of the source video
    2. Creates a processing job
    3. Initiates async video processing
    4. Returns a job ID for status tracking
    
    The video will be:
    - Converted to vertical format (9:16)
    - Processed according to specified options
    - Stored back in Firebase Storage
    
    Authentication:
    - Requires a valid Firebase ID token
    - Token must be provided in Authorization header
    
    Example:
    ```
    POST /api/v1/videos/process
    Authorization: Bearer <firebase-id-token>
    
    {
        "source_url": "https://storage.googleapis.com/bucket/video.mp4"
    }
    ```
    """
    try:
        processor = VideoProcessor()
        result = await processor.process_video(request.source_url, current_user.uid)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/status/{job_id}",
    response_model=VideoStatusResponse,
    responses={
        200: {
            "description": "Successfully retrieved job status",
            "content": {
                "application/json": {
                    "example": {
                        "status": "processing",
                        "progress": 60.0,
                        "processed_url": None,
                        "error": None,
                        "created_at": "2024-02-13T12:00:00Z",
                        "updated_at": "2024-02-13T12:01:00Z",
                        "user_id": "user123"
                    }
                }
            }
        },
        401: {
            "description": "Invalid authentication credentials"
        },
        403: {
            "description": "Not authorized to access this job"
        },
        404: {
            "description": "Job not found"
        }
    }
)
async def get_video_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get the status of a video processing job.
    
    This endpoint:
    1. Verifies the user owns the job
    2. Returns current processing status
    3. Includes progress percentage and result URL if complete
    
    Status Values:
    - queued: Job is waiting to start
    - processing: Job is currently being processed
    - completed: Processing finished successfully
    - failed: Processing encountered an error
    
    Authentication:
    - Requires a valid Firebase ID token
    - Token must be provided in Authorization header
    - User must own the job
    
    Example:
    ```
    GET /api/v1/videos/status/abc123
    Authorization: Bearer <firebase-id-token>
    ```
    """
    try:
        processor = VideoProcessor()
        status = await processor.get_job_status(job_id, current_user.uid)
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 