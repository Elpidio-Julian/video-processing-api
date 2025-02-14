from fastapi import APIRouter, HTTPException, Depends, status, Header
from app.schemas.video import VideoProcessRequest, VideoProcessResponse, VideoStatusResponse
from app.services.video_processor import VideoProcessor
from app.core.auth import verify_firebase_token

router = APIRouter(prefix="/videos", tags=["videos"])

@router.post(
    "/process",
    response_model=VideoProcessResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def process_video(
    request: VideoProcessRequest,
    user_id: str = Depends(verify_firebase_token)
):
    """
    Process a video using AI-powered video editing.
    Accepts videoUrl, videoId, userId and an optional prompt for HUD element detection.
    Authentication is handled via Firebase ID token in the Authorization header.
    """
    try:
        # Verify the requesting user matches the userId in the request
        if user_id != request.userId:
            raise HTTPException(
                status_code=403,
                detail="User ID in request does not match authenticated user"
            )
            
        processor = VideoProcessor()
        result = await processor.process_video(
            source_url=request.videoUrl,
            video_id=request.videoId,
            user_id=user_id,  # Use the verified user_id
            prompt=request.prompt
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/status/{job_id}",
    response_model=VideoStatusResponse,
)
async def get_video_status(
    job_id: str,
    user_id: str = Depends(verify_firebase_token)
):
    """
    Get the status of a video processing job.
    Authentication is handled via Firebase ID token in the Authorization header.
    """
    try:
        processor = VideoProcessor()
        status = await processor.get_job_status(job_id, user_id)
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 