from fastapi import APIRouter, HTTPException, Depends, status
from app.schemas.video import VideoProcessRequest, VideoProcessResponse, VideoStatusResponse
from app.services.video_processor import VideoProcessor
from app.core.auth import get_current_user

router = APIRouter(prefix="/videos", tags=["videos"])

@router.post(
    "/process",
    response_model=VideoProcessResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def process_video(
    request: VideoProcessRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Process a video from Firebase Storage using OpenShot Cloud API.
    Only requires a video URL and authentication token.
    """
    try:
        processor = VideoProcessor()
        result = await processor.process_video(request.source_url, user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/status/{job_id}",
    response_model=VideoStatusResponse,
)
async def get_video_status(
    job_id: str,
    user_id: str = Depends(get_current_user)
):
    """
    Get the status of a video processing job.
    Only requires job ID and authentication token.
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