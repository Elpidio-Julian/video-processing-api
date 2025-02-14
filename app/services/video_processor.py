import os
import json
import firebase_admin
from firebase_admin import credentials, storage, firestore
from datetime import datetime
import aiohttp
import asyncio
from fastapi import HTTPException
import ffmpeg
import tempfile
import aiofiles
import subprocess

class VideoProcessor:
    def __init__(self):
        # Initialize Firebase if not already initialized
        if not firebase_admin._apps:
            cred = credentials.Certificate({
                "type": "service_account",
                "project_id": os.getenv("FIREBASE_PROJECT_ID"),
                "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace("\\n", "\n"),
                "client_email": os.getenv("FIREBASE_CLIENT_EMAIL")
            })
            firebase_admin.initialize_app(cred, {
                'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET")
            })
        
        self.db = firestore.client()
        self.bucket = storage.bucket()

    async def process_video(self, source_url: str, user_id: str):
        """
        Process a video using FFmpeg
        """
        # Create a new document in Firestore
        job_id = self.db.collection('VideoProcessing').document().id
        
        # Initialize job document
        self.db.collection('VideoProcessing').document(job_id).set({
            'jobId': job_id,
            'userId': user_id,
            'originalUrl': source_url,
            'status': 'queued',
            'progress': 0,
            'createdAt': datetime.utcnow(),
            'updatedAt': datetime.utcnow(),
            'processingOptions': {
                'targetAspectRatio': '9:16',
                'backgroundBlur': False,
                'quality': 'high'
            }
        })

        # Start processing in background
        asyncio.create_task(self._process_video_task(job_id, source_url))

        return {
            'job_id': job_id,
            'status': 'queued',
            'estimated_time': 300  # 5 minutes estimated time
        }

    async def get_job_status(self, job_id: str, user_id: str):
        """
        Get the current status of a processing job
        """
        doc = self.db.collection('VideoProcessing').document(job_id).get()
        if not doc.exists:
            raise ValueError(f"Job {job_id} not found")
        
        data = doc.to_dict()
        
        # Verify user owns this job
        if data.get('userId') != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to access this job")
        
        return {
            'status': data.get('status'),
            'progress': data.get('progress', 0),
            'processed_url': data.get('processedUrl'),
            'error': data.get('error'),
            'created_at': data.get('createdAt'),
            'updated_at': data.get('updatedAt'),
            'user_id': data.get('userId')
        }

    async def _process_video_task(self, job_id: str, source_url: str):
        """
        Background task to process the video
        """
        try:
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Update status to processing
                self._update_job_status(job_id, 'processing', progress=10)

                # Download video from Firebase
                input_path = os.path.join(temp_dir, 'input.mp4')
                await self._download_video(source_url, input_path)
                self._update_job_status(job_id, 'processing', progress=30)

                # Process video with FFmpeg
                output_path = os.path.join(temp_dir, 'output.mp4')
                await self._process_with_ffmpeg(input_path, output_path)
                self._update_job_status(job_id, 'processing', progress=70)

                # Upload processed video to Firebase
                processed_url = await self._upload_to_firebase(output_path, job_id)
                
                # Update final status
                self._update_job_status(job_id, 'completed', processed_url=processed_url)

        except Exception as e:
            self._update_job_status(job_id, 'failed', error=str(e))
            raise

    def _update_job_status(self, job_id: str, status: str, progress: float = None, 
                          processed_url: str = None, error: str = None):
        """
        Update the job status in Firestore
        """
        update_data = {
            'status': status,
            'updatedAt': datetime.utcnow()
        }
        if progress is not None:
            update_data['progress'] = progress
        if processed_url is not None:
            update_data['processedUrl'] = processed_url
        if error is not None:
            update_data['error'] = error

        self.db.collection('VideoProcessing').document(job_id).update(update_data)

    async def _download_video(self, source_url: str, output_path: str):
        """
        Download video from Firebase URL to local file
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(source_url) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to download video from Firebase: {response.status}"
                    )
                async with aiofiles.open(output_path, 'wb') as f:
                    await f.write(await response.read())

    async def _process_with_ffmpeg(self, input_path: str, output_path: str):
        """
        Process video using FFmpeg to convert to 9:16 format
        """
        try:
            # Get video metadata
            probe = ffmpeg.probe(input_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            width = int(video_info['width'])
            height = int(video_info['height'])

            # Calculate scaling to fit 9:16 ratio
            target_width = 1080
            target_height = 1920
            scale = min(target_width/width, target_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)

            # Calculate padding
            pad_x = (target_width - new_width) // 2
            pad_y = (target_height - new_height) // 2

            # Build FFmpeg command
            stream = ffmpeg.input(input_path)
            
            # Apply scaling and padding
            stream = ffmpeg.filter(stream, 'scale', new_width, new_height)
            stream = ffmpeg.filter(stream, 'pad', target_width, target_height, 
                                pad_x, pad_y, color='black')

            # Set output options for high quality
            stream = ffmpeg.output(stream, output_path,
                vcodec='libx264',
                acodec='aac',
                video_bitrate='4M',
                audio_bitrate='192k',
                preset='slow',  # Higher quality encoding
                movflags='faststart'  # Enables streaming
            )

            # Run FFmpeg command asynchronously
            await asyncio.to_thread(
                ffmpeg.run,
                stream,
                capture_stdout=True,
                capture_stderr=True
            )

        except ffmpeg.Error as e:
            raise HTTPException(
                status_code=500,
                detail=f"FFmpeg processing failed: {e.stderr.decode()}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Video processing failed: {str(e)}"
            )

    async def _upload_to_firebase(self, video_path: str, job_id: str):
        """
        Upload the processed video to Firebase Storage
        """
        try:
            # Upload to Firebase in processed directory
            blob = self.bucket.blob(f"processed/{job_id}.mp4")
            blob.upload_from_filename(video_path, content_type='video/mp4')
            
            # Make the blob publicly accessible
            blob.make_public()
            
            return blob.public_url
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload processed video to Firebase: {str(e)}"
            ) 