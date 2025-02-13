import os
import json
import firebase_admin
from firebase_admin import credentials, storage, firestore
from datetime import datetime
import aiohttp
import asyncio
from fastapi import HTTPException

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
        self.openshot_api_key = os.getenv("OPENSHOT_API_KEY")
        self.openshot_api_endpoint = os.getenv("OPENSHOT_API_ENDPOINT")

    async def process_video(self, source_url: str, user_id: str):
        """
        Process a video using OpenShot Cloud API
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
            # Update status to processing
            self._update_job_status(job_id, 'processing', progress=10)

            # Create OpenShot project
            project_data = await self._create_openshot_project(job_id)
            
            # Add clip to project
            await self._add_clip_to_project(project_data['id'], source_url)
            self._update_job_status(job_id, 'processing', progress=30)

            # Apply transformations
            await self._apply_transformations(project_data['id'])
            self._update_job_status(job_id, 'processing', progress=60)

            # Export video
            export_url = await self._export_video(project_data['id'])
            self._update_job_status(job_id, 'processing', progress=90)

            # Upload to Firebase
            processed_url = await self._upload_to_firebase(export_url, job_id)
            
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

    async def _handle_openshot_response(self, response: aiohttp.ClientResponse, operation: str):
        """
        Handle OpenShot API response and errors
        """
        if response.status == 401:
            raise HTTPException(status_code=500, detail="OpenShot API authentication failed")
        elif response.status == 404:
            raise HTTPException(status_code=500, detail=f"OpenShot API resource not found: {operation}")
        elif response.status >= 400:
            error_data = await response.text()
            raise HTTPException(
                status_code=500,
                detail=f"OpenShot API error during {operation}: {error_data}"
            )
        
        try:
            return await response.json()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse OpenShot API response: {str(e)}"
            )

    async def _create_openshot_project(self, job_id: str):
        """
        Create a new project in OpenShot Cloud
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.openshot_api_endpoint}/projects/",
                headers={"Authorization": f"Bearer {self.openshot_api_key}"},
                json={
                    "name": f"video_{job_id}",
                    "width": 1080,
                    "height": 1920
                }
            ) as response:
                return await self._handle_openshot_response(response, "project creation")

    async def _add_clip_to_project(self, project_id: str, source_url: str):
        """
        Add a clip to the OpenShot project
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.openshot_api_endpoint}/projects/{project_id}/clips",
                headers={"Authorization": f"Bearer {self.openshot_api_key}"},
                json={
                    "file": source_url,
                    "position": 0,
                    "layer": 1
                }
            ) as response:
                return await self._handle_openshot_response(response, "adding clip")

    async def _apply_transformations(self, project_id: str):
        """
        Apply video transformations in OpenShot to convert to 9:16 format
        """
        async with aiohttp.ClientSession() as session:
            # First, get the clip properties to calculate scaling
            async with session.get(
                f"{self.openshot_api_endpoint}/projects/{project_id}/clips",
                headers={"Authorization": f"Bearer {self.openshot_api_key}"}
            ) as response:
                clips_data = await self._handle_openshot_response(response, "getting clips")
                if not clips_data:
                    raise HTTPException(
                        status_code=500,
                        detail="No clips found in project"
                    )
                
                clip = clips_data[0]  # Get the first clip
                original_width = clip.get('width', 1920)
                original_height = clip.get('height', 1080)

            # Calculate scaling to fit 9:16 ratio while maintaining aspect ratio
            target_width = 1080  # For 9:16 vertical video
            target_height = 1920

            # Calculate scale factor to fit video within 9:16 frame
            width_scale = target_width / original_width
            height_scale = target_height / original_height
            scale = min(width_scale, height_scale)

            # Calculate new dimensions
            new_width = original_width * scale
            new_height = original_height * scale

            # Calculate position to center the video
            x_offset = (target_width - new_width) / 2
            y_offset = (target_height - new_height) / 2

            # Update clip properties
            async with session.put(
                f"{self.openshot_api_endpoint}/projects/{project_id}/clips/{clip['id']}",
                headers={"Authorization": f"Bearer {self.openshot_api_key}"},
                json={
                    "scale": scale,
                    "location_x": x_offset,
                    "location_y": y_offset,
                    "gravity": "center",
                    "background_color": "#000000"
                }
            ) as response:
                return await self._handle_openshot_response(response, "applying transformations")

    async def _export_video(self, project_id: str):
        """
        Export the video from OpenShot
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.openshot_api_endpoint}/projects/{project_id}/exports",
                headers={"Authorization": f"Bearer {self.openshot_api_key}"},
                json={
                    "format": "mp4",
                    "video_codec": "libx264",
                    "video_bitrate": "4000k",
                    "audio_codec": "aac",
                    "audio_bitrate": "128k",
                    "width": 1080,
                    "height": 1920
                }
            ) as response:
                export_data = await self._handle_openshot_response(response, "exporting video")
                export_url = export_data.get('url')
                if not export_url:
                    raise HTTPException(
                        status_code=500,
                        detail="No export URL received from OpenShot"
                    )
                return export_url

    async def _upload_to_firebase(self, video_url: str, job_id: str):
        """
        Upload the processed video to Firebase Storage
        """
        # Download video from OpenShot
        async with aiohttp.ClientSession() as session:
            async with session.get(video_url) as response:
                video_data = await response.read()

        # Upload to Firebase in user-specific directory
        blob = self.bucket.blob(f"processed/{job_id}.mp4")
        blob.upload_from_string(video_data, content_type='video/mp4')
        
        # Make the blob publicly accessible
        blob.make_public()
        
        return blob.public_url 