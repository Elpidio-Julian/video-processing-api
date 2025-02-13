from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI(
    title=os.getenv("APP_NAME", "Video Processing API"),
    version=os.getenv("APP_VERSION", "1.0.0"),
    description="""
    API for processing videos using OpenShot Cloud and Firebase Storage.
    
    ## Authentication
    This API uses Firebase Authentication. You need to include a Firebase ID token in the Authorization header:
    ```
    Authorization: Bearer <firebase-id-token>
    ```
    
    ## Features
    - Video processing to vertical format (9:16)
    - Background effects (blur/color)
    - Progress tracking
    - Firebase Storage integration
    
    ## OpenShot Cloud Integration
    This API integrates with OpenShot Cloud API for video processing. The following operations are supported:
    
    ### Project Creation
    - Creates a new project with 1080x1920 dimensions
    - Configures project settings for vertical video
    
    ### Video Processing
    - Adds video clips to the project
    - Applies transformations (aspect ratio, background effects)
    - Exports in high quality MP4 format
    
    ### Technical Specifications
    - Video Codec: libx264
    - Video Bitrate: 4000k
    - Audio Codec: aac
    - Audio Bitrate: 128k
    """
)

# Configure CORS
origins = eval(os.getenv("CORS_ORIGINS", '["http://localhost:3000"]'))
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Security scheme for Firebase Authentication
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Firebase ID token"
        }
    }

    # Apply security globally
    openapi_schema["security"] = [{"bearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Import and include routers
from app.api.v1.routes import videos
app.include_router(videos.router, prefix="/api/v1") 