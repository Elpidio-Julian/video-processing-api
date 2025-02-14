from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer
from dotenv import load_dotenv
import os
import firebase_admin
from firebase_admin import credentials

# Load environment variables
load_dotenv()

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate("create-tok-firebase-adminsdk-fbsvc-21d11da1ed.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET')
    })

app = FastAPI(
    title=os.getenv("APP_NAME", "Video Processing API"),
    version=os.getenv("APP_VERSION", "1.0.0"),
    description="""
    API for processing videos using AI-powered video editing.
    
    ## Authentication
    This API uses Firebase Authentication. You need to include a Firebase ID token in the Authorization header:
    ```
    Authorization: Bearer <firebase-id-token>
    ```
    
    ## Features
    - Video processing to vertical format (9:16)
    - Intelligent HUD element detection and repositioning
    - Progress tracking
    - Firebase Storage integration
    
    ## Video Processing
    The API uses an AI-powered video editing agent that:
    - Analyzes video content
    - Detects and tracks HUD elements
    - Optimizes layout for vertical format
    - Maintains video quality
    
    ### Technical Specifications
    - Output Format: MP4
    - Resolution: 1080x1920 (9:16)
    - Video Codec: libx264
    - Audio Codec: aac
    """
)

# Configure CORS for Expo Go development
origins = [
    "http://localhost",
    "http://localhost:19000",
    "http://localhost:19001",
    "http://localhost:19002",
    "http://localhost:19006",
    "exp://localhost:19000",
    "exp://localhost:19001",
    "exp://localhost:19002",
    "exp://localhost:19006",
]

# Add dynamic IP-based origins for Expo Go
import socket
def get_local_ip():
    try:
        # Get local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return None

local_ip = get_local_ip()
if local_ip:
    # Add IP-based origins for Expo Go
    ip_origins = [
        f"http://{local_ip}:19000",
        f"http://{local_ip}:19001",
        f"http://{local_ip}:19002",
        f"http://{local_ip}:19006",
        f"exp://{local_ip}:19000",
        f"exp://{local_ip}:19001",
        f"exp://{local_ip}:19002",
        f"exp://{local_ip}:19006",
    ]
    origins.extend(ip_origins)

# Add any additional origins from environment variable
env_origins = os.getenv("CORS_ORIGINS")
if env_origins:
    try:
        additional_origins = eval(env_origins)
        if isinstance(additional_origins, list):
            origins.extend(additional_origins)
    except:
        pass

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