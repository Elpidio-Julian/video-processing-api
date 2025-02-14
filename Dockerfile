# Use Python 3.11 slim image
FROM python:3.11-slim

# Add labels for Google Container Registry
LABEL maintainer="Video Processing API"
LABEL org.opencontainers.image.source="https://github.com/your-repo/video-processing-api"

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables with default values
ENV PORT=8000 \
    ENVIRONMENT=production \
    APP_NAME=video-processing-api \
    APP_VERSION=1.0.0

# Make startup script executable
RUN chmod +x scripts/startup.py

# Expose port (GKE will use the PORT env variable)
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/docs || exit 1

# Command to run the application with startup script
CMD ["sh", "-c", "python /app/scripts/startup.py && uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"] 