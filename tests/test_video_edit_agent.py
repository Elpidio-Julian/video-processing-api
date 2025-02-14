import pytest
import os
from app.services.video_edit_agent import VideoEditAgent
import ffmpeg
import tempfile
import json

# Use the existing test video
TEST_VIDEO_URL = "https://firebasestorage.googleapis.com/v0/b/create-tok.firebasestorage.app/o/videos%2F1739480092012-Battlefield%202042%202023.03.13%20-%2000.22.07.07.DVR%20-%20HELI%20WEAVER.MP4?alt=media&token=7c501db7-e5fe-4a5d-a139-33418897486d"

@pytest.fixture
def video_agent():
    return VideoEditAgent()

@pytest.fixture(scope="session")
def test_video(tmp_path_factory):
    """Download the test video once and reuse it for all tests"""
    import aiohttp
    import asyncio
    import aiofiles
    
    # Create a temporary directory that persists for the whole test session
    temp_dir = tmp_path_factory.mktemp('videos')
    output_path = os.path.join(temp_dir, 'test_video.mp4')
    
    # Only download if we haven't already
    if not os.path.exists(output_path):
        async def download_video():
            async with aiohttp.ClientSession() as session:
                async with session.get(TEST_VIDEO_URL) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download test video: {response.status}")
                    async with aiofiles.open(output_path, 'wb') as f:
                        await f.write(await response.read())
        
        # Run the async download in a sync context
        asyncio.run(download_video())
        print("\n✓ Downloaded test video")
    else:
        print("\n✓ Using cached test video")
    
    return output_path

def test_video_analysis(video_agent, test_video):
    """Test video analysis functionality"""
    metadata = video_agent.analyze_video(test_video)
    
    # Check that all required fields are present
    required_fields = [
        'width', 'height', 'aspect_ratio', 'duration',
        'frame_rate', 'bit_rate', 'codec_name', 'is_vertical'
    ]
    for field in required_fields:
        assert field in metadata
    
    # Print metadata for inspection
    print("\nVideo Metadata:")
    print(json.dumps(metadata, indent=2))
    
    # Basic sanity checks
    assert metadata['width'] > 0
    assert metadata['height'] > 0
    assert metadata['duration'] > 0
    assert isinstance(metadata['is_vertical'], bool)

def test_processing_steps(video_agent, test_video):
    """Test processing steps generation"""
    metadata = video_agent.analyze_video(test_video)
    steps = video_agent.determine_processing_steps(metadata)
    
    # We should have gameplay processing and quality steps
    assert len(steps) == 2
    
    # Verify we have the right steps
    step_types = [step['type'] for step in steps]
    assert 'gameplay_processing' in step_types
    assert 'quality' in step_types
    
    # Verify gameplay processing parameters
    gameplay_step = next(s for s in steps if s['type'] == 'gameplay_processing')
    assert gameplay_step['params']['mode'] == 'center_crop_with_hud'
    assert 'heli_hp' in gameplay_step['params']['hud_elements']
    
    print("\nProcessing Steps:")
    print(json.dumps(steps, indent=2))

def test_ffmpeg_command_generation(video_agent, test_video):
    """Test FFmpeg command generation"""
    with tempfile.NamedTemporaryFile(suffix='.mp4') as output_file:
        command, steps = video_agent.process_video(test_video, output_file.name)
        
        # Check that command is a valid FFmpeg stream
        assert isinstance(command, ffmpeg.Stream)
        
        # Compile command and verify it contains required filters
        cmd = " ".join(command.compile())
        assert 'crop=' in cmd  # For center crop
        assert 'overlay=' in cmd  # For HUD overlay
        assert 'scale=' in cmd  # For scaling
        assert 'libx264' in cmd  # Video codec
        assert 'aac' in cmd  # Audio codec
        
        print("\nFFmpeg Command:")
        print(cmd)

def test_end_to_end_processing(video_agent, test_video):
    """Test complete video processing pipeline"""
    # Create test_outputs directory if it doesn't exist
    os.makedirs('test_outputs', exist_ok=True)
    output_path = os.path.join('test_outputs', 'processed_video.mp4')
    
    # Process the video
    command, steps = video_agent.process_video(test_video, output_path)
    
    try:
        print("\nProcessing video...")
        # Run the FFmpeg command
        command.run(capture_stdout=True, capture_stderr=True)
        print("✓ FFmpeg processing completed")
        
        # Verify the output file exists and is not empty
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        print(f"✓ Output file size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        print(f"✓ Output file saved to: {output_path}")
        
        # Analyze the output video
        output_metadata = video_agent.analyze_video(output_path)
        
        # Verify dimensions match our target vertical format
        assert output_metadata['width'] == video_agent.target_width   # 1080
        assert output_metadata['height'] == video_agent.target_height # 1920
        assert output_metadata['is_vertical']  # Should be vertical now
        print("✓ Output video dimensions verified")
        
        print("\nOutput Video Metadata:")
        print(json.dumps(output_metadata, indent=2))
        
    except ffmpeg.Error as e:
        print("FFmpeg error output:")
        print(e.stderr.decode())
        raise

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 