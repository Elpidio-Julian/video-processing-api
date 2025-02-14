import pytest
import os
from dotenv import load_dotenv
from app.services.video_edit_agent import VideoEditAgent, VideoTools, HUDElement
import ffmpeg
import tempfile
import json
import cv2
import numpy as np

# Load environment variables
load_dotenv()

# Use the existing test video
TEST_VIDEO_URL = "https://firebasestorage.googleapis.com/v0/b/create-tok.firebasestorage.app/o/videos%2F1739480092012-Battlefield%202042%202023.03.13%20-%2000.22.07.07.DVR%20-%20HELI%20WEAVER.MP4?alt=media&token=7c501db7-e5fe-4a5d-a139-33418897486d"

@pytest.fixture
def video_agent():
    """Create VideoEditAgent in deterministic mode for testing"""
    return VideoEditAgent(use_llm=False)

@pytest.fixture(scope="session")
def test_video(tmp_path_factory):
    """Download the test video once and reuse it for all tests"""
    import aiohttp
    import asyncio
    import aiofiles
    import ffmpeg
    
    # Create a temporary directory that persists for the whole test session
    temp_dir = tmp_path_factory.mktemp('videos')
    output_path = os.path.join(temp_dir, 'test_video.mp4')
    sample_path = os.path.join(temp_dir, 'sample_video.mp4')
    
    # Only download if we haven't already
    if not os.path.exists(sample_path):
        async def download_video():
            async with aiohttp.ClientSession() as session:
                async with session.get(TEST_VIDEO_URL) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download test video: {response.status}")
                    async with aiofiles.open(output_path, 'wb') as f:
                        await f.write(await response.read())
            
            # Create a 5-second sample using ffmpeg
            stream = ffmpeg.input(output_path, ss=0, t=5)  # Take first 5 seconds
            stream = ffmpeg.output(stream, sample_path, acodec='copy', vcodec='copy')
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            # Remove the full video to save space
            os.remove(output_path)
        
        # Run the async download in a sync context
        asyncio.run(download_video())
        print("\n✓ Created 5-second test video sample")
    else:
        print("\n✓ Using cached video sample")
    
    return sample_path

def test_video_analysis(video_agent, test_video):
    """Test video analysis functionality"""
    metadata = video_agent.analyze_video(test_video)
    
    # Check that all required fields are present
    required_fields = [
        'width', 'height', 'aspect_ratio', 'duration',
        'frame_rate', 'bit_rate', 'codec_name', 'is_vertical',
        'game_type', 'hud_elements', 'main_action_area'
    ]
    for field in required_fields:
        assert field in metadata, f"Missing required field: {field}"
    
    # Verify HUD elements structure
    assert isinstance(metadata['hud_elements'], list)
    if metadata['hud_elements']:
        hud = metadata['hud_elements'][0]
        assert 'type' in hud
        assert 'position' in hud
        assert all(k in hud['position'] for k in ['x', 'y', 'width', 'height'])
    
    # Basic sanity checks
    assert metadata['width'] > 0
    assert metadata['height'] > 0
    assert metadata['duration'] > 0
    assert isinstance(metadata['is_vertical'], bool)

def test_processing_steps(video_agent, test_video):
    """Test processing steps generation"""
    metadata = video_agent.analyze_video(test_video)
    steps = video_agent.determine_processing_steps(metadata)
    
    # Verify we have exactly the required steps
    assert len(steps) == 2, f"Expected 2 steps, got {len(steps)}"
    step_types = {step['type'] for step in steps}
    assert step_types == {'gameplay_processing', 'quality'}, f"Unexpected step types: {step_types}"
    
    # Verify gameplay processing parameters
    gameplay_step = next(s for s in steps if s['type'] == 'gameplay_processing')
    assert gameplay_step['params']['mode'] == 'center_crop_with_hud'
    assert 'hud_elements' in gameplay_step['params']
    assert gameplay_step['params']['hud_elements'] == ['heli_hp']
    
    # Verify quality parameters
    quality_step = next(s for s in steps if s['type'] == 'quality')
    assert 'video_bitrate' in quality_step['params']
    assert 'audio_bitrate' in quality_step['params']
    assert 'preset' in quality_step['params']

def test_ffmpeg_command_generation(video_agent, test_video):
    """Test FFmpeg command generation"""
    with tempfile.NamedTemporaryFile(suffix='.mp4') as output_file:
        command, steps = video_agent.process_video(test_video, output_file.name)
        
        # Check that command is a valid FFmpeg stream
        assert isinstance(command, ffmpeg.Stream)
        
        # Compile command and verify it contains required filters
        cmd = " ".join(command.compile())
        print("\nFFmpeg Command:")
        print(cmd)
        
        # Check for required elements
        required_elements = [
            'crop=', 'scale=1080:1920',  # Vertical format
            'libx264', 'aac',  # Codecs
            '-y'  # Overwrite flag
        ]
        for element in required_elements:
            assert element in cmd, f"Missing required element in command: {element}"

def test_video_tools():
    """Test individual video transformation tools"""
    tools = VideoTools()
    
    # Create a dummy stream for testing
    stream = ffmpeg.input('dummy.mp4')['v']
    
    # Test center_crop
    cropped = tools.center_crop(stream, 1920, 1080, (9, 16))
    cmd = " ".join(ffmpeg.output(cropped, 'out.mp4').compile())
    assert 'crop=' in cmd
    
    # Test scale
    scaled = tools.scale(stream, 1080, 1920)
    cmd = " ".join(ffmpeg.output(scaled, 'out.mp4').compile())
    assert 'scale=1080:1920' in cmd
    
    # Test extract_region
    region = tools.extract_region(stream, 100, 100, 200, 200)
    cmd = " ".join(ffmpeg.output(region, 'out.mp4').compile())
    assert 'crop=200:200:100:100' in cmd
    
    # Test quality settings
    quality = tools.adjust_quality(None, '4M')
    assert quality['video_bitrate'] == '4M'
    assert quality['vcodec'] == 'libx264'

def test_action_sequence(video_agent, test_video):
    """Test the generation and execution of action sequence"""
    command, actions = video_agent.process_video(test_video, 'test_outputs/processed.mp4')
    
    # Verify we have all required actions in the correct order
    expected_actions = ['center_crop', 'scale', 'extract_region', 'overlay', 'adjust_quality']
    actual_actions = [a['action'] for a in actions]
    assert actual_actions == expected_actions, f"Expected {expected_actions}, got {actual_actions}"
    
    # Verify action parameters
    for action in actions:
        if action['action'] == 'center_crop':
            assert 'input_width' in action['params']
            assert 'input_height' in action['params']
            assert 'target_ratio' in action['params']
            assert action['params']['target_ratio'] == video_agent.target_aspect_ratio
        elif action['action'] == 'scale':
            assert action['params']['width'] == video_agent.target_width
            assert action['params']['height'] == video_agent.target_height
        elif action['action'] == 'extract_region':
            assert all(k in action['params'] for k in ['x', 'y', 'width', 'height'])
        elif action['action'] == 'overlay':
            assert 'x' in action['params']
            assert 'y' in action['params']
        elif action['action'] == 'adjust_quality':
            assert 'video_bitrate' in action['params']

def test_end_to_end_processing(video_agent, test_video):
    """Test complete video processing pipeline with tools"""
    os.makedirs('test_outputs', exist_ok=True)
    output_path = os.path.join('test_outputs', 'processed_video.mp4')
    
    try:
        # Process the video
        command, actions = video_agent.process_video(test_video, output_path)
        print("\nProcessing video with actions:", json.dumps(actions, indent=2))
        
        # Run the FFmpeg command
        command.run(capture_stdout=True, capture_stderr=True)
        print("✓ FFmpeg processing completed")
        
        # Verify output
        assert os.path.exists(output_path), "Output file was not created"
        assert os.path.getsize(output_path) > 0, "Output file is empty"
        
        # Verify dimensions
        probe = ffmpeg.probe(output_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        assert int(video_info['width']) == video_agent.target_width
        assert int(video_info['height']) == video_agent.target_height
        
        print(f"✓ Output file size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        print(f"✓ Output file saved to: {output_path}")
        
    except ffmpeg.Error as e:
        print("FFmpeg error output:")
        print(e.stderr.decode())
        raise

def test_opencv_functionality(test_video):
    """Test that OpenCV can read video frames"""
    cap = cv2.VideoCapture(test_video)
    ret, frame = cap.read()
    cap.release()
    
    assert ret == True, "Failed to read frame from video"
    assert isinstance(frame, np.ndarray), "Frame should be a numpy array"
    assert len(frame.shape) == 3, "Frame should be a 3D array (height, width, channels)"
    assert frame.shape[2] == 3, "Frame should have 3 color channels (BGR)"
    
    print("\nOpenCV successfully read video frame with shape:", frame.shape)

def test_hud_detection(video_agent, test_video):
    """Test HUD element detection functionality"""
    # Process a single frame
    cap = cv2.VideoCapture(test_video)
    ret, frame = cap.read()
    cap.release()
    assert ret, "Failed to read test video frame"
    
    # Detect HUD elements
    elements = video_agent._detect_hud_elements(frame)
    
    # Verify detection results
    assert isinstance(elements, list), "Should return a list of HUD elements"
    for element in elements:
        assert isinstance(element, HUDElement), "Each element should be a HUDElement instance"
        assert element.element_type in ['text_element'] + list(video_agent.hud_templates.keys())
        assert all(k in element.position for k in ['x', 'y', 'width', 'height'])
        assert 0 <= element.confidence <= 1, "Confidence should be between 0 and 1"

def test_hud_tracking(video_agent):
    """Test HUD element tracking across frames"""
    # Create synthetic frames with a moving HUD element
    frames = []
    element_positions = []
    frame_size = (640, 480)
    
    for i in range(5):
        # Create blank frame
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        # Add moving HUD element (white rectangle)
        x = 100 + i * 10  # Move right by 10 pixels each frame
        y = 100
        cv2.rectangle(frame, (x, y), (x + 50, y + 30), (255, 255, 255), -1)
        frames.append(frame)
        element_positions.append({'x': x, 'y': y, 'width': 50, 'height': 30})
    
    # Track element across frames
    all_tracked_elements = []
    for i, frame in enumerate(frames):
        elements = [HUDElement(
            element_type='synthetic_hud',
            position=element_positions[i],
            confidence=1.0
        )]
        tracked = video_agent.hud_tracker.update(elements, frame)
        all_tracked_elements.append(tracked)
    
    # Verify tracking
    assert len(all_tracked_elements) == 5, "Should track element in all frames"
    
    # Check if the same element maintains its track_id
    first_element_id = all_tracked_elements[0][0].track_id
    for frame_elements in all_tracked_elements[1:]:
        assert frame_elements[0].track_id == first_element_id, "Track ID should remain consistent"
    
    # Verify position history
    track_positions = video_agent.hud_tracker.history[first_element_id]
    assert len(track_positions) == 5, "Should have position history for all frames"
    
    # Check if positions show expected movement
    for i in range(1, len(track_positions)):
        x_diff = track_positions[i]['x'] - track_positions[i-1]['x']
        assert x_diff == 10, "Element should move right by 10 pixels each frame"

def test_stable_element_detection(video_agent):
    """Test detection of stable HUD elements"""
    # Create synthetic frames with stable and unstable elements
    frames = []
    stable_pos = {'x': 100, 'y': 100, 'width': 50, 'height': 30}
    unstable_positions = [
        {'x': 200, 'y': 100, 'width': 50, 'height': 30},
        {'x': 250, 'y': 120, 'width': 50, 'height': 30},
        {'x': 180, 'y': 90, 'width': 50, 'height': 30},
    ]
    
    for i in range(3):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add stable element
        cv2.rectangle(
            frame,
            (stable_pos['x'], stable_pos['y']),
            (stable_pos['x'] + stable_pos['width'], stable_pos['y'] + stable_pos['height']),
            (255, 255, 255),
            -1
        )
        # Add unstable element
        unstable_pos = unstable_positions[i]
        cv2.rectangle(
            frame,
            (unstable_pos['x'], unstable_pos['y']),
            (unstable_pos['x'] + unstable_pos['width'], unstable_pos['y'] + unstable_pos['height']),
            (255, 255, 255),
            -1
        )
        frames.append(frame)
    
    # Process frames
    for i, frame in enumerate(frames):
        elements = [
            HUDElement(
                element_type='stable_element',
                position=stable_pos,
                confidence=1.0
            ),
            HUDElement(
                element_type='unstable_element',
                position=unstable_positions[i],
                confidence=1.0
            )
        ]
        video_agent.hud_tracker.update(elements, frame)
    
    # Get stable elements
    stable_elements = video_agent.hud_tracker.get_stable_elements()
    
    # Verify results
    assert len(stable_elements) == 1, "Should detect exactly one stable element"
    assert stable_elements[0].element_type == 'stable_element'
    assert stable_elements[0].position == stable_pos

def test_end_to_end_hud_processing(video_agent, test_video):
    """Test complete HUD detection and tracking pipeline"""
    # Analyze video
    metadata = video_agent.analyze_video(test_video)
    
    # Verify HUD elements in metadata
    assert 'hud_elements' in metadata
    assert isinstance(metadata['hud_elements'], list)
    
    # Check frame samples
    assert len(video_agent.state['frame_samples']) > 0
    for sample in video_agent.state['frame_samples']:
        assert 'hud_elements' in sample
        for element in sample['hud_elements']:
            assert 'type' in element
            assert 'position' in element
            assert 'confidence' in element
            assert 'track_id' in element
    
    # Process video
    with tempfile.NamedTemporaryFile(suffix='.mp4') as output_file:
        command, actions = video_agent.process_video(test_video, output_file.name)
        
        # Verify HUD-related actions
        hud_actions = [a for a in actions if a['action'] in ['extract_region', 'overlay']]
        assert len(hud_actions) > 0, "Should include HUD processing actions"
        
        # Check if command includes HUD processing filters
        cmd = " ".join(command.compile())
        assert 'crop=' in cmd, "Should include crop filter for HUD extraction"
        assert 'overlay=' in cmd, "Should include overlay filter for HUD elements"

def test_specific_hud_templates(video_agent, test_video):
    """Test detection of specific HUD templates"""
    # Process a single frame
    cap = cv2.VideoCapture(test_video)
    ret, frame = cap.read()
    cap.release()
    assert ret, "Failed to read test video frame"
    
    # Detect HUD elements with a lower confidence threshold
    elements = video_agent._detect_hud_elements(frame)
    
    # Print detection results
    print("\nDetected HUD Elements:")
    for element in elements:
        print(f"- Type: {element.element_type}")
        print(f"  Position: {element.position}")
        print(f"  Confidence: {element.confidence:.2f}")
    
    # Verify we can detect our specific templates
    element_types = {element.element_type for element in elements}
    print("\nDetected element types:", element_types)
    
    # Check for expected template types
    expected_types = {'ground_unit_hud', 'vehicle_hud', 'incoming_missile', 'incoming_missile1'}
    found_types = expected_types.intersection(element_types)
    print("\nFound expected types:", found_types)
    
    # Assert we found at least one of our templates
    assert len(found_types) > 0, "No expected HUD templates were detected"
    
    # For detected elements, verify reasonable confidence
    for element in elements:
        if element.element_type in expected_types:
            assert element.confidence > 0.3, f"Low confidence ({element.confidence}) for {element.element_type}"

def test_full_system_processing(video_agent, test_video):
    """Test complete system including aspect ratio conversion and HUD overlays"""
    os.makedirs('test_outputs', exist_ok=True)
    output_path = os.path.join('test_outputs', 'full_system_test.mp4')
    
    try:
        # 1. Process the video
        print("\nProcessing video with full system test...")
        command, actions = video_agent.process_video(test_video, output_path)
        command.run(capture_stdout=True, capture_stderr=True)
        print("✓ Video processing completed")
        
        # 2. Verify output video properties
        probe = ffmpeg.probe(output_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        
        # Check dimensions (should be 9:16)
        width = int(video_info['width'])
        height = int(video_info['height'])
        aspect_ratio = width / height
        target_ratio = 9 / 16
        print(f"\nOutput dimensions: {width}x{height}")
        print(f"Aspect ratio: {aspect_ratio:.3f} (target: {target_ratio:.3f})")
        assert abs(aspect_ratio - target_ratio) < 0.01, "Output aspect ratio should be 9:16"
        
        # 3. Verify content preservation by sampling frames
        cap = cv2.VideoCapture(output_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames from beginning, middle, and end
        sample_positions = [0, total_frames // 2, total_frames - 1]
        for pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            assert ret, f"Failed to read frame at position {pos}"
            
            # Check for black bars (should not have significant black regions in center)
            center_region = frame[height//4:3*height//4, width//4:3*width//4]
            is_black = np.mean(center_region) < 10  # Average pixel value < 10 indicates mostly black
            assert not is_black, f"Center content missing at frame {pos}"
            
            # Detect HUD elements in frame
            elements = video_agent._detect_hud_elements(frame)
            print(f"\nFrame {pos}: Found {len(elements)} HUD elements")
            for element in elements:
                print(f"- {element.element_type}: confidence={element.confidence:.2f}")
                
            # Verify at least some HUD elements are detected
            if pos == total_frames // 2:  # Check middle frame more strictly
                assert len(elements) > 0, "No HUD elements detected in middle frame"
        
        cap.release()
        print("\n✓ Content preservation verified")
        print(f"✓ Output saved to: {output_path}")
        
    except ffmpeg.Error as e:
        print("\nFFmpeg error output:")
        print(e.stderr.decode())
        raise
    except Exception as e:
        print(f"\nError during full system test: {str(e)}")
        raise

if __name__ == "__main__":
    pytest.main([__file__ + "::test_full_system_processing", "-v"]) 