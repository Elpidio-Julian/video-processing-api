import ffmpeg
import json
from typing import Dict, List, Optional, Tuple, Any
import math
import cv2
import numpy as np
from openai import OpenAI
import os
import tempfile
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class HUDElement:
    """Represents a detected HUD element with tracking info"""
    element_type: str
    position: Dict[str, int]
    confidence: float
    template: Optional[np.ndarray] = None
    track_id: Optional[int] = None
    
class HUDTracker:
    """Tracks HUD elements across video frames"""
    def __init__(self):
        self.next_track_id = 0
        self.tracked_elements = {}  # track_id -> HUDElement
        self.history = defaultdict(list)  # track_id -> list of positions
        
    def update(self, elements: List[HUDElement], frame: np.ndarray) -> List[HUDElement]:
        """Update tracking info for detected elements"""
        if not self.tracked_elements:
            # First frame, assign new track IDs
            for element in elements:
                element.track_id = self.next_track_id
                self.tracked_elements[self.next_track_id] = element
                self.history[self.next_track_id].append(element.position)
                self.next_track_id += 1
            return elements
            
        # Match new detections with existing tracks
        matched_elements = []
        for element in elements:
            best_match = None
            best_iou = 0
            
            for track_id, tracked in self.tracked_elements.items():
                iou = self._calculate_iou(element.position, tracked.position)
                if iou > 0.5 and iou > best_iou:  # IOU threshold of 0.5
                    best_match = track_id
                    best_iou = iou
            
            if best_match is not None:
                # Update existing track
                element.track_id = best_match
                self.tracked_elements[best_match] = element
                self.history[best_match].append(element.position)
            else:
                # Create new track
                element.track_id = self.next_track_id
                self.tracked_elements[self.next_track_id] = element
                self.history[self.next_track_id].append(element.position)
                self.next_track_id += 1
                
            matched_elements.append(element)
            
        return matched_elements
    
    def _calculate_iou(self, box1: Dict[str, int], box2: Dict[str, int]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1 = max(box1['x'], box2['x'])
        y1 = max(box1['y'], box2['y'])
        x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
        y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = box1['width'] * box1['height']
        box2_area = box2['width'] * box2['height']
        
        return intersection / float(box1_area + box2_area - intersection)
    
    def get_stable_elements(self) -> List[HUDElement]:
        """Return elements that have been consistently tracked"""
        stable_elements = []
        for track_id, positions in self.history.items():
            if len(positions) >= 3:  # Require at least 3 detections
                # Calculate position stability
                position_std = np.std([list(pos.values()) for pos in positions], axis=0)
                if np.all(position_std < 20):  # Position varies less than 20 pixels
                    stable_elements.append(self.tracked_elements[track_id])
        return stable_elements

class VideoTools:
    """A collection of basic video transformation tools"""
    
    @staticmethod
    def center_crop(stream, input_width: int, input_height: int, target_ratio: Tuple[int, int]) -> ffmpeg.Stream:
        """Crop the center portion of the video to match target ratio"""
        target_width = int((input_height * target_ratio[0]) / target_ratio[1])
        x_offset = (input_width - target_width) // 2
        return ffmpeg.filter(stream, 'crop', target_width, input_height, x_offset, 0)
    
    @staticmethod
    def scale(stream, width: int, height: int) -> ffmpeg.Stream:
        """Scale video to target dimensions"""
        return ffmpeg.filter(stream, 'scale', width, height)
    
    @staticmethod
    def extract_region(stream, x: int, y: int, width: int, height: int) -> ffmpeg.Stream:
        """Extract a specific region from the video"""
        return ffmpeg.filter(stream, 'crop', width, height, x, y)
    
    @staticmethod
    def overlay(base: ffmpeg.Stream, overlay: ffmpeg.Stream, x: int, y: int) -> ffmpeg.Stream:
        """Overlay one video stream on top of another"""
        return ffmpeg.filter([base, overlay], 'overlay', x, y)
    
    @staticmethod
    def adjust_quality(stream, video_bitrate: str, audio_bitrate: str = '192k', preset: str = 'slow') -> Dict:
        """Return quality settings for final output"""
        return {
            'vcodec': 'libx264',
            'acodec': 'aac',
            'video_bitrate': video_bitrate,
            'audio_bitrate': audio_bitrate,
            'preset': preset,
            'movflags': 'faststart'
        }

class VideoEditAction:
    """Represents a single video editing action with its parameters"""
    def __init__(self, tool_name: str, params: Dict):
        self.tool_name = tool_name
        self.params = params
    
    def execute(self, stream: ffmpeg.Stream, tools: VideoTools) -> ffmpeg.Stream:
        tool = getattr(tools, self.tool_name)
        return tool(stream, **self.params)

class VideoEditAgent:
    """
    A ReAct (Reasoning and Acting) agent for video editing.
    Uses LLM to:
    1. Analyze video content and identify key elements
    2. Reason about optimal processing strategy
    3. Generate and refine FFmpeg commands
    4. Evaluate results and improve
    """

    def __init__(self, use_llm: bool = True):
        self.target_aspect_ratio = (9, 16)
        self.target_width = 1080
        self.target_height = 1920
        self.min_bitrate = "2M"
        self.max_bitrate = "4M"
        
        # Make LLM optional for testing
        self.use_llm = use_llm
        self.llm = OpenAI() if use_llm else None
        
        # State management for ReAct loop
        self.state = {
            'observations': [],
            'thoughts': [],
            'actions': [],
            'frame_samples': [],  # Store analyzed frame data
            'detected_elements': {},  # Store detected HUD elements
            'pending_evaluation': None
        }

        self.tools = VideoTools()
        self.actions = []
        self.hud_tracker = HUDTracker()
        self.hud_templates = self._load_hud_templates()

    def _load_hud_templates(self) -> Dict[str, np.ndarray]:
        """Load template images for common HUD elements"""
        templates = {}
        template_dir = os.path.join(os.path.dirname(__file__), 'hud_templates')
        if os.path.exists(template_dir):
            for filename in os.listdir(template_dir):
                if filename.endswith('.png'):
                    element_type = filename[:-4]  # Remove .png
                    template_path = os.path.join(template_dir, filename)
                    templates[element_type] = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        return templates
        
    def _calculate_iou(self, box1: Dict[str, int], box2: Dict[str, int]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1 = max(box1['x'], box2['x'])
        y1 = max(box1['y'], box2['y'])
        x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
        y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = box1['width'] * box1['height']
        box2_area = box2['width'] * box2['height']
        
        return intersection / float(box1_area + box2_area - intersection)

    def _detect_hud_elements(self, frame: np.ndarray) -> List[HUDElement]:
        """Detect HUD elements using template matching and image processing"""
        elements = []
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Debug info
        print(f"\nLoaded templates: {list(self.hud_templates.keys())}")
        
        # 1. Template Matching with multiple thresholds
        for element_type, template in self.hud_templates.items():
            # Ensure template is grayscale
            if len(template.shape) > 2:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # Use TM_CCOEFF_NORMED for best results
            result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.5  # Lower threshold for more matches
            
            # Find local maxima
            loc = np.where(result >= threshold)
            for pt in zip(*loc[::-1]):  # Switch columns and rows
                confidence = result[pt[1], pt[0]]
                print(f"Found {element_type} at {pt} with confidence {confidence:.2f}")
                
                # Add element if we haven't already found one at this location
                new_element = HUDElement(
                    element_type=element_type,
                    position={
                        'x': int(pt[0]),
                        'y': int(pt[1]),
                        'width': template.shape[1],
                        'height': template.shape[0]
                    },
                    confidence=float(confidence),
                    template=template
                )
                
                # Check if this element overlaps with any existing ones
                overlap = False
                for existing in elements:
                    if self._calculate_iou(new_element.position, existing.position) > 0.5:
                        overlap = True
                        if new_element.confidence > existing.confidence:
                            elements.remove(existing)
                            elements.append(new_element)
                        break
                
                if not overlap:
                    elements.append(new_element)
        
        # 2. Text Detection (only if no HUD elements found)
        if not elements:
            try:
                thresh = cv2.adaptiveThreshold(
                    gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if 0.2 < w/h < 3 and 20 < w < 200 and 15 < h < 100:
                        roi = gray_frame[y:y+h, x:x+w]
                        if np.var(roi) > 1000:
                            elements.append(HUDElement(
                                element_type='text_element',
                                position={'x': x, 'y': y, 'width': w, 'height': h},
                                confidence=0.7
                            ))
            except Exception as e:
                print(f"Text detection error: {str(e)}")
        
        return elements
        
    def _sample_frames(self, input_path: str, num_samples: int = 5):
        """Extract sample frames and detect HUD elements with tracking"""
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for i in range(num_samples):
            frame_pos = int((i / (num_samples - 1)) * total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if ret:
                # Detect HUD elements
                elements = self._detect_hud_elements(frame)
                # Update tracking
                tracked_elements = self.hud_tracker.update(elements, frame)
                
                self.state['frame_samples'].append({
                    'position': frame_pos,
                    'frame': frame,
                    'hud_elements': [
                        {
                            'type': e.element_type,
                            'position': e.position,
                            'confidence': e.confidence,
                            'track_id': e.track_id
                        }
                        for e in tracked_elements
                    ]
                })
        
        cap.release()

    def _analyze_content(self, technical_metadata: Dict) -> Dict:
        """Enhanced content analysis with computer vision results"""
        if not self.use_llm:
            return self._default_content_analysis(technical_metadata)
            
        # Get stable HUD elements from tracker
        stable_elements = self.hud_tracker.get_stable_elements()
        
        # Convert frame samples to text descriptions
        frame_descriptions = []
        for sample in self.state['frame_samples']:
            frame = sample['frame']
            edges = cv2.Canny(frame, 100, 200)
            motion = np.mean(edges)
            
            description = f"Frame at position {sample['position']}: "
            description += f"Motion intensity: {'high' if motion > 50 else 'low'}, "
            description += "HUD elements: " + ", ".join([
                f"{e['type']} at ({e['position']['x']}, {e['position']['y']})"
                for e in sample['hud_elements']
            ])
            
            frame_descriptions.append(description)
            
        prompt = f"""
        Analyze this gameplay video:
        
        Technical details:
        {json.dumps(technical_metadata, indent=2)}
        
        Frame samples:
        {json.dumps(frame_descriptions, indent=2)}
        
        Please identify:
        1. What type of gameplay is this?
        2. Where are the important HUD elements located?
        3. What's the main action area?
        4. How should we optimize this for vertical format?
        
        Provide your analysis in JSON format with these keys:
        - game_type
        - hud_elements (list of elements with positions)
        - main_action_area
        - vertical_optimization_strategy
        """
        
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            content_analysis = json.loads(response.choices[0].message.content)
            self.state['observations'].append({
                'type': 'content_analysis',
                'data': content_analysis
            })
            return content_analysis
        except json.JSONDecodeError:
            return {
                'game_type': 'unknown',
                'hud_elements': [],
                'main_action_area': 'center',
                'vertical_optimization_strategy': 'default'
            }

    def _default_content_analysis(self, technical_metadata: Dict) -> Dict:
        """Reliable default content analysis without LLM"""
        return {
            'game_type': 'fps',
            'hud_elements': [{
                'type': 'heli_hp',
                'position': {
                    'x': technical_metadata['width'] - 200,  # Right side
                    'y': technical_metadata['height'] - 100,  # Bottom
                    'width': 180,
                    'height': 80
                }
            }],
            'main_action_area': 'center',
            'vertical_optimization_strategy': 'center_crop'
        }

    def determine_processing_steps(self, metadata: Dict) -> List[Dict]:
        """Determine processing steps with reliable fallback"""
        if not self.use_llm:
            return self._default_processing_steps(metadata)

        try:
            # Existing LLM code...
            steps = self._get_llm_processing_steps(metadata)
            if not steps or not self._validate_processing_steps(steps):
                return self._default_processing_steps(metadata)
            return steps
        except Exception as e:
            print(f"LLM processing failed: {str(e)}, using default steps")
            return self._default_processing_steps(metadata)

    def _validate_processing_steps(self, steps: List[Dict]) -> bool:
        """Validate that processing steps contain required elements"""
        required_types = {'gameplay_processing', 'quality'}
        step_types = {step['type'] for step in steps}
        return required_types.issubset(step_types)

    def _get_llm_processing_steps(self, metadata: Dict) -> List[Dict]:
        """Get processing steps from LLM with proper error handling"""
        prompt = f"""
        Based on this video analysis:
        {json.dumps(metadata, indent=2)}
        
        And these observations:
        {json.dumps(self.state['observations'], indent=2)}
        
        Determine the optimal video processing steps.
        Consider:
        1. How to crop the main action area
        2. How to handle HUD elements
        3. What quality settings to use
        
        Provide your response as a JSON array of processing steps.
        Each step must have:
        - type: Either 'gameplay_processing' or 'quality'
        - params: Parameters for the processing
        """
        
        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            steps = json.loads(response.choices[0].message.content)
            if self._validate_processing_steps(steps):
                self.state['actions'].append({
                    'type': 'processing_steps',
                    'data': steps
                })
                return steps
            return self._default_processing_steps(metadata)
        except Exception:
            return self._default_processing_steps(metadata)

    def _default_processing_steps(self, metadata: Dict) -> List[Dict]:
        """Fallback processing steps if LLM fails"""
        return [
            {
                'type': 'gameplay_processing',
                'params': {
                    'mode': 'center_crop_with_hud',
                    'hud_elements': ['heli_hp']
                }
            },
            {
                'type': 'quality',
                'params': {
                    'video_bitrate': self._calculate_target_bitrate(
                        metadata['bit_rate'],
                        metadata['width'] * metadata['height'],
                        self.target_width * self.target_height
                    ),
                    'audio_bitrate': '192k',
                    'preset': 'slow'
                }
            }
        ]

    def generate_ffmpeg_command(self, input_path: str, output_path: str, steps: List[Dict]) -> ffmpeg.Stream:
        """Generate FFmpeg command based on processing steps"""
        # Get video metadata first
        probe = ffmpeg.probe(input_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        metadata = {
            'width': int(video_info['width']),
            'height': int(video_info['height'])
        }
        
        # Start with input stream
        stream = ffmpeg.input(input_path)
        video_stream = stream['v']
        audio_stream = stream['a']
        
        # Process each step
        for step in steps:
            if step['type'] == 'gameplay_processing':
                video_stream = self._apply_gameplay_processing(
                    video_stream, 
                    step['params'],
                    metadata  # Pass metadata directly
                )
        
        # Get quality settings
        quality_step = next(s for s in steps if s['type'] == 'quality')
        
        # Output with quality settings
        stream = ffmpeg.output(
            video_stream,
            audio_stream,
            output_path,
            vcodec='libx264',
            acodec='aac',
            video_bitrate=quality_step['params']['video_bitrate'],
            audio_bitrate=quality_step['params']['audio_bitrate'],
            preset=quality_step['params']['preset'],
            movflags='faststart',
            **{'y': None}
        )
        
        return stream

    def _calculate_target_bitrate(self, original_bitrate: int, original_pixels: int, target_pixels: int) -> str:
        """Calculate target bitrate based on resolution change"""
        if original_bitrate == 0:
            return self.min_bitrate
            
        # Scale bitrate based on resolution change, with a floor and ceiling
        ratio = target_pixels / original_pixels
        target_bitrate = int(original_bitrate * ratio)
        
        # Convert to Mbps and apply limits
        target_mbps = max(2, min(4, target_bitrate / 1_000_000))
        return f"{target_mbps}M"

    def _apply_gameplay_processing(self, video_stream, params: Dict, content_analysis: Dict) -> ffmpeg.Stream:
        """Apply processing based on content analysis"""
        # Get main action area
        action_area = content_analysis.get('main_action_area', 'center')
        
        # Get input dimensions from content analysis
        input_width = content_analysis.get('width', 1920)  # Default to 1920x1080 if not found
        input_height = content_analysis.get('height', 1080)
        
        # Calculate crop based on action area
        if action_area == 'center':
            crop_width = int((self.target_height * 9) / 16)
            crop_x = (input_width - crop_width) // 2
            video_stream = ffmpeg.filter(
                video_stream,
                'crop',
                crop_width,
                input_height,  # Use full height
                crop_x,
                0
            )
        
        # Scale to target size
        video_stream = ffmpeg.filter(
            video_stream,
            'scale',
            self.target_width,
            self.target_height
        )
        
        # Process HUD elements
        hud_elements = content_analysis.get('hud_elements', [])
        for element in hud_elements:
            if 'position' in element:
                # Extract and overlay HUD element
                hud_stream = self._extract_hud_element(
                    video_stream,
                    element['position']
                )
                video_stream = ffmpeg.filter(
                    [video_stream, hud_stream],
                    'overlay',
                    element['position']['x'],
                    element['position']['y']
                )
        
        return video_stream

    def _extract_hud_element(self, stream, position: Dict) -> ffmpeg.Stream:
        """Extract a HUD element from the video"""
        return ffmpeg.filter(
            stream,
            'crop',
            position['width'],
            position['height'],
            position['x'],
            position['y']
        )

    def evaluate_result(self, output_path: str) -> Dict:
        """
        Use LLM to evaluate the processed video
        """
        # Sample the output video
        self._sample_frames(output_path)
        
        prompt = f"""
        Evaluate the processed video:
        
        Original observations:
        {json.dumps(self.state['observations'], indent=2)}
        
        Actions taken:
        {json.dumps(self.state['actions'], indent=2)}
        
        Please evaluate:
        1. Is the main action clearly visible?
        2. Are HUD elements readable?
        3. What improvements could be made?
        
        Provide your evaluation in JSON format.
        """
        
        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            evaluation = json.loads(response.choices[0].message.content)
            self.state['observations'].append({
                'type': 'evaluation',
                'data': evaluation
            })
            return evaluation
        except json.JSONDecodeError:
            return {
                'success': True,
                'improvements': []
            }

    def process_video(self, input_path: str, output_path: str) -> Tuple[ffmpeg.Stream, List[Dict]]:
        """Process video using sequence of tool-based actions"""
        try:
            # Get video metadata
            metadata = self._get_video_metadata(input_path)
            
            # Create action sequence for main video transformation
            actions = [
                # 1. Center crop to 9:16
                VideoEditAction('center_crop', {
                    'input_width': metadata['width'],
                    'input_height': metadata['height'],
                    'target_ratio': self.target_aspect_ratio
                }),
                # 2. Scale to target size
                VideoEditAction('scale', {
                    'width': self.target_width,
                    'height': self.target_height
                })
            ]
            
            # Start with input stream
            stream = ffmpeg.input(input_path)
            video_stream = stream['v']
            audio_stream = stream['a']
            
            # Apply initial transformations (crop and scale)
            for action in actions[:2]:
                video_stream = action.execute(video_stream, self.tools)
            
            # Split the stream for main video and HUD processing
            split_streams = ffmpeg.filter_multi_output(video_stream, 'split', 2)
            main_stream = split_streams[0]
            hud_stream = split_streams[1]
            
            # Sample a frame to detect HUD elements
            cap = cv2.VideoCapture(input_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                elements = self._detect_hud_elements(frame)
                ground_unit_hud = next((e for e in elements if e.element_type == 'ground_unit_hud'), None)
                
                if ground_unit_hud:
                    # Extract the HUD from original position
                    hud_stream = self.tools.extract_region(
                        hud_stream,
                        ground_unit_hud.position['x'],
                        ground_unit_hud.position['y'],
                        ground_unit_hud.position['width'],
                        ground_unit_hud.position['height']
                    )
                    
                    # Calculate new position for vertical format
                    # Place at bottom-middle of the frame
                    new_x = (self.target_width - ground_unit_hud.position['width']) // 2
                    new_y = self.target_height - ground_unit_hud.position['height'] - 20  # 20px padding from bottom
                    
                    # Create a semi-transparent dark background using color protocol
                    color_stream = ffmpeg.input(
                        f"color=c=black:s={ground_unit_hud.position['width']}x{ground_unit_hud.position['height']}:r={metadata.get('r_frame_rate', '60')}:d=5",
                        f='lavfi'
                    )
                    # Make it semi-transparent
                    color_stream = ffmpeg.filter(color_stream, 'format', 'rgba')
                    color_stream = ffmpeg.filter(color_stream, 'colorchannelmixer', aa=0.5)
                    
                    # Overlay the black box first
                    main_stream = ffmpeg.filter(
                        [main_stream, color_stream],
                        'overlay',
                        new_x,
                        new_y
                    )
                    
                    # Then overlay the HUD element
                    main_stream = self.tools.overlay(
                        main_stream,
                        hud_stream,
                        new_x,
                        new_y
                    )
                    
                    print(f"\nRepositioned ground_unit_hud:")
                    print(f"Original position: {ground_unit_hud.position}")
                    print(f"New position: x={new_x}, y={new_y}")
            
            # Get quality settings
            quality_settings = self.tools.adjust_quality(None, self._calculate_target_bitrate(
                metadata['bit_rate'],
                metadata['width'] * metadata['height'],
                self.target_width * self.target_height
            ))
            
            # Create output
            stream = ffmpeg.output(
                main_stream,
                audio_stream,
                output_path,
                **quality_settings,
                **{'y': None}
            )
            
            # Store actions for reference
            self.actions = actions
            
            return stream, [{'action': a.tool_name, 'params': a.params} for a in actions]
            
        except Exception as e:
            print(f"Error in process_video: {str(e)}")
            raise
            
    def _get_video_metadata(self, input_path: str) -> Dict:
        """Get basic video metadata needed for processing"""
        probe = ffmpeg.probe(input_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        
        return {
            'width': int(video_info['width']),
            'height': int(video_info['height']),
            'bit_rate': int(probe['format'].get('bit_rate', 0)),
            'r_frame_rate': video_info.get('r_frame_rate', '60')
        } 