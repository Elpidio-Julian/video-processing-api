import ffmpeg
import json
from typing import Dict, List, Optional, Tuple
import math

class VideoEditAgent:
    """
    A ReAct (Reasoning and Acting) agent for video editing.
    Uses a systematic approach to:
    1. Analyze input video
    2. Determine optimal processing parameters
    3. Generate FFmpeg commands
    4. Handle various aspect ratios and quality requirements
    """

    def __init__(self):
        self.target_aspect_ratio = (9, 16)  # Vertical video
        self.target_width = 1080
        self.target_height = 1920
        self.min_bitrate = "2M"
        self.max_bitrate = "4M"
        
        # HUD element regions (in relative coordinates 0-1)
        self.hud_regions = {
            'heli_hp': {
                'x': 0.85,  # Right side
                'y': 0.85,  # Bottom
                'width': 0.15,
                'height': 0.1,
                'overlay_x': 0.75,  # Where to place in vertical format
                'overlay_y': 0.85
            }
        }

    def analyze_video(self, input_path: str) -> Dict:
        """
        Analyze input video and return relevant metadata
        """
        try:
            # Get video metadata using ffprobe
            probe = ffmpeg.probe(input_path)
            
            # Find video stream
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            # Extract relevant information
            width = int(video_info['width'])
            height = int(video_info['height'])
            
            # Calculate current aspect ratio
            gcd = math.gcd(width, height)
            current_aspect = (width // gcd, height // gcd)
            
            # Get duration, frame rate, and other metadata
            duration = float(probe['format'].get('duration', 0))
            frame_rate = eval(video_info.get('r_frame_rate', '30/1'))
            if isinstance(frame_rate, str):
                frame_rate = 30  # Default if we can't parse
                
            # Get bitrate
            bit_rate = int(probe['format'].get('bit_rate', 0))
            
            return {
                'width': width,
                'height': height,
                'aspect_ratio': current_aspect,
                'duration': duration,
                'frame_rate': frame_rate,
                'bit_rate': bit_rate,
                'codec_name': video_info.get('codec_name', 'unknown'),
                'is_vertical': height > width,
                'rotation': int(video_info.get('rotation', 0)) if 'rotation' in video_info else 0
            }
        except Exception as e:
            raise Exception(f"Failed to analyze video: {str(e)}")

    def determine_processing_steps(self, metadata: Dict) -> List[Dict]:
        """
        Determine necessary processing steps based on video analysis
        """
        steps = []
        
        # Add center crop and HUD overlay step
        steps.append({
            'type': 'gameplay_processing',
            'params': {
                'mode': 'center_crop_with_hud',
                'hud_elements': ['heli_hp']
            }
        })
        
        # Add quality settings
        target_bitrate = self._calculate_target_bitrate(
            metadata['bit_rate'],
            metadata['width'] * metadata['height'],
            self.target_width * self.target_height
        )
        
        steps.append({
            'type': 'quality',
            'params': {
                'video_bitrate': target_bitrate,
                'audio_bitrate': '192k',
                'preset': 'slow'  # High quality encoding
            }
        })

        return steps

    def generate_ffmpeg_command(self, input_path: str, output_path: str, steps: List[Dict]) -> ffmpeg.Stream:
        """
        Generate FFmpeg command with advanced filtering for gameplay footage
        """
        # Start with input stream
        stream = ffmpeg.input(input_path)
        
        # Get video metadata
        metadata = self.analyze_video(input_path)
        original_width = metadata['width']
        original_height = metadata['height']
        
        # Split audio and video streams
        video_stream = stream['v']
        audio_stream = stream['a']
        
        # Calculate center crop dimensions
        # We want to maintain height and crop width for 9:16
        crop_width = int((original_height * 9) / 16)
        crop_x = (original_width - crop_width) // 2
        
        # 1. Create main center-cropped stream
        center_stream = ffmpeg.filter(
            video_stream,
            'crop',
            crop_width,
            original_height,
            crop_x,
            0
        )
        
        # Scale to target dimensions
        center_stream = ffmpeg.filter(
            center_stream,
            'scale',
            self.target_width,
            self.target_height
        )
        
        # 2. Create HUD element stream (heli HP)
        heli_hp_region = self.hud_regions['heli_hp']
        hud_x = int(original_width * heli_hp_region['x'])
        hud_y = int(original_height * heli_hp_region['y'])
        hud_width = int(original_width * heli_hp_region['width'])
        hud_height = int(original_height * heli_hp_region['height'])
        
        hud_stream = ffmpeg.filter(
            video_stream,
            'crop',
            hud_width,
            hud_height,
            hud_x,
            hud_y
        )
        
        # Scale HUD element to appropriate size for overlay
        overlay_width = int(self.target_width * 0.25)  # 25% of width
        overlay_height = int((hud_height * overlay_width) / hud_width)
        
        hud_stream = ffmpeg.filter(
            hud_stream,
            'scale',
            overlay_width,
            overlay_height
        )
        
        # Calculate overlay position
        overlay_x = int(self.target_width * heli_hp_region['overlay_x']) - overlay_width
        overlay_y = int(self.target_height * heli_hp_region['overlay_y']) - overlay_height
        
        # 3. Overlay HUD on main stream
        final_video_stream = ffmpeg.filter(
            [center_stream, hud_stream],
            'overlay',
            overlay_x,
            overlay_y
        )
        
        # Get quality settings
        quality_step = next(s for s in steps if s['type'] == 'quality')
        
        # Output with quality settings and both video and audio streams
        stream = ffmpeg.output(
            final_video_stream,  # Processed video
            audio_stream,        # Original audio
            output_path,
            vcodec='libx264',
            acodec='aac',
            video_bitrate=quality_step['params']['video_bitrate'],
            audio_bitrate=quality_step['params']['audio_bitrate'],
            preset=quality_step['params']['preset'],
            movflags='faststart',  # Enable streaming
            **{'y': None}  # Force overwrite output file
        )

        return stream

    def _calculate_scaling(self, width: int, height: int) -> Dict:
        """
        Calculate scaling parameters to fit target aspect ratio
        """
        # Calculate scale factor to fit within target dimensions
        scale_w = self.target_width / width
        scale_h = self.target_height / height
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Calculate padding
        pad_x = (self.target_width - new_width) // 2
        pad_y = (self.target_height - new_height) // 2
        
        return {
            'width': new_width,
            'height': new_height,
            'pad_x': pad_x,
            'pad_y': pad_y
        }

    def _calculate_target_bitrate(self, original_bitrate: int, original_pixels: int, target_pixels: int) -> str:
        """
        Calculate target bitrate based on resolution change and quality requirements
        """
        if original_bitrate == 0:
            # Default to 4Mbps for 1080x1920 if we can't determine original
            return self.max_bitrate
            
        # Scale bitrate based on resolution change
        scale_factor = target_pixels / original_pixels
        target_bitrate = int(original_bitrate * scale_factor)
        
        # Convert to Mbps and clamp between min and max
        target_mbps = max(2, min(4, target_bitrate / 1_000_000))
        
        return f"{target_mbps}M"

    def process_video(self, input_path: str, output_path: str) -> Tuple[ffmpeg.Stream, List[Dict]]:
        """
        Main method to process a video file
        Returns the FFmpeg command and processing steps
        """
        # 1. Analyze video
        metadata = self.analyze_video(input_path)
        
        # 2. Determine processing steps
        steps = self.determine_processing_steps(metadata)
        
        # 3. Generate FFmpeg command
        command = self.generate_ffmpeg_command(input_path, output_path, steps)
        
        return command, steps 