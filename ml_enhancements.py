import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, vfx
from moviepy.video.fx.all import speedx, freeze, colorx, fadein, fadeout, resize
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from PIL import Image

class MLVideoEnhancer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.style_model = None
        self.motion_model = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize ML models for different enhancements"""
        # Style transfer model (for color grading and effects)
        self.style_model = models.resnet50(pretrained=True)
        self.style_model = self.style_model.to(self.device)
        self.style_model.eval()
        
        # Motion estimation model (for transitions and speed ramping)
        self.motion_model = models.resnet18(pretrained=True)
        self.motion_model = self.motion_model.to(self.device)
        self.motion_model.eval()
        
    def apply_transition(self, video_path, transition_type='fade', duration=1.0):
        """Apply ML-based transitions between video segments"""
        video = VideoFileClip(video_path)
        
        if transition_type == 'fade':
            # Apply fade transition
            video = fadein(video, duration)
            video = fadeout(video, duration)
        elif transition_type == 'dissolve':
            # Apply dissolve transition
            def dissolve_effect(get_frame, t):
                frame = get_frame(t)
                if t < duration:
                    # Fade in
                    alpha = t / duration
                    return (frame * alpha).astype(np.uint8)
                elif t > video.duration - duration:
                    # Fade out
                    alpha = (video.duration - t) / duration
                    return (frame * alpha).astype(np.uint8)
                return frame

            video = video.fl(dissolve_effect)
        elif transition_type == 'wipe':
            # Apply wipe transition
            def wipe_effect(get_frame, t):
                frame = get_frame(t)
                h, w = frame.shape[:2]
                if t < duration:
                    # Wipe in from left
                    progress = t / duration
                    mask = np.zeros((h, w), dtype=np.uint8)
                    wipe_width = int(w * progress)
                    mask[:, :wipe_width] = 255
                    # Apply mask
                    result = frame.copy()
                    result[mask == 0] = 0
                    return result
                elif t > video.duration - duration:
                    # Wipe out to right
                    progress = (video.duration - t) / duration
                    mask = np.zeros((h, w), dtype=np.uint8)
                    wipe_width = int(w * progress)
                    mask[:, w-wipe_width:] = 255
                    # Apply mask
                    result = frame.copy()
                    result[mask == 0] = 0
                    return result
                return frame

            video = video.fl(wipe_effect)
            
        return video
        
    def _extract_frames(self, video):
        """Extract frames from video"""
        frames = []
        for frame in video.iter_frames():
            frames.append(frame)
        return frames
        
    def _generate_transition_frames(self, frames, duration):
        """Generate transition frames using ML-based interpolation"""
        # Implement frame interpolation using optical flow
        transition_frames = []
        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY),
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Generate intermediate frames
            for t in np.linspace(0, 1, int(duration * video.fps)):
                intermediate = self._interpolate_frame(frame1, frame2, flow, t)
                transition_frames.append(intermediate)
        
        return transition_frames
        
    def _interpolate_frame(self, frame1, frame2, flow, t):
        """Interpolate between two frames using optical flow"""
        h, w = frame1.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Apply flow to coordinates
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        
        # Interpolate coordinates
        coords_x = x + flow_x * t
        coords_y = y + flow_y * t
        
        # Remap frames
        frame1_warped = cv2.remap(frame1, coords_x, coords_y, cv2.INTER_LINEAR)
        frame2_warped = cv2.remap(frame2, coords_x - flow_x, coords_y - flow_y, cv2.INTER_LINEAR)
        
        # Blend frames
        return cv2.addWeighted(frame1_warped, 1-t, frame2_warped, t, 0)
        
    def apply_color_grading(self, video_path, style='cinematic'):
        """Apply ML-based color grading"""
        video = VideoFileClip(video_path)
        
        def color_grade_frame(frame):
            # Convert frame to PIL Image
            frame_pil = Image.fromarray(frame)
            
            # Apply ML-based color grading
            if style == 'cinematic':
                # Enhance contrast and saturation
                frame_enhanced = self._enhance_contrast(frame)
                frame_enhanced = self._enhance_saturation(frame_enhanced)
                # Add slight blue tint
                frame_enhanced = self._apply_color_tint(frame_enhanced, (0.95, 0.95, 1.05))
            elif style == 'vintage':
                # Apply vintage color grading
                frame_enhanced = self._apply_vintage_look(frame)
            elif style == 'warm':
                # Apply warm color grading
                frame_enhanced = self._apply_warm_look(frame)
            elif style == 'cool':
                # Apply cool color grading
                frame_enhanced = self._apply_cool_look(frame)
            elif style == 'noir':
                # Apply black and white with high contrast
                frame_enhanced = self._apply_noir_look(frame)
            elif style == 'vibrant':
                # Apply vibrant, saturated look
                frame_enhanced = self._apply_vibrant_look(frame)
            
            return frame_enhanced
        
        return video.fl_image(color_grade_frame)
        
    def _enhance_contrast(self, frame):
        """Enhance contrast using adaptive histogram equalization"""
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
    def _enhance_saturation(self, frame):
        """Enhance saturation"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 1.2)  # Increase saturation
        enhanced = cv2.merge((h, s, v))
        return cv2.cvtColor(enhanced, cv2.COLOR_HSV2RGB)
        
    def _apply_vintage_look(self, frame):
        """Apply vintage color grading"""
        # Convert to sepia tone
        sepia = cv2.transform(frame, np.array([[0.393, 0.769, 0.189],
                                             [0.349, 0.686, 0.168],
                                             [0.272, 0.534, 0.131]]))
        
        # Add vignette effect
        rows, cols = frame.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/4)
        kernel_y = cv2.getGaussianKernel(rows, rows/4)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        sepia = sepia * mask[..., np.newaxis]
        
        return sepia
        
    def _apply_warm_look(self, frame):
        """Apply warm color grading"""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Increase yellow/red tones
        b = cv2.add(b, 10)
        
        # Merge channels
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Add warm tint
        enhanced = self._apply_color_tint(enhanced, (1.1, 0.9, 0.9))
        
        return enhanced

    def _apply_cool_look(self, frame):
        """Apply cool color grading"""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Increase blue tones
        b = cv2.subtract(b, 10)
        
        # Merge channels
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Add cool tint
        enhanced = self._apply_color_tint(enhanced, (0.9, 0.9, 1.1))
        
        return enhanced

    def _apply_noir_look(self, frame):
        """Apply black and white with high contrast"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Apply high contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        # Add slight blue tint
        enhanced = self._apply_color_tint(enhanced, (0.95, 0.95, 1.05))
        
        return enhanced

    def _apply_vibrant_look(self, frame):
        """Apply vibrant, saturated look"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Increase saturation
        s = cv2.multiply(s, 1.5)
        
        # Increase value
        v = cv2.multiply(v, 1.2)
        
        # Merge channels
        enhanced = cv2.merge((h, s, v))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_HSV2RGB)
        
        # Enhance contrast
        enhanced = self._enhance_contrast(enhanced)
        
        return enhanced

    def _apply_color_tint(self, frame, tint_factors):
        """Apply color tint to frame"""
        # Split channels
        b, g, r = cv2.split(frame)
        
        # Apply tint factors
        b = cv2.multiply(b, tint_factors[0])
        g = cv2.multiply(g, tint_factors[1])
        r = cv2.multiply(r, tint_factors[2])
        
        # Merge channels
        return cv2.merge((b, g, r))
        
    def apply_speed_ramping(self, video_path, target_speed=1.5):
        """Apply ML-based speed ramping"""
        video = VideoFileClip(video_path)
        
        # Analyze motion in video
        motion_scores = self._analyze_motion(video)
        
        # Generate speed curve based on motion analysis
        speed_curve = self._generate_speed_curve(motion_scores, target_speed)
        
        # Create a list of subclips with different speeds
        clips = []
        duration = 0
        chunk_duration = 1.0  # Process in 1-second chunks
        
        while duration < video.duration:
            end_time = min(duration + chunk_duration, video.duration)
            chunk = video.subclip(duration, end_time)
            
            # Get the speed factor for this chunk
            idx = int(duration * len(speed_curve) / video.duration)
            speed_factor = float(speed_curve[idx])
            
            # Apply speed change to chunk
            speed_chunk = chunk.speedx(speed_factor)
            clips.append(speed_chunk)
            
            duration += chunk_duration
        
        # Concatenate all chunks
        final_clip = concatenate_videoclips(clips)
        return final_clip
        
    def _analyze_motion(self, video):
        """Analyze motion in video using ML"""
        motion_scores = []
        prev_frame = None
        
        # Sample frames at regular intervals
        for t in np.arange(0, video.duration, 1/video.fps):
            frame = video.get_frame(t)
            
            if prev_frame is not None:
                # Calculate motion between frames
                diff = np.abs(frame - prev_frame).mean()
                motion_scores.append(float(diff))
            else:
                motion_scores.append(0.0)
            
            prev_frame = frame
        
        # Convert to numpy array and normalize
        scores = np.array(motion_scores)
        if len(scores) > 0:
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        
        return scores
        
    def _generate_speed_curve(self, motion_scores, target_speed):
        """Generate speed curve based on motion analysis"""
        if len(motion_scores) == 0:
            return np.array([target_speed])
            
        # Ensure we have at least one valid motion score
        if np.all(np.isnan(motion_scores)):
            return np.array([target_speed])
            
        # Replace any remaining NaN values with 0
        motion_scores = np.nan_to_num(motion_scores, 0.0)
        
        # Generate base speed curve
        base_curve = 1.0 + (target_speed - 1.0) * motion_scores
        
        # Smooth the curve using a Savitzky-Golay filter instead of Gaussian
        window_size = min(len(base_curve), 11)  # Must be odd
        if window_size % 2 == 0:
            window_size -= 1
        if window_size >= 3:  # Need at least 3 points for Savitzky-Golay
            smoothed_curve = signal.savgol_filter(base_curve, window_size, 2)
        else:
            smoothed_curve = base_curve
        
        # Ensure minimum speed is not too slow
        min_speed = 0.5
        smoothed_curve = np.maximum(smoothed_curve, min_speed)
        
        return smoothed_curve
        
    def apply_effects(self, video_path, effect_type='freeze'):
        """Apply ML-based video effects"""
        video = VideoFileClip(video_path)
        
        if effect_type == 'freeze':
            return self._apply_freeze_frame(video)
        elif effect_type == 'motion_blur':
            return self._apply_motion_blur(video)
        elif effect_type == 'gaussian_blur':
            return self._apply_gaussian_blur(video)
        elif effect_type == 'sepia':
            return self._apply_sepia_effect(video)
        elif effect_type == 'negative':
            return self._apply_negative_effect(video)
        elif effect_type == 'mirror':
            return self._apply_mirror_effect(video)
        elif effect_type == 'pixelate':
            return self._apply_pixelate_effect(video)
        elif effect_type == 'edge_detection':
            return self._apply_edge_detection(video)
            
    def _apply_freeze_frame(self, video):
        """Apply ML-optimized freeze frame effect"""
        # Analyze video for optimal freeze points
        frames = self._analyze_frames(video)
        freeze_points = self._detect_freeze_points(frames)
        
        # Apply freeze effect
        clips = []
        for point in freeze_points:
            clip = video.subclip(point, point + 1)
            frozen = freeze(clip, t=point)
            clips.append(frozen)
            
        return concatenate_videoclips(clips)
        
    def _analyze_frames(self, video):
        """Analyze video frames using ML"""
        frames = []
        for frame in video.iter_frames():
            # Convert frame to tensor
            frame_tensor = torch.from_numpy(frame).float().to(self.device)
            frames.append(frame_tensor)
        return frames
        
    def _detect_fade_points(self, frames):
        """Detect optimal fade points using ML"""
        fade_points = []
        for i in range(len(frames) - 1):
            # Compare consecutive frames
            diff = torch.abs(frames[i] - frames[i + 1]).mean()
            if diff < 0.1:  # Threshold for fade detection
                fade_points.append(i)
        return fade_points
        
    def _detect_freeze_points(self, frames):
        """Detect optimal freeze points using ML"""
        freeze_points = []
        for i in range(len(frames) - 1):
            # Compare consecutive frames
            diff = torch.abs(frames[i] - frames[i + 1]).mean()
            if diff < 0.05:  # Threshold for freeze detection
                freeze_points.append(i)
        return freeze_points
        
    def _calculate_speed_factors(self, motion_scores, target_speed):
        """Calculate speed factors based on motion analysis"""
        # Normalize motion scores
        scores = np.array(motion_scores)
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        # Calculate speed factors
        factors = 1 + (target_speed - 1) * scores
        return factors
        
    def _apply_motion_blur(self, video):
        """Apply motion blur effect"""
        def blur_frame(frame):
            # Calculate motion blur kernel
            kernel_size = 15
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            
            # Apply motion blur
            blurred = cv2.filter2D(frame, -1, kernel)
            return blurred
        
        return video.fl_image(blur_frame)
        
    def _frames_to_video(self, frames, fps):
        """Convert frames to video"""
        return VideoFileClip(frames, fps=fps)

    def apply_animation(self, video_path, animation_type='zoom', duration=2.0):
        """Apply animation effects to video"""
        video = VideoFileClip(video_path)
        
        if animation_type == 'zoom':
            return self._apply_zoom_animation(video, duration)
        elif animation_type == 'rotate':
            return self._apply_rotation_animation(video, duration)
        elif animation_type == 'slide':
            return self._apply_slide_animation(video, duration)
        elif animation_type == 'fade':
            return self._apply_fade_animation(video, duration)
        elif animation_type == 'bounce':
            return self._apply_bounce_animation(video, duration)
        else:
            return video

    def _apply_zoom_animation(self, video, duration):
        """Apply zoom in/out animation"""
        def zoom_effect(get_frame, t):
            frame = get_frame(t)
            # Create a zoom effect that goes from 1.0 to 1.5 and back
            zoom_factor = 1.0 + 0.5 * np.sin(t * np.pi / float(duration))
            # Calculate new dimensions
            h, w = frame.shape[:2]
            new_h = int(h * zoom_factor)
            new_w = int(w * zoom_factor)
            # Resize frame
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # Calculate center crop
            start_h = max(0, (new_h - h) // 2)
            start_w = max(0, (new_w - w) // 2)
            # Ensure we don't exceed the resized frame dimensions
            end_h = min(new_h, start_h + h)
            end_w = min(new_w, start_w + w)
            # Get the cropped region
            cropped = resized[start_h:end_h, start_w:end_w]
            # If the cropped region is smaller than the original frame, pad it
            if cropped.shape[:2] != (h, w):
                padded = np.zeros((h, w, 3), dtype=np.uint8)
                padded[:cropped.shape[0], :cropped.shape[1]] = cropped
                return padded
            return cropped

        return video.fl(zoom_effect)

    def _apply_rotation_animation(self, video, duration):
        """Apply rotation animation"""
        def rotate_effect(get_frame, t):
            frame = get_frame(t)
            # Rotate video from 0 to 360 degrees
            angle = (t / float(duration)) * 360.0
            # Get image dimensions
            h, w = frame.shape[:2]
            # Calculate the center of the image
            center = (w // 2, h // 2)
            # Get the rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            # Apply the rotation
            rotated = cv2.warpAffine(frame, rotation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            return rotated

        return video.fl(rotate_effect)

    def _apply_slide_animation(self, video, duration):
        """Apply sliding animation"""
        def slide_effect(get_frame, t):
            frame = get_frame(t)
            # Create sliding effect from left to right
            progress = t / float(duration)
            offset = int((1.0 - progress) * float(video.w))
            # Create translation matrix
            translation_matrix = np.float32([[1, 0, offset], [0, 1, 0]])
            # Apply translation with border handling
            result = cv2.warpAffine(frame, translation_matrix, (frame.shape[1], frame.shape[0]), 
                                  borderMode=cv2.BORDER_REPLICATE)
            return result

        return video.fl(slide_effect)

    def _apply_fade_animation(self, video, duration):
        """Apply fade in/out animation"""
        def fade_effect(get_frame, t):
            frame = get_frame(t)
            # Calculate fade factor based on time
            if t < duration/2:  # Fade in
                fade_factor = t / (duration/2)
            else:  # Fade out
                fade_factor = 1 - (t - duration/2) / (duration/2)
            # Ensure fade factor is between 0 and 1
            fade_factor = max(0, min(1, fade_factor))
            # Apply fade effect
            result = (frame * fade_factor).astype(np.uint8)
            return result

        return video.fl(fade_effect)

    def _apply_bounce_animation(self, video, duration):
        """Apply bouncing animation"""
        def bounce_effect(get_frame, t):
            frame = get_frame(t)
            # Create a bouncing effect using sine wave
            offset = int(50.0 * abs(np.sin(t * np.pi * 2.0 / float(duration))))
            # Create translation matrix
            translation_matrix = np.float32([[1, 0, 0], [0, 1, offset]])
            # Apply translation with border handling
            result = cv2.warpAffine(frame, translation_matrix, (frame.shape[1], frame.shape[0]), 
                                  borderMode=cv2.BORDER_REPLICATE)
            return result

        return video.fl(bounce_effect)

    def _rotate_frame(self, frame, angle):
        """Rotate a frame by given angle"""
        height, width = frame.shape[:2]
        center = (width/2, height/2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(frame, rotation_matrix, (width, height))

    def _slide_frame(self, frame, offset):
        """Slide frame horizontally"""
        height, width = frame.shape[:2]
        translation_matrix = np.float32([[1, 0, offset], [0, 1, 0]])
        return cv2.warpAffine(frame, translation_matrix, (width, height))

    def _bounce_frame(self, frame, offset):
        """Apply bounce effect to frame"""
        height, width = frame.shape[:2]
        translation_matrix = np.float32([[1, 0, 0], [0, 1, offset]])
        return cv2.warpAffine(frame, translation_matrix, (width, height))

    def _apply_gaussian_blur(self, video):
        """Apply Gaussian blur effect"""
        def blur_frame(frame):
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(frame, (15, 15), 0)
            return blurred
        
        return video.fl_image(blur_frame)
        
    def _apply_sepia_effect(self, video):
        """Apply sepia effect"""
        def sepia_frame(frame):
            # Convert to sepia tone
            sepia = cv2.transform(frame, np.array([[0.393, 0.769, 0.189],
                                                 [0.349, 0.686, 0.168],
                                                 [0.272, 0.534, 0.131]]))
            return sepia
        
        return video.fl_image(sepia_frame)
        
    def _apply_negative_effect(self, video):
        """Apply negative effect"""
        def negative_frame(frame):
            # Invert colors
            return 255 - frame
        
        return video.fl_image(negative_frame)
        
    def _apply_mirror_effect(self, video):
        """Apply mirror effect"""
        def mirror_frame(frame):
            # Flip frame horizontally
            return cv2.flip(frame, 1)
        
        return video.fl_image(mirror_frame)
        
    def _apply_pixelate_effect(self, video):
        """Apply pixelation effect"""
        def pixelate_frame(frame):
            # Get frame dimensions
            h, w = frame.shape[:2]
            # Calculate new dimensions (smaller)
            small_h, small_w = h//8, w//8
            # Resize down
            temp = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            # Resize up
            return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return video.fl_image(pixelate_frame)
        
    def _apply_edge_detection(self, video):
        """Apply edge detection effect"""
        def edge_frame(frame):
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 100, 200)
            # Convert back to RGB
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return video.fl_image(edge_frame) 