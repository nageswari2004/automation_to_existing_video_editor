from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session, flash
import moviepy.editor as mp
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from functools import wraps
from ml_processor import MLVideoProcessor
from ml_enhancements import MLVideoEnhancer
from flask_sqlalchemy import SQLAlchemy
import time
import subprocess
import sys
import moviepy.config as mp_config
import json
import requests
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import numpy as np
import yt_dlp
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from bs4 import BeautifulSoup
import re
import gc  # Add this at the top with other imports
from concurrent.futures import ThreadPoolExecutor
import cv2

# Load environment variables
load_dotenv()

# Define ImageMagick path
IMAGEMAGICK_PATH = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"

# Add this after other API configurations
PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Set session lifetime to 7 days
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Create upload and output folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.session_protection = 'strong'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, username, email, password_hash):
        self.username = username
        self.email = email
        self.password_hash = password_hash

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.before_request
def before_request():
    # Debug logging for session and authentication
    print(f"[DEBUG] before_request: path={request.path}, method={request.method}")
    print(f"[DEBUG] session: {dict(session)}")
    print(f"[DEBUG] current_user.is_authenticated: {getattr(current_user, 'is_authenticated', None)}")
    print(f"[DEBUG] request.headers: {dict(request.headers)}")
    if current_user.is_authenticated:
        session.permanent = True  # Make session permanent for authenticated users
        session.modified = True  # Update session timestamp

@app.after_request
def after_request(response):
    if current_user.is_authenticated:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# Create database tables
with app.app_context():
    db.create_all()

# Setup ImageMagick
def setup_imagemagick():
    """Setup ImageMagick configuration with detailed error reporting."""
    print("Setting up ImageMagick configuration...")
    
    # First try the explicit path
    if os.path.exists(IMAGEMAGICK_PATH):
        print(f"Found ImageMagick at explicit path: {IMAGEMAGICK_PATH}")
        mp_config.change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGICK_PATH})
        return True
    
    # Try to find ImageMagick in PATH
    try:
        result = subprocess.run(['where', 'magick'], capture_output=True, text=True)
        if result.returncode == 0:
            magick_path = result.stdout.strip().split('\n')[0]
            print(f"Found ImageMagick in PATH: {magick_path}")
            mp_config.change_settings({"IMAGEMAGICK_BINARY": magick_path})
            return True
    except Exception as e:
        print(f"Error checking PATH for ImageMagick: {str(e)}")
    
    # Common installation paths
    common_paths = [
        r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
        r"C:\Program Files\ImageMagick-7.1.0-Q16-HDRI\magick.exe",
        r"C:\Program Files\ImageMagick-7.0.11-Q16-HDRI\magick.exe",
        r"C:\Program Files (x86)\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
        r"C:\Program Files (x86)\ImageMagick-7.1.0-Q16-HDRI\magick.exe",
        r"C:\Program Files (x86)\ImageMagick-7.0.11-Q16-HDRI\magick.exe",
        r"C:\Program Files\ImageMagick\magick.exe",
        r"C:\Program Files (x86)\ImageMagick\magick.exe"
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            print(f"Found ImageMagick at: {path}")
            mp_config.change_settings({"IMAGEMAGICK_BINARY": path})
            return True
    
    print("WARNING: ImageMagick not found in common locations or PATH")
    return False

# Setup ImageMagick at startup
imagemagick_configured = setup_imagemagick()

# Initialize ML models
ml_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create database tables
with app.app_context():
    db.create_all()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_video_path(filename):
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)

def get_output_path(filename):
    return os.path.join(app.config['OUTPUT_FOLDER'], filename)

# Test route
@app.route('/test')
def test():
    return "Flask is working!"

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=True)  # Enable remember me
            session.permanent = True  # Make session permanent
            session['user_id'] = user.id
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('editor'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        print(f"Registration attempt - Username: {username}, Email: {email}")  # Debug print
        
        # Check if username already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return render_template('register.html')
            
        # Check if email already exists
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
            
        try:
            user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password)
            )
            db.session.add(user)
            db.session.commit()
            print(f"Registration successful for user: {username}")  # Debug print
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            print(f"Registration error: {str(e)}")  # Debug print
            db.session.rollback()
            flash('An error occurred during registration. Please try again.', 'danger')
            return render_template('register.html')
        
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/editor')
@login_required
def editor():
    print(f"Accessing editor page - User: {current_user.username if current_user.is_authenticated else 'Not authenticated'}")  # Debug print
    return render_template('editor.html')

@app.route('/multi_video_editor')
@login_required
def multi_video_editor():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return render_template('multi_video_editor.html')

@app.route('/output/<filename>')
@login_required
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/trim', methods=['POST'])
@login_required
def trim_video():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        start_time = float(request.form.get('start_time', 0))
        end_time = float(request.form.get('end_time', 0))
        
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Load the video
        video = mp.VideoFileClip(input_path)
        
        # Trim the video
        trimmed_video = video.subclip(start_time, end_time)
        
        # Generate output filename
        output_filename = f'trimmed_{int(time.time())}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Write the trimmed video
        trimmed_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            bitrate='8000k',
            fps=30,
            preset='slow',
            threads=4,
            ffmpeg_params=[
                '-crf', '18',
                '-profile:v', 'high',
                '-level', '4.0',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p'
            ]
        )
        
        # Close the video
        video.close()
        trimmed_video.close()
        
        # Clean up input file
        os.remove(input_path)
        
        return jsonify({
            'success': True,
            'output_file': output_filename
        })
        
    except Exception as e:
        print(f"Error in trim_video: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

def cleanup_files(file_paths, max_retries=5, initial_delay=2):
    """Clean up files with retries and exponential backoff."""
    for path in file_paths:
        if os.path.exists(path):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    # Force garbage collection before each attempt
                    gc.collect()
                    time.sleep(delay) # Add a delay before attempting deletion
                    os.remove(path)
                    print(f"Successfully deleted file: {path}")
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed to delete {path}: {str(e)}")
                    if attempt < max_retries - 1:
                        delay *= 2  # Exponential backoff
                        print(f"Waiting {delay} seconds before next attempt...")
                    else:
                        print(f"Warning: Could not delete file {path} after {max_retries} attempts")

def cleanup_videos(video_objects, delay=2):
    """Clean up video objects safely with delay."""
    for video in video_objects:
        try:
            if video is not None:
                video.close()
                time.sleep(delay)  # Wait after closing each video
        except Exception as e:
            print(f"Warning: Error closing video object: {str(e)}")

@app.route('/merge', methods=['POST'])
@login_required
def merge_videos():
    # Check if user is authenticated
    if not current_user.is_authenticated:
        return jsonify({
            'success': False,
            'error': 'Authentication required',
            'redirect': url_for('login')
        }), 401

    videos = []
    input_paths = []
    resized_videos = []
    final_video = None
    
    try:
        if 'files[]' not in request.files:
            return jsonify({'success': False, 'error': 'No files uploaded'})
        
        files = request.files.getlist('files[]')
        if not files:
            return jsonify({'success': False, 'error': 'No files selected'})
        
        # Save uploaded files
        for file in files:
            if file:
                filename = secure_filename(file.filename)
                input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(input_path)
                input_paths.append(input_path)
                time.sleep(1)  # Wait after saving each file
                
        # Load all videos
        try:
            for path in input_paths:
                video = mp.VideoFileClip(path)
                videos.append(video)
                time.sleep(1)  # Wait after loading each video
        except Exception as e:
            cleanup_videos(videos)
            cleanup_files(input_paths)
            return jsonify({'success': False, 'error': f'Error loading videos: {str(e)}'})
        
        try:
            # Get the highest resolution
            target_width = max(v.w for v in videos)
            target_height = max(v.h for v in videos)
            
            # Ensure dimensions are even
            target_width = target_width - (target_width % 2)
            target_height = target_height - (target_height % 2)
        
            # Resize videos to match the highest resolution
            for video in videos:
                if video.w != target_width or video.h != target_height:
                    resized_video = video.resize(width=target_width, height=target_height)
                    resized_videos.append(resized_video)
                else:
                    resized_videos.append(video)
                time.sleep(1)  # Wait after each resize operation
            
            # Concatenate videos
            final_video = mp.concatenate_videoclips(resized_videos)
            
            # Generate output filename
            output_filename = f'merged_{int(time.time())}.mp4'
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            # Write the final video
            final_video.write_videofile(
                output_path,
                codec='libx264', 
                audio_codec='aac',
                bitrate='8000k',
                fps=30,
                preset='slow',
                threads=4,
                ffmpeg_params=[
                    '-crf', '18',
                    '-profile:v', 'high',
                    '-level', '4.0',
                    '-movflags', '+faststart',
                    '-pix_fmt', 'yuv420p'
                ]
            )
            
            # Wait before cleanup
            time.sleep(3)
            
            # Clean up video objects
            cleanup_videos(videos)
            cleanup_videos(resized_videos)
            if final_video:
                cleanup_videos([final_video])
            
            # Force garbage collection
            gc.collect()
            time.sleep(2)
            
            # Clean up input files
            cleanup_files(input_paths)
            
            return jsonify({
                'success': True,
                'output_file': output_filename
            })
            
        except Exception as e:
            # Clean up on error
            cleanup_videos(videos)
            cleanup_videos(resized_videos)
            if final_video:
                cleanup_videos([final_video])
            cleanup_files(input_paths)
            return jsonify({'success': False, 'error': f'Error processing videos: {str(e)}'})
            
    except Exception as e:
        # Clean up on unexpected error
        cleanup_videos(videos)
        cleanup_videos(resized_videos)
        if final_video:
            cleanup_videos([final_video])
        cleanup_files(input_paths)
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})

@app.route('/extract-audio', methods=['POST'])
@login_required
def extract_audio():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Load the video
        video = mp.VideoFileClip(input_path)
        
        # Extract audio
        audio = video.audio
        
        # Generate output filename
        output_filename = f'audio_{int(time.time())}.mp3'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Write the audio
        audio.write_audiofile(output_path)
        
        # Close the video and audio
        video.close()
        audio.close()
        
        # Clean up input file
        os.remove(input_path)
        
        return jsonify({
            'success': True,
            'output_file': output_filename
        })
        
    except Exception as e:
        print(f"Error in extract_audio: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/change-speed', methods=['POST'])
@login_required
def change_speed():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        speed_factor = float(request.form.get('speed_factor', 1.0))
        
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'})
            
        # Save the uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Load the video
        video = mp.VideoFileClip(input_path)
        
        # Change speed
        speeded_video = video.speedx(speed_factor)
        
        # Generate output filename
        output_filename = f'speed_{int(time.time())}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Write the speeded video
        speeded_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            bitrate='8000k',
            fps=30,
            preset='slow',
            threads=4,
            ffmpeg_params=[
                '-crf', '18',
                '-profile:v', 'high',
                '-level', '4.0',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p'
            ]
        )
        
        # Close the video
        video.close()
        speeded_video.close()
        
        # Clean up input file
        os.remove(input_path)
        
        return jsonify({
            'success': True,
            'output_file': output_filename
        })
        
    except Exception as e:
        print(f"Error in change_speed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/resize', methods=['POST'])
@login_required
def resize_video():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        width = int(request.form.get('width', 0))
        height = int(request.form.get('height', 0))
        
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'})
            
        if width <= 0 or height <= 0:
            return jsonify({'success': False, 'error': 'Invalid dimensions'})
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Load the video
        video = mp.VideoFileClip(input_path)
        
        # Resize the video
        resized_video = video.resize(width=width, height=height)
        
        # Generate output filename
        output_filename = f'resized_{int(time.time())}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Write the resized video
        resized_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            bitrate='8000k',
            fps=30,
            preset='slow',
            threads=4,
            ffmpeg_params=[
                '-crf', '18',
                '-profile:v', 'high',
                '-level', '4.0',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p'
            ]
        )
        
        # Close the video
        video.close()
        resized_video.close()
        
        # Clean up input file
        os.remove(input_path)
        
        return jsonify({
            'success': True,
            'output_file': output_filename
        })
        
    except Exception as e:
        print(f"Error in resize_video: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/apply-transition', methods=['POST'])
@login_required
def apply_transition():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        transition_type = request.form.get('transition_type', 'fade')
        duration = float(request.form.get('duration', 1.0))
        
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Load the video
        video = mp.VideoFileClip(input_path)
        
        # Apply transition
        if transition_type == 'fade':
            transitioned_video = video.fadein(duration).fadeout(duration)
        elif transition_type == 'dissolve':
            transitioned_video = video.crossfadein(duration)
        else:
            transitioned_video = video
            
        # Generate output filename
        output_filename = f'transition_{int(time.time())}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Write the transitioned video
        transitioned_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            bitrate='8000k',
            fps=30,
            preset='slow',
            threads=4,
            ffmpeg_params=[
                '-crf', '18',
                '-profile:v', 'high',
                '-level', '4.0',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p'
            ]
        )
        
        # Close the video
        video.close()
        transitioned_video.close()
        
        # Clean up input file
        os.remove(input_path)
        
        return jsonify({
            'success': True,
            'output_file': output_filename
        })
        
    except Exception as e:
        print(f"Error in apply_transition: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/apply-color-grading', methods=['POST'])
@login_required
def apply_color_grading():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        color_style = request.form.get('color_style', 'cinematic')
        
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Load the video
        video = mp.VideoFileClip(input_path)
        
        # Apply color grading
        if color_style == 'cinematic':
            graded_video = video.fx(mp.vfx.colorx, 1.2)
        elif color_style == 'vintage':
            graded_video = video.fx(mp.vfx.colorx, 0.8)
        elif color_style == 'warm':
            graded_video = video.fx(mp.vfx.colorx, 1.1)
        elif color_style == 'cool':
            graded_video = video.fx(mp.vfx.colorx, 0.9)
        elif color_style == 'noir':
            graded_video = video.fx(mp.vfx.blackwhite)
        else:
            graded_video = video
            
        # Generate output filename
        output_filename = f'color_graded_{int(time.time())}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Write the graded video
        graded_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            bitrate='8000k',
            fps=30,
            preset='slow',
            threads=4,
            ffmpeg_params=[
                '-crf', '18',
                '-profile:v', 'high',
                '-level', '4.0',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p'
            ]
        )
        
        # Close the video
        video.close()
        graded_video.close()
        
        # Clean up input file
        os.remove(input_path)
        
        return jsonify({
            'success': True,
            'output_file': output_filename
        })
        
    except Exception as e:
        print(f"Error in apply_color_grading: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/apply-speed-ramping', methods=['POST'])
@login_required
def apply_speed_ramping():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        target_speed = float(request.form.get('target_speed', 1.5))
        
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Load the video
        video = mp.VideoFileClip(input_path)
        
        # Apply speed ramping
        duration = video.duration
        half_duration = duration / 2
        
        def speed_func(t):
            if t < half_duration:
                return 1 + (target_speed - 1) * (t / half_duration)
            else:
                return target_speed - (target_speed - 1) * ((t - half_duration) / half_duration)
                
        ramped_video = video.fl_time(speed_func)
        ramped_video = ramped_video.set_duration(duration)
        
        # Generate output filename
        output_filename = f'speed_ramped_{int(time.time())}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Write the ramped video
        ramped_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            bitrate='8000k',
            fps=30,
            preset='slow',
            threads=4,
            ffmpeg_params=[
                '-crf', '18',
                '-profile:v', 'high',
                '-level', '4.0',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p'
            ]
        )
        
        # Close the video
        video.close()
        ramped_video.close()
        
        # Clean up input file
        os.remove(input_path)
        
        return jsonify({
            'success': True,
            'output_file': output_filename
        })
        
    except Exception as e:
        print(f"Error in apply_speed_ramping: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/apply-effects', methods=['POST'])
@login_required
def apply_effects():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        effect_type = request.form.get('effect_type', 'freeze')
        
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Load the video
        video = mp.VideoFileClip(input_path)
        
        # Apply effect
        if effect_type == 'freeze':
            # Freeze frame at the middle
            middle_frame = video.get_frame(video.duration / 2)
            freeze_clip = mp.ImageClip(middle_frame).set_duration(2)
            effected_video = mp.concatenate_videoclips([
                video.subclip(0, video.duration / 2),
                freeze_clip,
                video.subclip(video.duration / 2)
            ])
        elif effect_type == 'motion_blur':
            effected_video = video.fx(mp.vfx.motion_blur, 2)
        elif effect_type == 'gaussian_blur':
            effected_video = video.fx(mp.vfx.gaussian_blur, sigma=2)
        elif effect_type == 'sepia':
            effected_video = video.fx(mp.vfx.colorx, 0.8)
        elif effect_type == 'negative':
            effected_video = video.fx(mp.vfx.invert_colors)
        elif effect_type == 'mirror':
            effected_video = video.fx(mp.vfx.mirror_x)
        elif effect_type == 'pixelate':
            effected_video = video.fx(mp.vfx.pixelate, 10)
        elif effect_type == 'edge_detection':
            effected_video = video.fx(mp.vfx.blackwhite).fx(mp.vfx.invert_colors)
        else:
            effected_video = video
            
        # Generate output filename
        output_filename = f'effect_{int(time.time())}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Write the effected video
        effected_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            bitrate='8000k',
            fps=30,
            preset='slow',
            threads=4,
            ffmpeg_params=[
                '-crf', '18',
                '-profile:v', 'high',
                '-level', '4.0',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p'
            ]
        )
        
        # Close the video
        video.close()
        effected_video.close()
        
        # Clean up input file
        os.remove(input_path)
        
        return jsonify({
            'success': True,
            'output_file': output_filename
        })
        
    except Exception as e:
        print(f"Error in apply_effects: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/apply-animation', methods=['POST'])
@login_required
def apply_animation():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        animation_type = request.form.get('animation_type', 'zoom')
        duration = float(request.form.get('duration', 2.0))
        
        if not file:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Load the video
        video = mp.VideoFileClip(input_path)
        
        # Apply animation
        if animation_type == 'zoom':
            def zoom_effect(t):
                if t < duration:
                    return 1 + t / duration
                elif t > video.duration - duration:
                    return 1 + (video.duration - t) / duration
                else:
                    return 2
            animated_video = video.resize(lambda t: zoom_effect(t))
        elif animation_type == 'rotate':
            def rotate_effect(t):
                if t < duration:
                    return 360 * t / duration
                elif t > video.duration - duration:
                    return 360 * (video.duration - t) / duration
                else:
                    return 0
            animated_video = video.rotate(lambda t: rotate_effect(t))
        elif animation_type == 'slide':
            def slide_effect(t):
                if t < duration:
                    return (t / duration) * video.w
                elif t > video.duration - duration:
                    return ((video.duration - t) / duration) * video.w
                else:
                    return 0
            animated_video = video.set_position(lambda t: (slide_effect(t), 0))
        elif animation_type == 'fade':
            animated_video = video.fadein(duration).fadeout(duration)
        elif animation_type == 'bounce':
            def bounce_effect(t):
                if t < duration:
                    return 1 + math.sin(t * math.pi / duration)
                elif t > video.duration - duration:
                    return 1 + math.sin((video.duration - t) * math.pi / duration)
                else:
                    return 1
            animated_video = video.resize(lambda t: bounce_effect(t))
        else:
            animated_video = video
            
        # Generate output filename
        output_filename = f'animation_{int(time.time())}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Write the animated video
        animated_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            bitrate='8000k',
            fps=30,
            preset='slow',
            threads=4,
            ffmpeg_params=[
                '-crf', '18',
                '-profile:v', 'high',
                '-level', '4.0',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p'
            ]
        )
        
        # Close the video
        video.close()
        animated_video.close()
        
        # Clean up input file
        os.remove(input_path)
        
        return jsonify({
            'success': True,
            'output_file': output_filename
        })
        
    except Exception as e:
        print(f"Error in apply_animation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@login_manager.unauthorized_handler
def unauthorized():
    if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            'success': False,
            'error': 'Authentication required',
            'redirect': url_for('login')
        }), 401
    flash('Please login to access this page', 'warning')
    return redirect(url_for('login'))

@app.route('/search-youtube-clips', methods=['POST'])
@login_required
def search_youtube_clips():
    try:
        # Check if user is authenticated
        if not current_user.is_authenticated:
            return jsonify({
                'success': False,
                'error': 'Authentication required',
                'redirect': url_for('login')
            }), 401

        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
            
        script_description = data.get('script_description')
        if not script_description:
            return jsonify({'success': False, 'error': 'No script description provided'})
            
        print(f"Debug - Searching YouTube for: {script_description}")
        
        # Configure request session with retry mechanism
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        
        # Headers for the request
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'VideoEditor/1.0'
        }
        
        # Build the search URL with parameters
        base_url = 'https://www.googleapis.com/youtube/v3/search'
        params = {
            'part': 'snippet',
            'q': script_description,
            'maxResults': 50,
            'type': 'video',
            'videoEmbeddable': 'true',
            'videoSyndicated': 'true',
            'key': os.getenv('YOUTUBE_API_KEY')
        }
        
        try:
            print(f"Debug - Making request to: {base_url}")
            print(f"Debug - With parameters: {params}")
            
            # Make request to YouTube API
            response = session.get(base_url, params=params, headers=headers)
            
            print(f"Debug - API Response Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Debug - API Error: {response.status_code} - {response.text}")
                return jsonify({
                    'success': True,
                    'clips': []
                })
            
            data = response.json()
            videos = data.get('items', [])
            print(f"Debug - Found {len(videos)} videos")
            
            clips = []
            for video in videos:
                try:
                    video_id = video['id']['videoId']
                    snippet = video['snippet']
                    
                    # Format the published date
                    try:
                        published_at = datetime.strptime(snippet['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                        formatted_date = published_at.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        formatted_date = snippet.get('publishedAt', '')
                    
                    clip = {
                        'id': video_id,
                        'title': snippet.get('title', ''),
                        'thumbnail': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
                        'description': snippet.get('description', '')[:200] + '...' if snippet.get('description') else '',
                        'channel': snippet.get('channelTitle', ''),
                        'published_at': formatted_date,
                        'watch_url': f'https://www.youtube.com/watch?v={video_id}',
                        'embed_url': f'https://www.youtube.com/embed/{video_id}',
                        'duration': '0',  # Add default duration
                        'view_count': '0',  # Add default view count
                        'like_count': '0'   # Add default like count
                    }
                    
                    clips.append(clip)
                    print(f"Debug - Added clip: {snippet['title']}")
                    
                except Exception as e:
                    print(f"Debug - Error processing video: {str(e)}")
                    continue
            
            print(f"Debug - Successfully processed {len(clips)} clips")
            return jsonify({
                'success': True,
                'clips': clips
            })
            
        except requests.exceptions.RequestException as e:
            print(f"Debug - Network error during API request: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Network error during YouTube search: {str(e)}'
            })
        except Exception as e:
            print(f"Debug - Unexpected error during API request: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Unexpected error during YouTube search: {str(e)}'
            })
            
    except Exception as e:
        print(f"Debug - General error in search_youtube_clips: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download-youtube-clip', methods=['POST'])
@login_required
def download_youtube_clip():
    try:
        # Check if user is authenticated
        if not current_user.is_authenticated:
            return jsonify({
                'success': False,
                'error': 'Authentication required',
                'redirect': url_for('login')
            }), 401

        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
            
        video_id = data.get('video_id')
        if not video_id:
            return jsonify({'success': False, 'error': 'No video ID provided'})
            
        print(f"Debug - Downloading YouTube video ID: {video_id}")
        
        # Download YouTube video
        output_filename = f'youtube_{video_id}_{int(time.time())}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Use yt-dlp to download the video with enhanced options
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': output_path,
            'verbose': True,
            'no_warnings': False,
            'ignoreerrors': True,
            'quiet': False,
            'no_color': True,
            'extract_flat': False,
            'force_generic_extractor': False,
            'cookiesfrombrowser': None,
            'cookiefile': None,
            'nocheckcertificate': True,
            'prefer_insecure': True,
            'geo_bypass': True,
            'geo_verification_proxy': None,
            'socket_timeout': 30,
            'retries': 3,
            'fragment_retries': 3,
            'skip_unavailable_fragments': True,
            'keepvideo': False,
            'writedescription': False,
            'writeinfojson': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'postprocessors': [],
            'merge_output_format': 'mp4',
            'updatetime': False,
            'consoletitle': False,
            'noprogress': False,
            'progress_with_newline': True,
            'progress_hooks': [],
            'postprocessor_hooks': [],
            'match_filter': None,
            'source_address': None,
            'call_home': False,
            'sleep_interval': 0,
            'max_sleep_interval': 10,
            'external_downloader_args': None,
            'listformats': False,
            'list_thumbnails': False,
            'playlist_items': None,
            'playlist_random': False,
            'playlist_reverse': False,
            'playlist_start': 1,
            'playlist_end': None,
            'playlist_min_files': 1,
            'playlist_max_files': None,
            'playlist_filters': [],
            'age_limit': None,
            'download_archive': None,
            'break_on_existing': False,
            'break_per_url': False,
            'skip_download': False,
            'cachedir': None,
            'youtube_include_dash_manifest': True,
            'youtube_include_hls_manifest': True,
            'youtube_include_drm_manifest': True,
            'youtube_include_webm': True,
            'youtube_include_3d': True,
            'youtube_include_playlist_metafiles': True,
            'youtube_include_dash_audio': True,
            'youtube_include_dash_video': True,
            'youtube_include_hls_audio': True,
            'youtube_include_hls_video': True,
            'youtube_include_drm_audio': True,
            'youtube_include_drm_video': True,
            'youtube_include_webm_audio': True,
            'youtube_include_webm_video': True,
            'youtube_include_3d_audio': True,
            'youtube_include_3d_video': True,
            'youtube_include_playlist_metafiles_audio': True,
            'youtube_include_playlist_metafiles_video': True,
        }
        
        print(f"Debug - Downloading to: {output_path}")
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # First try to extract info to verify the video exists
                try:
                    info = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
                    if not info:
                        print(f"Debug - Could not extract video info")
                        return jsonify({'success': False, 'error': 'Video not found or unavailable'})
                except Exception as e:
                    print(f"Debug - Error extracting video info: {str(e)}")
                    return jsonify({'success': False, 'error': f'Error accessing video: {str(e)}'})
                
                # Now try to download
                try:
                    ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
                except Exception as e:
                    print(f"Debug - Error during download: {str(e)}")
                    if os.path.exists(output_path):
                        try:
                            os.remove(output_path)
                        except:
                            pass
                    return jsonify({'success': False, 'error': f'Error downloading video: {str(e)}'})
                
            if not os.path.exists(output_path):
                print(f"Debug - Output file not found after download")
                return jsonify({'success': False, 'error': 'Failed to download video'})
                
            # Verify the file is not empty
            if os.path.getsize(output_path) == 0:
                print(f"Debug - Downloaded file is empty")
                os.remove(output_path)
                return jsonify({'success': False, 'error': 'Downloaded file is empty'})
                
            print(f"Debug - Successfully downloaded video to: {output_path}")
            
            return jsonify({
                'success': True,
                'filename': output_filename
            })
        except Exception as e:
            print(f"Debug - yt-dlp error: {str(e)}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            return jsonify({'success': False, 'error': f'Error downloading video: {str(e)}'})
            
    except Exception as e:
        print(f"Debug - General error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/add-youtube-to-video', methods=['POST'])
@login_required
def add_youtube_to_video():
    """Add a YouTube clip to the user's video."""
    videos = []
    input_paths = []
    final_video = None
    youtube_path = None # Define youtube_path outside try for cleanup

    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'})
            
        video_file = request.files['file']
        video_id = request.form.get('video_id')
        
        if not video_id:
            return jsonify({'success': False, 'error': 'No YouTube video ID provided'})
            
        if not video_file.filename:
            return jsonify({'success': False, 'error': 'No selected file'})
            
        print(f"Debug - Processing merge request for YouTube video ID: {video_id}")
            
        # Save the uploaded video
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
        video_file.save(video_path)
        input_paths.append(video_path) # Add to input_paths for cleanup
        
        # Download the YouTube clip
        youtube_filename = f'youtube_{video_id}_{int(time.time())}.mp4'
        youtube_path = os.path.join(app.config['OUTPUT_FOLDER'], youtube_filename)
        input_paths.append(youtube_path) # Add to input_paths for cleanup
        
        try:
            print(f"Debug - Downloading YouTube video to: {youtube_path}")
            
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'outtmpl': youtube_path,
                'verbose': True,
                'no_warnings': False,
                'ignoreerrors': True,
                'quiet': False,
                'no_color': True,
                'extract_flat': False,
                'force_generic_extractor': False,
                'cookiesfrombrowser': None,
                'cookiefile': None,
                'nocheckcertificate': True,
                'prefer_insecure': True,
                'geo_bypass': True,
                'socket_timeout': 60, # Increased socket timeout
                'retries': 10, # Increased retries for robustness
                'fragment_retries': 10,
                'skip_unavailable_fragments': True,
                'keepvideo': False,
                'writedescription': False,
                'writeinfojson': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
                'postprocessors': [],
                'merge_output_format': 'mp4',
                'updatetime': False,
                'consoletitle': False,
                'noprogress': False,
                'progress_with_newline': True,
                'progress_hooks': [],
                'postprocessor_hooks': [],
                'match_filter': None,
                'source_address': None,
                'call_home': False,
                'sleep_interval': 0,
                'max_sleep_interval': 10,
                'external_downloader_args': None,
                'listformats': False,
                'list_thumbnails': False,
                'playlist_items': None,
                'playlist_random': False,
                'playlist_reverse': False,
                'playlist_start': 1,
                'playlist_end': None,
                'playlist_min_files': 1,
                'playlist_max_files': None,
                'playlist_filters': [],
                'age_limit': None,
                'download_archive': None,
                'break_on_existing': False,
                'break_per_url': False,
                'skip_download': False,
                'cachedir': None,
                'youtube_include_dash_manifest': True,
                'youtube_include_hls_manifest': True,
                'youtube_include_drm_manifest': True,
                'youtube_include_webm': True,
                'youtube_include_3d': True,
                'youtube_include_playlist_metafiles': True,
                'youtube_include_dash_audio': True,
                'youtube_include_dash_video': True,
                'youtube_include_hls_audio': True,
                'youtube_include_hls_video': True,
                'youtube_include_drm_audio': True,
                'youtube_include_drm_video': True,
                'youtube_include_webm_audio': True,
                'youtube_include_webm_video': True,
                'youtube_include_3d_audio': True,
                'youtube_include_3d_video': True,
                'youtube_include_playlist_metafiles_audio': True,
                'youtube_include_playlist_metafiles_video': True,
            }
            
            try:
                print(f"Debug - Starting YouTube download with yt-dlp")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # First try to extract info to verify the video exists
                    try:
                        info = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
                        if not info:
                            print(f"Debug - Could not extract video info")
                            return jsonify({'success': False, 'error': 'Video not found or unavailable'})
                    except Exception as e:
                        print(f"Debug - Error extracting video info: {str(e)}")
                        return jsonify({'success': False, 'error': f'Error accessing video: {str(e)}'})
                    
                    # Now try to download
                    try:
                        ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
                    except Exception as e:
                        print(f"Debug - Error during download: {str(e)}")
                        if os.path.exists(youtube_path):
                            try:
                                os.remove(youtube_path)
                            except:
                                pass
                        return jsonify({'success': False, 'error': f'Error downloading video: {str(e)}'})
                    
                if not os.path.exists(youtube_path):
                    print(f"Debug - YouTube video not found after download")
                    return jsonify({'success': False, 'error': 'Failed to download YouTube video'})
                    
                # Verify the file is not empty
                if os.path.getsize(youtube_path) == 0:
                    print(f"Debug - Downloaded YouTube file is empty")
                    os.remove(youtube_path)
                    return jsonify({'success': False, 'error': 'Downloaded YouTube file is empty'})
                    
                print(f"Debug - Successfully downloaded YouTube video")
                
                # Load both videos
                print(f"Debug - Loading videos for merging")
                video1 = mp.VideoFileClip(video_path)
                video2 = mp.VideoFileClip(youtube_path)
                videos.extend([video1, video2]) # Add to videos list for cleanup
                
                # Get the highest resolution
                target_width = max(video1.w, video2.w)
                target_height = max(video1.h, video2.h)
                
                # Ensure dimensions are even
                target_width = target_width - (target_width % 2)
                target_height = target_height - (target_height % 2)
                
                print(f"Debug - Resizing videos to {target_width}x{target_height}")
                
                # Resize videos to match the highest resolution
                if video1.w != target_width or video1.h != target_height:
                    video1 = video1.resize(width=target_width, height=target_height)
                if video2.w != target_width or video2.h != target_height:
                    video2 = video2.resize(width=target_width, height=target_height)
                
                # Concatenate videos
                print(f"Debug - Concatenating videos")
                final_video = mp.concatenate_videoclips([video1, video2])
                
                # Generate output filename
                output_filename = f'merged_youtube_{int(time.time())}.mp4'
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                
                print(f"Debug - Writing final video to {output_path}")
                print(f"Debug - Starting write_videofile with preset 'fast' and progress logger")
                
                # Write the final video
                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    bitrate='8000k',
                    fps=30,
                    preset='fast', # Changed from slow to fast
                    threads=4,
                    ffmpeg_params=[
                        '-crf', '18',
                        '-profile:v', 'high',
                        '-level', '4.0',
                        '-movflags', '+faststart',
                        '-pix_fmt', 'yuv420p'
                    ],
                    logger='bar' # Add progress bar to console
                )
                
                print(f"Debug - Finished write_videofile")
                
                print(f"Debug - Cleaning up video objects")
                
                # Close all videos
                video1.close()
                video2.close()
                final_video.close()
                
                # Clean up temporary files
                print(f"Debug - Cleaning up temporary files")
                if os.path.exists(video_path):
                    os.remove(video_path)
                if os.path.exists(youtube_path):
                    os.remove(youtube_path)
                
                print(f"Debug - Merge completed successfully")
                return jsonify({
                    'success': True,
                    'output_file': output_filename
                })
                
            except Exception as e:
                print(f"Debug - Error during processing: {str(e)}")
                # Clean up any files that were saved
                if os.path.exists(video_path):
                    os.remove(video_path)
                if os.path.exists(youtube_path):
                    os.remove(youtube_path)
                return jsonify({'success': False, 'error': f'Error processing videos: {str(e)}'})
                
        except Exception as e:
            print(f"Debug - Unexpected error in add_youtube_to_video: {str(e)}")
            return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})

    except Exception as e:
        print(f"Debug - Unexpected error in add_youtube_to_video: {str(e)}")
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})

@app.route('/search-pexels-clips', methods=['POST'])
@login_required
def search_pexels_clips():
    try:
        # Check if user is authenticated
        if not current_user.is_authenticated:
            return jsonify({
                'success': False,
                'error': 'Authentication required',
                'redirect': url_for('login')
            }), 401

        script_description = request.form.get('script_description')
        if not script_description:
            return jsonify({'success': False, 'error': 'No script description provided'})
            
        print(f"Debug - Searching Pexels for: {script_description}")
        
        # Configure request session with retry mechanism
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        
        # Headers for the request
        headers = {
            'Authorization': PEXELS_API_KEY,
            'Accept': 'application/json',
            'User-Agent': 'VideoEditor/1.0'
        }
        
        # Build the search URL with parameters
        base_url = 'https://api.pexels.com/videos/search'
        params = {
            'query': script_description,
            'per_page': 15,
            'orientation': 'landscape'
        }
        
        try:
            print(f"Debug - Making request to: {base_url}")
            
            # Make request to Pexels API
            response = session.get(base_url, params=params, headers=headers)
            
            print(f"Debug - API Response Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Debug - API Error: {response.status_code} - {response.text}")
                return jsonify({
                    'success': False,
                    'error': f'Pexels API error: {response.status_code}'
                })
            
            data = response.json()
            videos = data.get('videos', [])
            print(f"Debug - Found {len(videos)} videos")
            
            clips = []
            for video in videos:
                try:
                    # Get the best quality video file
                    video_files = video.get('video_files', [])
                    best_quality = max(video_files, key=lambda x: x.get('width', 0) * x.get('height', 0))
                    
                    clip = {
                        'id': str(video['id']),
                        'title': video.get('url', '').split('/')[-1],
                        'thumbnail': video.get('image', ''),
                        'url': best_quality.get('link', ''),
                        'width': best_quality.get('width', 0),
                        'height': best_quality.get('height', 0),
                        'duration': video.get('duration', 0),
                        'user': video.get('user', {}).get('name', 'Unknown'),
                        'download_url': best_quality.get('link', '')
                    }
                    
                    clips.append(clip)
                    print(f"Debug - Added clip: {clip['title']}")
                    
                except Exception as e:
                    print(f"Debug - Error processing video: {str(e)}")
                    continue
            
            print(f"Debug - Successfully processed {len(clips)} clips")
            return jsonify({
                'success': True,
                'clips': clips
            })
            
        except requests.exceptions.RequestException as e:
            print(f"Debug - Network error during API request: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Network error during Pexels search: {str(e)}'
            })
        except Exception as e:
            print(f"Debug - Unexpected error during API request: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Unexpected error during Pexels search: {str(e)}'
            })
            
    except Exception as e:
        print(f"Debug - General error in search_pexels_clips: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

def process_video_segment(video_clip, target_size):
    """Process video segment using ML-based optimization."""
    def process_frame(frame):
        # Convert to numpy array for ML processing
        frame = np.array(frame)
        
        # Apply ML-based enhancement
        # 1. Smart scaling using OpenCV's INTER_AREA for downscaling
        if frame.shape[0] > target_size[1] or frame.shape[1] > target_size[0]:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        
        # 2. Apply ML-based sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        frame = cv2.filter2D(frame, -1, kernel)
        
        return frame
    
    return video_clip.fl_image(process_frame)

@app.route('/merge-with-pexels', methods=['POST'])
@login_required
def merge_with_pexels():
    """Merge a Pexels clip with the user's video using ML optimization."""
    videos = []
    input_paths = []
    processed_videos = []
    final_video = None
    pexels_path = None # Define pexels_path outside try for cleanup

    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'})
            
        video_file = request.files['file']
        pexels_clip_url = request.form.get('pexels_clip_url')
        
        if not pexels_clip_url:
            return jsonify({'success': False, 'error': 'No Pexels clip URL provided'})
            
        if not video_file.filename:
            return jsonify({'success': False, 'error': 'No selected file'})
            
        print(f"Debug - Processing merge request with Pexels clip from: {pexels_clip_url}")
            
        # Save the uploaded file
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
        video_file.save(video_path)
        input_paths.append(video_path) # Add to input_paths for cleanup
        
        # Download Pexels video
        pexels_filename = f'pexels_{int(time.time())}.mp4'
        pexels_path = os.path.join(app.config['OUTPUT_FOLDER'], pexels_filename)
        input_paths.append(pexels_path) # Add to input_paths for cleanup
        
        try:
            print(f"Debug - Attempting to download Pexels clip for merge from {pexels_clip_url} to {pexels_path}")
            # Download the Pexels video
            # Use a session with retry logic for robustness
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount('https://', adapter)

            # Add a timeout to the request
            response = session.get(pexels_clip_url, stream=True, timeout=30)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
            print(f"Debug - Pexels merge download response status: {response.status_code}")
            
            with open(pexels_path, 'wb') as f:
                print(f"Debug - Writing Pexels video content for merge to {pexels_path}")
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                print(f"Debug - Finished writing Pexels video content for merge")
            
            if not os.path.exists(pexels_path):
                print(f"Debug - Pexels merge output file not found after download")
                # Ensure cleanup is called even on download failure
                cleanup_files(input_paths)
                return jsonify({'success': False, 'error': 'Failed to download Pexels video for merge: Output file not created.'})
                
            # Verify the file is not empty
            if os.path.getsize(pexels_path) == 0:
                print(f"Debug - Downloaded Pexels file for merge is empty: {pexels_path}")
                # Ensure cleanup is called even on empty file
                cleanup_files(input_paths)
                return jsonify({'success': False, 'error': 'Downloaded Pexels video file for merge is empty.'})
                
            print(f"Debug - Successfully downloaded Pexels video for merge")
            
            print(f"Debug - Loading videos for merging")
            # Load both videos
            video1 = mp.VideoFileClip(video_path, audio=True)
            video2 = mp.VideoFileClip(pexels_path, audio=True)
            videos.extend([video1, video2]) # Add to videos list for cleanup
            
            # Get target dimensions (capped at 1080p)
            target_width = min(max(video1.w, video2.w), 1920)
            target_height = min(max(video1.h, video2.h), 1080)
            target_size = (target_width, target_height)
            
            print(f"Debug - Processing videos in parallel...")
            # Process videos in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit video processing tasks
                future1 = executor.submit(process_video_segment, video1, target_size)
                future2 = executor.submit(process_video_segment, video2, target_size)
                
                # Get processed videos
                processed_video1 = future1.result()
                processed_video2 = future2.result()
                processed_videos.extend([processed_video1, processed_video2]) # Add to processed_videos for cleanup
            
            print(f"Debug - Concatenating videos...")
            # Concatenate processed videos
            final_video = mp.concatenate_videoclips([processed_video1, processed_video2], method="compose")
            
            # Generate output filename
            output_filename = f'merged_pexels_{int(time.time())}.mp4'
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            print(f"Debug - Writing final video...")
            print(f"Debug - Starting write_videofile with preset 'ultrafast' and progress logger for Pexels merge")
            
            # Write the final video with optimized settings
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                bitrate='6000k',
                fps=30,
                preset='ultrafast',  # Use ultrafast preset for faster processing
                threads=8,  # Increased thread count
                ffmpeg_params=[
                    '-crf', '23',  # Slightly reduced quality for speed
                    '-profile:v', 'main',
                    '-level', '4.0',
                    '-movflags', '+faststart',
                    '-pix_fmt', 'yuv420p',
                    '-tune', 'fastdecode',  # Optimize for fast decoding
                    '-threads', '8'  # Use 8 threads for encoding
                ],
                logger='bar' # Add progress bar
            )
            
            print(f"Debug - Finished write_videofile for Pexels merge")
            
            print(f"Debug - Cleaning up video objects and files after Pexels merge")
            # Close all videos
            cleanup_videos(videos)
            cleanup_videos(processed_videos)
            if final_video:
                final_video.close()
            
            # Clean up temporary files
            cleanup_files(input_paths)
            
            print(f"Debug - Pexels merge completed successfully")
            return jsonify({
                'success': True,
                'output_file': output_filename
            })
            
        except requests.exceptions.RequestException as e:
            print(f"Debug - Pexels merge download RequestException: {str(e)}")
            # Clean up any files that were saved
            cleanup_videos(videos)
            cleanup_videos(processed_videos)
            if final_video:
                final_video.close()
            cleanup_files(input_paths)
            return jsonify({'success': False, 'error': f'Error downloading Pexels video for merge: {str(e)}'})
        except Exception as e:
            print(f"Debug - Pexels merge processing unexpected error: {str(e)}")
            # Clean up on error
            cleanup_videos(videos)
            cleanup_videos(processed_videos)
            if final_video:
                final_video.close()
            cleanup_files(input_paths)
            return jsonify({'success': False, 'error': f'Error processing videos: {str(e)}'})
            
    except Exception as e:
        print(f"Error in merge_with_pexels: {str(e)}")
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})

@app.route('/download-pexels-clip', methods=['POST'])
@login_required
def download_pexels_clip():
    """Download a Pexels clip."""
    try:
        # Check if user is authenticated
        if not current_user.is_authenticated:
            return jsonify({
                'success': False,
                'error': 'Authentication required',
                'redirect': url_for('login')
            }), 401

        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
            
        clip_url = data.get('clip_url')
        if not clip_url:
            return jsonify({'success': False, 'error': 'No clip URL provided'})
            
        print(f"Debug - Downloading Pexels video from: {clip_url}")
        
        # Generate output filename
        output_filename = f'pexels_{int(time.time())}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        try:
            print(f"Debug - Attempting to download Pexels clip using requests from {clip_url} to {output_path}")
            # Download the video
            # Use a session with retry logic for robustness
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount('https://', adapter)

            # Add a timeout to the request
            response = session.get(clip_url, stream=True, timeout=30)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
            print(f"Debug - Pexels download response status: {response.status_code}")
            
            with open(output_path, 'wb') as f:
                print(f"Debug - Writing Pexels video content to {output_path}")
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                print(f"Debug - Finished writing Pexels video content")
            
            if not os.path.exists(output_path):
                print(f"Debug - Output file not found after Pexels download")
                return jsonify({'success': False, 'error': 'Failed to download video: Output file not created.'})
                
            # Verify the file is not empty
            if os.path.getsize(output_path) == 0:
                print(f"Debug - Downloaded Pexels file is empty: {output_path}")
                os.remove(output_path) # Clean up empty file
                return jsonify({'success': False, 'error': 'Downloaded video file is empty.'})
                
            print(f"Debug - Successfully downloaded Pexels video to: {output_path}")
            
            return jsonify({
                'success': True,
                'filename': output_filename
            })
        except requests.exceptions.RequestException as e:
            print(f"Debug - Pexels download RequestException: {str(e)}")
            # Clean up any partial file if the request failed
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except: # nosec
                    pass # Ignore errors during cleanup attempt
            return jsonify({'success': False, 'error': f'Error downloading video: {str(e)}'})
        except Exception as e:
            print(f"Debug - Pexels download unexpected error: {str(e)}")
            # Clean up any partial file in case of other exceptions
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except: # nosec
                    pass # Ignore errors during cleanup attempt
            return jsonify({'success': False, 'error': f'Error downloading video: {str(e)}'})
            
    except Exception as e:
        print(f"Debug - General error in download_pexels_clip: {str(e)}")
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})

def formatDuration(seconds):
    """Format duration in seconds to HH:MM:SS format."""
    try:
        seconds = int(float(seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except:
        return "00:00:00"

@app.route('/search-dailymotion-clips', methods=['POST'])
@login_required
def search_dailymotion_clips():
    """Search for Dailymotion clips based on the provided description."""
    try:
        data = request.get_json()
        if not data or 'script_description' not in data:
            return jsonify({'success': False, 'error': 'No search query provided'})

        search_query = data['script_description']
        
        # Set up the request session with retry logic
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5)
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        # Prepare headers for the API request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make the API request to Dailymotion
        api_url = f'https://api.dailymotion.com/videos?search={search_query}&limit=10&fields=id,title,description,thumbnail_url,duration,views_total,owner.username,embed_url'
        response = session.get(api_url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        clips = []
        
        for item in data.get('list', []):
            clip = {
                'id': item.get('id'),
                'title': item.get('title'),
                'description': item.get('description', ''),
                'thumbnail': item.get('thumbnail_url'),
                'duration': item.get('duration'),
                'view_count': item.get('views_total', 0),
                'uploader': item.get('owner.username', 'Unknown'),
                'watch_url': f'https://www.dailymotion.com/video/{item.get("id")}'
            }
            clips.append(clip)
        
        return jsonify({'success': True, 'clips': clips})
        
    except requests.exceptions.RequestException as e:
        return jsonify({'success': False, 'error': f'Error searching Dailymotion: {str(e)}'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})

@app.route('/download-dailymotion-clip', methods=['POST'])
@login_required
def download_dailymotion_clip():
    print(f"Debug - download_dailymotion_clip function called.")
    """Download a Dailymotion clip using yt-dlp."""
    try:
        # Check if user is authenticated
        if not current_user.is_authenticated:
            return jsonify({
                'success': False,
                'error': 'Authentication required',
                'redirect': url_for('login')
            }), 401

        data = request.get_json()
        if not data or 'video_id' not in data:
            return jsonify({'success': False, 'error': 'No video ID provided'})

        video_id = data['video_id']
        print(f"Debug - Downloading Dailymotion video ID: {video_id} using yt-dlp")

        # Generate output filename
        output_filename = f'dailymotion_{video_id}_{int(time.time())}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Configure yt-dlp options for Dailymotion
        ydl_opts = {
            'format': 'best', # Let yt-dlp pick the best format
            'outtmpl': output_path,
            'verbose': True,
            'no_warnings': False,
            'ignoreerrors': True,
            'quiet': False,
            'no_color': True,
            'extract_flat': False,
            'force_generic_extractor': False,
            'nocheckcertificate': True,
            'prefer_insecure': True,
            'geo_bypass': True,
            'socket_timeout': 60, # Increased socket timeout
            'retries': 10, # Increased retries for robustness
            'fragment_retries': 10,
            'skip_unavailable_fragments': True,
            'keepvideo': False,
            'writedescription': False,
            'writeinfojson': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'postprocessors': [],
            'merge_output_format': 'mp4',
            'progress_with_newline': True,
        }

        try:
            print(f"Debug - Starting yt-dlp download to: {output_path}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # First try to extract info to verify the video exists and is downloadable
                try:
                    # Try with different URL formats
                    urls_to_try = [
                        f'https://www.dailymotion.com/video/{video_id}',
                        f'https://dailymotion.com/video/{video_id}',
                        f'https://www.dailymotion.com/embed/video/{video_id}'
                    ]

                    info = None
                    for url in urls_to_try:
                        try:
                            print(f"Debug - Trying to extract info from: {url}")
                            info = ydl.extract_info(url, download=False)
                            if info:
                                print(f"Debug - Successfully extracted video info from: {url}")
                                break
                        except Exception as e:
                            print(f"Debug - Failed to extract info from {url}: {str(e)}")
                            continue

                    if not info:
                        print(f"Debug - Could not extract video info for {video_id} from any URL format")
                        return jsonify({'success': False, 'error': 'Video not found or unavailable'})

                except Exception as e:
                    print(f"Debug - Error extracting video info for {video_id}: {type(e).__name__} - {str(e)}")
                    return jsonify({'success': False, 'error': f'Error accessing video: {str(e)}'})

                # Now try to download the video
                try:
                    print(f"Debug - Initiating download for {video_id}")
                    # Use the first successful URL or the original one for download
                    download_url = urls_to_try[0] if info else f'https://www.dailymotion.com/video/{video_id}'
                    ydl.download([download_url])
                    print(f"Debug - yt-dlp download process finished for {video_id}")
                except Exception as e:
                    print(f"Debug - Error during download for {video_id}: {type(e).__name__} - {str(e)}")
                    if os.path.exists(output_path):
                        try:
                            os.remove(output_path)
                        except: # nosec
                            pass
                    return jsonify({'success': False, 'error': f'Error downloading video: {str(e)}'})

            # Verify the downloaded file
            if not os.path.exists(output_path):
                print(f"Debug - Output file not found after download at {output_path}")
                return jsonify({'success': False, 'error': 'Failed to download video: Output file not created.'})

            if os.path.getsize(output_path) == 0:
                print(f"Debug - Downloaded file is empty at {output_path}")
                os.remove(output_path)
                return jsonify({'success': False, 'error': 'Downloaded video file is empty.'})

            print(f"Debug - Successfully downloaded Dailymotion clip to {output_path}")

            # --- NEW: Re-encode video for browser compatibility ---
            try:
                print(f"Debug - Loading downloaded Dailymotion clip for re-encoding: {output_path}")
                clip = mp.VideoFileClip(output_path)

                # Define temporary re-encoded path
                reencoded_filename = f'reencoded_dailymotion_{int(time.time())}.mp4'
                reencoded_path = os.path.join(app.config['OUTPUT_FOLDER'], reencoded_filename)

                print(f"Debug - Re-encoding Dailymotion clip to: {reencoded_path}")
                clip.write_videofile(
                    reencoded_path,
                    codec='libx264',
                    audio_codec='aac',
                    bitrate='3000k', # Adjust bitrate as needed for quality vs file size
                    fps=clip.fps, # Preserve original FPS
                    preset='medium', # Use a balanced preset for quality and speed
                    threads=4,
                    ffmpeg_params=[
                        '-movflags', '+faststart', # For faster web playback
                        '-pix_fmt', 'yuv420p', # Essential for broad browser compatibility
                        '-crf', '23' # Constant Rate Factor: 23 is a good balance
                    ]
                )
                clip.close()
                os.remove(output_path) # Remove the original downloaded file
                output_filename = reencoded_filename # Use the re-encoded filename for output
                print(f"Debug - Successfully re-encoded Dailymotion clip to: {output_filename}")
            except Exception as e:
                print(f"Debug - Error during Dailymotion re-encoding: {str(e)}")
                if os.path.exists(output_path): os.remove(output_path) # Clean up original if re-encoding fails
                return jsonify({'success': False, 'error': f'Error re-encoding video for browser: {str(e)}'})
            # --- END NEW RE-ENCODE ---

            return jsonify({
                'success': True,
                'filename': output_filename # Return the re-encoded filename
            })

        except Exception as e:
            print(f"Debug - yt-dlp execution error for {video_id}: {str(e)}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except: # nosec
                    pass
            return jsonify({'success': False, 'error': f'Error processing download: {str(e)}'})

    except Exception as e:
        print(f"Debug - General error in download_dailymotion_clip: {str(e)}")
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})

@app.route('/add-dailymotion-to-video', methods=['POST'])
@login_required
def add_dailymotion_to_video():
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'})
        
        video_file = request.files['video']
        dailymotion_filename = request.form.get('dailymotion_filename')
        
        if not dailymotion_filename:
            return jsonify({'success': False, 'error': 'No Dailymotion filename provided'})
        
        # Get the paths for both videos
        dailymotion_path = get_output_path(dailymotion_filename)
        if not os.path.exists(dailymotion_path):
            return jsonify({'success': False, 'error': 'Dailymotion video not found'})
        
        # Save the uploaded video temporarily
        temp_video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
        video_file.save(temp_video_path)
        
        try:
            # Load both videos with high quality settings
            main_video = mp.VideoFileClip(temp_video_path, audio=True)
            dailymotion_video = mp.VideoFileClip(dailymotion_path, audio=True)
            
            # Ensure both videos have the same resolution (use the higher resolution)
            target_width = max(main_video.w, dailymotion_video.w)
            target_height = max(main_video.h, dailymotion_video.h)
            
            # Resize videos if needed while maintaining aspect ratio
            if main_video.w != target_width or main_video.h != target_height:
                main_video = main_video.resize(width=target_width, height=target_height)
            if dailymotion_video.w != target_width or dailymotion_video.h != target_height:
                dailymotion_video = dailymotion_video.resize(width=target_width, height=target_height)
            
            # Concatenate videos
            final_video = mp.concatenate_videoclips([main_video, dailymotion_video])
            
            # Generate output filename
            timestamp = int(time.time())
            output_filename = f'merged_{timestamp}.mp4'
            output_path = get_output_path(output_filename)
            
            # Write the final video with high quality settings
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                bitrate='8000k',  # High bitrate for better quality
                fps=30,  # Maintain high frame rate
                preset='slow',  # Better compression quality
                threads=4,  # Use multiple threads for faster processing
                ffmpeg_params=[
                    '-crf', '18',  # Constant Rate Factor (lower = better quality, 18 is visually lossless)
                    '-movflags', '+faststart',  # Enable fast start for web playback
                    '-pix_fmt', 'yuv420p'  # Ensure compatibility with most players
                ]
            )
            
            # Clean up
            final_video.close()
            main_video.close()
            dailymotion_video.close()
            
            # Remove temporary files
            cleanup_files([temp_video_path, dailymotion_path])
            
            return jsonify({
                'success': True,
                'filename': output_filename
            })
            
        except Exception as e:
            print(f"Error in video processing: {str(e)}")
            # Clean up in case of error
            cleanup_files([temp_video_path, dailymotion_path])
            return jsonify({'success': False, 'error': f'Error processing video: {str(e)}'})
            
    except Exception as e:
        print(f"Error in add_dailymotion_to_video: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Create a test user if none exists
    with app.app_context():
        if not User.query.filter_by(username='admin').first():
            print("Creating test user...")  # Debug print
            test_user = User(
                username='admin',
                email='admin@example.com',
                password_hash=generate_password_hash('password')
            )
            db.session.add(test_user)
            db.session.commit()
            print("Test user created successfully")  # Debug print
    
    print("Starting Flask application...")  # Debug print
    app.run(debug=True, port=5000) 