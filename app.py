from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session
import moviepy.editor as mp
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from functools import wraps
from ml_processor import MLVideoProcessor
from models import db, User

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generate a secure random key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db.init_app(app)

# Create database tables
with app.app_context():
    db.create_all()

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

def get_video_path(filename):
    return os.path.join(UPLOAD_FOLDER, filename)

def get_output_path(filename):
    return os.path.join(OUTPUT_FOLDER, filename)

@app.route('/')
def index():
    print("Accessing root route")  # Debug log
    # Clear any existing session
    session.clear()
    print("Redirecting to login page")  # Debug log
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    print(f"Login route accessed with method: {request.method}")  # Debug log
    if request.method == 'GET':
        print("Rendering login template")  # Debug log
        return render_template('login.html')
    
    print("Processing login POST request")  # Debug log
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    user = User.query.filter_by(email=email).first()
    
    if user and user.check_password(password):
        session['user_id'] = user.id
        print(f"Login successful for user: {email}")  # Debug log
        return jsonify({'success': True, 'redirect': url_for('editor')})
    
    print(f"Login failed for email: {email}")  # Debug log
    return jsonify({'success': False, 'error': 'Invalid credentials'})

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    print(f"Signup route accessed with method: {request.method}")  # Debug log
    if request.method == 'GET':
        print("Rendering signup template")  # Debug log
        return render_template('signup.html')
    
    print("Processing signup POST request")  # Debug log
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    # Check if user already exists
    if User.query.filter_by(email=email).first():
        print(f"Email already registered: {email}")  # Debug log
        return jsonify({'success': False, 'error': 'Email already registered'})
    
    if User.query.filter_by(username=username).first():
        print(f"Username already taken: {username}")  # Debug log
        return jsonify({'success': False, 'error': 'Username already taken'})
    
    # Create new user
    user = User(username=username, email=email)
    user.set_password(password)
    
    try:
        db.session.add(user)
        db.session.commit()
        session['user_id'] = user.id
        print(f"User created successfully: {username}")  # Debug log
        return jsonify({'success': True, 'redirect': url_for('editor')})
    except Exception as e:
        db.session.rollback()
        print(f"Error creating user: {str(e)}")  # Debug log
        return jsonify({'success': False, 'error': 'Error creating account'})

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/editor')
@login_required
def editor():
    print("Accessing editor route")  # Debug log
    if 'user_id' not in session:
        print("No user in session, redirecting to landing page")  # Debug log
        return redirect(url_for('index'))
    print("User authenticated, showing editor")  # Debug log
    return render_template('editor.html')

@app.route('/output/<filename>')
@login_required
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/trim', methods=['POST'])
@login_required
def trim_video():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"})
        
        # Save the uploaded file
        input_path = get_video_path(file.filename)
        file.save(input_path)
        
        start_time = float(request.form.get('start_time', 0))
        end_time = float(request.form.get('end_time', 0))
        
        video = mp.VideoFileClip(input_path)
        trimmed = video.subclip(start_time, end_time)
        
        output_file = f"trimmed_{file.filename}"
        output_path = get_output_path(output_file)
        trimmed.write_videofile(output_path)
        
        video.close()
        trimmed.close()
        
        return jsonify({"success": True, "output_file": output_file})
    except Exception as e:
        print(f"Error in trim_video: {str(e)}")  # Debug log
        return jsonify({"success": False, "error": str(e)})

@app.route('/merge', methods=['POST'])
@login_required
def merge_videos():
    try:
        if 'files' not in request.files:
            return jsonify({"success": False, "error": "No files provided"})
        
        files = request.files.getlist('files')
        if len(files) < 2:
            return jsonify({"success": False, "error": "At least 2 files required"})
        
        clips = []
        for file in files:
            if file.filename:
                input_path = get_video_path(file.filename)
                file.save(input_path)
                clips.append(mp.VideoFileClip(input_path))
        
        final_clip = mp.concatenate_videoclips(clips)
        output_file = "merged_video.mp4"
        output_path = get_output_path(output_file)
        final_clip.write_videofile(output_path)
        
        for clip in clips:
            clip.close()
        final_clip.close()
        
        return jsonify({"success": True, "output_file": output_file})
    except Exception as e:
        print(f"Error in merge_videos: {str(e)}")  # Debug log
        return jsonify({"success": False, "error": str(e)})

@app.route('/extract-audio', methods=['POST'])
@login_required
def extract_audio():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"})
        
        input_path = get_video_path(file.filename)
        file.save(input_path)
        
        video = mp.VideoFileClip(input_path)
        audio = video.audio
        
        output_file = f"{os.path.splitext(file.filename)[0]}.mp3"
        output_path = get_output_path(output_file)
        audio.write_audiofile(output_path)
        
        video.close()
        audio.close()
        
        return jsonify({"success": True, "output_file": output_file})
    except Exception as e:
        print(f"Error in extract_audio: {str(e)}")  # Debug log
        return jsonify({"success": False, "error": str(e)})

@app.route('/change-speed', methods=['POST'])
@login_required
def change_speed():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"})
        
        input_path = get_video_path(file.filename)
        file.save(input_path)
        
        speed_factor = float(request.form.get('speed_factor', 1.0))
        
        video = mp.VideoFileClip(input_path)
        modified = video.speedx(speed_factor)
        
        output_file = f"speed_{speed_factor}x_{file.filename}"
        output_path = get_output_path(output_file)
        modified.write_videofile(output_path)
        
        video.close()
        modified.close()
        
        return jsonify({"success": True, "output_file": output_file})
    except Exception as e:
        print(f"Error in change_speed: {str(e)}")  # Debug log
        return jsonify({"success": False, "error": str(e)})

@app.route('/resize', methods=['POST'])
@login_required
def resize_video():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"})
        
        input_path = get_video_path(file.filename)
        file.save(input_path)
        
        width = int(request.form.get('width', 0))
        height = int(request.form.get('height', 0))
        
        if width <= 0 or height <= 0:
            return jsonify({"success": False, "error": "Invalid dimensions"})
        
        video = mp.VideoFileClip(input_path)
        resized = video.resize((width, height))
        
        output_file = f"resized_{width}x{height}_{file.filename}"
        output_path = get_output_path(output_file)
        resized.write_videofile(output_path)
        
        video.close()
        resized.close()
        
        return jsonify({"success": True, "output_file": output_file})
    except Exception as e:
        print(f"Error in resize_video: {str(e)}")  # Debug log
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 