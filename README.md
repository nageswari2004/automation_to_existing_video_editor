# Video Processing Web Application

A Flask-based web application for video processing and editing with user authentication and various video manipulation features.

## Features

- User Authentication (Signup/Login)
- Video Processing Capabilities:
  - Video Trimming
  - Video Merging
  - Audio Extraction
  - Speed Adjustment
  - Video Resizing
- Secure File Handling
- User Session Management
- SQLite Database for User Management

## Tech Stack

- **Backend**: Flask 2.0.1
- **Database**: SQLite with SQLAlchemy
- **Video Processing**: MoviePy, OpenCV
- **Machine Learning**: PyTorch, TorchVision
- **Frontend**: HTML, CSS, JavaScript (templates in the templates directory)

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## System Requirements

### Hardware Requirements
- CPU: Dual-core processor or better
- RAM: Minimum 4GB (8GB recommended for video processing)
- Storage: At least 10GB free space for video processing
- Graphics: Any modern GPU (dedicated GPU recommended for faster video processing)

### Software Requirements
- Operating System: Windows 10/11, macOS, or Linux
- Web Browser: Chrome, Firefox, or Edge (latest versions)
- FFmpeg (required for video processing)
  - Windows: Download from [FFmpeg official website](https://ffmpeg.org/download.html)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg`

### Additional Resources
- FFmpeg installation guide: [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- OpenCV installation guide: [OpenCV Installation](https://docs.opencv.org/master/d5/de5/tutorial_py_setup_in_windows.html)
- PyTorch installation guide: [PyTorch Installation](https://pytorch.org/get-started/locally/)

### Development Tools (Optional)
- Code Editor: VS Code, PyCharm, or any preferred IDE
- Git for version control
- Postman or similar tool for API testing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── app.py              # Main Flask application
├── models.py           # Database models
├── ml_processor.py     # Machine learning processing logic
├── requirements.txt    # Project dependencies
├── templates/          # HTML templates
├── static/            # Static files (CSS, JS, images)
├── uploads/           # Temporary storage for uploaded videos
├── output/            # Processed video output directory
└── users.db           # SQLite database file
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Create an account or login to access the video editor

4. Use the web interface to:
   - Upload videos
   - Trim videos
   - Merge multiple videos
   - Extract audio
   - Adjust video speed
   - Resize videos

## Security Features

- Secure password hashing
- Session-based authentication
- Secure file handling
- Input validation
- Protected routes

## Development

The application uses Flask's development server by default. For production deployment, consider using a production-grade WSGI server like Gunicorn.

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here] 