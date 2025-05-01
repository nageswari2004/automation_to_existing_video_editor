import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
import os

class MLVideoProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the ML model for video processing"""
        # Load pre-trained ResNet model
        base_model = models.resnet50(pretrained=True)
        
        # Modify the final layer for our needs
        num_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 3)  # 3 classes: enhance, basic, skip
        )
        
        # Move model to device
        self.model = base_model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        frame_tensor = self.transform(frame)
        
        # Add batch dimension
        frame_tensor = frame_tensor.unsqueeze(0)
        
        return frame_tensor.to(self.device)
        
    def process_video(self, video_path, output_path):
        """Process video using ML model"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        with torch.no_grad():  # Disable gradient calculation
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process every 5th frame to improve performance
                if frame_count % 5 == 0:
                    processed_frame = self.process_frame(frame)
                else:
                    processed_frame = frame
                    
                # Write processed frame
                out.write(processed_frame)
                frame_count += 1
                
        cap.release()
        out.release()
        
    def process_frame(self, frame):
        """Process a single frame using ML model"""
        # Preprocess frame
        frame_tensor = self.preprocess_frame(frame)
        
        # Get model prediction
        with torch.no_grad():
            prediction = self.model(frame_tensor)
            action = torch.argmax(prediction[0]).item()
        
        # Apply processing based on prediction
        if action == 0:  # Enhance
            frame = self.enhance_frame(frame)
        elif action == 1:  # Basic
            frame = self.basic_process(frame)
        # If action == 2 (Skip), keep original frame
        
        return frame
        
    def enhance_frame(self, frame):
        """Apply enhanced processing to frame"""
        # Convert to float32
        frame = frame.astype(np.float32)
        
        # Apply advanced contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl,a,b))
        frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        frame = cv2.filter2D(frame, -1, kernel)
        
        # Apply color correction
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
        
        return frame
        
    def basic_process(self, frame):
        """Apply basic processing to frame"""
        # Convert to float32
        frame = frame.astype(np.float32)
        
        # Apply basic adjustments
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=5)
        
        # Apply slight sharpening
        kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
        frame = cv2.filter2D(frame, -1, kernel)
        
        return frame
        
    def auto_trim(self, video_path, output_path, threshold=0.5):
        """Automatically trim video based on ML model predictions"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                frame_tensor = self.preprocess_frame(frame)
                prediction = self.model(frame_tensor)
                prob = torch.softmax(prediction[0], dim=0)[0].item()
                
                # Only write frames that meet threshold (high quality frames)
                if prob > threshold:
                    out.write(frame)
                    
        cap.release()
        out.release()
        
    def auto_stabilize(self, video_path, output_path):
        """Automatically stabilize video using ML-based motion estimation"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize stabilizer
        stabilizer = cv2.VideoStabilizer_create()
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            return
            
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Stabilize frame
            stabilized_frame = stabilizer.stabilize(frame)
            
            # Write stabilized frame
            out.write(stabilized_frame)
            
            # Update previous frame
            prev_gray = gray
            
        cap.release()
        out.release() 