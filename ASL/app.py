from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from gtts import gTTS
import os
import time
app = Flask(__name__)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

# CNN-LSTM Model
class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm_dropout = nn.Dropout(0.4)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc_out = nn.Linear(1024, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)
        attn_weights = torch.softmax(self.attention(x), dim=1)
        x = torch.sum(attn_weights * x, dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc_out(x))
        return x

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

try:
    model = CNNLSTMModel(input_dim=3, hidden_dim=64, output_dim=27).to(device)
    model.load_state_dict(torch.load('sequential_model.pth', map_location=device))
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

classes = sorted(os.listdir('dataset/train'))

# Global variable to store the latest prediction
current_prediction = ""
last_prediction_time = 0
prediction_cooldown = 2  # seconds

def process_frame(frame):
    global current_prediction, last_prediction_time
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = torch.tensor([[
                [landmark.x, landmark.y, landmark.z] 
                for landmark in hand_landmarks.landmark
            ]], dtype=torch.float32).to(device)
            
            try:
                current_time = time.time()
                if current_time - last_prediction_time >= prediction_cooldown:
                    with torch.no_grad():
                        output = model(landmarks)
                        _, predicted_idx = torch.max(output, 1)
                        current_prediction = classes[predicted_idx.item()]
                        last_prediction_time = current_time
                
                cv2.putText(frame, f"Predicted: {current_prediction}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
            except Exception as e:
                print(f"Prediction error: {e}")
    
    return frame

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    return jsonify({'prediction': current_prediction})

@app.route('/text_to_speech')
def text_to_speech():
    text = current_prediction
    if text:
        tts = gTTS(text=text, lang='en')
        audio_path = "static/speech.mp3"
        tts.save(audio_path)
        return send_file(audio_path, mimetype='audio/mp3')
    return '', 404

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)