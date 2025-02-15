# app.py
import os
import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify
from gtts import gTTS
import playsound
import base64

app = Flask(__name__)

# CNN-LSTM Model definition (same as your original code)
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

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 27
model = None
classes = None
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def load_model(model_path, input_dim, hidden_dim, output_dim):
    global model, classes
    model = CNNLSTMModel(input_dim, hidden_dim, output_dim)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    classes = sorted(os.listdir('dataset/train'))
    return model

def process_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    predicted_class = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = torch.tensor([[
                [landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark
            ]], dtype=torch.float32)
            
            try:
                with torch.no_grad():
                    output = model(landmarks)
                    _, predicted_class_index = torch.max(output, 1)
                predicted_class = classes[predicted_class_index.item()]
                
                # Draw prediction on frame
                cv2.putText(image, f"Predicted: {predicted_class}", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 0, 255), 2)
            except RuntimeError as e:
                print(f"RuntimeError during inference: {e}")
                predicted_class = "Error"

    return image, predicted_class

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        processed_frame, prediction = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
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

if __name__ == '__main__':
    # Load model before starting the app
    load_model('sequential_model.pth', input_dim=3, hidden_dim=64, output_dim=num_classes)
    app.run(debug=True)