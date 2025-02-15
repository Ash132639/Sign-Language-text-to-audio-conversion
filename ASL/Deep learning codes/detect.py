import os
import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import playsound

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
        # Adjust input to fit Conv1d layers
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # Adjust for LSTM
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
def load_model(model_path, input_dim, hidden_dim, output_dim):
    model = CNNLSTMModel(input_dim, hidden_dim, output_dim)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model
# Load the trained model
def detect_frame(image, model, classes, device):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        predicted_class = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks
               landmarks = landmarks.permute(0, 2, 1)

                # Debug: Print the tensor shape
               print(f"Landmarks shape passed to model: {landmarks.shape}")  # Should be [1, 3, 21]

                # Perform inference with error handling
            try:
                    with torch.no_grad():
                        output = model(landmarks)
                        _, predicted_class_index = torch.max(output, 1)
                    predicted_class = classes[predicted_class_index.item()]
            except RuntimeError as e:
                    print(f"RuntimeError during inference: {e}")
                    predicted_class = "Error"

                # Display result on screen
                    cv2.putText(image, f"Predicted: {predicted_class}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Text-to-speech conversion
                    sound = gTTS(predicted_class, lang="en")
                    sound.save("C:\Users\aswin\Desktop\Sign-Language-text-to-audio-conversion\ASL\sign.mp3")
                    playsound.playsound("C:\Users\aswin\Desktop\Sign-Language-text-to-audio-conversion\ASL\sign.mp3")
                    os.remove("sC:\Users\aswin\Desktop\Sign-Language-text-to-audio-conversion\ASL\sign.mp3")

        return image, predicted_class

# Main detection function
def detect():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set the correct number of classes (27 in this case)
    num_classes = 27
    model = load_model('sequential_model.pth', input_dim=3, hidden_dim=64, output_dim=num_classes).to(device)
    
    # Ensure class labels are loaded in the same order as training
    classes = sorted(os.listdir('dataset/train'))
    if len(classes) != num_classes:
        print(f"Error: Mismatch between model output classes ({num_classes}) and dataset classes ({len(classes)}).")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open the webcam.")
        return

    print("Press 'ESC' to exit.")
    last_prediction = ""
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, last_prediction = detect_frame(frame, model, classes, device)
        cv2.imshow("Sign Language Detection", frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    detect()
