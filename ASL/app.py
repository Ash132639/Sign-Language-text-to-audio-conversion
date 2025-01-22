import os
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from main import model, classes  # Importing model and classes from main.py

app = Flask(__name__)

# Initialize the camera globally
camera = cv2.VideoCapture(0)

def get_frame():
    """Captures and processes frames for ASL detection"""
    success, frame = camera.read()
    if not success:
        return None

    # Preprocess the frame for prediction
    processed_frame = cv2.resize(frame, (224, 224))
    processed_frame = np.expand_dims(processed_frame, axis=0)
    processed_frame = processed_frame / 255.0  # Normalize the image

    # Predict ASL gesture
    prediction = model.predict(processed_frame)
    predicted_class = classes[np.argmax(prediction)]

    # Draw prediction on frame
    cv2.putText(frame, predicted_class, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

@app.route('/')
def index():
    """Loads the main webpage."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    def generate():
        while True:
            frame = get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about():
    """Loads the About page."""
    return render_template('about.html')

@app.route('/team')
def team():
    """Loads the Team page."""
    return render_template('team.html')

if __name__ == '__main__':
    app.run(debug=True)
