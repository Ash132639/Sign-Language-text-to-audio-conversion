import os
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('sequencial_model.h5')
classes = os.listdir('dataset/train')
camera = None

def get_frame():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    
    success, frame = camera.read()
    if not success:
        return None
        
    # Process frame for ASL detection
    processed_frame = cv2.resize(frame, (224, 224))
    processed_frame = np.expand_dims(processed_frame, axis=0)
    prediction = model.predict(processed_frame)
    predicted_class = classes[np.argmax(prediction)]
    
    # Draw prediction on frame
    cv2.putText(frame, predicted_class, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
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
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

if __name__ == '__main__':
    app.run(debug=True)