import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import time
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import gtts
import playsound




def detect_frame(image, model, classes):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    fps, predicted_class = float(), str()
    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        t1 = time.time()
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                current_data = []
                for i in range(0, 21):
                    current_data.append(np.asarray([hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y, hand_landmarks.landmark[i].z], dtype=np.float))

            pred = model.predict(np.expand_dims(np.asarray(current_data), axis=0))
            predicted_class_indices = np.argmax(pred, axis = 1)
            predicted_class = classes[predicted_class_indices[0]]
            

            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        t2 = time.time()
        fps = int(1./ (t2 - t1))
        image = cv2.flip(image, 1)
        cv2.putText(image, f"FPS: {fps}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if predicted_class == "SS":
            predicted_class = "Space"
        cv2.putText(image, f"Predicted Class: {predicted_class}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if len(predicted_class) != 0:
            sound= gtts.gTTS(predicted_class,lang="en")
            sound.save("sign.mp3") 
            playsound.playsound("C:/Users/aswin/OneDrive/Desktop/ASL/sign.mp3")
            os.remove("C:/Users/aswin/OneDrive/Desktop/ASL/sign.mp3")
        return image, predicted_class


def detect():
    model = load_model('sequencial_model.h5')
    print(model.summary())
    print("Model Loaded...")
    classes = os.listdir('dataset/train')

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)
    fps, predicted_class = float(), str()
    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            t1 = time.time()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    current_data = []
                    for i in range(0, 21):
                        current_data.append(np.asarray([hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y, hand_landmarks.landmark[i].z], dtype=np.float))

                pred = model.predict(np.expand_dims(np.asarray(current_data), axis=0))
                predicted_class_indices = np.argmax(pred, axis = 1)
                predicted_class = classes[predicted_class_indices[0]]
                

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            t2 = time.time()
            fps = int(1./ (t2 - t1))
            image = cv2.flip(image, 1)
            cv2.putText(image, f"FPS: {fps}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image, f"Predicted Class: {predicted_class}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)        
                
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

if __name__ == "__main__":
    detect()
