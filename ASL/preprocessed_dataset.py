import os
import cv2
import pickle
import mediapipe as mp
import numpy as np
from tqdm import tqdm
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def preprocessed_dataset(DATASET_PATH, save_path):
    whole_dataset_x = []
    whole_dataset_Y = []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        for i, main_dir in enumerate(os.listdir(DATASET_PATH)):
            sub_dir = os.path.join(DATASET_PATH, main_dir)
            for file_name in tqdm(os.listdir(sub_dir), desc=f"Step {i+1}/27, {main_dir}"):
                file_name = os.path.join(sub_dir, file_name)

                current_file = []
                image = cv2.flip(cv2.imread(file_name), 1)
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if not results.multi_hand_landmarks:
                    continue
                image_height, image_width, _ = image.shape
                annotated_image = image.copy()
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(0, 21):
                        current_file.append(np.asarray([hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y, hand_landmarks.landmark[i].z], dtype=np.float))
                    
                    whole_dataset_x.append(np.array(current_file))
                    whole_dataset_Y.append(main_dir)
                    break
            
    print(np.array(whole_dataset_x).shape)
    print(np.array(whole_dataset_Y).shape)

    pickle.dump({'X': np.array(whole_dataset_x), 'Y': np.array(whole_dataset_Y)},open(save_path, 'wb'))


if __name__ == "__main__":
    preprocessed_dataset("dataset/train", 'preprocessed_dataset_train.h5')
    preprocessed_dataset("dataset/valid", 'preprocessed_dataset_valid.h5')
