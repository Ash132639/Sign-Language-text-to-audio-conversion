import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tqdm import tqdm

mp_hands = mp.solutions.hands

def process_dataset(DATASET_PATH, save_path):
    
    DATASET_PATH = os.path.abspath(DATASET_PATH)
    save_path = os.path.abspath(save_path)
    
    print(f"Processing directory: {DATASET_PATH}")
    
    # Check if directory exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Directory {DATASET_PATH} not found")
        return
        
    dataset_X, dataset_Y = [], []
    
    try:
        labels = [d for d in os.listdir(DATASET_PATH) 
                 if os.path.isdir(os.path.join(DATASET_PATH, d))]
        labels = sorted(labels)
        print(f"Found labels: {labels}")
        
    except Exception as e:
        print(f"Error reading directory {DATASET_PATH}: {str(e)}")
        return
    
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        for label_idx, label in enumerate(labels):
            label_path = os.path.join(DATASET_PATH, label)
            print(f"\nProcessing class directory: {label_path}")
            
            try:
                files = [f for f in os.listdir(label_path) 
                        if f.endswith(('.jpg', '.png', '.jpeg'))]
                print(f"Found {len(files)} image files in {label}")
                
                for file_name in files:
                    file_path = os.path.join(label_path, file_name)
                    print(f"Processing: {file_path}", end='\r')
                    
                    # Loading the image and process with MediaPipe
                    image = cv2.imread(file_path)
                    if image is None:
                        print(f"\nFailed to load image: {file_path}")
                        continue
                        
                    image = cv2.flip(image, 1)
                    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    
                    # Extracting the hand landmarks if detected
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            landmarks = []
                            for i in range(21):  # 21 keypoints
                                landmarks.append([
                                    hand_landmarks.landmark[i].x,
                                    hand_landmarks.landmark[i].y,
                                    hand_landmarks.landmark[i].z
                                ])
                            dataset_X.append(np.array(landmarks))
                            dataset_Y.append(label_idx)
                            break
                            
            except Exception as e:
                print(f"\nError processing directory {label_path}: {str(e)}")
                continue
    
    if len(dataset_X) > 0:
        # Create directory for save_path if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save dataset
        with open(save_path, 'wb') as f:
            pickle.dump({'X': np.array(dataset_X), 'Y': np.array(dataset_Y)}, f)
        print(f"\nDataset saved at {save_path}")
        print(f"Processed {len(dataset_X)} images successfully")
    else:
        print("\nNo valid hand landmarks were detected in any images")

if __name__ == "__main__":
    # Get the current working directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Define paths relative to the script location
    dataset_dir = os.path.join(current_dir, "dataset")
    output_dir = os.path.join(current_dir, "processed_data")
    
    # Process train and validation data
    print("\nProcessing training data...")
    process_dataset(
        os.path.join(dataset_dir, "train"),
        os.path.join(output_dir, "train_data.pkl")
    )
    
    print("\nProcessing validation data...")
    process_dataset(
        os.path.join(dataset_dir, "valid"),
        os.path.join(output_dir, "valid_data.pkl")
    )