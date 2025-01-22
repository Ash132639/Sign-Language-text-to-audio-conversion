import os
import cv2
import mediapipe as mp
import torch
from torchvision import transforms
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def remove_existing_files(directory):
    """
    Remove files with `_annotated.png` and `_landmarks.pt` extensions in the directory.

    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    for file in os.listdir(directory):
        if file.endswith("_annotated.png") or file.endswith("_landmarks.pt"):
            file_path = os.path.join(directory, file)
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except OSError as e:
                print(f"Error removing file {file_path}: {e}")

def process_image(image_path, hands):
    """
    Process a single image to detect hands and return hand landmarks as PyTorch tensors.

    Args:
        image_path (str): Path to the image file.
        hands: Mediapipe Hands object.

    Returns:
        image_tensor (torch.Tensor): Transformed image tensor.
        hand_tensors (list): List of PyTorch tensors representing hand landmarks.
    """
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None, None

    image = cv2.flip(cv2.imread(image_path), 1)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None, None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None, None

    annotated_image = image.copy()
    hand_tensors = []

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        hand_tensors.append(torch.tensor(landmarks, dtype=torch.float32))

    image_tensor = transform(Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)))
    return image_tensor, hand_tensors

def test_dataset():
    """
    Process all images in the dataset directory to detect hands and save annotated images and landmarks.
    """
    dataset_dir = 'dataset/test'
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return

    print("Cleaning up existing annotated images and landmarks...")
    remove_existing_files(dataset_dir)

    IMAGE_FILES = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if x.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not IMAGE_FILES:
        print(f"No valid image files found in {dataset_dir}")
        return

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:
        for idx, file in enumerate(IMAGE_FILES):
            print(f"Processing file {idx + 1}/{len(IMAGE_FILES)}: {file}")

            image_tensor, hand_tensors = process_image(file, hands)

            if image_tensor is not None and hand_tensors is not None:
                output_image = torch.permute(image_tensor, (1, 2, 0)).numpy()
                output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
                annotated_path = file[:-4] + '_annotated.png'
                cv2.imwrite(annotated_path, cv2.flip(output_image, 1))
                
                landmarks_path = file[:-4] + '_landmarks.pt'
                torch.save(hand_tensors, landmarks_path)

                print(f"Processed {file}: Found {len(hand_tensors)} hands")
                print(f"Annotated image saved to: {annotated_path}")
                print(f"Landmarks saved to: {landmarks_path}")
            else:
                print(f"No hands detected in {file}")

if __name__ == "__main__":
    test_dataset()
