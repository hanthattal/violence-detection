import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

def preprocess_video(video_file, seq_length=30, frame_size=(128, 128)):
    """
    Preprocess a single video file by extracting frames.
    """
    cap = cv2.VideoCapture(video_file)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)
        if len(frames) == seq_length:  # Stop once 30 frames are collected
            break
    cap.release()

    # Check if enough frames were extracted
    if len(frames) < seq_length:
        print(f"Skipping {video_file}: Not enough frames (found {len(frames)})")
        return None

    # Normalize and return
    return np.array(frames)[np.newaxis, ...] / 255.0  # Add batch dimension

def evaluate_model(dataset_dir, model_path="models/violence_detection_model.h5"):
    """
    Evaluate the model on the entire dataset and calculate accuracy.
    """
    # Load the trained model
    model = load_model(model_path)

    # Prepare lists for predictions and ground truth
    y_true = []
    y_pred = []

    for class_name, label in [("violent", 1), ("non-violent", 0)]:
        class_dir = os.path.join(dataset_dir, class_name)
        for root, _, files in os.walk(class_dir):
            for file_name in files:
                if file_name.endswith(".mp4"):
                    video_path = os.path.join(root, file_name)
                    # Preprocess video
                    input_data = preprocess_video(video_path)
                    if input_data is None:  # Skip if not enough frames
                        continue
                    # Predict
                    prediction = model.predict(input_data)
                    predicted_label = 1 if prediction[0][0] > 0.5 else 0
                    # Append results
                    y_true.append(label)
                    y_pred.append(predicted_label)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy on dataset: {accuracy * 100:.2f}%")

    # Return predictions and ground truth for further analysis if needed
    return y_true, y_pred

if __name__ == "__main__":
    # Evaluate the model on the dataset
    dataset_dir = "data/violence-detection-dataset"
    y_true, y_pred = evaluate_model(dataset_dir)