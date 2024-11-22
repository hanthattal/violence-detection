import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def predict_violence(video_file, model_path="models/violence_detection_model.h5"):
    """
    Predict whether the video contains violence.
    """
    # Check if the video file exists
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"File not found: {video_file}")

    # Load the trained model
    model = load_model(model_path)

    # Preprocess video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError(f"Error: Couldn't open video stream for file: {video_file}")

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (128, 128))
        frames.append(frame)
        if len(frames) == 30:  # Stop once 30 frames are collected
            break
    cap.release()

    # Check if there are enough frames
    if len(frames) < 30:
        raise ValueError(f"Video {video_file} does not have enough frames. Found {len(frames)} frames.")

    # Prepare data for prediction
    input_data = np.array(frames)[np.newaxis, ...] / 255.0  # Normalize and add batch dimension

    # Make prediction
    prediction = model.predict(input_data)
    return "Violent" if prediction[0][0] > 0.5 else "Non-Violent"

if __name__ == "__main__":
    try:
        # Test on a file called 1.mp4 in the dataset
        result = predict_violence("data/violence-detection-dataset/violent/cam1/1.mp4")
        print("Prediction:", result)
    except Exception as e:
        print("Error:", e)