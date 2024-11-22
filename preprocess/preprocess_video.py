import os
import cv2
import numpy as np

def preprocess_videos(input_dir, output_dir, seq_length=30, frame_size=(128, 128)):
    """
    Extract frames from videos and save them as .npy files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for class_name in ["violent", "non-violent"]:
        class_dir = os.path.join(input_dir, class_name)
        for cam_folder in ["cam1", "cam2"]:
            cam_dir = os.path.join(class_dir, cam_folder)
            for video_file in os.listdir(cam_dir):
                if video_file.endswith(".mp4"):
                    video_path = os.path.join(cam_dir, video_file)
                    cap = cv2.VideoCapture(video_path)
                    frames = []
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.resize(frame, frame_size)
                        frames.append(frame)
                        if len(frames) == seq_length:
                            output_file = os.path.join(
                                output_dir, f"{class_name}_{cam_folder}_{video_file.replace('.mp4', '.npy')}"
                            )
                            np.save(output_file, np.array(frames) / 255.0)  # Normalize frames
                            frames = []
                    cap.release()
                    print(f"Processed {video_file}")

if __name__ == "__main__":
    preprocess_videos("data/violence-detection-dataset", "data/processed_data", seq_length=30, frame_size=(128, 128))