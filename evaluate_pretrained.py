import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(processed_data_dir):
    """
    Load preprocessed video data and assign labels based on filenames.
    """
    X = []
    y = []
    for file_name in os.listdir(processed_data_dir):
        if file_name.endswith(".npy"):
            file_path = os.path.join(processed_data_dir, file_name)
            X.append(np.load(file_path))

            # Assign label based on filename
            if "violent" in file_name:
                y.append(1)  # Violent
            elif "nonviolent" in file_name:
                y.append(0)  # Non-violent
            else:
                print(f"Skipping file with unknown label: {file_name}")

    return np.array(X), np.array(y)

def evaluate_model_pretrained(processed_data_dir, model_path="models/pretrained_violence_model.h5"):
    """
    Evaluate the pretrained model on the dataset and calculate metrics.
    """
    # Load the trained model
    model = load_model(model_path)

    # Load data and labels
    X, y = load_data(processed_data_dir)

    # Predict on the dataset
    predictions = model.predict(X)
    y_pred = (predictions > 0.5).astype(int)  # Convert probabilities to binary labels

    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Non-Violent", "Violent"], labels=[0, 1]))

    # Generate confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

if __name__ == "__main__":
    processed_data_dir = "data/processed_data"  
    evaluate_model_pretrained(processed_data_dir)