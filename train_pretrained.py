import os
import numpy as np
from sklearn.model_selection import train_test_split
from models.pretrained_model import build_pretrained_model

def load_data(processed_data_dir):
    """
    Load preprocessed video data and assign labels based on filenames.
    Assumes filenames include 'violent' or 'nonviolent' as identifiers.
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

if __name__ == "__main__":
    # Load preprocessed data
    processed_data_dir = "data/processed_data"
    X, y = load_data(processed_data_dir)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Build and train the model
    model = build_pretrained_model(input_shape=(30, 128, 128, 3))
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)

    # Save the trained model
    model.save("models/pretrained_violence_model.h5")