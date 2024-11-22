import os
import numpy as np
from sklearn.model_selection import train_test_split
from models.violence_model import build_model

def load_data(processed_data_dir):
    """
    Load preprocessed .npy files and generate labels.
    """
    X = []
    y = []
    for file_name in os.listdir(processed_data_dir):
        if file_name.endswith(".npy"):
            file_path = os.path.join(processed_data_dir, file_name)
            X.append(np.load(file_path))
            if "violent" in file_name:
                y.append(1)  # Violent
            else:
                y.append(0)  # Non-violent
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Load data
    X, y = load_data("data/processed_data")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model(input_shape=X_train.shape[1:])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)

    # Save the trained model
    model.save("models/violence_detection_model.h5")