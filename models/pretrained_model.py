from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, LSTM, TimeDistributed, Input
from tensorflow.keras import Sequential

def build_pretrained_model(input_shape=(30, 128, 128, 3)):
    """
    Build a violence detection model using pretrained ResNet50 for feature extraction.
    """
    # Input for video frames
    video_input = Input(shape=input_shape)

    # Pretrained ResNet50 for spatial feature extraction (frame-level)
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False  # Freeze the base model weights

    # Apply ResNet50 to each frame using TimeDistributed
    frame_features = TimeDistributed(base_model)(video_input)
    pooled_features = TimeDistributed(GlobalAveragePooling2D())(frame_features)

    # Temporal processing using LSTM
    lstm_layer = LSTM(128, activation="relu", return_sequences=False)(pooled_features)
    dropout_layer = Dropout(0.5)(lstm_layer)

    # Fully connected layer for classification
    output = Dense(1, activation="sigmoid")(dropout_layer)

    # Compile the model
    model = Model(inputs=video_input, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model