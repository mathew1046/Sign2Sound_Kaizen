"""Bi-LSTM classifier for glove sensor sequences."""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    BatchNormalization,
    Bidirectional,
    LSTM,
    SpatialDropout1D,
    Dense,
    Dropout,
)


def build_bilstm_classifier(
    window_size: int,
    num_features: int,
    num_classes: int,
    bilstm_units_1: int = 128,
    bilstm_units_2: int = 64,
    dense_units: int = 64,
    spatial_dropout: float = 0.3,
    dropout: float = 0.4,
) -> Sequential:
    """Build Bi-LSTM stack: (None, T, F) -> softmax."""
    model = Sequential(
        [
            Input(shape=(window_size, num_features)),
            BatchNormalization(),
            Bidirectional(LSTM(bilstm_units_1, return_sequences=True)),
            SpatialDropout1D(spatial_dropout),
            Bidirectional(LSTM(bilstm_units_2, return_sequences=False)),
            Dense(dense_units, activation="relu"),
            Dropout(dropout),
            Dense(num_classes, activation="softmax"),
        ],
        name="glovetalk_bilstm",
    )
    return model
