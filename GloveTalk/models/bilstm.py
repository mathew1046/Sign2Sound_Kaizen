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
from tensorflow.keras.regularizers import l2


def build_bilstm_classifier(
    window_size: int,
    num_features: int,
    num_classes: int,
    bilstm_units_1: int = 128,
    bilstm_units_2: int = 64,
    dense_units: int = 64,
    spatial_dropout: float = 0.3,
    dropout: float = 0.4,
    lstm_dropout: float = 0.0,
    recurrent_dropout: float = 0.0,
    l2_reg: float = 0.0,
) -> Sequential:
    """Build Bi-LSTM stack: (None, T, F) -> softmax."""
    reg = l2(l2_reg) if l2_reg > 0 else None
    model = Sequential(
        [
            Input(shape=(window_size, num_features)),
            BatchNormalization(),
            Bidirectional(
                LSTM(
                    bilstm_units_1,
                    return_sequences=True,
                    dropout=lstm_dropout,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=reg,
                )
            ),
            SpatialDropout1D(spatial_dropout),
            Bidirectional(
                LSTM(
                    bilstm_units_2,
                    return_sequences=False,
                    dropout=lstm_dropout,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=reg,
                )
            ),
            Dense(dense_units, activation="relu", kernel_regularizer=reg),
            Dropout(dropout),
            Dense(num_classes, activation="softmax", kernel_regularizer=reg),
        ],
        name="glovetalk_bilstm",
    )
    return model
