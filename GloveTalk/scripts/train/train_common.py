"""Shared training utilities."""
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.bilstm import build_bilstm_classifier
from paths import TRAINING_CONFIG


def load_config():
    with open(TRAINING_CONFIG) as f:
        return yaml.safe_load(f)


def train_model(
    X_train, y_train, X_val, y_val, class_names,
    window_size, num_features, task_cfg, model_cfg, optimizer_cfg,
    model_path,
):
    num_classes = len(class_names)
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)

    present_classes = np.unique(y_train)
    class_weights = compute_class_weight(
        "balanced", classes=present_classes, y=y_train
    )
    class_weight = {int(i): w for i, w in zip(present_classes, class_weights)}

    model = build_bilstm_classifier(
        window_size=window_size,
        num_features=num_features,
        num_classes=num_classes,
        bilstm_units_1=model_cfg["bilstm_units_1"],
        bilstm_units_2=model_cfg["bilstm_units_2"],
        dense_units=model_cfg["dense_units"],
        spatial_dropout=model_cfg["spatial_dropout"],
        dropout=model_cfg["dropout"],
    )

    model.compile(
        optimizer=Adam(learning_rate=optimizer_cfg["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=task_cfg["early_stopping_patience"],
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=task_cfg["reduce_lr_patience"],
            min_lr=1e-6,
        ),
        ModelCheckpoint(str(model_path), monitor="val_accuracy", save_best_only=True),
    ]

    print(f"Training {model_path.name}...")
    model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=task_cfg["epochs"],
        batch_size=task_cfg["batch_size"],
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_val, y_val_cat, verbose=0)
    print(f"Val accuracy: {accuracy * 100:.2f}%")

    preds = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print(
        classification_report(
            y_val,
            preds,
            labels=np.arange(num_classes),
            target_names=class_names,
            zero_division=0,
        )
    )

    model.save(model_path)
    return model
