"""Train Bi-LSTM alphabet classifier."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paths import ALPHABET_CLASSES, ALPHABET_CLASSES_PREPROCESSED, ALPHABET_MODEL, ALPHABET_TRAIN_NPZ, ALPHABET_VAL_NPZ
from scripts.train.train_common import load_config, train_model


def main():
    cfg = load_config()
    train = np.load(ALPHABET_TRAIN_NPZ)
    val = np.load(ALPHABET_VAL_NPZ)
    class_names = np.load(ALPHABET_CLASSES_PREPROCESSED, allow_pickle=True)

    train_model(
        train["X"], train["y"], val["X"], val["y"], class_names,
        window_size=cfg["window_size"],
        num_features=cfg["num_features"],
        task_cfg=cfg["alphabet"],
        model_cfg=cfg["model"],
        optimizer_cfg=cfg["optimizer"],
        model_path=ALPHABET_MODEL,
    )
    np.save(ALPHABET_CLASSES, class_names)
    print(f"Saved model to {ALPHABET_MODEL}")


if __name__ == "__main__":
    main()
