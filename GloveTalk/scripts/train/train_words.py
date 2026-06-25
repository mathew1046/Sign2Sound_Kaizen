"""Train Bi-LSTM word classifier."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paths import WORDS_CLASSES, WORDS_CLASSES_PREPROCESSED, WORDS_MODEL, WORDS_TRAIN_NPZ, WORDS_VAL_NPZ
from scripts.train.train_common import load_config, train_model


def main():
    cfg = load_config()
    train = np.load(WORDS_TRAIN_NPZ)
    val = np.load(WORDS_VAL_NPZ)
    class_names = np.load(WORDS_CLASSES_PREPROCESSED, allow_pickle=True)

    train_model(
        train["X"], train["y"], val["X"], val["y"], class_names,
        window_size=cfg["window_size"],
        num_features=cfg["num_features"],
        task_cfg=cfg["words"],
        model_cfg=cfg["model"],
        optimizer_cfg=cfg["optimizer"],
        model_path=WORDS_MODEL,
    )
    np.save(WORDS_CLASSES, class_names)
    print(f"Saved model to {WORDS_MODEL}")


if __name__ == "__main__":
    main()
