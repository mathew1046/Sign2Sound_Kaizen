"""Augmentation, windowing, and dataset preparation for GloveTalk Bi-LSTM."""
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paths import (  # noqa: E402
    ALPHABET_CLEAN_CSV,
    ALPHABET_SCALER,
    ALPHABET_TRAIN_NPZ,
    ALPHABET_VAL_NPZ,
    FEATURE_CONFIG,
    PREPROCESSED,
    TRAINING_CONFIG,
    WORDS_CLEAN_CSV,
    WORDS_SCALER,
    WORDS_TRAIN_NPZ,
    WORDS_VAL_NPZ,
)
from scripts.vocabulary import load_words_vocabulary
from scripts.preprocess.feature_engineering import (  # noqa: E402
    NUM_FEATURES,
    preprocess_raw_frame,
    raw_sequence_to_features,
    save_feature_config,
    sliding_windows,
)

RAW_COLUMNS = [
    "L_qw", "L_qx", "L_qy", "L_qz",
    "L_f1", "L_f2", "L_f3", "L_f4", "L_f5",
    "R_qw", "R_qx", "R_qy", "R_qz",
    "R_f1", "R_f2", "R_f3", "R_f4", "R_f5",
]


def load_config():
    with open(TRAINING_CONFIG) as f:
        return yaml.safe_load(f)


def add_gaussian_noise(window: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    noisy = window + np.random.normal(0, sigma, window.shape).astype(np.float32)
    return noisy.astype(np.float32)


def time_shift(window: np.ndarray, max_shift: int = 3) -> np.ndarray:
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return window.copy()
    return np.roll(window, shift, axis=0).astype(np.float32)


def time_scale_window(window: np.ndarray, min_len: int = 24, max_len: int = 36) -> np.ndarray:
    target_len = window.shape[0]
    new_len = np.random.randint(min_len, max_len + 1)
    old_idx = np.linspace(0, target_len - 1, new_len)
    scaled = np.zeros((target_len, window.shape[1]), dtype=np.float32)
    for feat in range(window.shape[1]):
        scaled[:, feat] = np.interp(
            np.linspace(0, new_len - 1, target_len),
            np.arange(new_len),
            np.interp(old_idx, np.arange(target_len), window[:, feat]),
        )
    return scaled.astype(np.float32)


def augment_window(window: np.ndarray, apply_time_ops: bool = True) -> np.ndarray:
    aug = add_gaussian_noise(window)
    if apply_time_ops:
        if np.random.rand() < 0.5:
            aug = time_shift(aug)
        if np.random.rand() < 0.5:
            aug = time_scale_window(aug)
    return aug.astype(np.float32)


def jitter_raw_frame(frame: np.ndarray, flex_sigma: float = 0.02) -> np.ndarray:
    noisy = preprocess_raw_frame(frame).copy()
    for idx in [4, 5, 6, 7, 8, 13, 14, 15, 16, 17]:
        noisy[idx] = np.clip(noisy[idx] + np.random.normal(0, flex_sigma), 0.0, 1.0)
    axis = np.random.randn(3).astype(np.float32)
    axis /= np.linalg.norm(axis) + 1e-8
    angle = np.random.uniform(-0.05, 0.05)
    half = angle / 2.0
    dq = np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float32)
    from scripts.preprocess.feature_engineering import quaternion_multiply

    for slc in [slice(0, 4), slice(9, 13)]:
        q = noisy[slc]
        noisy[slc] = quaternion_multiply(dq, q)
        norm = np.linalg.norm(noisy[slc])
        if norm > 1e-8:
            noisy[slc] /= norm
    return noisy


def build_pseudo_sequence(base_frame: np.ndarray, seq_len: int) -> np.ndarray:
    return np.stack([jitter_raw_frame(base_frame) for _ in range(seq_len)])


def balance_samples(X_list, y_list, target_count: int | None = None):
    if not X_list:
        return X_list, y_list
    counts = Counter(y_list)
    max_count = target_count or max(counts.values())
    balanced_X, balanced_y = list(X_list), list(y_list)
    rng = np.random.default_rng(42)
    for label, count in counts.items():
        indices = [i for i, y in enumerate(y_list) if y == label]
        while counts.get(label, 0) < max_count:
            idx = rng.choice(indices)
            balanced_X.append(X_list[idx].copy())
            balanced_y.append(label)
            counts[label] = counts.get(label, 0) + 1
    return balanced_X, balanced_y


def fit_scaler(X: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    n, t, f = X.shape
    scaler.fit(X.reshape(n * t, f))
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, t, f = X.shape
    scaled = scaler.transform(X.reshape(n * t, f))
    return scaled.reshape(n, t, f).astype(np.float32)


def apply_augmentation(X_list, y_list, multiplier: int, apply_time_ops: bool):
    aug_X, aug_y = [], []
    for x, y in zip(X_list, y_list):
        aug_X.append(x)
        aug_y.append(y)
        for _ in range(multiplier - 1):
            aug_X.append(augment_window(x, apply_time_ops=apply_time_ops))
            aug_y.append(y)
    return aug_X, aug_y


def prepare_words_dataset(cfg: dict) -> None:
    df = pd.read_csv(WORDS_CLEAN_CSV)
    window_size = cfg["window_size"]
    stride = cfg["window_stride"]
    target_frames = cfg["words"]["target_frames"]
    dt = cfg["dt"]
    augment_mult = cfg["words"]["augment_multiplier"]

    take_windows = []
    take_labels = []
    num_takes = len(df) // target_frames
    for take_idx in range(num_takes):
        chunk = df.iloc[take_idx * target_frames : (take_idx + 1) * target_frames]
        label = chunk["label"].iloc[0]
        raw = chunk[RAW_COLUMNS].values.astype(np.float32)
        features = raw_sequence_to_features(raw, dt=dt)
        windows = sliding_windows(features, window_size, stride)
        if windows:
            take_windows.append(windows)
            take_labels.append(label)

    take_indices = np.arange(len(take_labels))
    train_takes, val_takes = train_test_split(
        take_indices,
        test_size=cfg["train_val_split"],
        random_state=cfg["random_state"],
        stratify=take_labels,
    )

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    for idx in train_takes:
        for window in take_windows[idx]:
            X_train_list.append(window)
            y_train_list.append(take_labels[idx])
    for idx in val_takes:
        for window in take_windows[idx]:
            X_val_list.append(window)
            y_val_list.append(take_labels[idx])

    print(f"  Words base windows: train={len(X_train_list)}, val={len(X_val_list)} (split by take)")
    X_train_list, y_train_list = balance_samples(X_train_list, y_train_list)
    X_train_list, y_train_list = apply_augmentation(
        X_train_list, y_train_list, augment_mult, apply_time_ops=True
    )

    vocabulary = load_words_vocabulary()
    label_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
    y_train = np.array([label_to_idx[y] for y in y_train_list], dtype=np.int64)
    y_val = np.array([label_to_idx[y] for y in y_val_list], dtype=np.int64)
    X_train = np.stack(X_train_list).astype(np.float32)
    X_val = np.stack(X_val_list).astype(np.float32)

    scaler = fit_scaler(X_train)
    X_train = apply_scaler(X_train, scaler)
    X_val = apply_scaler(X_val, scaler)

    PREPROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez(WORDS_TRAIN_NPZ, X=X_train, y=y_train)
    np.savez(WORDS_VAL_NPZ, X=X_val, y=y_val)
    joblib.dump(scaler, WORDS_SCALER)
    np.save(PREPROCESSED / "words_classes.npy", np.array(vocabulary))
    save_feature_config(FEATURE_CONFIG, window_size, dt)
    print(f"  Words train: {X_train.shape}, val: {X_val.shape}, classes: {len(vocabulary)} (fixed 22-word vocabulary)")


def prepare_alphabet_dataset(cfg: dict) -> None:
    df = pd.read_csv(ALPHABET_CLEAN_CSV)
    seq_len = cfg["alphabet"]["pseudo_sequence_length"]
    variants = cfg["alphabet"]["variants_per_sample"]
    augment_mult = cfg["alphabet"]["augment_multiplier"]
    dt = cfg["dt"]
    window_size = cfg["window_size"]

    sample_sequences = []
    sample_labels = []
    for row_idx, row in df.iterrows():
        label = row["label"]
        base = row[RAW_COLUMNS].values.astype(np.float32)
        sequences = []
        for _ in range(variants):
            pseudo = build_pseudo_sequence(base, seq_len)
            features = raw_sequence_to_features(pseudo, dt=dt)
            if len(features) >= window_size:
                sequences.append(features[:window_size])
        if sequences:
            sample_sequences.append(sequences)
            sample_labels.append(label)

    sample_indices = np.arange(len(sample_labels))
    label_counts = Counter(sample_labels)
    singletons = {lbl for lbl, cnt in label_counts.items() if cnt < 2}
    if singletons:
        train_samples = [i for i in sample_indices if sample_labels[i] in singletons]
        pool = [i for i in sample_indices if sample_labels[i] not in singletons]
        pool_labels = [sample_labels[i] for i in pool]
        pool_train, pool_val = train_test_split(
            pool,
            test_size=cfg["train_val_split"],
            random_state=cfg["random_state"],
            stratify=pool_labels,
        )
        train_samples = train_samples + list(pool_train)
        val_samples = list(pool_val)
    else:
        train_samples, val_samples = train_test_split(
            sample_indices,
            test_size=cfg["train_val_split"],
            random_state=cfg["random_state"],
            stratify=sample_labels,
        )

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    for idx in train_samples:
        for seq in sample_sequences[idx]:
            X_train_list.append(seq)
            y_train_list.append(sample_labels[idx])
    for idx in val_samples:
        X_val_list.append(sample_sequences[idx][0])
        y_val_list.append(sample_labels[idx])

    print(f"  Alphabet base sequences: train={len(X_train_list)}, val={len(X_val_list)} (split by sample)")
    X_train_list, y_train_list = balance_samples(X_train_list, y_train_list, target_count=30)
    X_train_list, y_train_list = apply_augmentation(
        X_train_list, y_train_list, augment_mult, apply_time_ops=False
    )

    encoder = LabelEncoder()
    encoder.fit(y_train_list + y_val_list)
    y_train = encoder.transform(y_train_list)
    y_val = encoder.transform(y_val_list)
    X_train = np.stack(X_train_list).astype(np.float32)
    X_val = np.stack(X_val_list).astype(np.float32)

    scaler = fit_scaler(X_train)
    X_train = apply_scaler(X_train, scaler)
    X_val = apply_scaler(X_val, scaler)

    np.savez(ALPHABET_TRAIN_NPZ, X=X_train, y=y_train)
    np.savez(ALPHABET_VAL_NPZ, X=X_val, y=y_val)
    joblib.dump(scaler, ALPHABET_SCALER)
    np.save(PREPROCESSED / "alphabet_classes.npy", encoder.classes_)
    print(f"  Alphabet train: {X_train.shape}, val: {X_val.shape}, classes: {len(encoder.classes_)}")


def run_augmentation(task: str = "both") -> None:
    cfg = load_config()
    if task in ("both", "words"):
        print("Preparing words dataset...")
        prepare_words_dataset(cfg)
    if task in ("both", "alphabet"):
        print("Preparing alphabet dataset...")
        prepare_alphabet_dataset(cfg)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["both", "words", "alphabet"], default="both")
    args = parser.parse_args()
    run_augmentation(args.task)
