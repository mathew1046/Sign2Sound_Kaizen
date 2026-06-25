"""Resolve paths relative to the GloveTalk project root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_RAW = ROOT / "data" / "raw"
PREPROCESSED = ROOT / "preprocessed"
WEIGHTS = ROOT / "weights"
CONFIG = ROOT / "config"

ALPHABET_RAW_CSV = DATA_RAW / "alphabet_dataset.csv"
WORDS_RAW_CSV = DATA_RAW / "sign_language_dataset.csv"

ALPHABET_CLEAN_CSV = PREPROCESSED / "alphabet_clean.csv"
WORDS_CLEAN_CSV = PREPROCESSED / "words_clean.csv"

ALPHABET_TRAIN_NPZ = PREPROCESSED / "alphabet_train.npz"
ALPHABET_VAL_NPZ = PREPROCESSED / "alphabet_val.npz"
WORDS_TRAIN_NPZ = PREPROCESSED / "words_train.npz"
WORDS_VAL_NPZ = PREPROCESSED / "words_val.npz"

ALPHABET_SCALER = PREPROCESSED / "alphabet_scaler.pkl"
WORDS_SCALER = PREPROCESSED / "words_scaler.pkl"
FEATURE_CONFIG = PREPROCESSED / "feature_config.json"

ALPHABET_MODEL = WEIGHTS / "alphabet_bilstm.keras"
ALPHABET_CLASSES = WEIGHTS / "alphabet_classes.npy"
ALPHABET_CLASSES_PREPROCESSED = PREPROCESSED / "alphabet_classes.npy"
WORDS_MODEL = WEIGHTS / "words_bilstm.keras"
WORDS_CLASSES = WEIGHTS / "words_classes.npy"
WORDS_CLASSES_PREPROCESSED = PREPROCESSED / "words_classes.npy"

TRAINING_CONFIG = CONFIG / "training.yaml"
