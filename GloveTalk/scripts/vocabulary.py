"""Canonical 22-word vocabulary for the GloveTalk words model."""
from pathlib import Path

import yaml

from paths import CONFIG

VOCABULARY_PATH = CONFIG / "words_vocabulary.yaml"
NUM_WORDS = 22


def load_words_vocabulary(path: Path = VOCABULARY_PATH) -> list[str]:
    with open(path) as f:
        data = yaml.safe_load(f)
    words = data["words"]
    if len(words) != NUM_WORDS:
        raise ValueError(f"Expected {NUM_WORDS} words in vocabulary, got {len(words)}")
    return words


def is_valid_word(label: str, vocabulary: list[str] | None = None) -> bool:
    vocab = vocabulary or load_words_vocabulary()
    return label.strip().lower() in vocab
