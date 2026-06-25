#!/usr/bin/env python3
"""GloveTalk Bi-LSTM training entry point."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.preprocess.augment import run_augmentation
from scripts.preprocess.clean_data import run_cleaning
from scripts.train.train_alphabet import main as train_alphabet
from scripts.train.train_words import main as train_words


def main():
    parser = argparse.ArgumentParser(description="Train GloveTalk Bi-LSTM models")
    parser.add_argument("--task", choices=["both", "words", "alphabet"], default="both")
    parser.add_argument("--skip-preprocess", action="store_true")
    args = parser.parse_args()

    if not args.skip_preprocess:
        print("=== Cleaning data ===")
        run_cleaning()
        print("=== Augmenting & preparing tensors ===")
        run_augmentation(args.task)

    if args.task in ("both", "words"):
        print("=== Training words model ===")
        train_words()

    if args.task in ("both", "alphabet"):
        print("=== Training alphabet model ===")
        train_alphabet()


if __name__ == "__main__":
    main()
