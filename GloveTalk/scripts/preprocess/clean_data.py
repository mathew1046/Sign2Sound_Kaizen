"""Clean and validate GloveTalk CSV datasets."""
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paths import ALPHABET_CLEAN_CSV, ALPHABET_RAW_CSV, PREPROCESSED, WORDS_CLEAN_CSV, WORDS_RAW_CSV
from scripts.vocabulary import load_words_vocabulary

RAW_COLUMNS = [
    "L_qw", "L_qx", "L_qy", "L_qz",
    "L_f1", "L_f2", "L_f3", "L_f4", "L_f5",
    "R_qw", "R_qx", "R_qy", "R_qz",
    "R_f1", "R_f2", "R_f3", "R_f4", "R_f5",
]

ALPHABET_COLUMNS = ["label"] + RAW_COLUMNS
WORDS_COLUMNS = ["timestamp", "label"] + RAW_COLUMNS
TARGET_FRAMES = 50


def _is_valid_quaternion_row(row) -> bool:
    quat_cols = ["L_qw", "L_qx", "L_qy", "L_qz", "R_qw", "R_qx", "R_qy", "R_qz"]
    for col in quat_cols:
        if abs(float(row[col])) > 1.5:
            return False
    return True


def clean_alphabet() -> pd.DataFrame:
    print(f"Cleaning alphabet data from {ALPHABET_RAW_CSV}...")
    valid_rows = []
    dropped = 0

    with open(ALPHABET_RAW_CSV, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) != len(ALPHABET_COLUMNS):
                dropped += 1
                continue
            record = dict(zip(ALPHABET_COLUMNS, row))
            try:
                series = pd.Series(
                    {c: float(record[c]) if c != "label" else record[c] for c in ALPHABET_COLUMNS}
                )
            except ValueError:
                dropped += 1
                continue
            if not _is_valid_quaternion_row(series):
                dropped += 1
                continue
            valid_rows.append(record)

    df = pd.DataFrame(valid_rows, columns=ALPHABET_COLUMNS)
    PREPROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(ALPHABET_CLEAN_CSV, index=False)
    print(f"  Alphabet: kept {len(df)} rows, dropped {dropped} corrupted rows")
    return df


def clean_words() -> pd.DataFrame:
    print(f"Cleaning words data from {WORDS_RAW_CSV}...")
    vocabulary = set(load_words_vocabulary())
    expected_cols = len(WORDS_COLUMNS)
    valid_rows = []
    fixed = 0
    previous_valid = None

    with open(WORDS_RAW_CSV, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) == expected_cols:
                valid_rows.append(row)
                previous_valid = row
            elif previous_valid is not None:
                valid_rows.append(previous_valid)
                fixed += 1
            else:
                fixed += 1

    remainder = len(valid_rows) % TARGET_FRAMES
    if remainder:
        valid_rows = valid_rows[:-remainder]
        print(f"  Dropped {remainder} trailing incomplete frames")

    df = pd.DataFrame(valid_rows, columns=WORDS_COLUMNS)
    for col in RAW_COLUMNS:
        df[col] = df[col].astype(float)
    df["timestamp"] = df["timestamp"].astype(float)
    df["label"] = df["label"].str.strip().str.lower()

    unknown = sorted(set(df["label"].unique()) - vocabulary)
    if unknown:
        dropped = len(df[df["label"].isin(unknown)])
        df = df[df["label"].isin(vocabulary)]
        print(f"  Dropped {dropped} frames with labels outside the 22-word vocabulary: {unknown}")

    if len(set(df["label"].unique())) != 22:
        present = sorted(df["label"].unique())
        raise ValueError(
            f"Words dataset must contain exactly 22 labels; found {len(present)}: {present}"
        )

    PREPROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(WORDS_CLEAN_CSV, index=False)
    num_takes = len(df) // TARGET_FRAMES
    print(f"  Words: {len(df)} frames ({num_takes} takes), repaired {fixed} rows")
    return df


def run_cleaning():
    return clean_alphabet(), clean_words()


if __name__ == "__main__":
    run_cleaning()
