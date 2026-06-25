"""Label map and gloss normalization from repo vocabulary CSV."""

from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path

from dashboard.config import LABEL_MAP_PATH, VOCAB_CSV

ALIASES: dict[str, str] = {
    "hi": "hello",
    "thanks": "thank_you",
    "thankyou": "thank_you",
    "goodmorning": "good_morning",
    "goodafternoon": "good_afternoon",
    "goodevening": "good_evening",
    "howareyou": "how_are_you",
    "cellphone": "cell_phone",
    "big": "big_large",
    "large": "big_large",
}


def slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def build_label_map_from_vocab(vocab_csv: Path = VOCAB_CSV) -> dict[str, int]:
    label_map: dict[str, int] = {}
    extra_id = 50
    with vocab_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gloss = row["canonical_gloss"].strip()
            in50 = row.get("in_include50_mspt", "").strip().lower() == "yes"
            if in50:
                label_id = int(row["include50_label_id"])
            else:
                label_id = extra_id
                extra_id += 1
            label_map[gloss] = label_id
    return label_map


def ensure_label_map(path: Path = LABEL_MAP_PATH) -> Path:
    if path.is_file():
        return path
    label_map = build_label_map_from_vocab()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(label_map, indent=2, sort_keys=True), encoding="utf-8")
    return path


@lru_cache(maxsize=1)
def load_label_map(path: str | None = None) -> tuple[dict[str, int], dict[int, str]]:
    p = Path(path) if path else ensure_label_map()
    with p.open(encoding="utf-8") as f:
        label_map = json.load(f)
    idx_to_label = {int(v): k for k, v in label_map.items()}
    return label_map, idx_to_label


def canonical_label(slug: str, valid: set[str]) -> str | None:
    if slug in valid:
        return slug
    return None


def signer_id_from_path(path: str) -> str:
    stem = Path(path).stem
    m = re.match(r"MVI_(\d+)", stem)
    return m.group(1) if m else stem[:8]
