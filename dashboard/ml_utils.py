from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ML_MAPPING_PATH = Path(__file__).resolve().parent / "ml_mapping.json"


@lru_cache(maxsize=1)
def load_ml_mapping(path: str | None = None) -> dict[str, str]:
    p = Path(path) if path else ML_MAPPING_PATH
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def ml_display_name(gloss: str) -> str | None:
    mapping = load_ml_mapping()
    return mapping.get(gloss.replace("-", "_"))
