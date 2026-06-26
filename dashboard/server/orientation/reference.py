"""Load orientation reference documents built from RTMLIB corpus."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from dashboard.config import ORIENTATION_REFS_DIR, PROJECT_ROOT
from dashboard.ml_utils import ml_display_name
from dashboard.server.orientation.schemas import OrientationReferenceMeta


def _resolve_skeleton_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_file():
        return p
    candidate = PROJECT_ROOT / path_str
    if candidate.is_file():
        return candidate
    return p


@lru_cache(maxsize=1)
def load_orientation_index() -> dict:
    index_path = ORIENTATION_REFS_DIR / "index.json"
    if not index_path.is_file():
        return {"glosses": {}, "num_glosses": 0}
    return json.loads(index_path.read_text(encoding="utf-8"))


def list_orientation_glosses() -> list[str]:
    idx = load_orientation_index()
    return sorted(idx.get("glosses", {}).keys())


def get_orientation_reference(gloss: str) -> dict | None:
    gloss = gloss.strip().lower().replace(" ", "_")
    ref_path = ORIENTATION_REFS_DIR / f"{gloss}.json"
    if not ref_path.is_file():
        idx = load_orientation_index()
        rel = idx.get("glosses", {}).get(gloss)
        if rel:
            ref_path = ORIENTATION_REFS_DIR / rel
    if not ref_path.is_file():
        return None
    return json.loads(ref_path.read_text(encoding="utf-8"))


def reference_meta(gloss: str, display_name: str | None = None) -> OrientationReferenceMeta | None:
    ref = get_orientation_reference(gloss)
    if ref is None:
        return None
    return OrientationReferenceMeta(
        sign_id=ref["sign_id"],
        display_name=display_name or ref.get("display_name", gloss.replace("_", " ").title()),
        display_name_ml=ml_display_name(gloss),
        sign_type=ref.get("sign_type", "dynamic"),
        active_hand=ref.get("active_hand", "right"),
        critical_features=ref.get("critical_features", []),
        tolerance=ref.get("tolerance", {}),
        num_reference_frames=len(ref.get("reference_sequence", [])),
    )
