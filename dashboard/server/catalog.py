"""Load and query dashboard catalog."""

from __future__ import annotations

import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

from dashboard.config import (
    CATALOG_PATH,
    LABEL_MAP_PATH,
    MANIFEST_PATH,
    PROJECT_ROOT,
    WHOLEBODY_DIR,
    ensure_combined_manifest,
)
from dashboard.label_utils import ensure_label_map
from dashboard.ml_utils import ml_display_name


def ensure_catalog() -> Path:
    """Build catalog from rtmlib wholebody cache if missing."""
    if CATALOG_PATH.is_file():
        return CATALOG_PATH
    ensure_combined_manifest()
    ensure_label_map()
    script = PROJECT_ROOT / "scripts" / "build_dashboard_catalog.py"
    spec = importlib.util.spec_from_file_location("build_dashboard_catalog", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_dashboard_catalog"] = mod
    spec.loader.exec_module(mod)
    mod.build_catalog(MANIFEST_PATH, WHOLEBODY_DIR, LABEL_MAP_PATH, CATALOG_PATH)
    return CATALOG_PATH


@lru_cache(maxsize=1)
def load_catalog(path: str | None = None) -> dict:
    p = Path(path) if path else ensure_catalog()
    if not p.exists():
        raise FileNotFoundError(
            f"Catalog not found at {p}. Run: python scripts/build_dashboard_catalog.py"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def get_gloss_entry(catalog: dict, gloss: str) -> dict | None:
    for g in catalog["glosses"]:
        if g["gloss"] == gloss:
            return g
    return None


def vocab_list(catalog: dict) -> list[dict]:
    return [
        {
            "gloss": g["gloss"],
            "display_name": g["display_name"],
            "display_name_ml": ml_display_name(g["gloss"]),
            "label_id": g["label_id"],
            "variant_count": g["variant_count"],
            "default_exemplar_id": g.get("default_exemplar_id"),
            "has_sign": g.get("default_exemplar_id") is not None,
        }
        for g in catalog["glosses"]
    ]
