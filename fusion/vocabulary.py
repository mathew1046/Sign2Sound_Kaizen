"""Canonical glove → MSPT gloss mapping for fusion."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOCAB_PATH = REPO_ROOT / "config" / "fusion_vocabulary.yaml"


def load_fusion_vocabulary(path: Path | None = None) -> dict:
    vocab_path = path or DEFAULT_VOCAB_PATH
    with open(vocab_path) as f:
        return yaml.safe_load(f)


class FusionVocabulary:
    def __init__(self, path: Path | None = None):
        data = load_fusion_vocabulary(path)
        self.glove_to_mspt: dict[str, str] = {
            str(k).strip().lower(): str(v).strip()
            for k, v in (data.get("glove_to_mspt") or {}).items()
        }
        self.glove_letters: set[str] = {str(x).strip().lower() for x in (data.get("glove_letters") or [])}
        self.reject_labels: set[str] = {str(x).strip().lower() for x in (data.get("reject_labels") or ["rest"])}

    def glove_to_mspt_slug(self, glove_label: str) -> str | None:
        return self.glove_to_mspt.get(glove_label.strip().lower())

    def is_glove_letter(self, glove_label: str) -> bool:
        return glove_label.strip().lower() in self.glove_letters

    def is_overlap_word(self, glove_label: str) -> bool:
        key = glove_label.strip().lower()
        return key in self.glove_to_mspt and key not in self.glove_letters

    def should_reject(self, glove_label: str) -> bool:
        return glove_label.strip().lower() in self.reject_labels
