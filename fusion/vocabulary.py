"""Canonical glove → MSPT gloss mapping for fusion."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

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

    def validate_against_glove_classes(self, classes: list[str]) -> list[str]:
        """Validate that all non-rejected glove classes have a fusion mapping.

        Args:
            classes: List of glove model class labels (e.g. from words_classes.npy).

        Returns:
            List of unmapped class names (excluding reject labels and letters).
        """
        unmapped: list[str] = []
        for cls in classes:
            key = str(cls).strip().lower()
            if key in self.reject_labels:
                continue
            if key in self.glove_letters:
                continue
            if self.glove_to_mspt_slug(key) is None:
                unmapped.append(key)
        return unmapped

    def validate_completeness(self) -> list[str]:
        """Auto-load GloveTalk word classes and check for missing mappings.

        Returns:
            List of unmapped class names. Empty means all classes are mapped.
        """
        classes_path = REPO_ROOT / "GloveTalk" / "weights" / "words_classes.npy"
        if not classes_path.is_file():
            logger.warning(
                "Cannot validate fusion vocabulary: %s not found", classes_path
            )
            return []

        try:
            import numpy as np

            classes = np.load(classes_path, allow_pickle=True)
            unmapped = self.validate_against_glove_classes([str(c) for c in classes])
            if unmapped:
                logger.warning(
                    "Fusion vocabulary missing mappings for glove classes: %s",
                    unmapped,
                )
            else:
                logger.info(
                    "Fusion vocabulary validated: all %d glove classes mapped",
                    len(classes),
                )
            return unmapped
        except Exception as exc:
            logger.warning("Fusion vocabulary validation failed: %s", exc)
            return []
