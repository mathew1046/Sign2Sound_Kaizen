"""Label maps for MSPT checkpoints (INCLUDE-50 and rtmlib 263-class vocab)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import slr_common as C


def load_rtmlib_label_map(manifest_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Build label maps from rtmlib lab ``manifest.csv`` (label_id column)."""
    man = pd.read_csv(manifest_path)
    if "label_id" not in man.columns or "label" not in man.columns:
        raise ValueError(f"manifest must have label + label_id: {manifest_path}")
    label_to_id: dict[str, int] = {}
    for row in man.drop_duplicates("label_id").itertuples():
        label_to_id[str(row.label).strip()] = int(row.label_id)
    idx_to_label = {v: k for k, v in label_to_id.items()}
    return label_to_id, idx_to_label


def num_classes_from_checkpoint(ckpt: dict, state_key: str = "model") -> int | None:
    """Read output class count from a saved MSPT checkpoint."""
    if "num_classes" in ckpt and ckpt["num_classes"] is not None:
        return int(ckpt["num_classes"])
    state = ckpt.get(state_key, ckpt)
    if not isinstance(state, dict):
        return None
    for key in ("classifier.weight", "head.weight", "fc.weight"):
        if key in state:
            return int(state[key].shape[0])
    return None


def label_names_for_checkpoint(
    ckpt: dict,
    lab_root: Path | None = None,
    label_map_path: Path | None = None,
) -> list[str]:
    """Resolve display names for each class index in a checkpoint."""
    n = num_classes_from_checkpoint(ckpt)
    if n is None:
        n = C.NUM_CLASSES

    if lab_root is not None and (lab_root / "manifest.csv").is_file():
        _, idx_to_label = load_rtmlib_label_map(lab_root / "manifest.csv")
        if len(idx_to_label) >= n:
            return [idx_to_label.get(i, f"class_{i}") for i in range(n)]

    _, idx_to_label = C.load_label_map(label_map_path or C.LABEL_MAP_PATH)
    return [idx_to_label.get(i, f"class_{i}") for i in range(n)]
