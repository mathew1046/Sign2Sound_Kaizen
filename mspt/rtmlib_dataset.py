"""MSPT dataset backed by rtmlib COCO-WholeBody caches."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

import sys

NOTEBOOKS = Path(__file__).resolve().parent.parent
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))

from mspt.augment import apply_augmentations  # noqa: E402
from mspt.normalize import flatten_xy  # noqa: E402
from mspt.rtmlib_io import load_streams_rtmlib  # noqa: E402


class MSPTRtmlibDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        cache_root: Path,
        max_frames: int = 128,
        split: str | None = None,
        training: bool = False,
        repeat: int = 1,
    ):
        df = pd.read_csv(manifest_csv)
        if split:
            df = df[df["split"] == split]
        self.cache_root = Path(cache_root)
        self.max_frames = max_frames
        self.training = training
        self.repeat = repeat if training else 1
        self.rows: list[tuple[Path, Path, Path, Path, int]] = []

        for _, row in df.iterrows():
            word, stem = row["label"], row["stem"]
            paths = (
                self.cache_root / "left_hand" / word / f"{stem}.npy",
                self.cache_root / "right_hand" / word / f"{stem}.npy",
                self.cache_root / "body" / word / f"{stem}.npy",
                self.cache_root / "face" / word / f"{stem}.npy",
            )
            if not all(p.is_file() for p in paths):
                continue
            self.rows.append((*paths, int(row["label_id"])))

    def __len__(self):
        return len(self.rows) * self.repeat

    def __getitem__(self, idx):
        base_idx = idx % len(self.rows)
        lh, rh, body, face, label = self.rows[base_idx]
        hands, body_arr, face_arr, t = load_streams_rtmlib(
            lh, rh, body, face, self.max_frames
        )

        if self.training:
            hands, body_arr, face_arr = apply_augmentations(
                hands, body_arr, face_arr, training=True, max_len=self.max_frames
            )
            t = int((hands != 0).any(axis=-1).sum())
            t = max(1, min(t, len(hands), self.max_frames))
            hands, body_arr, face_arr = hands[:t], body_arr[:t], face_arr[:t]
        else:
            t = max(1, t)

        return {
            "hand": torch.from_numpy(flatten_xy(hands)),
            "body": torch.from_numpy(flatten_xy(body_arr)),
            "face": torch.from_numpy(flatten_xy(face_arr)),
            "length": t,
            "label": torch.tensor(label, dtype=torch.long),
        }
