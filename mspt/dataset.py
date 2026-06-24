"""MSPT PyTorch dataset — lazy mmap load, per-sample CPU aug, variable length."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import sys

NOTEBOOKS = Path(__file__).resolve().parent.parent
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))

import slr_common as C  # noqa: E402
from face_landmarks import FACE_IDXS, NUM_FACE  # noqa: E402
from mspt.augment import apply_augmentations  # noqa: E402
from mspt.normalize import flatten_xy, normalize_body, normalize_face_from_mesh, normalize_hands  # noqa: E402

NUM_HAND_KP = 42
NUM_BODY_KP = 33
NUM_FACE_KP = NUM_FACE  # 72 expression landmarks

HAND_DIM = NUM_HAND_KP * 2
BODY_DIM = NUM_BODY_KP * 2
FACE_DIM = NUM_FACE_KP * 2


def _load_npy_xy(path: Path, kp_slice: slice | None = None) -> np.ndarray:
    """Memory-safe: mmap read, copy only xy slice we need, float32."""
    arr = np.load(path, mmap_mode="r")
    if kp_slice is not None:
        xy = np.array(arr[:, kp_slice, :2], dtype=np.float32, copy=True)
    else:
        xy = np.array(arr[..., :2], dtype=np.float32, copy=True)
    del arr
    return xy


def _subsample_frames(arr: np.ndarray, max_frames: int) -> np.ndarray:
    """Uniform subsample long clips instead of zero-padding entire max length upfront."""
    t = len(arr)
    if t <= max_frames:
        return arr
    idx = np.linspace(0, t - 1, max_frames, dtype=int)
    return arr[idx]


def load_streams(
    lm_path: Path,
    body_path: Path | None,
    face_path: Path | None,
    max_frames: int,
    require_body: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Lazy-load three streams; one clip at a time, no global preload."""
    lm = _load_npy_xy(lm_path)
    t_raw = len(lm)
    hands = lm[:, 12:54]  # (T, 42, 2) view already copied

    if body_path is not None and body_path.exists():
        body = _load_npy_xy(body_path)
        if body.shape[1] != NUM_BODY_KP:
            body = body[:, :NUM_BODY_KP]
    elif require_body:
        raise FileNotFoundError(f"Missing body cache: {body_path}")
    else:
        upper = lm[:, :12]
        body = np.zeros((t_raw, NUM_BODY_KP, 2), dtype=np.float32)
        body[:, :12] = upper

    if face_path is not None and face_path.exists():
        face = _load_npy_xy(face_path)
    else:
        face = np.zeros((t_raw, NUM_FACE_KP, 2), dtype=np.float32)

    del lm

    hands = _subsample_frames(hands, max_frames)
    body = _subsample_frames(body, max_frames)
    face = _subsample_frames(face, max_frames)
    t = len(hands)
    return hands, body, face, t


class MSPTDataset(Dataset):
    """Isolated SLR dataset for MSPT.

    Memory choices:
    - Stores only file paths + labels (no keypoint preload).
    - ``repeat`` multiplies epoch length for stochastic aug without disk copies.
    - ``__getitem__`` loads one clip via mmap, aug on CPU, returns variable-length tensors.
    """

    def __init__(
        self,
        manifest_csv: Path,
        landmarks_dir: Path,
        body_dir: Path | None = None,
        face_dir: Path | None = None,
        max_frames: int = C.MAX_FRAMES,
        split: str | None = None,
        training: bool = False,
        repeat: int = 1,
        require_body: bool = False,
    ):
        df = pd.read_csv(manifest_csv)
        if split:
            df = df[df["split"] == split]
        self.landmarks_dir = Path(landmarks_dir)
        self.body_dir = Path(body_dir) if body_dir else None
        self.face_dir = Path(face_dir) if face_dir else None
        self.max_frames = max_frames
        self.training = training
        self.repeat = repeat if training else 1
        self.require_body = require_body
        self.face_mesh_idxs = FACE_IDXS
        self.rows: list[tuple[Path, Path | None, Path | None, int]] = []
        for _, row in df.iterrows():
            stem = Path(row["path"]).stem
            lp = self.landmarks_dir / row["label"] / f"{stem}.npy"
            if not lp.exists():
                continue
            bp = self.body_dir / row["label"] / f"{stem}.npy" if self.body_dir else None
            fp = self.face_dir / row["label"] / f"{stem}.npy" if self.face_dir else None
            if self.require_body and (bp is None or not bp.exists()):
                continue
            self.rows.append((lp, bp, fp, int(row["label_id"])))

    def __len__(self):
        return len(self.rows) * self.repeat

    def __getitem__(self, idx):
        base_idx = idx % len(self.rows)
        lp, bp, fp, label = self.rows[base_idx]
        hands, body, face, t = load_streams(lp, bp, fp, self.max_frames, self.require_body)

        hands = normalize_hands(hands)
        body = normalize_body(body)
        face = normalize_face_from_mesh(face, self.face_mesh_idxs)

        if self.training:
            hands, body, face = apply_augmentations(
                hands, body, face, training=True, max_len=self.max_frames
            )
            t = int((hands != 0).any(axis=-1).sum())
            t = max(1, min(t, len(hands), self.max_frames))
            hands, body, face = hands[:t], body[:t], face[:t]
        else:
            t = max(1, t)

        return {
            "hand": torch.from_numpy(flatten_xy(hands)),
            "body": torch.from_numpy(flatten_xy(body)),
            "face": torch.from_numpy(flatten_xy(face)),
            "length": t,
            "label": torch.tensor(label, dtype=torch.long),
        }
