"""Shared utilities for the INCLUDE-50 SignBERT+ notebooks.

This module centralises everything the four notebooks need so each notebook can
stay focused on *training logic*:

* paths into the pre-processed ``include50_lab`` cache,
* the landmark layout produced by ``include50_lab/preprocess/landmarks.py``
  (12 upper-pose + 21 left-hand + 21 right-hand joints, each ``(x, y, z, vis)``),
* the multi-level masking used by SignBERT+ self-supervised pre-training
  (joint / frame / clip masking -- a re-implementation of
  ``signbert/data_modules/utils.py``),
* PyTorch ``Dataset`` classes for isolated clips (pose + RGB features) and for
  the synthetic continuous sessions,
* metrics: top-1 accuracy and Word Error Rate (WER) for CTC decoding.

Everything is plain NumPy / PyTorch so the notebooks run without
PyTorch-Lightning or the git submodules that the upstream repo relies on.
"""

from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Paths -- point these at the pre-processed include50_lab folder.
# Override with env var:  export INCLUDE50_LAB_ROOT=/path/to/include50_lab
# ---------------------------------------------------------------------------
INCLUDE_ML_ROOT = Path(
    os.environ.get(
        "INCLUDE_ML_ROOT",
        "/media/mathew/OS/Users/augus/INCLUDE_ML",
    )
)
_MANIFEST_PATH_PREFIX_OLD = Path("/home/mathew/Downloads/INCLUDE_ML")

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_LAB_ROOTS = [
    str(_REPO_ROOT / "data" / "include50_rtmlib_1080"),
    str(INCLUDE_ML_ROOT / "include50_lab"),
    "/home/mathew/Downloads/INCLUDE_ML/include50_lab",
]


def resolve_video_path(path: str | Path) -> Path:
    """Manifests may list old Downloads paths; videos live under INCLUDE_ML_ROOT."""
    p = Path(path)
    if p.is_file():
        return p
    try:
        rel = p.relative_to(_MANIFEST_PATH_PREFIX_OLD)
        alt = INCLUDE_ML_ROOT / rel
        if alt.is_file():
            return alt
    except ValueError:
        pass
    return p


def _resolve_lab_root() -> Path:
    if env := os.environ.get("INCLUDE50_LAB_ROOT"):
        return Path(env)
    for p in _DEFAULT_LAB_ROOTS:
        if (Path(p) / "manifests" / "train.csv").exists():
            return Path(p)
    return Path(_DEFAULT_LAB_ROOTS[0])


def landmarks_ready(lab_root: Path | str | None = None, min_fraction: float = 0.9) -> bool:
    """True when landmark cache covers most isolated skeleton clips."""
    root = Path(lab_root) if lab_root is not None else _resolve_lab_root()
    skel_dir = root / "cache" / "skeleton"
    lm_dir = root / "cache" / "landmarks"
    if not skel_dir.is_dir() or not lm_dir.is_dir():
        return False
    n_skel = sum(1 for _ in skel_dir.rglob("*.npy"))
    n_lm = sum(1 for _ in lm_dir.rglob("*.npy"))
    if n_skel == 0:
        return n_lm > 0
    return n_lm >= min_fraction * n_skel


def face_landmarks_ready(lab_root: Path | str | None = None, min_fraction: float = 0.9) -> bool:
    """True when face cache covers most body+hand landmark clips."""
    root = Path(lab_root) if lab_root is not None else _resolve_lab_root()
    lm_dir = root / "cache" / "landmarks"
    face_dir = root / "cache" / "landmarks_face"
    if not face_dir.is_dir():
        return False
    n_lm = sum(1 for _ in lm_dir.rglob("*.npy")) if lm_dir.is_dir() else 0
    n_face = sum(1 for _ in face_dir.rglob("*.npy"))
    if n_lm == 0:
        return n_face > 0
    return n_face >= min_fraction * n_lm


def split_face(frames: np.ndarray) -> dict[str, np.ndarray]:
    """Split ``(T, N, 4)`` face landmarks into xy + visibility scores."""
    xy = frames[..., :2].astype(np.float32)
    score = frames[..., 3].astype(np.float32)
    return {"face": xy, "face_score": score}


LAB_ROOT = _resolve_lab_root()

MANIFESTS_DIR = LAB_ROOT / "manifests"
CACHE_DIR = LAB_ROOT / "cache"
SKELETON_DIR = CACHE_DIR / "skeleton"          # rendered skeleton RGB rasters (T,224,224,3)
LANDMARKS_DIR = CACHE_DIR / "landmarks"        # keypoints (T,54,4) -- populated by landmarks.py
LANDMARKS_FACE_DIR = CACHE_DIR / "landmarks_face"  # face subset (T,N,4) -- extract_face_landmarks.py
FEATURES_DIR = CACHE_DIR / "features"          # CNN features per backbone (T,D)
CONTINUOUS_DIR = CACHE_DIR / "continuous"      # synthetic continuous sessions

LABEL_MAP_PATH = LAB_ROOT.parent / "saved_models" / "label_map.json"

NUM_CLASSES = 50
MAX_FRAMES = 128

# Landmark layout (see include50_lab/preprocess/landmarks.py).
#   indices  0..11  -> 12 upper-body pose joints (sorted MediaPipe ids 11..22)
#   indices 12..32  -> 21 left-hand joints
#   indices 33..53  -> 21 right-hand joints
NUM_POSE = 12
NUM_HAND = 21
NUM_LANDMARKS = NUM_POSE + 2 * NUM_HAND  # 54

POSE_SLICE = slice(0, NUM_POSE)
LHAND_SLICE = slice(NUM_POSE, NUM_POSE + NUM_HAND)
RHAND_SLICE = slice(NUM_POSE + NUM_HAND, NUM_LANDMARKS)

# The first 6 pose joints (sorted ids 11..16) are L/R shoulder, L/R elbow,
# L/R wrist. SignBERT+'s ArmsExtractor picks right arm = (1,3,5),
# left arm = (0,2,4) from this ordering.
ARMS_SLICE = slice(0, 6)
RIGHT_ARM_IDXS = (1, 3, 5)
LEFT_ARM_IDXS = (0, 2, 4)


def load_label_map(path: Path = LABEL_MAP_PATH) -> tuple[dict[str, int], dict[int, str]]:
    with open(path, encoding="utf-8") as f:
        label_map = json.load(f)
    idx_to_label = {int(v): k for k, v in label_map.items()}
    return label_map, idx_to_label


# ---------------------------------------------------------------------------
# Multi-level masking (re-implementation of signbert/data_modules/utils.py).
# Returns the corrupted sequence and the indices of masked frames so the loss
# can be applied only on those frames (as in the paper).
# ---------------------------------------------------------------------------
def _mask_joint(frame: np.ndarray, max_disturbance: float, m: int) -> np.ndarray:
    """Zero-mask or spatially disturb up to ``m`` joints of a single frame."""
    n_joints = frame.shape[0]
    m = np.random.randint(1, m + 1)
    joint_idxs = np.random.choice(n_joints, size=min(m, n_joints), replace=False)
    op = np.random.binomial(1, 0.5, size=len(joint_idxs)).reshape(-1, 1)
    disturb = frame[joint_idxs] + np.random.uniform(
        -max_disturbance, max_disturbance, size=frame[joint_idxs].shape
    )
    frame[joint_idxs] = np.where(op, disturb, 0.0)
    return frame


def _mask_clip(frame_idx: int, seq: np.ndarray, n_frames: int, K: int) -> list[int]:
    """Zero a contiguous clip of up to ``K`` frames centred on ``frame_idx``."""
    n = np.random.randint(2, K + 1)
    half = n // 2
    start, end = frame_idx - half, frame_idx + (n - half)
    if start < 0:
        end += -start
        start = 0
    if end > n_frames:
        start -= end - n_frames
        end = n_frames
    start = max(0, start)
    idxs = list(range(start, end))
    seq[idxs] = 0.0
    return idxs


def mask_keypoints(
    seq: np.ndarray,
    R: float = 0.3,
    m: int = 5,
    K: int = 8,
    max_disturbance: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply joint/frame/clip masking to a ``(T, V, 2)`` keypoint sequence.

    Mirrors ``mask_transform`` from the upstream repo: a fraction ``R`` of valid
    frames is chosen; each chosen frame is masked at the joint level, frame
    level, or clip level with equal probability.
    """
    out = seq.copy()
    n_frames = int((out != 0.0).all((1, 2)).sum())
    if n_frames == 0:
        return out, np.array([], dtype=np.int64)
    n_to_mask = int(np.ceil(R * n_frames))
    chosen = np.random.choice(n_frames, size=n_to_mask, replace=False)
    clipped: list[int] = []
    for f in chosen:
        op = np.random.choice(3)  # 0 joint, 1 frame, 2 clip
        if op == 0:
            out[f] = _mask_joint(out[f], max_disturbance, m)
        elif op == 1:
            out[f] = 0.0
        else:
            clipped.extend(_mask_clip(int(f), out, n_frames, K))
    masked_idx = np.unique(np.concatenate([chosen, np.array(clipped, dtype=np.int64)]))
    return out, masked_idx.astype(np.int64)


# ---------------------------------------------------------------------------
# Pose normalisation: centre on the shoulder mid-point and scale by shoulder
# width. landmarks.py already applies scale-invariance, this just re-centres
# robustly and drops the z / visibility channels (visibility kept as score).
# ---------------------------------------------------------------------------
def split_landmarks(frames: np.ndarray) -> dict[str, np.ndarray]:
    """Split a ``(T, 54, 4)`` landmark array into arms/hands xy + scores."""
    xy = frames[..., :2].astype(np.float32)
    score = frames[..., 3].astype(np.float32)  # visibility channel as confidence
    return {
        "arms": xy[:, ARMS_SLICE],            # (T, 6, 2)
        "lhand": xy[:, LHAND_SLICE],          # (T, 21, 2)
        "rhand": xy[:, RHAND_SLICE],          # (T, 21, 2)
        "lhand_score": score[:, LHAND_SLICE],  # (T, 21)
        "rhand_score": score[:, RHAND_SLICE],  # (T, 21)
    }


def _pad_or_trim(arr: np.ndarray, max_frames: int) -> tuple[np.ndarray, int]:
    T = min(len(arr), max_frames)
    out = np.zeros((max_frames, *arr.shape[1:]), dtype=arr.dtype)
    out[:T] = arr[:T]
    return out, T


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class PoseClipDataset(Dataset):
    """Isolated-clip pose dataset for SignBERT+.

    Loads ``(T, 54, 4)`` landmark npy files written by
    ``include50_lab/preprocess/landmarks.py`` and returns arms/hands tensors,
    confidence scores, an optionally masked copy (for self-supervised
    pre-training) and the class id.
    """

    def __init__(
        self,
        manifest_csv: Path,
        landmarks_dir: Path = LANDMARKS_DIR,
        face_dir: Path | None = None,
        max_frames: int = MAX_FRAMES,
        split: str | None = None,
        mask: bool = False,
        mask_kwargs: dict | None = None,
        require_face: bool = False,
    ):
        df = pd.read_csv(manifest_csv)
        if split:
            df = df[df["split"] == split]
        self.landmarks_dir = Path(landmarks_dir)
        self.face_dir = Path(face_dir) if face_dir else None
        self.max_frames = max_frames
        self.mask = mask
        self.mask_kwargs = mask_kwargs or {}
        self.require_face = require_face
        self.rows = []
        for _, row in df.iterrows():
            stem = Path(row["path"]).stem
            p = self.landmarks_dir / row["label"] / f"{stem}.npy"
            if not p.exists():
                continue
            if self.face_dir is not None:
                fp = self.face_dir / row["label"] / f"{stem}.npy"
                if require_face and not fp.exists():
                    continue
            else:
                fp = None
            self.rows.append((p, fp, int(row["label_id"])))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, face_path, label = self.rows[idx]
        frames = np.load(path).astype(np.float32)  # (T,54,4)
        parts = split_landmarks(frames)
        arms, T = _pad_or_trim(parts["arms"], self.max_frames)
        lhand, _ = _pad_or_trim(parts["lhand"], self.max_frames)
        rhand, _ = _pad_or_trim(parts["rhand"], self.max_frames)
        lscore, _ = _pad_or_trim(parts["lhand_score"], self.max_frames)
        rscore, _ = _pad_or_trim(parts["rhand_score"], self.max_frames)

        sample = {
            "arms": torch.from_numpy(arms),
            "rhand": torch.from_numpy(rhand),
            "lhand": torch.from_numpy(lhand),
            "rhand_score": torch.from_numpy(rscore),
            "lhand_score": torch.from_numpy(lscore),
            "length": T,
            "label": torch.tensor(label, dtype=torch.long),
        }
        if self.face_dir is not None:
            if face_path is not None and face_path.exists():
                face_frames = np.load(face_path).astype(np.float32)
            else:
                from face_landmarks import NUM_FACE

                face_frames = np.zeros((T, NUM_FACE, 4), np.float32)
            face_parts = split_face(face_frames)
            face_xy, _ = _pad_or_trim(face_parts["face"], self.max_frames)
            face_sc, _ = _pad_or_trim(face_parts["face_score"], self.max_frames)
            sample["face"] = torch.from_numpy(face_xy)
            sample["face_score"] = torch.from_numpy(face_sc)
        if self.mask:
            rmask, rmask_idx = mask_keypoints(rhand, **self.mask_kwargs)
            lmask, lmask_idx = mask_keypoints(lhand, **self.mask_kwargs)
            rsel = np.zeros(self.max_frames, dtype=np.float32)
            lsel = np.zeros(self.max_frames, dtype=np.float32)
            rsel[rmask_idx] = 1.0
            lsel[lmask_idx] = 1.0
            sample["rhand_masked"] = torch.from_numpy(rmask)
            sample["lhand_masked"] = torch.from_numpy(lmask)
            sample["rhand_mask_sel"] = torch.from_numpy(rsel)
            sample["lhand_mask_sel"] = torch.from_numpy(lsel)
        return sample


class FeatureClipDataset(Dataset):
    """Isolated-clip RGB-feature dataset (mobilenet/timm cache).

    Loads ``(T, D)`` CNN features written by
    ``include50_lab/preprocess/extract_features.py``.
    """

    def __init__(
        self,
        manifest_csv: Path,
        feature_dir: Path,
        max_frames: int = MAX_FRAMES,
        split: str | None = None,
    ):
        df = pd.read_csv(manifest_csv)
        if split:
            df = df[df["split"] == split]
        self.feature_dir = Path(feature_dir)
        self.max_frames = max_frames
        self.rows = []
        for _, row in df.iterrows():
            stem = Path(row["path"]).stem
            p = self.feature_dir / row["label"] / f"{stem}_feat.npy"
            if p.exists():
                self.rows.append((p, int(row["label_id"])))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label = self.rows[idx]
        feats = np.load(path).astype(np.float32)
        x, T = _pad_or_trim(feats, self.max_frames)
        mask = np.zeros(self.max_frames, dtype=np.float32)
        mask[:T] = 1.0
        return {
            "feats": torch.from_numpy(x),
            "mask": torch.from_numpy(mask),
            "length": T,
            "label": torch.tensor(label, dtype=torch.long),
        }


def collapse_gloss_sequence(per_frame_gloss: np.ndarray, blank: int = -1) -> list[int]:
    """Collapse a per-frame gloss array into an ordered gloss sequence.

    Drops blank/idle frames (``blank``) and merges consecutive identical ids,
    yielding the CTC target sequence, e.g. ``[34,34,...,14,14] -> [34, 14]``.
    """
    seq: list[int] = []
    prev = None
    for v in per_frame_gloss.tolist():
        if v == blank:
            prev = None
            continue
        if v != prev:
            seq.append(int(v))
        prev = v
    return seq


class ContinuousFeatureDataset(Dataset):
    """Continuous synthetic sessions for CTC training (RGB-feature variant).

    Returns ``(T, D)`` features, a length, and the collapsed gloss target
    sequence built from ``session_*_gloss.npy``.
    """

    def __init__(
        self,
        continuous_dir: Path = CONTINUOUS_DIR,
        feature_dir: Path | None = None,
        max_frames: int = MAX_FRAMES,
        split: str = "train",
        feature_dim: int = 1280,
    ):
        self.continuous_dir = Path(continuous_dir)
        self.feature_dir = Path(feature_dir) if feature_dir else None
        self.max_frames = max_frames
        self.feature_dim = feature_dim
        manifest = pd.read_csv(self.continuous_dir / "manifest.csv")
        self.rows = manifest[manifest["split"] == split].to_dict("records")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        sid = self.rows[idx]["session_id"]
        gloss = np.load(self.continuous_dir / f"{sid}_gloss.npy")
        if self.feature_dir is not None:
            fp = self.feature_dir / "continuous" / f"{sid}_feat.npy"
            feats = np.load(fp).astype(np.float32) if fp.exists() else \
                np.zeros((len(gloss), self.feature_dim), np.float32)
        else:
            feats = np.zeros((len(gloss), self.feature_dim), np.float32)
        T = min(len(feats), len(gloss), self.max_frames)
        x, _ = _pad_or_trim(feats[:T], self.max_frames)
        mask = np.zeros(self.max_frames, dtype=np.float32)
        mask[:T] = 1.0
        target = collapse_gloss_sequence(gloss[:T])
        return {
            "feats": torch.from_numpy(x),
            "mask": torch.from_numpy(mask),
            "length": T,
            "target": torch.tensor(target, dtype=torch.long),
            "target_len": len(target),
        }


class ContinuousPoseDataset(Dataset):
    """Continuous synthetic sessions for CTC training (pose variant).

    Expects stitched landmark sessions ``(T, 54, 4)`` produced by
    :func:`build_continuous_pose_sessions`. Falls back to zeros (so the notebook
    runs end-to-end) if a session has not been stitched yet.
    """

    def __init__(
        self,
        continuous_dir: Path = CONTINUOUS_DIR,
        pose_dir: Path | None = None,
        max_frames: int = MAX_FRAMES,
        split: str = "train",
    ):
        self.continuous_dir = Path(continuous_dir)
        self.pose_dir = Path(pose_dir) if pose_dir else (continuous_dir / "landmarks")
        self.max_frames = max_frames
        manifest = pd.read_csv(self.continuous_dir / "manifest.csv")
        self.rows = manifest[manifest["split"] == split].to_dict("records")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        sid = self.rows[idx]["session_id"]
        gloss = np.load(self.continuous_dir / f"{sid}_gloss.npy")
        fp = self.pose_dir / f"{sid}.npy"
        if fp.exists():
            frames = np.load(fp).astype(np.float32)
        else:
            frames = np.zeros((len(gloss), NUM_LANDMARKS, 4), np.float32)
        parts = split_landmarks(frames)
        arms, T = _pad_or_trim(parts["arms"], self.max_frames)
        lhand, _ = _pad_or_trim(parts["lhand"], self.max_frames)
        rhand, _ = _pad_or_trim(parts["rhand"], self.max_frames)
        mask = np.zeros(self.max_frames, dtype=np.float32)
        mask[: min(T, len(gloss))] = 1.0
        target = collapse_gloss_sequence(gloss[: self.max_frames])
        return {
            "arms": torch.from_numpy(arms),
            "rhand": torch.from_numpy(rhand),
            "lhand": torch.from_numpy(lhand),
            "mask": torch.from_numpy(mask),
            "length": min(T, len(gloss)),
            "target": torch.tensor(target, dtype=torch.long),
            "target_len": len(target),
        }


def ctc_collate(batch: list[dict]) -> dict:
    """Collate variable-length CTC targets into the flat form ``nn.CTCLoss`` wants."""
    out: dict = {}
    skip = {"target", "target_len", "length"}
    for k in batch[0]:
        if k in skip or not torch.is_tensor(batch[0][k]):
            continue
        out[k] = torch.stack([b[k] for b in batch])
    if any(b["target_len"] for b in batch):
        targets = torch.cat([b["target"] for b in batch])
    else:
        targets = torch.zeros(0, dtype=torch.long)
    out["targets"] = targets
    out["target_lengths"] = torch.tensor([b["target_len"] for b in batch], dtype=torch.long)
    out["input_lengths"] = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    return out


# ---------------------------------------------------------------------------
# Continuous pose-session stitching (mirror of data/continuous_synth.py but for
# landmark keypoints instead of skeleton rasters, so the pose CTC notebook has
# real inputs). Produces sessions + matching per-frame gloss arrays.
# ---------------------------------------------------------------------------
def build_continuous_pose_sessions(
    manifest_csv: Path,
    landmarks_dir: Path = LANDMARKS_DIR,
    out_dir: Path | None = None,
    num_sessions: int = 500,
    min_signs: int = 2,
    max_signs: int = 5,
    idle_min: int = 5,
    idle_max: int = 15,
    max_frames: int = MAX_FRAMES,
    seed: int = 42,
) -> int:
    """Stitch isolated landmark clips into continuous pose sessions.

    Writes ``<out_dir>/<sid>.npy`` (T,54,4) and ``<out_dir>/<sid>_gloss.npy``
    plus a ``manifest.csv`` with train/val split. Returns the number written.
    """
    import random

    out_dir = Path(out_dir) if out_dir else (CONTINUOUS_DIR / "landmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    landmarks_dir = Path(landmarks_dir)
    rng = random.Random(seed)

    df = pd.read_csv(manifest_csv)
    by_label: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        stem = Path(row["path"]).stem
        if (landmarks_dir / row["label"] / f"{stem}.npy").exists():
            by_label.setdefault(row["label"], []).append(
                {"stem": stem, "label": row["label"], "label_id": int(row["label_id"])}
            )
    labels = [k for k, v in by_label.items() if v]
    if not labels:
        return 0

    rows = []
    for i in range(num_sessions):
        sid = f"posesession_{i:05d}"
        n_signs = rng.randint(min_signs, min(max_signs, len(labels)))
        chosen = rng.sample(labels, n_signs)
        frames_acc, gloss_acc, names = [], [], []
        for lab in chosen:
            pick = rng.choice(by_label[lab])
            clip = np.load(landmarks_dir / lab / f"{pick['stem']}.npy").astype(np.float32)
            if len(frames_acc) + len(clip) > max_frames:
                clip = clip[: max_frames - len(frames_acc)]
            frames_acc.append(clip)
            gloss_acc.extend([pick["label_id"]] * len(clip))
            names.append(lab)
            idle = rng.randint(idle_min, idle_max)
            if len(frames_acc) and sum(len(f) for f in frames_acc) + idle <= max_frames:
                frames_acc.append(np.zeros((idle, NUM_LANDMARKS, 4), np.float32))
                gloss_acc.extend([-1] * idle)
            if sum(len(f) for f in frames_acc) >= max_frames:
                break
        if not frames_acc:
            continue
        session = np.concatenate(frames_acc, axis=0)[:max_frames]
        gloss = np.array(gloss_acc[:max_frames], dtype=np.int64)
        np.save(out_dir / f"{sid}.npy", session.astype(np.float32))
        np.save(out_dir / f"{sid}_gloss.npy", gloss)
        rows.append({"session_id": sid, "num_frames": len(session),
                     "glosses": "|".join(names), "split": "train"})

    rng.shuffle(rows)
    n_val = max(1, int(0.15 * len(rows)))
    for j, r in enumerate(rows):
        r["split"] = "val" if j < n_val else "train"
    pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)
    # gloss files live next to sessions; ContinuousPoseDataset reads gloss from
    # continuous_dir, so also drop copies there for a single source of truth.
    return len(rows)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(-1) == labels).float().mean().item()


def wer(reference: Sequence[int], hypothesis: Sequence[int]) -> float:
    """Word (gloss) error rate via Levenshtein edit distance."""
    r, h = list(reference), list(hypothesis)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return d[len(r), len(h)] / max(1, len(r))


def ctc_greedy_decode(log_probs: torch.Tensor, input_lengths: torch.Tensor, blank: int = 0) -> list[list[int]]:
    """Greedy CTC decode. ``log_probs``: (T, B, C). Returns list of id sequences."""
    preds = log_probs.argmax(-1).transpose(0, 1).cpu().numpy()  # (B, T)
    out = []
    for b, seq in enumerate(preds):
        L = int(input_lengths[b].item())
        collapsed, prev = [], None
        for v in seq[:L]:
            if v != prev and v != blank:
                collapsed.append(int(v))
            prev = v
        out.append(collapsed)
    return out
