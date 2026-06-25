"""Landmark spline bridges and skeleton rendering from landmark arrays."""

from __future__ import annotations

import numpy as np

from include50_lab.preprocess.landmarks import NUM_LANDMARKS, landmarks_from_results
from include50_lab.preprocess.skeleton import draw_skeleton

# Re-export for compose
__all__ = ["bridge_landmarks", "render_landmark_frame", "load_landmarks"]


def load_landmarks(path) -> np.ndarray | None:
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return None
    return np.load(p, mmap_mode="r")


def _interp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def bridge_landmarks(
    end_a: np.ndarray,
    start_b: np.ndarray,
    n_frames: int,
) -> np.ndarray:
    """Cubic ease blend between two (L, 4) landmark frames."""
    if n_frames <= 0:
        return np.zeros((0, NUM_LANDMARKS, 4), dtype=np.float32)
    out = np.zeros((n_frames, NUM_LANDMARKS, 4), dtype=np.float32)
    for i in range(n_frames):
        t = (i + 1) / (n_frames + 1)
        # smoothstep
        t = t * t * (3.0 - 2.0 * t)
        out[i] = _interp(end_a, start_b, t)
    return out


class _FakeLm:
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, z, vis):
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.visibility = float(vis)


class _FakePoseResult:
    def __init__(self, pose_lms):
        self.pose_landmarks = [pose_lms] if pose_lms else None


class _FakeHandResult:
    def __init__(self, hands):
        self.hand_landmarks = hands
        self.handedness = None


def _landmarks_to_fake_results(lm_frame: np.ndarray):
    """Rebuild minimal MediaPipe-like results from (54, 4) array for draw_skeleton."""
    from include50_lab.preprocess.landmarks import NUM_HAND, NUM_POSE, POSE_IDXS

    # Build full 33 pose slots (only upper indices filled)
    pose_slots = [_FakeLm(0, 0, 0, 0) for _ in range(33)]
    for j, idx in enumerate(POSE_IDXS):
        x, y, z, v = lm_frame[j]
        pose_slots[idx] = _FakeLm(x, y, z, v)

    left = []
    right = []
    lo = NUM_POSE
    ro = NUM_POSE + NUM_HAND
    for k in range(NUM_HAND):
        left.append(_FakeLm(*lm_frame[lo + k]))
        right.append(_FakeLm(*lm_frame[ro + k]))

    class _Cat:
        def __init__(self, name):
            self.category_name = name

    class _Handed:
        def __init__(self, name):
            self = [_Cat(name)]

    hands = []
    handedness = []
    if np.any(lm_frame[lo : lo + NUM_HAND, :2] > 0.01):
        hands.append(left)
        handedness.append([_Cat("Left")])
    if np.any(lm_frame[ro : ro + NUM_HAND, :2] > 0.01):
        hands.append(right)
        handedness.append([_Cat("Right")])

    pr = _FakePoseResult(pose_slots)
    hr = _FakeHandResult(hands)
    hr.handedness = handedness if handedness else None
    return pr, hr


def render_landmark_frame(lm_frame: np.ndarray, width: int = 224, height: int = 224) -> np.ndarray:
    pr, hr = _landmarks_to_fake_results(lm_frame)
    return draw_skeleton(pr, hr, width=width, height=height)
