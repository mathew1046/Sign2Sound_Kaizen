#!/usr/bin/env python3
"""Live webcam inference for MSPT — fully automatic, no manual triggers.

Default capture: 1280x960 (960p) at 10 FPS.

While motion is detected, frames accumulate and the model predicts continuously.
Every ``--buffer-clear-interval`` seconds (default 2s) the buffer is hard-cleared.

Only key: q / ESC to quit.

Usage:
  cd notebooks
  python run_mspt_webcam.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from repo_paths import MSPT_CHECKPOINTS, REPO_ROOT, RTMLIB_LAB, SCRIPTS_MSPT

NOTEBOOKS = SCRIPTS_MSPT
sys.path.insert(0, str(SCRIPTS_MSPT))
sys.path.insert(0, str(REPO_ROOT))

import slr_common as C  # noqa: E402
from mspt.dataset import BODY_DIM, FACE_DIM, HAND_DIM  # noqa: E402
from mspt.live_extract import LiveStreamExtractor  # noqa: E402
from mspt.model import MSPT  # noqa: E402
from mspt.normalize import (  # noqa: E402
    flatten_xy,
    normalize_body,
    normalize_face_from_mesh,
    normalize_hands,
)
from face_landmarks import FACE_IDXS  # noqa: E402
from mspt.skeleton_viz import composite_bottom_right, render_skeleton_panel  # noqa: E402

DEFAULT_WIDTH = 1280
DEFAULT_SKELETON_PANEL = 420
DEFAULT_HEIGHT = 960
DEFAULT_FPS = 10
DEFAULT_BUFFER_CLEAR_SEC = 2.0
DEFAULT_PREDICT_INTERVAL_SEC = 0.4


def load_model(ckpt_path: Path, max_seq_len: int, device: str) -> MSPT:
    model = MSPT(
        hand_dim=HAND_DIM,
        body_dim=BODY_DIM,
        face_dim=FACE_DIM,
        num_classes=C.NUM_CLASSES,
        max_len=max_seq_len,
        use_checkpoint=False,
        sequential_streams=True,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def sequences_to_tensors(
    hands_buf: list[np.ndarray],
    body_buf: list[np.ndarray],
    face_buf: list[np.ndarray],
    max_seq_len: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hands = np.stack(hands_buf, axis=0)
    body = np.stack(body_buf, axis=0)
    face = np.stack(face_buf, axis=0)

    t = len(hands)
    if t > max_seq_len:
        idx = np.linspace(0, t - 1, max_seq_len, dtype=int)
        hands, body, face = hands[idx], body[idx], face[idx]
        t = max_seq_len

    hands = normalize_hands(hands)
    body = normalize_body(body)
    face = normalize_face_from_mesh(face, FACE_IDXS)

    hand_t = torch.from_numpy(flatten_xy(hands)).unsqueeze(0).to(device)
    body_t = torch.from_numpy(flatten_xy(body)).unsqueeze(0).to(device)
    face_t = torch.from_numpy(flatten_xy(face)).unsqueeze(0).to(device)
    mask = torch.ones(1, t, device=device)
    return hand_t, body_t, face_t, mask


def frame_motion(hands: np.ndarray, body: np.ndarray, prev: np.ndarray | None) -> float:
    cur = np.concatenate([hands.reshape(-1), body.reshape(-1)])
    if prev is None or prev.shape != cur.shape:
        return 0.0
    valid = (cur != 0) & (prev != 0)
    if not valid.any():
        return 0.0
    return float(np.mean(np.abs(cur[valid] - prev[valid])))


def clear_buffers(
    hands_buf: list,
    body_buf: list,
    face_buf: list,
    prev_kp: list,
) -> None:
    hands_buf.clear()
    body_buf.clear()
    face_buf.clear()
    prev_kp.clear()


@torch.inference_mode()
def predict(
    model: MSPT,
    hands_buf: list[np.ndarray],
    body_buf: list[np.ndarray],
    face_buf: list[np.ndarray],
    max_seq_len: int,
    device: str,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    if not hands_buf:
        return []
    hand_t, body_t, face_t, mask = sequences_to_tensors(
        hands_buf, body_buf, face_buf, max_seq_len, device
    )
    logits = model(hand_t, body_t, face_t, mask)
    probs = torch.softmax(logits, dim=-1)[0]
    k = min(top_k, probs.numel())
    vals, idxs = torch.topk(probs, k)
    _, idx_to_label = C.load_label_map()
    return [(idx_to_label[int(i)], float(v)) for i, v in zip(idxs.cpu(), vals.cpu())]


def draw_hud(
    frame: np.ndarray,
    moving: bool,
    n_frames: int,
    last_preds: list[tuple[str, float]],
    status: str,
    motion: float,
    secs_to_clear: float,
) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 88), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    color = (0, 0, 255) if moving else (120, 120, 120)
    cv2.circle(frame, (24, 28), 10, color, -1 if moving else 2)
    cv2.putText(
        frame,
        f"{'MOTION' if moving else 'still'}  frames={n_frames}  motion={motion:.4f}  clear in {secs_to_clear:.1f}s",
        (44, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT} @ {DEFAULT_FPS}fps | auto predict | q quit",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    if status:
        cv2.putText(frame, status, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    y0 = 100
    for i, (label, prob) in enumerate(last_preds[:5]):
        text = f"{i + 1}. {label}: {prob * 100:.1f}%"
        cv2.putText(frame, text, (12, y0 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def configure_camera(cap: cv2.VideoCapture, width: int, height: int, fps: int) -> tuple[int, int, float]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps)
    return actual_w, actual_h, actual_fps


def run_webcam(args: argparse.Namespace) -> None:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[webcam] CUDA unavailable, using CPU")

    ckpt = args.checkpoint.resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"[webcam] checkpoint: {ckpt}")
    print(
        f"[webcam] auto mode: predict every {args.predict_interval}s while motion, "
        f"buffer clear every {args.buffer_clear_interval}s"
    )

    model = load_model(ckpt, args.max_seq_len, device)
    hands_buf: list[np.ndarray] = []
    body_buf: list[np.ndarray] = []
    face_buf: list[np.ndarray] = []
    prev_kp: list[np.ndarray] = []
    last_preds: list[tuple[str, float]] = []
    status = ""
    last_motion = 0.0

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    aw, ah, afps = configure_camera(cap, args.width, args.height, args.fps)
    frame_interval = 1.0 / args.fps
    print(f"[webcam] camera {aw}x{ah} @ {afps:.1f} fps")

    mirror = not args.no_mirror
    loop_start = time.monotonic()
    last_predict_time = 0.0
    last_clear_time = loop_start
    motion_in_window = False

    with LiveStreamExtractor(frame_size=args.frame_size) as extractor:
        next_frame_time = time.monotonic()
        while True:
            now = time.monotonic()
            if now < next_frame_time:
                if cv2.waitKey(max(1, int((next_frame_time - now) * 1000))) & 0xFF in (ord("q"), 27):
                    break
                continue
            next_frame_time = now + frame_interval

            ok, frame = cap.read()
            if not ok:
                print("[webcam] camera read failed")
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            try:
                hands, body, face = extractor.process_frame(frame)
            except Exception as exc:
                status = f"landmark error: {exc}"
                secs_left = max(0.0, args.buffer_clear_interval - (now - last_clear_time))
                draw_hud(frame, False, len(hands_buf), last_preds, status, last_motion, secs_left)
                cv2.imshow(args.window, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue

            prev = prev_kp[0] if prev_kp else None
            cur_kp = np.concatenate([hands.reshape(-1), body.reshape(-1)])
            last_motion = frame_motion(hands, body, prev)
            prev_kp.clear()
            prev_kp.append(cur_kp)

            moving = last_motion >= args.motion_threshold

            if moving:
                motion_in_window = True
                hands_buf.append(hands)
                body_buf.append(body)
                face_buf.append(face)
                if len(hands_buf) > args.max_buffer:
                    hands_buf.pop(0)
                    body_buf.pop(0)
                    face_buf.pop(0)

                # Continuous prediction while motion (buffer grows until 2s clear)
                if (
                    len(hands_buf) >= args.min_frames
                    and (now - last_predict_time) >= args.predict_interval
                ):
                    last_preds = predict(
                        model, hands_buf, body_buf, face_buf, args.max_seq_len, device, args.top_k
                    )
                    if last_preds:
                        status = f"{last_preds[0][0]} ({last_preds[0][1] * 100:.0f}%)"
                    last_predict_time = now

            # Hard buffer reset every N seconds
            if (now - last_clear_time) >= args.buffer_clear_interval:
                if motion_in_window and len(hands_buf) >= args.min_frames:
                    last_preds = predict(
                        model, hands_buf, body_buf, face_buf, args.max_seq_len, device, args.top_k
                    )
                    if last_preds:
                        status = f"{last_preds[0][0]} ({last_preds[0][1] * 100:.0f}%)"
                    last_predict_time = now
                clear_buffers(hands_buf, body_buf, face_buf, prev_kp)
                last_clear_time = now
                motion_in_window = False

            secs_left = max(0.0, args.buffer_clear_interval - (now - last_clear_time))
            draw_hud(frame, moving, len(hands_buf), last_preds, status, last_motion, secs_left)
            skel_panel = render_skeleton_panel(
                hands, body, face, panel_size=args.skeleton_panel_size
            )
            composite_bottom_right(frame, skel_panel, margin=args.skeleton_margin)
            cv2.imshow(args.window, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MSPT live webcam — automatic motion-driven inference",
    )
    ap.add_argument("--checkpoint", type=Path, default=MSPT_CHECKPOINTS / "mspt_best.pt")
    ap.add_argument("--label-map", type=Path, default=C.LABEL_MAP_PATH)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--frame-size", type=int, default=224)
    ap.add_argument("--max-seq-len", type=int, default=96)
    ap.add_argument("--max-buffer", type=int, default=96)
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--window", default="MSPT webcam")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--motion-threshold", type=float, default=0.008)
    ap.add_argument(
        "--buffer-clear-interval",
        type=float,
        default=DEFAULT_BUFFER_CLEAR_SEC,
        help="Clear buffer every N seconds (default 2)",
    )
    ap.add_argument(
        "--predict-interval",
        type=float,
        default=DEFAULT_PREDICT_INTERVAL_SEC,
        help="Min seconds between predictions while motion continues",
    )
    ap.add_argument(
        "--skeleton-panel-size",
        type=int,
        default=DEFAULT_SKELETON_PANEL,
        help="Bottom-right skeleton overlay size in pixels (square)",
    )
    ap.add_argument(
        "--skeleton-margin",
        type=int,
        default=20,
        help="Margin from bottom-right edge for skeleton panel",
    )
    args = ap.parse_args()
    run_webcam(args)


if __name__ == "__main__":
    main()
