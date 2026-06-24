#!/usr/bin/env python3
"""Live webcam inference for MSPT — fixed 2.5 s clip windows.

Each cycle:
  1. Clear keypoint buffer
  2. Record for ``--clip-sec`` seconds (default 2.5)
  3. Run model on the full clip
  4. Show top-3 predictions (top-right) + skeleton panel (bottom-left)
  5. Repeat

Usage (conda base):
  cd notebooks
  python run_mspt_webcam_finetuned.py

  # Original model from an HTTP MJPEG feed (e.g. phone webcam app on :8080)
  python run_mspt_webcam_finetuned.py \\
    --checkpoint checkpoints/mspt_best.pt \\
    --video-url http://localhost:8080/video \\
    --no-mirror
"""

from __future__ import annotations

import argparse
import sys
import threading
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
from mspt.skeleton_viz import composite_bottom_left, render_skeleton_panel  # noqa: E402

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 960
DEFAULT_FPS = 30
DEFAULT_DISPLAY_FPS = 0  # 0 = uncapped (smooth preview)
DEFAULT_CLIP_SEC = 2.5
DEFAULT_SKELETON_PANEL = 420


class FrameGrabber:
    """Continuously read frames so the UI always shows the latest image."""

    def __init__(self, cap: cv2.VideoCapture, mirror: bool):
        self._cap = cap
        self._mirror = mirror
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._alive = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="frame-grabber")
        self._thread.start()

    def _loop(self) -> None:
        while True:
            ok, frame = self._cap.read()
            if not ok:
                with self._lock:
                    self._alive = False
                return
            if self._mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._frame is None:
                return self._alive, None
            return self._alive, self._frame

    def stop(self) -> None:
        self._thread.join(timeout=1.0)


class AsyncExtractor:
    """Run MediaPipe off the display thread; keep only the newest pending frame."""

    def __init__(self, extractor: LiveStreamExtractor):
        self._extractor = extractor
        self._lock = threading.Lock()
        self._pending: np.ndarray | None = None
        self._latest: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._error: str | None = None
        self._new = False
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="mp-extract")
        self._thread.start()

    def submit(self, bgr: np.ndarray) -> None:
        with self._lock:
            self._pending = bgr

    def _loop(self) -> None:
        while not self._stop:
            with self._lock:
                frame = self._pending
                self._pending = None
            if frame is None:
                time.sleep(0.001)
                continue
            try:
                result = self._extractor.process_frame(frame)
                err = None
            except Exception as exc:
                result = None
                err = str(exc)
            with self._lock:
                if result is not None:
                    self._latest = result
                    self._new = True
                self._error = err

    def consume(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        with self._lock:
            if not self._new or self._latest is None:
                return None
            self._new = False
            return self._latest

    def last_error(self) -> str | None:
        with self._lock:
            return self._error

    def stop(self) -> None:
        self._stop = True
        self._thread.join(timeout=1.0)


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


def _display_gloss(label: str) -> str:
    return label.replace("_", " ").title()


@torch.inference_mode()
def predict_clip(
    model: MSPT,
    hands_buf: list[np.ndarray],
    body_buf: list[np.ndarray],
    face_buf: list[np.ndarray],
    max_seq_len: int,
    device: str,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    if not hands_buf:
        return []
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
    logits = model(hand_t, body_t, face_t, mask)
    probs = torch.softmax(logits, dim=-1)[0]
    k = min(top_k, probs.numel())
    vals, idxs = torch.topk(probs, k)
    _, idx_to_label = C.load_label_map()
    return [(idx_to_label[int(i)], float(v)) for i, v in zip(idxs.cpu(), vals.cpu())]


def draw_top_right_predictions(
    frame: np.ndarray,
    preds: list[tuple[str, float]],
    top_k: int = 3,
) -> None:
    h, w = frame.shape[:2]
    line_h = 56
    pad = 16
    header_h = 40
    box_h = pad * 2 + header_h + line_h * min(top_k, len(preds))
    box_w = 560
    x0 = w - box_w - pad
    y0 = pad
    x1, y1 = w - pad, y0 + box_h
    roi = frame[y0:y1, x0:x1]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, roi, 0.4, 0, roi)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 255), 3, cv2.LINE_AA)
    cv2.putText(
        frame, "Top predictions", (x0 + 14, y0 + 34),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 3, cv2.LINE_AA,
    )
    for i, (label, prob) in enumerate(preds[:top_k]):
        text = f"{i + 1}. {_display_gloss(label)}: {prob * 100:.1f}%"
        font_scale = 1.15 if i == 0 else 0.95
        thickness = 3 if i == 0 else 2
        cv2.putText(
            frame, text, (x0 + 14, y0 + header_h + pad + 38 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 120), thickness, cv2.LINE_AA,
        )


def draw_status_bar(
    frame: np.ndarray,
    phase: str,
    elapsed: float,
    clip_sec: float,
    n_frames: int,
    fps: float,
    display_fps: float = 0.0,
) -> None:
    h, w = frame.shape[:2]
    bar_h = 56
    roi = frame[0:bar_h, 0:w]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, roi, 0.45, 0, roi)
    if phase == "recording":
        color = (0, 0, 255)
        msg = f"RECORDING {elapsed:.1f}/{clip_sec:.1f}s  frames={n_frames}  extract={fps:.0f}fps"
    else:
        color = (0, 255, 255)
        msg = f"READY — next clip in {max(0.0, clip_sec - elapsed):.1f}s  extract={fps:.0f}fps"
    if display_fps > 0:
        msg += f"  display={display_fps:.0f}fps"
    msg += "  |  q quit"
    cv2.circle(frame, (20, 28), 8, color, -1)
    cv2.putText(frame, msg, (38, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def configure_camera(cap: cv2.VideoCapture, width: int, height: int, fps: int) -> tuple[int, int, float]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        cap.get(cv2.CAP_PROP_FPS) or float(fps),
    )


def open_video_capture(args: argparse.Namespace) -> tuple[cv2.VideoCapture, str]:
    if args.video_url:
        url = args.video_url.strip()
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video URL: {url}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
        afps = cap.get(cv2.CAP_PROP_FPS) or float(args.fps)
        return cap, f"stream {url} ({aw}x{ah} @ {afps:.1f} fps)"
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    aw, ah, afps = configure_camera(cap, args.width, args.height, args.fps)
    return cap, f"camera {args.camera} ({aw}x{ah} @ {afps:.1f} fps)"


def run_webcam(args: argparse.Namespace) -> None:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[webcam] CUDA unavailable, using CPU")

    ckpt = args.checkpoint.resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    target_frames = max(1, round(args.clip_sec * args.fps))
    print(f"[webcam] checkpoint: {ckpt}")
    print(f"[webcam] clip mode: {args.clip_sec}s (~{target_frames} frames @ {args.fps} fps), top-{args.top_k} UI")

    model = load_model(ckpt, args.max_seq_len, device)
    last_preds: list[tuple[str, float]] = []
    last_hands = np.zeros((42, 2), dtype=np.float32)
    last_body = np.zeros((33, 2), dtype=np.float32)
    last_face = np.zeros((72, 2), dtype=np.float32)

    cap, source_desc = open_video_capture(args)
    extract_interval = 1.0 / max(args.fps, 1)
    display_interval = (1.0 / args.display_fps) if args.display_fps > 0 else 0.0
    print(f"[webcam] {source_desc}")
    print(f"[webcam] display: {'uncapped' if display_interval == 0 else f'{args.display_fps} fps'}, extract: {args.fps} fps")

    mirror = not args.no_mirror
    hands_buf: list[np.ndarray] = []
    body_buf: list[np.ndarray] = []
    face_buf: list[np.ndarray] = []
    phase = "recording"
    clip_start = time.monotonic()
    gap_start = 0.0
    next_extract_time = time.monotonic()
    next_display_time = time.monotonic()
    cached_skel: np.ndarray | None = None
    display_frames = 0
    display_t0 = time.monotonic()
    measured_display_fps = 0.0

    grabber = FrameGrabber(cap, mirror=mirror)
    with LiveStreamExtractor(frame_size=args.frame_size) as extractor:
        worker = AsyncExtractor(extractor)
        try:
            while True:
                now = time.monotonic()
                if display_interval > 0 and now < next_display_time:
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                    continue
                if display_interval > 0:
                    next_display_time = now + display_interval

                alive, frame = grabber.read()
                if not alive and frame is None:
                    print("[webcam] frame read failed")
                    break
                if frame is None:
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                    continue

                if now >= next_extract_time:
                    worker.submit(frame.copy())
                    next_extract_time += extract_interval
                    if now - next_extract_time > extract_interval:
                        next_extract_time = now + extract_interval

                err = worker.last_error()
                if err:
                    display = frame.copy()
                    draw_status_bar(display, phase, 0.0, args.clip_sec, len(hands_buf), args.fps)
                    cv2.putText(display, err, (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.imshow(args.window, display)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                    continue

                result = worker.consume()
                if result is not None:
                    hands, body, face = result
                    last_hands, last_body, last_face = hands, body, face
                    cached_skel = render_skeleton_panel(
                        last_hands, last_body, last_face, panel_size=args.skeleton_panel_size,
                    )

                    if phase == "recording":
                        hands_buf.append(hands)
                        body_buf.append(body)
                        face_buf.append(face)
                        elapsed = now - clip_start
                        if len(hands_buf) >= target_frames or elapsed >= args.clip_sec:
                            if len(hands_buf) >= args.min_frames:
                                last_preds = predict_clip(
                                    model, hands_buf, body_buf, face_buf,
                                    args.max_seq_len, device, args.top_k,
                                )
                            hands_buf.clear()
                            body_buf.clear()
                            face_buf.clear()
                            phase = "gap"
                            gap_start = now
                elif phase == "recording":
                    elapsed = now - clip_start
                    if elapsed >= args.clip_sec and len(hands_buf) >= args.min_frames:
                        last_preds = predict_clip(
                            model, hands_buf, body_buf, face_buf,
                            args.max_seq_len, device, args.top_k,
                        )
                        hands_buf.clear()
                        body_buf.clear()
                        face_buf.clear()
                        phase = "gap"
                        gap_start = now

                if phase == "gap":
                    elapsed = now - gap_start
                    if elapsed >= args.gap_sec:
                        hands_buf.clear()
                        body_buf.clear()
                        face_buf.clear()
                        phase = "recording"
                        clip_start = now

                display = frame.copy()
                elapsed_rec = (now - clip_start) if phase == "recording" else 0.0
                draw_status_bar(
                    display, phase,
                    elapsed_rec if phase == "recording" else now - gap_start,
                    args.clip_sec, len(hands_buf), args.fps, measured_display_fps,
                )
                if last_preds:
                    draw_top_right_predictions(display, last_preds, args.top_k)
                if cached_skel is not None:
                    composite_bottom_left(display, cached_skel, margin=args.skeleton_margin)
                cv2.imshow(args.window, display)
                display_frames += 1
                if now - display_t0 >= 1.0:
                    measured_display_fps = display_frames / (now - display_t0)
                    display_frames = 0
                    display_t0 = now
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        finally:
            worker.stop()
            grabber.stop()

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description="MSPT webcam — 2.5s clip inference")
    ap.add_argument("--checkpoint", type=Path, default=MSPT_CHECKPOINTS / "mspt_finetuned.pt")
    ap.add_argument("--video-url", default="", help="HTTP MJPEG/video URL (e.g. http://localhost:8080/video)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Keyframe extract + clip sampling rate")
    ap.add_argument(
        "--display-fps",
        type=int,
        default=DEFAULT_DISPLAY_FPS,
        help="Max preview refresh rate (0 = uncapped, smoothest)",
    )
    ap.add_argument("--clip-sec", type=float, default=DEFAULT_CLIP_SEC, help="Record window per inference")
    ap.add_argument("--gap-sec", type=float, default=0.3, help="Pause after predict before next clip")
    ap.add_argument("--frame-size", type=int, default=224)
    ap.add_argument("--max-seq-len", type=int, default=96)
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--window", default="MSPT finetuned")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--skeleton-panel-size", type=int, default=DEFAULT_SKELETON_PANEL)
    ap.add_argument("--skeleton-margin", type=int, default=20)
    run_webcam(ap.parse_args())


if __name__ == "__main__":
    main()
