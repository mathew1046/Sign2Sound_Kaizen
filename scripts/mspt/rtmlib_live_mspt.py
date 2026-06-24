#!/usr/bin/env python3
"""Live rtmlib GPU + MSPT from an HTTP video feed (e.g. localhost:8080/video).

Smooth preview (async frame grab + pose worker), large prediction overlay,
rtmlib COCO-WholeBody skeleton panel bottom-left. Clip-based inference like
``run_mspt_webcam_finetuned.py``.

Usage:
  conda activate base
  cd notebooks
  python rtmlib_live_mspt.py \\
    --video-url http://localhost:8080/video \\
    --checkpoint checkpoints/mspt_rtmlib_263_best.pt \\
    --lab-root data/include50_rtmlib_1080
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from pathlib import Path
from typing import Iterator
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import torch

from repo_paths import MSPT_CHECKPOINTS, REPO_ROOT, RTMLIB_LAB, SCRIPTS_MSPT

NOTEBOOKS = SCRIPTS_MSPT
sys.path.insert(0, str(SCRIPTS_MSPT))
sys.path.insert(0, str(REPO_ROOT))

from mspt.dataset import BODY_DIM, FACE_DIM, HAND_DIM  # noqa: E402
from mspt.label_utils import label_names_for_checkpoint, num_classes_from_checkpoint  # noqa: E402
from mspt.model import MSPT  # noqa: E402
from mspt.normalize import flatten_xy  # noqa: E402
from mspt.rtmlib_io import streams_from_wholebody  # noqa: E402
from mspt.rtmlib_preprocess import (  # noqa: E402
    BODY_SLICE,
    FACE_SLICE,
    LEFT_HAND_SLICE,
    RIGHT_HAND_SLICE,
    RtmlibWholebodyExtractor,
)
from mspt.skeleton_viz import composite_bottom_left, render_skeleton_panel  # noqa: E402

DEFAULT_VIDEO_URL = "http://localhost:8080/video"
DEFAULT_LAB = RTMLIB_LAB
DEFAULT_CKPT = MSPT_CHECKPOINTS / "mspt_rtmlib_263_best.pt"
DEFAULT_CLIP_SEC = 2.5
DEFAULT_GAP_SEC = 1.0
DEFAULT_FPS = 10
DEFAULT_DISPLAY_FPS = 0
DEFAULT_SKELETON_PANEL = 420
DEFAULT_POSE_MAX_WIDTH = 960
DEFAULT_MIN_CONFIDENCE = 0.12
DEFAULT_HOLD_PRED_SEC = 2.0
DEFAULT_MOTION_THRESHOLD = 0.008


def _display_gloss(label: str) -> str:
    return label.replace("_", " ").title()


def resize_for_pose(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)


def _parse_mjpeg_boundary(content_type: str) -> bytes:
    m = re.search(r"boundary=([^;\s]+)", content_type, flags=re.I)
    if not m:
        return b"--frame"
    raw = m.group(1).strip().strip('"')
    return raw.encode() if not raw.startswith(b"--") else raw


def _iter_mjpeg_frames(url: str, timeout: float = 10.0) -> Iterator[np.ndarray]:
    req = Request(url, headers={"User-Agent": "signbert-rtmlib-live/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        boundary = _parse_mjpeg_boundary(resp.headers.get("Content-Type", ""))
        buf = b""
        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            buf += chunk
            while True:
                start = buf.find(boundary)
                if start < 0:
                    break
                next_b = buf.find(boundary, start + len(boundary))
                if next_b < 0:
                    buf = buf[start:]
                    break
                part = buf[start + len(boundary) : next_b]
                buf = buf[next_b:]
                header_end = part.find(b"\r\n\r\n")
                if header_end < 0:
                    continue
                jpeg = part[header_end + 4 :].strip(b"\r\n")
                if not jpeg:
                    continue
                frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame


class FrameGrabber:
    """Continuously read frames so the UI always shows the latest image."""

    def __init__(self, source: str | cv2.VideoCapture, mode: str):
        self._mode = mode
        self._source = source
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._alive = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="frame-grabber")
        self._thread.start()

    def _loop(self) -> None:
        if self._mode == "cv2":
            cap: cv2.VideoCapture = self._source
            while self._alive:
                ok, frame = cap.read()
                if not ok:
                    with self._lock:
                        self._alive = False
                    return
                with self._lock:
                    self._frame = frame
            return

        url = str(self._source)
        try:
            for frame in _iter_mjpeg_frames(url):
                if not self._alive:
                    break
                with self._lock:
                    self._frame = frame
        except Exception:
            with self._lock:
                self._alive = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._frame is None:
                return self._alive, None
            return self._alive, self._frame

    def stop(self) -> None:
        self._alive = False
        self._thread.join(timeout=2.0)


class PoseWorker:
    """Background rtmlib pose — display never blocks on inference."""

    def __init__(self, extractor: RtmlibWholebodyExtractor, pose_max_width: int, unmirror_pose: bool):
        self._extractor = extractor
        self._pose_max_width = pose_max_width
        self._unmirror_pose = unmirror_pose
        self._lock = threading.Lock()
        self._pending: np.ndarray | None = None
        self._latest: np.ndarray | None = None
        self._version = 0
        self._polled_version = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="pose-worker", daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            self._pending = frame

    def poll_update(self) -> np.ndarray | None:
        with self._lock:
            if self._latest is None or self._version == self._polled_version:
                return None
            self._polled_version = self._version
            return self._latest

    def discard_pending(self) -> None:
        """Drop queued frames so the next clip starts clean."""
        with self._lock:
            self._pending = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            frame = None
            with self._lock:
                if self._pending is not None:
                    frame = self._pending
                    self._pending = None
            if frame is None:
                time.sleep(0.002)
                continue
            if self._unmirror_pose:
                frame = cv2.flip(frame, 1)
            small = resize_for_pose(frame, self._pose_max_width)
            wb = self._extractor.process_frame(small)
            with self._lock:
                self._latest = wb
                self._version += 1

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def wholebody_to_viz(wb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = wb[..., :2].copy()
    valid = wb[..., 3] > 0
    xy[~valid] = 0.0
    lh = xy[LEFT_HAND_SLICE]
    rh = xy[RIGHT_HAND_SLICE]
    hands = np.concatenate([lh, rh], axis=0)
    body = np.zeros((33, 2), dtype=np.float32)
    body[:17] = xy[BODY_SLICE]
    face = xy[FACE_SLICE]
    return hands, body, face


def frame_motion(hands: np.ndarray, body: np.ndarray, prev: np.ndarray | None) -> float:
    cur = np.concatenate([hands.reshape(-1), body.reshape(-1)])
    if prev is None or prev.shape != cur.shape:
        return 0.0
    valid = (cur != 0) & (prev != 0)
    if not valid.any():
        return 0.0
    return float(np.mean(np.abs(cur[valid] - prev[valid])))


def open_video_capture(video_url: str) -> tuple[cv2.VideoCapture | str, str]:
    cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok and frame is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return cap, "cv2"
        cap.release()
    return video_url, "mjpeg"


def load_model(
    ckpt_path: Path,
    max_seq_len: int,
    device: str,
    lab_root: Path,
) -> tuple[MSPT, dict, dict[int, str]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_classes = num_classes_from_checkpoint(ckpt) or 50
    names = label_names_for_checkpoint(ckpt, lab_root=lab_root)
    idx_to_label = {i: names[i] for i in range(len(names))}
    model = MSPT(
        hand_dim=HAND_DIM,
        body_dim=BODY_DIM,
        face_dim=FACE_DIM,
        num_classes=n_classes,
        max_len=max_seq_len,
        use_checkpoint=False,
        sequential_streams=True,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt, idx_to_label


@torch.inference_mode()
def predict_clip(
    model: MSPT,
    wholebody_buf: list[np.ndarray],
    max_seq_len: int,
    device: str,
    idx_to_label: dict[int, str],
    top_k: int = 3,
    min_confidence: float = 0.0,
) -> list[tuple[str, float]]:
    if not wholebody_buf:
        return []
    seq = np.stack(wholebody_buf, axis=0).astype(np.float32)
    hands, body, face, t = streams_from_wholebody(seq, max_seq_len)
    t = max(1, t)
    hand_t = torch.from_numpy(flatten_xy(hands)).unsqueeze(0).to(device)
    body_t = torch.from_numpy(flatten_xy(body)).unsqueeze(0).to(device)
    face_t = torch.from_numpy(flatten_xy(face)).unsqueeze(0).to(device)
    mask = torch.zeros(1, t, device=device)
    mask[0, :t] = 1.0
    logits = model(hand_t[:, :t], body_t[:, :t], face_t[:, :t], mask)
    probs = torch.softmax(logits, dim=-1)[0]
    k = min(top_k, probs.numel())
    vals, idxs = torch.topk(probs, k)
    preds = [(idx_to_label.get(int(i), f"class_{int(i)}"), float(v)) for i, v in zip(idxs.cpu(), vals.cpu())]
    if preds and preds[0][1] < min_confidence:
        return [("uncertain", preds[0][1])] + preds[: top_k - 1]
    return preds


def _end_clip(wb_buf: list[np.ndarray], worker: PoseWorker) -> None:
    wb_buf.clear()
    worker.discard_pending()


def draw_predictions_large(
    frame: np.ndarray,
    preds: list[tuple[str, float]],
    top_k: int = 3,
    recording: bool = False,
) -> None:
    """Large primary prediction + smaller runner-ups (top-right)."""
    h, w = frame.shape[:2]
    pad = 20
    if recording:
        text = "Signing…"
        font_primary = 1.6
        thickness_primary = 4
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_primary, thickness_primary)
        x0 = max(pad, (w - tw) // 2)
        y0 = int(h * 0.14)
        cv2.putText(
            frame, text, (x0, y0),
            cv2.FONT_HERSHEY_SIMPLEX, font_primary, (0, 180, 255), thickness_primary, cv2.LINE_AA,
        )
        return
    if not preds:
        return
    primary_raw = preds[0][0]
    primary = "Uncertain" if primary_raw == "uncertain" else _display_gloss(primary_raw)
    conf = preds[0][1] * 100.0
    color = (0, 180, 255) if primary_raw == "uncertain" else (0, 255, 120)

    # Primary gloss — large, top center
    font_primary = 2.2
    thickness_primary = 5
    (tw, th), _ = cv2.getTextSize(primary, cv2.FONT_HERSHEY_SIMPLEX, font_primary, thickness_primary)
    x0 = max(pad, (w - tw) // 2)
    y0 = int(h * 0.14)
    roi = frame[max(0, y0 - th - pad) : y0 + pad, max(0, x0 - pad) : min(w, x0 + tw + pad)]
    if roi.size:
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (roi.shape[1], roi.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, roi, 0.35, 0, roi)
    cv2.putText(
        frame, primary, (x0, y0),
        cv2.FONT_HERSHEY_SIMPLEX, font_primary, color, thickness_primary, cv2.LINE_AA,
    )
    conf_text = f"{conf:.1f}%"
    cv2.putText(
        frame, conf_text, (x0, y0 + int(th * 0.9) + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 255), 3, cv2.LINE_AA,
    )

    # Runner-ups — top right
    line_h = 48
    box_w = 520
    header_h = 36
    n_show = min(top_k, len(preds))
    box_h = pad + header_h + line_h * max(0, n_show - 1) + pad
    rx0 = w - box_w - pad
    ry0 = pad
    roi2 = frame[ry0 : ry0 + box_h, rx0 : w - pad]
    if roi2.size:
        overlay2 = roi2.copy()
        cv2.rectangle(overlay2, (0, 0), (roi2.shape[1], roi2.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.55, roi2, 0.45, 0, roi2)
    cv2.putText(
        frame, "Top predictions", (rx0 + 12, ry0 + 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2, cv2.LINE_AA,
    )
    for i, (label, prob) in enumerate(preds[1:top_k], start=1):
        text = f"{i + 1}. {_display_gloss(label)}: {prob * 100:.1f}%"
        cv2.putText(
            frame, text, (rx0 + 12, ry0 + header_h + 34 + (i - 1) * line_h),
            cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 120), 2, cv2.LINE_AA,
        )


def draw_status_bar(
    frame: np.ndarray,
    phase: str,
    elapsed: float,
    clip_sec: float,
    gap_sec: float,
    n_frames: int,
    motion_frames: int,
    moving: bool,
    pose_fps: float,
    display_fps: float,
    rtmlib_device: str,
) -> None:
    h, w = frame.shape[:2]
    bar_h = 56
    roi = frame[0:bar_h, 0:w]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, roi, 0.45, 0, roi)
    if phase == "recording":
        color = (0, 0, 255) if moving else (120, 120, 120)
        msg = (
            f"{'MOTION' if moving else 'still'} {elapsed:.1f}/{clip_sec:.1f}s  "
            f"buf={n_frames} motion_frames={motion_frames}"
        )
    elif phase == "hold":
        color = (0, 255, 120)
        msg = f"PREDICTION {elapsed:.1f}/{gap_sec:.1f}s hold"
    else:
        color = (0, 255, 255)
        msg = f"READY — next clip in {max(0.0, gap_sec - elapsed):.1f}s"
    msg += f"  rtmlib:{rtmlib_device} pose={pose_fps:.0f}fps"
    if display_fps > 0:
        msg += f"  display={display_fps:.0f}fps"
    msg += "  |  q quit"
    cv2.circle(frame, (20, 28), 8, color, -1)
    cv2.putText(frame, msg, (38, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def run_live(args: argparse.Namespace) -> None:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[live] CUDA unavailable for MSPT, using CPU")

    ckpt = args.checkpoint.resolve()
    lab_root = args.lab_root.resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    video_url = (args.video_url or args.stream_url).strip()
    print(f"[live] stream: {video_url}")
    print(f"[live] checkpoint: {ckpt} ({lab_root})")
    print(
        f"[live] clip={args.clip_sec}s gap={args.gap_sec}s hold={args.hold_pred_sec}s "
        f"fps={args.fps} motion>={args.motion_threshold}"
    )

    extractor = RtmlibWholebodyExtractor(device=args.rtmlib_device)
    worker = PoseWorker(extractor, args.pose_max_width, unmirror_pose=args.unmirror_pose)
    print(f"[live] rtmlib device: {extractor.device} | model: yolox-x + rtmw-x")

    model, ckpt_obj, idx_to_label = load_model(ckpt, args.max_seq_len, device, lab_root)
    print(f"[live] MSPT loaded: {len(idx_to_label)} classes, val={ckpt_obj.get('val_acc')}")

    source, mode = open_video_capture(video_url)
    grabber = FrameGrabber(source, mode)

    wb_buf: list[np.ndarray] = []
    last_preds: list[tuple[str, float]] = []
    prev_kp: list[np.ndarray] = []
    last_motion = 0.0
    motion_frames = 0
    last_hands = np.zeros((42, 2), dtype=np.float32)
    last_body = np.zeros((33, 2), dtype=np.float32)
    last_face = np.zeros((68, 2), dtype=np.float32)
    cached_skel: np.ndarray | None = None

    phase = "recording"
    clip_start = time.monotonic()
    gap_start = 0.0
    hold_start = 0.0
    extract_interval = 1.0 / max(args.fps, 1)
    display_interval = (1.0 / args.display_fps) if args.display_fps > 0 else 0.0
    next_extract_time = time.monotonic()
    next_display_time = time.monotonic()
    pose_fps_ema = 0.0
    last_pose_wall = time.monotonic()
    display_frames = 0
    display_t0 = time.monotonic()
    measured_display_fps = 0.0

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
                print("[live] stream ended")
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

            wb = worker.poll_update()
            if wb is not None:
                dt = max(now - last_pose_wall, 1e-6)
                inst = 1.0 / dt
                pose_fps_ema = inst if pose_fps_ema == 0 else 0.85 * pose_fps_ema + 0.15 * inst
                last_pose_wall = now

                hands, body, face = wholebody_to_viz(wb)
                last_hands, last_body, last_face = hands, body, face
                cached_skel = render_skeleton_panel(
                    last_hands, last_body, last_face, panel_size=args.skeleton_panel_size,
                )

                prev = prev_kp[0] if prev_kp else None
                cur_kp = np.concatenate([hands.reshape(-1), body.reshape(-1)])
                last_motion = frame_motion(hands, body, prev)
                prev_kp.clear()
                prev_kp.append(cur_kp)
                moving = last_motion >= args.motion_threshold

                if phase == "recording":
                    if moving:
                        wb_buf.append(wb)
                        motion_frames += 1
                    elapsed = now - clip_start
                    if elapsed >= args.clip_sec:
                        if motion_frames >= args.min_motion_frames and len(wb_buf) >= args.min_frames:
                            last_preds = predict_clip(
                                model, wb_buf, args.max_seq_len, device, idx_to_label, args.top_k,
                                min_confidence=args.min_confidence,
                            )
                            hold_start = now
                            phase = "hold"
                        else:
                            last_preds = []
                        _end_clip(wb_buf, worker)
                        prev_kp.clear()
                        motion_frames = 0
                        gap_start = now
                        phase = "gap"

            elif phase == "recording":
                elapsed = now - clip_start
                if elapsed >= args.clip_sec:
                    if motion_frames >= args.min_motion_frames and len(wb_buf) >= args.min_frames:
                        last_preds = predict_clip(
                            model, wb_buf, args.max_seq_len, device, idx_to_label, args.top_k,
                            min_confidence=args.min_confidence,
                        )
                        hold_start = now
                        phase = "hold"
                    else:
                        last_preds = []
                    _end_clip(wb_buf, worker)
                    prev_kp.clear()
                    motion_frames = 0
                    gap_start = now
                    phase = "gap"

            if phase == "hold":
                if (now - hold_start) >= args.hold_pred_sec:
                    last_preds = []
                    gap_start = now
                    phase = "gap"

            if phase == "gap":
                elapsed = now - gap_start
                if elapsed >= args.gap_sec:
                    _end_clip(wb_buf, worker)
                    prev_kp.clear()
                    motion_frames = 0
                    last_preds = []
                    phase = "recording"
                    clip_start = now

            display = frame.copy()
            if phase == "recording":
                elapsed_ui = now - clip_start
            elif phase == "hold":
                elapsed_ui = now - hold_start
            else:
                elapsed_ui = now - gap_start
            draw_status_bar(
                display, phase, elapsed_ui, args.clip_sec, args.gap_sec, len(wb_buf),
                motion_frames, last_motion >= args.motion_threshold,
                pose_fps_ema, measured_display_fps, extractor.device,
            )
            draw_predictions_large(
                display, last_preds, args.top_k, recording=(phase == "recording"),
            )
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
    except KeyboardInterrupt:
        print("\n[live] stopped")
    except URLError as exc:
        print(f"[live] cannot open stream: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        grabber.stop()
        worker.close()
        extractor.close()
        if mode == "cv2":
            source.release()
        cv2.destroyAllWindows()


def main() -> int:
    ap = argparse.ArgumentParser(description="Live rtmlib GPU + MSPT from HTTP video feed")
    ap.add_argument("--video-url", "--stream-url", default=DEFAULT_VIDEO_URL, dest="video_url")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--lab-root", type=Path, default=DEFAULT_LAB, help="rtmlib lab (manifest.csv for 263-class names)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rtmlib-device", default=None, help="onnxruntime device for rtmlib (cuda/cpu)")
    ap.add_argument("--pose-max-width", type=int, default=DEFAULT_POSE_MAX_WIDTH)
    ap.add_argument("--clip-sec", type=float, default=DEFAULT_CLIP_SEC,
                    help="Wall-clock window per sign (buffer cleared after this)")
    ap.add_argument("--gap-sec", type=float, default=DEFAULT_GAP_SEC,
                    help="Pause after prediction before next clip")
    ap.add_argument("--hold-pred-sec", type=float, default=DEFAULT_HOLD_PRED_SEC,
                    help="How long to show prediction before clearing")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS,
                    help="Pose extract rate (match stream; dashboard webcam is 10)")
    ap.add_argument("--display-fps", type=int, default=DEFAULT_DISPLAY_FPS, help="0 = uncapped (smoothest)")
    ap.add_argument("--max-seq-len", type=int, default=96)
    ap.add_argument("--min-frames", type=int, default=8, help="Min pose frames in buffer to predict")
    ap.add_argument("--min-motion-frames", type=int, default=6,
                    help="Min frames with hand motion required to predict")
    ap.add_argument("--motion-threshold", type=float, default=DEFAULT_MOTION_THRESHOLD)
    ap.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                    help="Below this top-1 prob, show 'uncertain' (263-class softmax is naturally low)")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--window", default="MSPT rtmlib live")
    ap.add_argument(
        "--unmirror-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flip frame before rtmlib (use when stream is already mirrored)",
    )
    ap.add_argument("--skeleton-panel-size", type=int, default=DEFAULT_SKELETON_PANEL)
    ap.add_argument("--skeleton-margin", type=int, default=20)
    run_live(ap.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
