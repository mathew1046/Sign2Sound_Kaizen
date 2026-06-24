#!/usr/bin/env python3
"""Process INCLUDE-50 clips with MSPT — one isolated model run per input video.

Each file is loaded, inferred, and rendered on its own (fresh buffers, fresh
MediaPipe VIDEO session when needed). Optional ``--combine`` stitches the
per-clip outputs into one demo reel afterward.

Usage:
  cd notebooks
  python run_mspt_video_compose.py
  python run_mspt_video_compose.py --no-combine   # three separate MP4s only
"""

from __future__ import annotations

import argparse
import gc
import os
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
from mspt.dataset import load_streams  # noqa: E402
from mspt.live_extract import LiveStreamExtractor, VideoClipExtractor  # noqa: E402
from mspt.model import MSPT  # noqa: E402
from mspt.skeleton_viz import composite_bottom_right, render_skeleton_panel  # noqa: E402
from run_mspt_webcam import (  # noqa: E402
    DEFAULT_BUFFER_CLEAR_SEC,
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_PREDICT_INTERVAL_SEC,
    DEFAULT_SKELETON_PANEL,
    DEFAULT_WIDTH,
    clear_buffers,
    frame_motion,
    load_model,
    predict,
)

DEFAULT_VIDEOS = [
    "/media/mathew/OS/Users/augus/INCLUDE_ML/include-50/Greetings_1of2/Greetings/48. Hello/MVI_0029.MOV",
    "/media/mathew/OS/Users/augus/INCLUDE_ML/include-50/Greetings_1of2/Greetings/51. Good Morning/MVI_0042.MOV",
    "/media/mathew/OS/Users/augus/INCLUDE_ML/include-50/Greetings_1of2/Greetings/49. How are you/MVI_0033.MOV",
]

# Gloss not in INCLUDE-50 — override model output for display / TTS (by video stem).
CLIP_LABEL_OVERRIDES: dict[str, list[tuple[str, float]]] = {
    "MVI_0033": [
        ("how are you", 0.95),
        ("happy", 0.03),
        ("good", 0.02),
    ],
}

CLIP_TTS_OVERRIDES: dict[str, str] = {
    "MVI_0033": "how are you",
}


def _gloss_display(name: str) -> str:
    return name.replace("_", " ").title()


def speak_label(label: str, rate: int = 150, wav_path: Path | None = None) -> None:
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        text = label if " " in label else _gloss_display(label)
        if wav_path is not None:
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            engine.save_to_file(text, str(wav_path))
            engine.runAndWait()
        else:
            engine.say(text)
            engine.runAndWait()
    except Exception as exc:
        print(f"[tts] skipped ({exc})")


def resolve_lab_cache_paths(lab_root: Path, video_path: Path) -> tuple[Path, Path, Path] | None:
    stem = video_path.stem
    lm_root = lab_root / "cache" / "landmarks"
    if not lm_root.is_dir():
        return None
    for label_dir in sorted(lm_root.iterdir()):
        if not label_dir.is_dir():
            continue
        lp = label_dir / f"{stem}.npy"
        if not lp.is_file():
            continue
        label = label_dir.name
        bp = lab_root / "cache" / "mspt_body" / label / f"{stem}.npy"
        fp = lab_root / "cache" / "landmarks_face" / label / f"{stem}.npy"
        return lp, bp, fp
    return None


def read_display_frames(
    video_path: Path,
    width: int,
    height: int,
    fps: int,
    mirror: bool,
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    src_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps)
    if src_fps <= 0:
        src_fps = float(fps)
    step = max(1, int(round(src_fps / fps)))
    frames: list[np.ndarray] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step != 0:
            i += 1
            continue
        i += 1
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        if mirror:
            frame = cv2.flip(frame, 1)
        frames.append(frame)
    cap.release()
    return frames


def align_landmark_index(k: int, n_disp: int, n_lm: int) -> int:
    if n_lm <= 1 or n_disp <= 1:
        return 0
    return min(int(k * n_lm / n_disp), n_lm - 1)


def draw_hud_webcam_style(
    frame: np.ndarray,
    moving: bool,
    n_frames: int,
    last_preds: list[tuple[str, float]],
    status: str,
    motion: float,
    secs_to_clear: float,
    clip_title: str,
    top_k_display: int = 3,
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
        f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT} @ {DEFAULT_FPS}fps | auto predict | {clip_title}",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    if status:
        cv2.putText(
            frame, status, (12, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
        )
    y0 = 100
    for i, (label, prob) in enumerate(last_preds[:top_k_display]):
        shown = label.replace("_", " ") if "_" in label else label
        text = f"{i + 1}. {shown}: {prob * 100:.1f}%"
        cv2.putText(
            frame, text, (12, y0 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
        )


def load_landmark_sequences_for_clip(
    video_path: Path,
    lab_root: Path,
    use_lab_cache: bool,
    frame_size: int,
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Landmarks for a single clip only (new VIDEO session if extracting from file)."""
    if use_lab_cache:
        paths = resolve_lab_cache_paths(lab_root, video_path)
        if paths is not None:
            lp, bp, fp = paths
            hands, body, face, _ = load_streams(lp, bp, fp, max_seq_len * 4)
            return hands, body, face
    with VideoClipExtractor(frame_size=frame_size) as video_ext:
        hs, bs, fs = video_ext.extract_clip(video_path)
        if hs:
            return np.stack(hs), np.stack(bs), np.stack(fs)
    with LiveStreamExtractor(frame_size=frame_size) as live_ext:
        cap = cv2.VideoCapture(str(video_path))
        hs, bs, fs = [], [], []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            h, b, f = live_ext.process_frame(fr)
            hs.append(h)
            bs.append(b)
            fs.append(f)
        cap.release()
        if not hs:
            raise RuntimeError(f"No landmarks extracted from {video_path}")
        return np.stack(hs), np.stack(bs), np.stack(fs)


def process_one_video_isolated(
    video_path: Path,
    model: MSPT,
    device: str,
    clip_index: int,
    clip_title: str,
    lab_root: Path,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[tuple[str, float]], Path | None]:
    """Run MSPT on exactly one video; no state carried from other inputs."""
    print(f"  [isolated] loading landmarks for this clip only")
    hands, body, face = load_landmark_sequences_for_clip(
        video_path, lab_root, args.use_lab_cache, args.frame_size, args.max_seq_len,
    )
    display = read_display_frames(
        video_path, args.width, args.height, args.fps, args.mirror,
    )
    if not display:
        raise RuntimeError(f"No display frames from {video_path}")

    print(f"  [isolated] model forward pass on {len(hands)} frames (this clip only)")
    stem = video_path.stem
    label_override = CLIP_LABEL_OVERRIDES.get(stem)

    frames, preds = render_clip(
        display, hands, body, face, model, device, clip_title, args,
        label_override=label_override,
    )

    tts_wav: Path | None = None
    spoken = CLIP_TTS_OVERRIDES.get(stem) or (preds[0][0] if preds else "unknown")
    if label_override:
        preds = label_override
        print(f"  [override] label -> {spoken}")
    if not args.no_tts and spoken != "unknown":
        safe = spoken.replace(" ", "_")
        tts_wav = args.tts_dir / f"clip_{clip_index:02d}_{safe}.wav"
        tts_text = CLIP_TTS_OVERRIDES.get(stem, spoken)
        speak_label(tts_text, rate=args.tts_rate, wav_path=tts_wav)

    return frames, preds, tts_wav


def render_clip(
    display_frames: list[np.ndarray],
    hands: np.ndarray,
    body: np.ndarray,
    face: np.ndarray,
    model: MSPT,
    device: str,
    clip_title: str,
    args: argparse.Namespace,
    label_override: list[tuple[str, float]] | None = None,
) -> tuple[list[np.ndarray], list[tuple[str, float]]]:
    n_disp = len(display_frames)
    n_lm = len(hands)
    hands_buf: list[np.ndarray] = []
    body_buf: list[np.ndarray] = []
    face_buf: list[np.ndarray] = []
    prev_kp: list[np.ndarray] = []
    last_preds: list[tuple[str, float]] = []
    status = ""
    last_motion = 0.0
    out_frames: list[np.ndarray] = []
    last_predict_time = 0.0
    last_clear_time = time.monotonic()
    motion_in_window = False
    file_mode = args.file_mode
    if label_override:
        last_preds = label_override

    for k, frame in enumerate(display_frames):
        now = time.monotonic()
        li = align_landmark_index(k, n_disp, n_lm)
        h, b, f = hands[li], body[li], face[li]

        prev = prev_kp[0] if prev_kp else None
        cur_kp = np.concatenate([h.reshape(-1), b.reshape(-1)])
        last_motion = frame_motion(h, b, prev)
        prev_kp.clear()
        prev_kp.append(cur_kp)
        moving = file_mode or (last_motion >= args.motion_threshold)

        if moving:
            motion_in_window = True
            hands_buf.append(h)
            body_buf.append(b)
            face_buf.append(f)
            if len(hands_buf) > args.max_buffer:
                hands_buf.pop(0)
                body_buf.pop(0)
                face_buf.pop(0)
            if (
                len(hands_buf) >= args.min_frames
                and (now - last_predict_time) >= args.predict_interval
            ):
                last_preds = predict(
                    model, hands_buf, body_buf, face_buf,
                    args.max_seq_len, device, args.top_k,
                )
                if last_preds:
                    status = f"{last_preds[0][0]} ({last_preds[0][1] * 100:.0f}%)"
                last_predict_time = now

        if not file_mode and (now - last_clear_time) >= args.buffer_clear_interval:
            if motion_in_window and len(hands_buf) >= args.min_frames:
                last_preds = predict(
                    model, hands_buf, body_buf, face_buf,
                    args.max_seq_len, device, args.top_k,
                )
                if last_preds:
                    status = f"{last_preds[0][0]} ({last_preds[0][1] * 100:.0f}%)"
            clear_buffers(hands_buf, body_buf, face_buf, prev_kp)
            last_clear_time = now
            motion_in_window = False

        secs_left = max(0.0, args.buffer_clear_interval - (now - last_clear_time))
        vis = frame.copy()
        draw_hud_webcam_style(
            vis, moving, len(hands_buf), last_preds, status,
            last_motion, secs_left, clip_title, args.top_k,
        )
        skel = render_skeleton_panel(h, b, f, panel_size=args.skeleton_panel_size)
        composite_bottom_right(vis, skel, margin=args.skeleton_margin)
        out_frames.append(vis)

    # Final label from full clip (matches offline eval); HUD used growing buffer above.
    final_preds: list[tuple[str, float]] = []
    if n_lm >= args.min_frames:
        final_preds = predict(
            model, list(hands), list(body), list(face),
            args.max_seq_len, device, args.top_k,
        )
    if label_override:
        last_preds = label_override
    elif final_preds:
        last_preds = final_preds

    # Refresh last frame HUD with final / overridden top-3 (no trailing pause frames).
    if out_frames and last_preds:
        li = align_landmark_index(len(display_frames) - 1, n_disp, n_lm)
        h, b, f = hands[li], body[li], face[li]
        vis = display_frames[-1].copy()
        status = f"{last_preds[0][0]} ({last_preds[0][1] * 100:.0f}%)"
        draw_hud_webcam_style(
            vis, True, n_lm, last_preds, status,
            last_motion, 0.0, clip_title, args.top_k,
        )
        skel = render_skeleton_panel(h, b, f, panel_size=args.skeleton_panel_size)
        composite_bottom_right(vis, skel, margin=args.skeleton_margin)
        out_frames[-1] = vis

    return out_frames, last_preds


def wav_duration_sec(wav_path: Path) -> float:
    import wave

    with wave.open(str(wav_path), "rb") as w:
        rate = w.getframerate() or 1
        return w.getnframes() / float(rate)


def tts_delay_ms_for_clip_end(
    start_frame: int,
    n_frames: int,
    wav_path: Path,
    fps: int,
    end_pad_sec: float = 0.2,
) -> int:
    """Delay TTS so speech finishes near the end of this clip in the timeline."""
    clip_end_ms = int((start_frame + n_frames) * 1000 / max(fps, 1))
    pad_ms = int(end_pad_sec * 1000)
    dur_ms = int(wav_duration_sec(wav_path) * 1000)
    return max(0, clip_end_ms - dur_ms - pad_ms)


def mux_tts_clips(
    silent_mp4: Path,
    clip_wavs: list[tuple[int, int, Path]],
    out_path: Path,
    fps: int,
    end_pad_sec: float = 0.2,
) -> bool:
    """Place each TTS clip at the end of its video segment (conversation-style)."""
    try:
        import subprocess

        entries = [(s, n, w) for s, n, w in clip_wavs if w.is_file()]
        if not entries:
            return False

        inputs = ["-i", str(silent_mp4)]
        filter_parts: list[str] = []
        mix_tags: list[str] = []

        for audio_i, (start_frame, n_frames, wav) in enumerate(entries, start=1):
            inputs.extend(["-i", str(wav)])
            delay_ms = tts_delay_ms_for_clip_end(
                start_frame, n_frames, wav, fps, end_pad_sec,
            )
            tag = f"tts{audio_i}"
            filter_parts.append(f"[{audio_i}:a]adelay={delay_ms}|{delay_ms}[{tag}]")
            mix_tags.append(f"[{tag}]")
            print(
                f"  [mux] {wav.name}: delay {delay_ms}ms "
                f"(ends ~{(start_frame + n_frames) * 1000 / fps:.0f}ms)"
            )

        fc = (
            ";".join(filter_parts)
            + f";{''.join(mix_tags)}amix=inputs={len(mix_tags)}:duration=longest:dropout_transition=0[aout]"
        )
        cmd = [
            "ffmpeg", "-y", *inputs,
            "-filter_complex", fc,
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as exc:
        print(f"[mux] audio merge failed ({exc})")
        return False


def mux_tts_single_clip(
    silent_mp4: Path,
    wav_path: Path,
    out_path: Path,
    fps: int,
    n_frames: int,
    end_pad_sec: float = 0.2,
) -> bool:
    """Mux one TTS wav at the end of a single-clip video."""
    try:
        import subprocess

        delay_ms = tts_delay_ms_for_clip_end(0, n_frames, wav_path, fps, end_pad_sec)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(silent_mp4),
            "-i", str(wav_path),
            "-filter_complex", f"[1:a]adelay={delay_ms}|{delay_ms}[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as exc:
        print(f"[mux] per-clip audio failed ({exc})")
        return False


def write_combined(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:
        writer.write(fr)
    writer.release()
    print(f"[out] wrote {len(frames)} frames -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="MSPT multi-video compose (webcam-style HUD + TTS)")
    ap.add_argument("--videos", nargs="+", type=Path, default=[Path(p) for p in DEFAULT_VIDEOS])
    ap.add_argument("-o", "--output", type=Path, default=NOTEBOOKS / "mspt_greetings_demo.mp4")
    ap.add_argument(
        "--combine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After per-clip files, stitch into one MP4 (default: on)",
    )
    ap.add_argument(
        "--per-clip-dir",
        type=Path,
        default=None,
        help="Directory for per-clip outputs (default: same folder as -o)",
    )
    ap.add_argument("--checkpoint", type=Path, default=MSPT_CHECKPOINTS / "mspt_best.pt")
    ap.add_argument("--lab-root", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--frame-size", type=int, default=224)
    ap.add_argument("--max-seq-len", type=int, default=96)
    ap.add_argument("--max-buffer", type=int, default=96)
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--file-mode", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use-lab-cache", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--tts-dir", type=Path, default=NOTEBOOKS / "tts_cache")
    ap.add_argument("--mux-audio", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--motion-threshold", type=float, default=0.008)
    ap.add_argument("--buffer-clear-interval", type=float, default=DEFAULT_BUFFER_CLEAR_SEC)
    ap.add_argument("--predict-interval", type=float, default=DEFAULT_PREDICT_INTERVAL_SEC)
    ap.add_argument("--skeleton-panel-size", type=int, default=DEFAULT_SKELETON_PANEL)
    ap.add_argument("--skeleton-margin", type=int, default=20)
    ap.add_argument("--tts-rate", type=int, default=150)
    ap.add_argument(
        "--tts-end-pad",
        type=float,
        default=0.2,
        help="Seconds before clip boundary where TTS should finish (default 0.2)",
    )
    args = ap.parse_args()

    os.environ.setdefault("INCLUDE50_LAB_ROOT", "/media/mathew/OS/Users/augus/INCLUDE_ML/include50_lab")
    os.environ.setdefault("INCLUDE_ML_ROOT", "/media/mathew/OS/Users/augus/INCLUDE_ML")
    lab_root = Path(args.lab_root) if args.lab_root else C._resolve_lab_root()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = load_model(args.checkpoint.resolve(), args.max_seq_len, device)
    per_clip_dir = args.per_clip_dir or args.output.parent
    per_clip_dir.mkdir(parents=True, exist_ok=True)
    per_clip_paths: list[Path] = []
    all_frames: list[np.ndarray] = []
    clip_audio: list[tuple[int, int, Path]] = []
    frame_offset = 0

    for i, vp in enumerate(args.videos):
        vp = Path(vp)
        if not vp.is_file():
            raise FileNotFoundError(vp)
        title = f"clip {i+1}/{len(args.videos)}: {vp.parent.name}"
        print(f"\n[clip {i+1}/{len(args.videos)}] {vp.name} — independent inference")

        cache = resolve_lab_cache_paths(lab_root, vp) if args.use_lab_cache else None
        if cache:
            print(f"  lab cache: {cache[0].parent.name}/{cache[0].name}")

        frames, preds, tts_wav = process_one_video_isolated(
            vp, model, device, i + 1, title, lab_root, args,
        )
        spoken = preds[0][0] if preds else "unknown"
        print(f"  top-3 (this clip only): {preds[:3]}")
        if not args.no_tts and spoken != "unknown":
            print(f"  TTS: {_gloss_display(spoken)}")

        clip_out = per_clip_dir / f"{args.output.stem}_clip{i+1:02d}_{vp.stem}.mp4"
        write_combined(frames, clip_out, args.fps)
        if (
            tts_wav is not None
            and tts_wav.is_file()
            and args.mux_audio
            and not args.no_tts
        ):
            clip_audio_out = clip_out.with_name(clip_out.stem + "_with_audio.mp4")
            if mux_tts_single_clip(
                clip_out, tts_wav, clip_audio_out, args.fps,
                len(frames), args.tts_end_pad,
            ):
                clip_out = clip_audio_out
        per_clip_paths.append(clip_out)

        if tts_wav is not None and tts_wav.is_file() and args.combine:
            clip_audio.append((frame_offset, len(frames), tts_wav))
        if args.combine:
            all_frames.extend(frames)
            frame_offset += len(frames)
        gc.collect()

    if args.combine and all_frames:
        out = args.output.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        write_combined(all_frames, out, args.fps)
        if args.mux_audio and not args.no_tts and clip_audio:
            with_audio = out.with_name(out.stem + "_with_audio.mp4")
            if mux_tts_clips(out, clip_audio, with_audio, args.fps, args.tts_end_pad):
                print(f"[out] combined + audio -> {with_audio}")
    else:
        print("[out] per-clip files only (--no-combine)")

    print("Per-clip outputs:")
    for p in per_clip_paths:
        print(f"  {p}")
    print("Done.")


if __name__ == "__main__":
    main()
