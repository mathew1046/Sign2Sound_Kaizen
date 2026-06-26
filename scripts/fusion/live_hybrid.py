#!/usr/bin/env python3
"""Hybrid live fusion: MSPT + alphabet transformer + glove (confirm-only).

Usage:
  cd ~/Arrakis/Sign2Sound_Kaizen
  export PYTHONPATH=$PWD:$PWD/scripts/mspt
  adb forward tcp:8090 tcp:8080
  python scripts/fusion/live_hybrid.py \\
    --video-url http://localhost:8090/video \\
    --checkpoint checkpoints/mspt/mspt_rtmlib_263_best.pt

Keys: q quit | f flush composer | s cycle mode (auto / spell / word) | Space flush spell buffer
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_MSPT = REPO_ROOT / "scripts" / "mspt"
sys.path.insert(0, str(SCRIPTS_MSPT))
sys.path.insert(0, str(REPO_ROOT))

from fusion.alphabet_worker import AlphabetWorker  # noqa: E402
from fusion.mode_detector import ModeDetector, ModeDetectorConfig  # noqa: E402
from fusion.policy import FusionPolicy  # noqa: E402
from fusion.vocabulary import FusionVocabulary  # noqa: E402
from mspt.gloss_compose import GlossComposer, ensure_project_env, gemini_available  # noqa: E402
from mspt.segmentation import (  # noqa: E402
    FixedSegmenter,
    SegmentPhase,
    add_segmenter_args,
    make_segmenter,
    segmenter_config_from_args,
)
from repo_paths import MSPT_CHECKPOINTS, REPO_ROOT, RTMLIB_LAB  # noqa: E402

import rtmlib_live_mspt as mspt_live  # noqa: E402

DEFAULT_VIDEO_URL = mspt_live.DEFAULT_VIDEO_URL
DEFAULT_CKPT = MSPT_CHECKPOINTS / "mspt_rtmlib_263_best.pt"
DEFAULT_LAB = RTMLIB_LAB


def _apply_decision(
    decision,
    *,
    composer: GlossComposer | None,
    tts: mspt_live.TtsSpeaker | None,
    now: float,
) -> None:
    if decision.action == "accept_mspt" and composer is not None:
        if decision.meta.get("glove_agreement"):
            print(f"[fusion] MSPT {decision.gloss} (+ glove agree)")
        else:
            print(f"[fusion] MSPT {decision.gloss}")
        composer.add_gloss(decision.gloss, decision.confidence, now)
    elif decision.action == "accept_glove" and composer is not None:
        print(f"[fusion] glove fallback → {decision.gloss} ({decision.reason})")
        composer.add_gloss(decision.gloss, decision.confidence, now)
    elif decision.action == "append_letter":
        print(f"[fusion] spell +{decision.gloss} → {decision.meta.get('spell_buffer', '')}")
    elif decision.action == "flush_spell" and decision.gloss:
        print(f"[fusion] spell complete: {decision.gloss}")
        if composer is not None:
            composer.add_gloss(decision.gloss.lower(), 1.0, now)
        elif tts is not None:
            tts.speak_text(decision.gloss)


def draw_fusion_status(frame, policy: FusionPolicy, mode_debug: dict | None = None) -> None:
    h, _ = frame.shape[:2]
    y = int(h * 0.30)
    cv2.putText(
        frame,
        f"Mode: {policy.mode_label}",
        (16, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 180, 80) if policy.is_alphabet_mode else (80, 200, 255),
        2,
        cv2.LINE_AA,
    )
    if policy.spell_display:
        cv2.putText(
            frame,
            f"Spell: {policy.spell_display}",
            (16, y + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 220, 100),
            2,
            cv2.LINE_AA,
        )
    if mode_debug is not None:
        dbg = (
            f"h={mode_debug.get('hand_motion_ema', 0):.4f} "
            f"b={mode_debug.get('body_motion_ema', 0):.4f}"
        )
        cv2.putText(
            frame, dbg, (16, y + 64),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA,
        )


def draw_hand_crop_bbox(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> None:
    if bbox is None:
        return
    x0, y0, w, h = bbox
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (0, 255, 180), 2)
    cv2.putText(
        frame,
        "hand crop",
        (x0, max(20, y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 180),
        2,
        cv2.LINE_AA,
    )


def _hand_bbox(keypoints: np.ndarray, frame_w: int, frame_h: int, pad_frac: float = 0.15) -> tuple[int, int, int, int] | None:
    """Compute padded bbox for a single hand's keypoints (N, 2) in [0,1] coords."""
    valid = keypoints[np.any(keypoints != 0, axis=1)]
    if len(valid) < 5:
        return None
    xs = valid[:, 0] * frame_w
    ys = valid[:, 1] * frame_h
    x_min, x_max = int(np.min(xs)), int(np.max(xs))
    y_min, y_max = int(np.min(ys)), int(np.max(ys))
    w, h = x_max - x_min, y_max - y_min
    if w < 12 or h < 12:
        return None
    pad_w = int(w * pad_frac)
    pad_h = int(h * pad_frac)
    x0 = max(0, x_min - pad_w)
    y0 = max(0, y_min - pad_h)
    x1 = min(frame_w, x_max + pad_w)
    y1 = min(frame_h, y_max + pad_h)
    return (x0, y0, x1 - x0, y1 - y0)


def get_hand_bboxes_from_rtmlib(hands: np.ndarray, frame_w: int, frame_h: int, pad_frac: float = 0.15) -> list:
    """Return a list of individual hand bboxes from RTMLib keypoints.
    
    When hands are close together, returns a single union bbox.
    When hands are far apart (single-hand sign), returns separate bboxes.
    """
    lh = hands[:21]
    rh = hands[21:]
    
    lh_bbox = _hand_bbox(lh, frame_w, frame_h, pad_frac)
    rh_bbox = _hand_bbox(rh, frame_w, frame_h, pad_frac)
    
    if lh_bbox is None and rh_bbox is None:
        return []
    if lh_bbox is None:
        return [rh_bbox]
    if rh_bbox is None:
        return [lh_bbox]
    
    # Both hands present: check if they're far apart (single-hand sign)
    def _center(b):
        return (b[0] + b[2] // 2, b[1] + b[3] // 2)
    
    lx, ly = _center(lh_bbox)
    rx, ry = _center(rh_bbox)
    dist = ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5
    max_hand_size = max(lh_bbox[2], lh_bbox[3], rh_bbox[2], rh_bbox[3])
    
    if dist > max_hand_size * 3.0:
        return [lh_bbox, rh_bbox]
    
    # Close together → single union bbox
    x0 = min(lh_bbox[0], rh_bbox[0])
    y0 = min(lh_bbox[1], rh_bbox[1])
    x1 = max(lh_bbox[0] + lh_bbox[2], rh_bbox[0] + rh_bbox[2])
    y1 = max(lh_bbox[1] + lh_bbox[3], rh_bbox[1] + rh_bbox[3])
    return [(x0, y0, x1 - x0, y1 - y0)]


def run_live(args: argparse.Namespace) -> None:
    path = ensure_project_env()
    if path is not None:
        print(f"[hybrid] loaded env from {path}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[hybrid] CUDA unavailable for MSPT, using CPU")

    ckpt = args.checkpoint.resolve()
    lab_root = args.lab_root.resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    video_url = (args.video_url or args.stream_url).strip()
    print(f"[hybrid] stream: {video_url}")
    print(f"[hybrid] checkpoint: {ckpt}")

    tts: mspt_live.TtsSpeaker | None = None
    if not args.no_tts:
        tts = mspt_live.TtsSpeaker(rate=args.tts_rate, volume=args.tts_volume)

    use_gemini = not args.no_gemini and gemini_available()
    composer: GlossComposer | None = None
    if args.compose:

        def _on_composed(result) -> None:
            print(f"[compose] {result.source}: {result.speak}")
            if tts is not None and not args.no_tts and result.speak:
                tts.speak_text(result.speak)

        composer = GlossComposer(
            utterance_pause_sec=args.utterance_pause_sec,
            max_buffer=args.max_gloss_buffer,
            use_gemini=use_gemini,
            on_composed=_on_composed,
            speak_enabled=not args.no_tts,
        )

    policy = FusionPolicy(
        vocab=FusionVocabulary(),
        min_mspt_confidence=args.min_confidence,
        alphabet_threshold=args.alphabet_confidence,
        glove_fallback=args.glove_fallback,
        glove_fallback_silent_sec=args.glove_fallback_silent_sec,
        manual_mode="alphabet" if args.spell_mode else None,
    )

    mode_detector = ModeDetector(
        ModeDetectorConfig(
            hand_motion_min=args.mode_hand_min,
            body_spell_max=args.mode_body_spell_max,
            body_word_min=args.mode_body_word_min,
            hand_body_ratio_min=args.mode_hand_body_ratio,
            switch_frames=args.mode_switch_frames,
        )
    )

    glove = None
    if not args.no_glove:
        from fusion.glove_worker import GloveWorker  # noqa: WPS433

        glove = GloveWorker(
            host=args.glove_host,
            feed_port=args.glove_feed_port,
            connect_timeout_sec=args.glove_connect_timeout,
        )
        if not glove.start():
            print(f"[hybrid] glove feed unavailable ({glove.error}); continuing without glove")
            glove.close()
            glove = None
        else:
            print(f"[hybrid] glove enabled: feed {args.glove_host}:{args.glove_feed_port}")

    alphabet: AlphabetWorker | None = None
    if not args.no_alphabet:
        alphabet = AlphabetWorker(
            weights_path=Path(args.alphabet_weights),
            confidence_threshold=args.alphabet_confidence,
            use_crop=args.crop,
            crop_pad=args.crop_pad,
            hand_det_confidence=args.hand_det_confidence,
            device=args.alphabet_device,
        )
        if not alphabet.start():
            print(f"[hybrid] alphabet unavailable ({alphabet.error}); continuing without alphabet")
            alphabet.close()
            alphabet = None
        else:
            print("[hybrid] alphabet worker enabled")

    extractor = mspt_live.RtmlibWholebodyExtractor(
        device=args.rtmlib_device, det_interval=args.det_interval
    )
    worker = mspt_live.PoseWorker(extractor, args.pose_max_width, unmirror_pose=args.unmirror_pose)
    model, ckpt_obj, idx_to_label = mspt_live.load_model(ckpt, args.max_seq_len, device, lab_root)
    print(f"[hybrid] MSPT loaded: {len(idx_to_label)} classes")

    if args.cooldown_sec is not None:
        pass
    elif hasattr(args, "gap_sec"):
        args.cooldown_sec = args.gap_sec
    if args.conf_on is None:
        args.conf_on = args.min_confidence

    seg_cfg = segmenter_config_from_args(args)
    seg_cfg.device = device
    if args.segmenter_checkpoint is None:
        default_seg = REPO_ROOT / "checkpoints" / "mspt" / "sign_segmenter_best.pt"
        if default_seg.is_file():
            args.segmenter_checkpoint = default_seg
            seg_cfg.segmenter_checkpoint = default_seg

    def _predict_fn(buf: list[np.ndarray]) -> list[tuple[str, float]]:
        return mspt_live.predict_clip(
            model, buf, args.max_seq_len, device, idx_to_label, args.top_k,
            min_confidence=args.min_confidence,
        )

    segmenter = make_segmenter(args.segmenter, seg_cfg, predict_fn=_predict_fn)
    if isinstance(segmenter, FixedSegmenter):
        segmenter.start_session(time.monotonic())
    print(f"[hybrid] segmenter={args.segmenter}")

    source, mode = mspt_live.open_video_capture(video_url)
    grabber = mspt_live.FrameGrabber(source, mode)

    last_preds: list[tuple[str, float]] = []
    display_preds: list[tuple[str, float]] = []
    pred_visible_until = 0.0
    prev_kp: list = []
    last_motion = 0.0
    last_hands = np.zeros((42, 2), dtype=np.float32)
    last_body = np.zeros((33, 2), dtype=np.float32)
    last_face = np.zeros((68, 2), dtype=np.float32)
    cached_skel = None
    phase_elapsed_start = time.monotonic()
    extract_interval = 1.0 / max(args.fps, 1)
    display_interval = (1.0 / args.display_fps) if args.display_fps > 0 else 0.0
    next_extract_time = time.monotonic()
    next_display_time = time.monotonic()
    pose_fps_ema = 0.0
    last_pose_wall = time.monotonic()
    display_frames = 0
    display_t0 = time.monotonic()
    measured_display_fps = 0.0
    flush_key = ord(args.flush_key) if args.flush_key else ord("f")

    window_fullscreen_set = False

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
                print("[hybrid] stream ended")
                break
            if frame is None:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue

            if alphabet is not None:
                alphabet.submit_frame(frame)
                alphabet.set_enabled(policy.is_alphabet_mode or policy.alphabet_weight > 0.0)

            if glove is not None:
                for token in glove.poll():
                    decision = policy.on_glove(token)
                    _apply_decision(decision, composer=composer, tts=tts, now=now)

            if alphabet is not None:
                for token in alphabet.poll():
                    decision = policy.on_alphabet(token.gloss, token.confidence, token.timestamp)
                    _apply_decision(decision, composer=composer, tts=tts, now=token.timestamp)

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

                hands, body, face = mspt_live.wholebody_to_viz(wb)
                
                # Get hand bbox from active RTMLib keypoints and submit it to alphabet worker
                h_h, h_w = frame.shape[:2]
                hand_bboxes = get_hand_bboxes_from_rtmlib(hands, h_w, h_h, pad_frac=args.crop_pad)
                if alphabet is not None:
                    alphabet.submit_hand_bboxes(hand_bboxes)
                last_hands, last_body, last_face = hands, body, face
                detected_mode = mode_detector.update(hands, body)
                if policy.manual_mode is None:
                    policy.set_auto_mode(detected_mode)
                policy.set_alphabet_weight(mode_detector.alphabet_weight)
                cached_skel = mspt_live.render_skeleton_panel(
                    last_hands, last_body, last_face, panel_size=args.skeleton_panel_size,
                )

                prev = prev_kp[0] if prev_kp else None
                cur_kp = np.concatenate([hands.reshape(-1), body.reshape(-1)])
                last_motion = mspt_live.frame_motion(hands, body, prev)
                prev_kp.clear()
                prev_kp.append(cur_kp)

                events = segmenter.update(wb, last_motion, now)
                for ev in events:
                    if ev.kind == "end" and ev.buffer:
                        last_preds = mspt_live.predict_clip(
                            model, ev.buffer, args.max_seq_len, device, idx_to_label, args.top_k,
                            min_confidence=args.min_confidence,
                        )
                        if last_preds:
                            display_preds = last_preds
                            pred_visible_until = now + 1.5
                        if last_preds and last_preds[0][0] != "uncertain":
                            gloss, conf = last_preds[0]
                            decision = policy.on_mspt(gloss, conf, now)
                            _apply_decision(decision, composer=composer, tts=tts, now=now)
                        if isinstance(segmenter, FixedSegmenter):
                            phase_elapsed_start = now
                        mspt_live._end_clip([], worker)
                        prev_kp.clear()

            if composer is not None:
                composer.tick(now)

            # Auto-flush spell buffer if idle
            spell_timeout_decision = policy.check_spell_timeout(now)
            _apply_decision(spell_timeout_decision, composer=composer, tts=tts, now=now)

            display = frame.copy()
            display_phase = segmenter.display_phase
            elapsed_ui = now - phase_elapsed_start

            if isinstance(segmenter, FixedSegmenter):
                if segmenter.phase == SegmentPhase.HOLD:
                    if now - phase_elapsed_start >= args.hold_pred_sec:
                        segmenter.enter_gap(now)
                        phase_elapsed_start = now
                elif segmenter.phase == SegmentPhase.GAP:
                    if now - phase_elapsed_start >= args.gap_sec:
                        segmenter.enter_recording(now)
                        phase_elapsed_start = now
                        mspt_live._end_clip([], worker)

            mspt_live.draw_status_bar(
                display, display_phase, elapsed_ui, args.clip_sec, args.gap_sec, args.hold_pred_sec,
                segmenter.n_frames,
                segmenter.motion_frames, last_motion >= args.motion_threshold,
                pose_fps_ema, measured_display_fps, extractor.device,
                segmenter_mode=args.segmenter,
            )
            preds_visible = bool(display_preds) and now < pred_visible_until
            recording_ui = display_phase in ("recording", "signing")
            mspt_live.draw_predictions_large(
                display,
                display_preds if preds_visible else [],
                args.top_k,
                recording=recording_ui,
            )
            mspt_live.draw_composer_status(display, composer)
            draw_fusion_status(display, policy, mode_detector.debug)
            if policy.is_alphabet_mode and alphabet is not None:
                draw_hand_crop_bbox(display, alphabet.get_crop_bbox())
            if cached_skel is not None:
                mspt_live.composite_bottom_left(display, cached_skel, margin=args.skeleton_margin)
            if args.window_scale != 1.0 and args.window_scale > 0:
                display = cv2.resize(display, None, fx=args.window_scale, fy=args.window_scale,
                                     interpolation=cv2.INTER_LINEAR)
            cv2.imshow(args.window, display)
            if args.fullscreen and not window_fullscreen_set:
                cv2.setWindowProperty(args.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                window_fullscreen_set = True

            display_frames += 1
            if now - display_t0 >= 1.0:
                measured_display_fps = display_frames / (now - display_t0)
                display_frames = 0
                display_t0 = now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if composer is not None and key == flush_key:
                composer.flush()
            if key == ord("s"):
                label = policy.toggle_manual_mode()
                print(f"[hybrid] mode → {label}")
            if key == ord(" "):
                decision = policy.flush_spell_buffer()
                _apply_decision(decision, composer=composer, tts=tts, now=now)
    except KeyboardInterrupt:
        print("\n[hybrid] stopped")
    finally:
        if composer is not None:
            composer.close()
        if tts is not None:
            tts.close()
        if glove is not None:
            glove.close()
        if alphabet is not None:
            alphabet.close()
        grabber.stop()
        worker.close()
        extractor.close()
        if mode == "cv2":
            source.release()
        cv2.destroyAllWindows()


def main() -> int:
    ap = argparse.ArgumentParser(description="Hybrid MSPT + alphabet + glove fusion")
    ap.add_argument("--video-url", "--stream-url", default=DEFAULT_VIDEO_URL, dest="video_url")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--lab-root", type=Path, default=DEFAULT_LAB)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rtmlib-device", default=None)
    ap.add_argument("--pose-max-width", type=int, default=mspt_live.DEFAULT_POSE_MAX_WIDTH)
    ap.add_argument("--det-interval", type=int, default=5)
    ap.add_argument("--clip-sec", type=float, default=mspt_live.DEFAULT_CLIP_SEC)
    ap.add_argument("--gap-sec", type=float, default=mspt_live.DEFAULT_GAP_SEC)
    ap.add_argument("--hold-pred-sec", type=float, default=mspt_live.DEFAULT_HOLD_PRED_SEC)
    ap.add_argument("--fps", type=int, default=mspt_live.DEFAULT_FPS)
    ap.add_argument("--display-fps", type=int, default=mspt_live.DEFAULT_DISPLAY_FPS)
    ap.add_argument("--max-seq-len", type=int, default=96)
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--min-motion-frames", type=int, default=6)
    ap.add_argument("--motion-threshold", type=float, default=mspt_live.DEFAULT_MOTION_THRESHOLD)
    ap.add_argument("--min-confidence", type=float, default=mspt_live.DEFAULT_MIN_CONFIDENCE)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--window", default="Sign2Sound Hybrid")
    ap.add_argument("--fullscreen", action="store_true", help="Display window in fullscreen mode")
    ap.add_argument("--window-scale", type=float, default=1.0,
                    help="Scale the display frame (e.g. 1.5, 2.0) without changing capture resolution")
    ap.add_argument("--unmirror-pose", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--skeleton-panel-size", type=int, default=mspt_live.DEFAULT_SKELETON_PANEL)
    ap.add_argument("--skeleton-margin", type=int, default=20)
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--tts-rate", type=int, default=150)
    ap.add_argument("--tts-volume", type=float, default=1.0)
    ap.add_argument("--compose", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--utterance-pause-sec", type=float, default=mspt_live.DEFAULT_UTTERANCE_PAUSE_SEC)
    ap.add_argument("--max-gloss-buffer", type=int, default=mspt_live.DEFAULT_MAX_GLOSS_BUFFER)
    ap.add_argument("--no-gemini", action="store_true")
    ap.add_argument("--flush-key", default=mspt_live.DEFAULT_FLUSH_KEY)

    ap.add_argument("--no-glove", action="store_true")
    ap.add_argument("--glove-host", type=str, default="10.43.206.118",
                    help="GloveTalk prediction feed host (default 10.43.206.118)")
    ap.add_argument("--glove-feed-port", type=int, default=8081,
                    help="TCP port for GloveTalk prediction feed (default 8081)")
    ap.add_argument("--glove-connect-timeout", type=float, default=15.0)
    ap.add_argument("--glove-fallback", action="store_true",
                    help="Allow glove-only overlap words when MSPT silent")
    ap.add_argument("--glove-fallback-silent-sec", type=float, default=3.0)

    ap.add_argument("--no-alphabet", action="store_true")
    ap.add_argument("--alphabet-weights", type=str,
                    default=str(REPO_ROOT / "alphabet_transformer" / "weights" / "sign_transformer_alphabet.pth"))
    ap.add_argument("--alphabet-confidence", type=float, default=0.85)
    ap.add_argument("--alphabet-device", default="cpu",
                    help="PyTorch device for alphabet model (cpu avoids GPU OOM with MSPT)")
    ap.add_argument("--spell-mode", action="store_true", help="Start in manual spell mode")
    ap.add_argument("--crop", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--crop-pad", type=float, default=0.15)
    ap.add_argument("--hand-det-confidence", type=float, default=0.2)

    ap.add_argument("--mode-hand-min", type=float, default=0.003,
                    help="Min hand motion EMA to consider alphabet mode")
    ap.add_argument("--mode-body-spell-max", type=float, default=0.008,
                    help="Max body motion EMA for alphabet mode")
    ap.add_argument("--mode-body-word-min", type=float, default=0.014,
                    help="Body motion EMA that forces word mode")
    ap.add_argument("--mode-hand-body-ratio", type=float, default=2.5,
                    help="Hand/body motion ratio favoring alphabet mode")
    ap.add_argument("--mode-switch-frames", type=int, default=6,
                    help="Consecutive frames before mode switches")

    add_segmenter_args(ap)
    run_live(ap.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
