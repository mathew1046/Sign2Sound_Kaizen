"""FastAPI backend — unified ISL dashboard (learn, collect, explore)."""

from __future__ import annotations

import hashlib
import io
import json
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dashboard.config import (
    ASSETS_DIR,
    AUTO_START,
    CAMERA_ENABLED,
    CATALOG_PATH,
    COLLECTION_OUTPUT_DIR,
    COMPOSE_CACHE_DIR,
    CONSENT_PATH,
    COOLDOWN_SEC,
    FPS,
    FRAME_SIZE,
    FRONTEND_DIST,
    REF_PLAY_COUNTDOWN_SEC,
    SAMPLES_PER_WORD,
    VOCAB_VERSION,
    WHOLEBODY_DIR,
    load_vocabulary,
)
from dashboard.label_utils import ensure_label_map
from dashboard.server.catalog import get_gloss_entry, load_catalog, vocab_list
from dashboard.server.collection.engine import CollectionEngine
from dashboard.server.collection.manifest import ManifestStore
from dashboard.server.collection.references import ReferenceVideoResolver, extract_reference_zip
from dashboard.server.collection.transcode import is_browser_playable, reencode_for_browser
from dashboard.server.collection.webcam import WebcamCapture
from dashboard.server.compose import (
    compose_glosses,
    frames_to_timeline_payload,
    render_wholebody_frame,
    resolve_gloss_entries,
)
from dashboard.server.corpus import corpus
from dashboard.server.translate import translate_sentence
from dashboard.server.orientation.compare import compare_sequences
from dashboard.server.orientation.extract import ALLOWED_SUFFIXES, get_video_extractor
from dashboard.server.orientation.features import sequence_from_wholebody
from dashboard.server.orientation.feedback import feedback_with_gemma
from dashboard.server.orientation.progress import OrientationProgressStore
from dashboard.server.orientation.reference import get_orientation_reference, reference_meta
from dashboard.server.orientation.schemas import AnalyzeResponse

manifest = ManifestStore()
webcam = WebcamCapture()
refs = ReferenceVideoResolver()
engine = CollectionEngine(manifest, webcam, refs)
orientation_progress = OrientationProgressStore()


class TranslateRequest(BaseModel):
    sentence: str = Field(..., min_length=1)
    use_gemini: bool = True


class ComposeRequest(BaseModel):
    glosses: list[str] = Field(..., min_length=1)
    exemplar_ids: list[str] | None = None
    encode_frames: bool = True


class WordTimingUpdate(BaseModel):
    cooldown_sec: float | None = Field(default=None, ge=0, le=30)
    ref_countdown_sec: float | None = Field(default=None, ge=0, le=30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from dashboard.server.catalog import ensure_catalog

    ensure_catalog()
    ensure_label_map()
    corpus.refresh_video_index()

    manifest.load()
    extract_reference_zip()
    webcam.set_enabled(CAMERA_ENABLED)
    if not CAMERA_ENABLED:
        webcam.stop()

    if AUTO_START and CAMERA_ENABLED:
        webcam.ensure_reader()
        if webcam.ok:
            engine.start()
        else:
            manifest.update_engine(
                state="error",
                phase="error",
                message=webcam.error or "Camera unavailable",
            )
    else:
        manifest.update_engine(
            state="idle",
            phase="idle",
            message="Open Collect and click Start collection to begin",
        )
    summary = corpus.corpus_summary()
    print(
        f"[dashboard] lab={corpus.lab_root} backend={corpus.skeleton_backend} "
        f"words={summary['num_words']} clips={summary['num_clips']} "
        f"catalog={CATALOG_PATH.name}"
    )
    yield
    engine.stop()
    webcam.stop()


app = FastAPI(title="Sign2Sound ISL Dashboard", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _wholebody_path(gloss: str, exemplar_id: str) -> Path:
    return WHOLEBODY_DIR / gloss / f"{exemplar_id}.npy"


def _asset_frame_path(gloss: str, exemplar_id: str, frame_idx: int) -> Path:
    return ASSETS_DIR / gloss / exemplar_id / f"{frame_idx:05d}.png"


def _load_frame(gloss: str, exemplar_id: str, frame_idx: int) -> np.ndarray:
    ap = _asset_frame_path(gloss, exemplar_id, frame_idx)
    if ap.exists():
        img = cv2.imread(str(ap))
        if img is not None:
            return img
    wb_path = _wholebody_path(gloss, exemplar_id)
    if not wb_path.exists():
        raise FileNotFoundError(wb_path)
    seq = np.load(wb_path, mmap_mode="r")
    if frame_idx < 0 or frame_idx >= seq.shape[0]:
        raise IndexError(frame_idx)
    return render_wholebody_frame(seq[frame_idx])


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health():
    snap = manifest.snapshot()
    catalog_ok = CATALOG_PATH.exists()
    return {
        "status": "ok" if (not webcam.enabled or webcam.ok) else "degraded",
        "vocab_version": VOCAB_VERSION,
        "catalog": catalog_ok,
        "wholebody_dir": str(WHOLEBODY_DIR),
        "camera_enabled": webcam.enabled,
        "camera_ok": webcam.ok if webcam.enabled else False,
        "camera_error": webcam.error,
        "engine": snap["engine"],
        "output_dir": str(COLLECTION_OUTPUT_DIR),
        "corpus": corpus.corpus_summary(),
    }


# ---------------------------------------------------------------------------
# Learn / skeleton browser
# ---------------------------------------------------------------------------


@app.get("/api/vocab")
def api_vocab():
    catalog = load_catalog()
    return {
        "vocab_version": catalog.get("vocab_version", VOCAB_VERSION),
        "num_glosses": catalog["num_glosses"],
        "glosses_with_data": catalog["glosses_with_data"],
        "glosses": vocab_list(catalog),
    }


from dashboard.ml_utils import ml_display_name

@app.get("/api/signs/{gloss}")
def api_sign_detail(gloss: str):
    catalog = load_catalog()
    entry = get_gloss_entry(catalog, gloss)
    if not entry:
        raise HTTPException(404, f"Unknown gloss: {gloss}")
    entry["display_name_ml"] = ml_display_name(gloss)
    return entry


@app.get("/api/signs/{gloss}/{exemplar_id}/meta")
def api_sign_meta(gloss: str, exemplar_id: str):
    wb_path = _wholebody_path(gloss, exemplar_id)
    if not wb_path.exists():
        raise HTTPException(404, "Exemplar not found")
    seq = np.load(wb_path, mmap_mode="r")
    asset_dir = ASSETS_DIR / gloss / exemplar_id
    use_assets = asset_dir.is_dir() and any(asset_dir.glob("*.png"))
    return {
        "gloss": gloss,
        "exemplar_id": exemplar_id,
        "num_frames": int(seq.shape[0]),
        "fps": FPS,
        "frame_size": FRAME_SIZE,
        "assets_exported": use_assets,
    }


@app.get("/api/signs/{gloss}/{exemplar_id}/frames/{frame_idx}")
def api_sign_frame(gloss: str, exemplar_id: str, frame_idx: int):
    ap = _asset_frame_path(gloss, exemplar_id, frame_idx)
    if ap.exists():
        return FileResponse(ap, media_type="image/png")
    try:
        frame = _load_frame(gloss, exemplar_id, frame_idx)
    except FileNotFoundError:
        raise HTTPException(404, "Exemplar not found")
    except IndexError:
        raise HTTPException(404, "Frame index out of range")
    _, buf = cv2.imencode(".png", frame)
    return Response(content=buf.tobytes(), media_type="image/png")


@app.post("/api/translate")
def api_translate(body: TranslateRequest):
    result = translate_sentence(body.sentence, use_gemini=body.use_gemini)
    if not result["glosses"] and result.get("unknown"):
        raise HTTPException(
            422,
            detail={
                "message": "Could not map sentence to known glosses",
                "unknown": result["unknown"],
            },
        )
    return result


@app.post("/api/compose")
def api_compose(body: ComposeRequest):
    catalog = load_catalog()
    try:
        if body.exemplar_ids and len(body.exemplar_ids) == len(body.glosses):
            entries = list(zip(body.glosses, body.exemplar_ids))
        else:
            entries = resolve_gloss_entries(body.glosses, catalog)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    cache_key = hashlib.sha256(
        json.dumps({"g": body.glosses, "e": [e[1] for e in entries]}, sort_keys=True).encode()
    ).hexdigest()[:16]
    cache_path = COMPOSE_CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists() and body.encode_frames:
        return JSONResponse(json.loads(cache_path.read_text(encoding="utf-8")))

    frames, segments = compose_glosses(entries)
    payload = frames_to_timeline_payload(frames, segments, encode_b64=body.encode_frames)
    payload["glosses"] = body.glosses
    payload["exemplars"] = [{"gloss": g, "exemplar_id": e} for g, e in entries]

    if body.encode_frames:
        COMPOSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


@app.post("/api/sentence")
def api_sentence(body: TranslateRequest):
    tr = translate_sentence(body.sentence, use_gemini=body.use_gemini)
    if not tr["glosses"]:
        raise HTTPException(422, detail={"translate": tr, "message": "No glosses produced"})
    catalog = load_catalog()
    try:
        entries = resolve_gloss_entries(tr["glosses"], catalog)
    except ValueError as exc:
        raise HTTPException(422, detail={"translate": tr, "message": str(exc)})
    frames, segments = compose_glosses(entries)
    timeline = frames_to_timeline_payload(frames, segments, encode_b64=True)
    return {"translate": tr, "timeline": timeline}


# ---------------------------------------------------------------------------
# Orientation coach (Learn tab — Practice signs)
# ---------------------------------------------------------------------------


@app.get("/api/orientation/vocab")
def api_orientation_vocab():
    return api_vocab()


@app.get("/api/orientation/reference/{gloss}")
def api_orientation_reference(gloss: str):
    catalog = load_catalog()
    entry = get_gloss_entry(catalog, gloss)
    if not entry:
        raise HTTPException(404, f"Unknown gloss: {gloss}")
    meta = reference_meta(gloss, entry.get("display_name"))
    if meta is None:
        raise HTTPException(
            503,
            "Orientation references not built. Run: python scripts/build_orientation_references.py",
        )
    return meta.model_dump()


@app.get("/api/orientation/progress/{gloss}")
def api_orientation_progress(gloss: str):
    catalog = load_catalog()
    entry = get_gloss_entry(catalog, gloss)
    if not entry:
        raise HTTPException(404, f"Unknown gloss: {gloss}")
    return orientation_progress.get_progress(gloss).model_dump()


@app.post("/api/orientation/analyze")
async def api_orientation_analyze(
    gloss: str = Form(...),
    video: UploadFile = File(...),
    use_gemma: bool = Form(True),
):
    catalog = load_catalog()
    entry = get_gloss_entry(catalog, gloss)
    if not entry:
        raise HTTPException(404, f"Unknown gloss: {gloss}")

    reference = get_orientation_reference(gloss)
    if reference is None:
        raise HTTPException(
            503,
            "Orientation references not built. Run: python scripts/build_orientation_references.py",
        )

    suffix = Path(video.filename or "upload.mp4").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported video format. Use: {', '.join(sorted(ALLOWED_SUFFIXES))}")

    data = await video.read()
    if len(data) < 1024:
        raise HTTPException(400, "Video file too small or empty")

    try:
        extractor = get_video_extractor()
        wholebody, fps = extractor.extract_from_bytes(data, suffix=suffix)
    except Exception as exc:
        raise HTTPException(422, f"Could not extract pose from video: {exc}") from exc

    ref_hand = reference.get("active_hand")
    user_features, _ = sequence_from_wholebody(wholebody, fps=fps, active_hand=ref_hand)
    comparison = compare_sequences(gloss, user_features, reference, fps=fps)
    display_name = entry.get("display_name", gloss.replace("_", " ").title())
    display_name_ml = ml_display_name(gloss)
    feedback_text = feedback_with_gemma(display_name, comparison, use_api=use_gemma)
    progress = orientation_progress.log_attempt(
        gloss,
        comparison.overall_result,
        len(comparison.errors),
        feedback_text,
    )

    return AnalyzeResponse(
        comparison=comparison,
        feedback_text=feedback_text,
        progress=progress,
        display_name=display_name,
        display_name_ml=display_name_ml,
    ).model_dump()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


@app.get("/api/collection/vocab")
def collection_vocab():
    snap = manifest.snapshot()
    glosses = []
    for v in load_vocabulary():
        word = v["word"]
        entry = snap["words"].get(word, {})
        glosses.append(
            {
                **v,
                "completed_count": entry.get("completed_count", 0),
                "samples_per_word": SAMPLES_PER_WORD,
                "is_complete": entry.get("completed_count", 0) >= SAMPLES_PER_WORD,
                **manifest.get_word_timing(word),
            }
        )
    eng = snap["engine"]
    return {
        "num_glosses": len(glosses),
        "glosses": glosses,
        "total_completed": eng.get("total_completed", 0),
        "total_target": eng.get("total_target", len(glosses) * SAMPLES_PER_WORD),
        "default_cooldown_sec": COOLDOWN_SEC,
        "default_ref_countdown_sec": REF_PLAY_COUNTDOWN_SEC,
    }


@app.get("/api/camera/status")
def camera_status():
    return {
        "enabled": webcam.enabled,
        "active": webcam.ok and webcam.enabled,
        "error": webcam.error,
    }


@app.post("/api/camera/enable")
def camera_enable():
    webcam.set_enabled(True)
    ok = webcam.ensure_reader()
    return {"ok": ok, "enabled": True, "active": ok, "error": webcam.error}


@app.post("/api/camera/disable")
def camera_disable():
    was_running = engine.is_running()
    if was_running:
        engine.stop()
    webcam.set_enabled(False)
    manifest.update_engine(
        state="idle",
        phase="idle",
        paused=False,
        current_word=None,
        message="Camera off — open Collect and click Start collection to begin",
    )
    return {"ok": True, "enabled": False, "active": False, "stopped_engine": was_running}


@app.get("/api/engine/status")
def engine_status():
    return manifest.snapshot()["engine"]


@app.post("/api/engine/start")
def engine_start():
    """Enable collection — requires camera to be on."""
    if not webcam.enabled:
        raise HTTPException(400, "Turn on the camera first")
    if not webcam.ensure_reader():
        raise HTTPException(503, webcam.error or "Camera not available")

    import threading

    def _warm_transcodes() -> None:
        refs.warm_all_playable()

    threading.Thread(target=_warm_transcodes, daemon=True, name="ref-warm").start()

    if not engine.is_running():
        engine.start()
    manifest.update_engine(
        state="collecting",
        phase="reference",
        paused=False,
        message="Collection started — watch the reference, then sign",
    )
    return {"ok": True, "running": True}


@app.post("/api/engine/stop")
def engine_stop():
    """Stop collection and return to idle."""
    was_running = engine.is_running()
    engine.stop()
    manifest.update_engine(
        state="idle",
        phase="idle",
        paused=False,
        current_word=None,
        message="Collection stopped",
    )
    return {"ok": True, "was_running": was_running}


@app.post("/api/engine/pause")
def engine_pause():
    engine.pause()
    return {"ok": True, "paused": True}


@app.post("/api/engine/resume")
def engine_resume():
    engine.resume()
    if not engine.is_running():
        engine.start()
    return {"ok": True, "paused": False}


@app.post("/api/data/reset")
def reset_data():
    was_running = engine.is_running()
    engine.stop()
    manifest.reset_all()
    if was_running and webcam.enabled and webcam.ok:
        engine.start()
    return {"ok": True, "message": "All collected data has been reset"}


# ---------------------------------------------------------------------------
# Data collection consent
# ---------------------------------------------------------------------------


class ConsentRecord(BaseModel):
    agreed: bool = True
    consent_version: int = 1


@app.post("/api/consent")
def record_consent(body: ConsentRecord):
    """Record the user's data collection consent."""
    if not body.agreed:
        if CONSENT_PATH.exists():
            CONSENT_PATH.unlink()
        return {"ok": True, "consented": False}
    record = {
        "consented": True,
        "consent_version": body.consent_version,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "rgb_not_public": True,
        "skeleton_may_be_public": True,
    }
    CONSENT_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {"ok": True, "consented": True}


@app.get("/api/consent")
def consent_status():
    if CONSENT_PATH.exists():
        record = json.loads(CONSENT_PATH.read_text(encoding="utf-8"))
        return {"ok": True, "consented": record.get("consented", False), "record": record}
    return {"ok": True, "consented": False, "record": None}


@app.delete("/api/consent")
def withdraw_consent():
    if CONSENT_PATH.exists():
        CONSENT_PATH.unlink()
    return {"ok": True, "consented": False}


@app.get("/api/stream/live.mjpg")
def live_stream():
    if not webcam.enabled:
        raise HTTPException(503, "Camera disabled")
    if not webcam.ensure_reader():
        raise HTTPException(503, webcam.error or "Camera not available")

    def generate():
        for chunk in webcam.mjpeg_stream():
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stream/snapshot.jpg")
def live_snapshot():
    if not webcam.enabled:
        raise HTTPException(503, "Camera disabled")
    if not webcam.ensure_reader():
        raise HTTPException(503, webcam.error or "Camera not available")
    jpeg = webcam.wait_for_jpeg(timeout=2.0)
    if not jpeg:
        raise HTTPException(503, "No camera frame available yet")
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/api/words/{word}/timing")
def get_word_timing(word: str):
    snap = manifest.snapshot()
    if word not in snap["words"]:
        raise HTTPException(404, f"Unknown word: {word}")
    entry = snap["words"][word]
    timing = manifest.get_word_timing(word)
    return {
        "word": word,
        "display_name": entry.get("display_name", word.replace("_", " ")),
        "display_name_ml": ml_display_name(word),
        **timing,
        "default_cooldown_sec": COOLDOWN_SEC,
        "default_ref_countdown_sec": REF_PLAY_COUNTDOWN_SEC,
    }


@app.patch("/api/words/{word}/timing")
def patch_word_timing(word: str, body: WordTimingUpdate):
    snap = manifest.snapshot()
    if word not in snap["words"]:
        raise HTTPException(404, f"Unknown word: {word}")
    try:
        timing = manifest.set_word_timing(
            word,
            cooldown_sec=body.cooldown_sec,
            ref_countdown_sec=body.ref_countdown_sec,
        )
    except KeyError:
        raise HTTPException(404, f"Unknown word: {word}")
    entry = manifest.snapshot()["words"][word]
    return {
        "ok": True,
        "word": word,
        "display_name": entry.get("display_name", word.replace("_", " ")),
        "display_name_ml": ml_display_name(word),
        **timing,
    }


@app.get("/api/references/{word}")
def list_references(word: str):
    snap = manifest.snapshot()
    if word not in snap["words"]:
        raise HTTPException(404, f"Unknown word: {word}")
    playable = refs.resolve_playable(word)
    return {
        "word": word,
        "references": [
            {"index": i, "source": str(p), "url": f"/api/references/{word}/{i}"}
            for i, p in enumerate(playable)
        ],
    }


@app.get("/api/references/{word}/{idx}")
def serve_reference(word: str, idx: int):
    playable = refs.resolve_playable(word)
    if idx < 0 or idx >= len(playable):
        raise HTTPException(404, "Reference not found")
    path = playable[idx]
    if not path.exists():
        raise HTTPException(404, "Reference file missing")
    if path.suffix.lower() != ".mp4" or not is_browser_playable(path):
        raise HTTPException(503, "Reference video is still being prepared — retry in a few seconds")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=path.name,
        headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/collected/{word}/{slot}")
def serve_collected(word: str, slot: int):
    if slot < 0 or slot >= SAMPLES_PER_WORD:
        raise HTTPException(404, "Invalid slot")
    path = manifest.slot_path(word, slot)
    if not path.exists():
        raise HTTPException(404, "Clip not found")
    reencode_for_browser(path)
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=path.name,
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/collected/{word}")
def collected_word(word: str):
    snap = manifest.snapshot()
    if word not in snap["words"]:
        raise HTTPException(404, f"Unknown word: {word}")
    entry = snap["words"][word]
    slots = []
    for slot in entry["slots"]:
        idx = slot["index"]
        info = dict(slot)
        fpath = manifest.slot_path(word, idx)
        if slot.get("status") == "complete" and slot.get("file") and fpath.exists():
            info["url"] = f"/api/collected/{word}/{idx}"
        else:
            if slot.get("status") == "complete" and not fpath.exists():
                manifest.clear_slot(word, idx)
                info["status"] = "empty"
                info["file"] = None
            info["url"] = None
        slots.append(info)
    ref_urls = [
        {"index": i, "url": f"/api/references/{word}/{i}"}
        for i in range(len(refs.resolve_playable(word)))
    ]
    return {
        "word": word,
        "label_id": entry["label_id"],
        "display_name": entry.get("display_name", word.replace("_", " ")),
        "display_name_ml": ml_display_name(word),
        "completed_count": entry.get("completed_count", 0),
        "slots": slots,
        "references": ref_urls,
    }


@app.delete("/api/collected/{word}/{slot}")
def delete_collected(word: str, slot: int):
    snap = manifest.snapshot()
    if word not in snap["words"]:
        raise HTTPException(404, f"Unknown word: {word}")
    if slot < 0 or slot >= SAMPLES_PER_WORD:
        raise HTTPException(404, "Invalid slot")
    path = manifest.slot_path(word, slot)
    if path.exists():
        path.unlink()
    manifest.clear_slot(word, slot)
    return {"ok": True}


@app.post("/api/collected/{word}/{slot}/rerecord")
def rerecord_collected(word: str, slot: int):
    snap = manifest.snapshot()
    if word not in snap["words"]:
        raise HTTPException(404, f"Unknown word: {word}")
    if slot < 0 or slot >= SAMPLES_PER_WORD:
        raise HTTPException(404, "Invalid slot")
    path = manifest.slot_path(word, slot)
    if path.exists():
        path.unlink()
    manifest.mark_pending_rerecord(word, slot)
    engine.queue_rerecord(word, slot)
    return {"ok": True, "queued": True}


@app.get("/api/export/manifest.csv")
def export_manifest_csv():
    import csv

    snap = manifest.snapshot()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["path", "label", "label_id", "split"])
    for word, entry in snap["words"].items():
        for slot in entry["slots"]:
            if slot.get("status") != "complete" or not slot.get("file"):
                continue
            path = manifest.slot_path(word, slot["index"])
            writer.writerow([str(path.resolve()), word, entry["label_id"], "custom_train"])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=collected_manifest.csv"},
    )


@app.get("/api/overview")
def overview():
    snap = manifest.snapshot()
    eng = snap["engine"]
    complete_words = sum(
        1 for w in snap["words"].values() if w.get("completed_count", 0) >= SAMPLES_PER_WORD
    )
    incomplete = [
        {
            "word": word,
            "display_name": entry.get("display_name", word),
            "display_name_ml": ml_display_name(word),
            "completed_count": entry.get("completed_count", 0),
            "remaining": SAMPLES_PER_WORD - entry.get("completed_count", 0),
        }
        for word, entry in snap["words"].items()
        if entry.get("completed_count", 0) < SAMPLES_PER_WORD
    ]
    return {
        "total_words": len(snap["words"]),
        "complete_words": complete_words,
        "total_completed": eng.get("total_completed", 0),
        "total_target": eng.get("total_target", 500),
        "engine_state": eng.get("state"),
        "incomplete_words": incomplete,
    }


# ---------------------------------------------------------------------------
# Corpus explore
# ---------------------------------------------------------------------------


@app.get("/api/include50/corpus/summary")
def include50_corpus_summary():
    return corpus.corpus_summary()


@app.get("/api/include50/corpus/vocab")
def include50_corpus_vocab():
    return {
        "glosses": corpus.corpus_glosses(),
        "summary": corpus.corpus_summary(),
    }


@app.get("/api/include50/eval")
def include50_eval():
    analysis = corpus.eval_analysis()
    cm = corpus.confusion_matrix_png()
    return {
        "ready": analysis is not None and cm is not None,
        "analysis": analysis,
        "confusion_matrix_url": "/api/include50/eval/confusion-matrix.png" if cm else None,
    }


@app.get("/api/include50/eval/confusion-matrix.png")
def include50_confusion_matrix():
    path = corpus.confusion_matrix_png()
    if path is None:
        raise HTTPException(404, "Confusion matrix not generated")
    return FileResponse(path, media_type="image/png")


@app.get("/api/include50/{word}/clips")
def include50_clips(word: str):
    if word not in corpus.list_words():
        raise HTTPException(404, f"Unknown word: {word}")
    clips = []
    for c in corpus.clips_for_word(word):
        clips.append(corpus.clip_meta(c.word, c.stem))
    clips.sort(key=lambda x: (x["split"], x["stem"]))
    return {"word": word, "clips": clips, "count": len(clips)}


@app.get("/api/include50/clips/{word}/{stem}/video")
def include50_clip_video(word: str, stem: str):
    try:
        path = corpus.playable_video(word, stem)
    except KeyError:
        raise HTTPException(404, "Clip not found")
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    if not path.is_file():
        raise HTTPException(404, f"Video file missing: {path}")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=path.name,
        headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/include50/clips/{word}/{stem}/meta")
def include50_clip_meta(word: str, stem: str):
    try:
        return corpus.clip_meta(word, stem)
    except KeyError:
        raise HTTPException(404, "Clip not found")


@app.get("/api/include50/clips/{word}/{stem}/skeleton/{frame_idx}.png")
def include50_skeleton_frame(word: str, stem: str, frame_idx: int):
    try:
        data = corpus.skeleton_png(word, stem, frame_idx)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))
    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=300"},
    )


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

if FRONTEND_DIST.is_dir():
    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        icon = FRONTEND_DIST / "favicon.ico"
        if icon.is_file():
            return FileResponse(icon)
        return Response(status_code=204)

    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")


def run():
    import uvicorn

    from dashboard.config import SERVER_HOST, SERVER_PORT

    uvicorn.run(
        "dashboard.server.main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=False,
    )


if __name__ == "__main__":
    run()
