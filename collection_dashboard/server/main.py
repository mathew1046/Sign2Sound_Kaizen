"""FastAPI backend for INCLUDE-50 data collection dashboard."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from collection_dashboard.config import (
    AUTO_START,
    COLLECTION_OUTPUT_DIR,
    COOLDOWN_SEC,
    CAMERA_ENABLED,
    FRONTEND_DIST,
    REF_PLAY_COUNTDOWN_SEC,
    SAMPLES_PER_WORD,
    load_vocabulary,
)
from collection_dashboard.server.engine import CollectionEngine
from collection_dashboard.server.manifest import ManifestStore
from collection_dashboard.server.include50_corpus import corpus
from collection_dashboard.server.references import ReferenceVideoResolver, extract_reference_zip
from collection_dashboard.server.transcode import is_browser_playable, reencode_for_browser
from collection_dashboard.server.webcam import WebcamCapture

manifest = ManifestStore()
webcam = WebcamCapture()
refs = ReferenceVideoResolver()
engine = CollectionEngine(manifest, webcam, refs)


class WordTimingUpdate(BaseModel):
    cooldown_sec: float | None = Field(default=None, ge=0, le=30)
    ref_countdown_sec: float | None = Field(default=None, ge=0, le=30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    manifest.load()
    extract_reference_zip()
    if CAMERA_ENABLED:
        webcam.ensure_reader()
    else:
        webcam.set_enabled(False)
    # Pre-transcode first batch of references so playback works immediately
    import threading

    def _warm_transcodes() -> None:
        refs.warm_all_playable()

    threading.Thread(target=_warm_transcodes, daemon=True, name="ref-warm").start()
    if AUTO_START and webcam.enabled and webcam.ok:
        engine.start()
    elif AUTO_START and webcam.enabled:
        manifest.update_engine(
            state="error",
            phase="error",
            message=webcam.error or "Camera unavailable — connect webcam and restart",
        )
    elif AUTO_START:
        manifest.update_engine(
            state="idle",
            phase="idle",
            message="Camera disabled — enable camera to collect data",
        )
    summary = corpus.corpus_summary()
    print(
        f"[corpus] keypoints={corpus.lab_root} "
        f"rgb={corpus.video_root} backend={corpus.skeleton_backend} "
        f"words={summary['num_words']} clips={summary['num_clips']}"
    )
    yield
    engine.stop()
    webcam.stop()


app = FastAPI(title="INCLUDE-50 Data Collection", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    snap = manifest.snapshot()
    return {
        "status": "ok" if (not webcam.enabled or webcam.ok) else "error",
        "camera_enabled": webcam.enabled,
        "camera_ok": webcam.ok if webcam.enabled else False,
        "camera_error": webcam.error,
        "engine": snap["engine"],
        "output_dir": str(COLLECTION_OUTPUT_DIR),
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
        engine.pause()
    webcam.set_enabled(False)
    manifest.update_engine(
        message="Camera disabled — enable camera to collect data",
    )
    return {"ok": True, "enabled": False, "active": False, "paused_engine": was_running}


@app.get("/api/vocab")
def api_vocab():
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


@app.get("/api/engine/status")
def engine_status():
    return manifest.snapshot()["engine"]


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
    """Delete all collected videos and restart collection from scratch."""
    was_running = engine.is_running()
    engine.stop()
    manifest.reset_all()
    if was_running and webcam.enabled and webcam.ok:
        engine.start()
    return {"ok": True, "message": "All collected data has been reset"}


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
    """Single-frame JPEG — polled by the UI for reliable live preview."""
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
    import io

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
    from fastapi.responses import Response

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=collected_manifest.csv"},
    )


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
        raise HTTPException(404, "Confusion matrix not generated — run notebooks/eval_confusion_matrix.py")
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

    from collection_dashboard.config import SERVER_HOST, SERVER_PORT

    uvicorn.run(
        "collection_dashboard.server.main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=False,
    )


if __name__ == "__main__":
    run()
