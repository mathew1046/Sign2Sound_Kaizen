"""Video transcoding and probing for browser-compatible playback."""

from __future__ import annotations

import json
import subprocess
import threading
from functools import lru_cache
from pathlib import Path

from dashboard.config import REF_MAX_PLAY_SEC, TRANSCODE_CACHE_DIR

_lock_guard = threading.Lock()
_path_locks: dict[str, threading.Lock] = {}


def _run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)


def _path_lock(key: str) -> threading.Lock:
    with _lock_guard:
        if key not in _path_locks:
            _path_locks[key] = threading.Lock()
        return _path_locks[key]


def probe_duration_sec(path: Path) -> float:
    """Duration via ffprobe (no decode — safe while files are being streamed)."""
    if not path.exists():
        return REF_MAX_PLAY_SEC
    try:
        out = _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            timeout=30,
        )
        data = json.loads(out.stdout)
        duration = float(data.get("format", {}).get("duration", 0) or 0)
        if duration <= 0:
            return REF_MAX_PLAY_SEC
        return max(0.5, min(duration, REF_MAX_PLAY_SEC * 4))
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError):
        return REF_MAX_PLAY_SEC


def is_browser_h264(path: Path) -> bool:
    """True if file is H.264 + yuv420p in an MP4 container."""
    if not path.exists() or path.suffix.lower() != ".mp4":
        return False
    try:
        out = _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt",
                "-of",
                "json",
                str(path),
            ],
            timeout=30,
        )
        streams = json.loads(out.stdout).get("streams", [])
        if not streams:
            return False
        s = streams[0]
        return s.get("codec_name") == "h264" and s.get("pix_fmt") == "yuv420p"
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError):
        return False


def is_decodable_mp4(path: Path) -> bool:
    """Verify at least one video frame decodes (catches truncated/corrupt encodes)."""
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def is_browser_playable(path: Path) -> bool:
    return is_browser_h264(path) and is_decodable_mp4(path)


def ensure_mp4(source: Path) -> Path:
    """Return path to mp4 playable in browser; transcode if needed."""
    source = source.resolve()
    if not source.exists():
        return source
    if source.suffix.lower() == ".mp4" and is_browser_playable(source):
        return source

    cache_dir = TRANSCODE_CACHE_DIR / source.parent.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{source.stem}.mp4"

    with _path_lock(str(out)):
        if out.exists():
            if is_browser_playable(out):
                src_mtime = source.stat().st_mtime
                if out.stat().st_mtime >= src_mtime:
                    return out
            out.unlink(missing_ok=True)

        if _ffmpeg_transcode(source, out):
            return out
    return source


def reencode_for_browser(source: Path) -> Path:
    """Re-encode any video (e.g. OpenCV mp4v) to browser-safe H.264."""
    source = source.resolve()
    if not source.exists():
        return source
    if is_browser_playable(source):
        return source

    with _path_lock(str(source)):
        if is_browser_playable(source):
            return source
        tmp = source.with_name(f"{source.stem}.browser.tmp.mp4")
        if _ffmpeg_transcode(source, tmp):
            tmp.replace(source)
        elif tmp.exists():
            tmp.unlink(missing_ok=True)
    return source


def _ffmpeg_transcode(source: Path, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.stem}.transcode.tmp.mp4")
    if tmp.exists():
        tmp.unlink(missing_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    try:
        _run(cmd, timeout=180)
        if not tmp.exists() or tmp.stat().st_size < 1024 or not is_browser_playable(tmp):
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(out)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        tmp.unlink(missing_ok=True)
        return False


@lru_cache(maxsize=512)
def video_duration_sec(path: str) -> float:
    """Cached duration lookup (string key for lru_cache)."""
    return probe_duration_sec(Path(path))
