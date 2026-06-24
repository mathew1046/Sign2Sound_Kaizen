"""INCLUDE-50 original corpus browsing — manifests, video serve, skeleton frames."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from collection_dashboard.config import (
    INCLUDE50_LAB_ROOT,
    INCLUDE50_VIDEO_MANIFEST_ROOT,
    INCLUDE50_VIDEO_ROOT,
    PROJECT_ROOT,
    TRANSCODE_CACHE_DIR,
)

MSPT_SCRIPTS = PROJECT_ROOT / "scripts" / "mspt"
if str(MSPT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MSPT_SCRIPTS))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import slr_common as C  # noqa: E402
from mspt.rtmlib_skeleton_viz import render_rtmlib_skeleton_panel  # noqa: E402
from mspt.skeleton_viz import render_skeleton_panel  # noqa: E402

CORPUS_TRANSCODE_DIR = TRANSCODE_CACHE_DIR.parent / "_include50_corpus"
EVALS_DIR = PROJECT_ROOT / "collection_dashboard" / "evals"
MODAL_MANIFEST_ROOT = PROJECT_ROOT / "modal" / "data"
_VIDEO_SUFFIXES = (".mov", ".mp4", ".avi", ".mkv", ".webm")


@dataclass(frozen=True)
class ClipRecord:
    word: str
    stem: str
    label_id: int
    split: str
    source_path: Path

    @property
    def clip_id(self) -> str:
        return f"{self.word}/{self.stem}"


class Include50Corpus:
    def __init__(
        self,
        lab_root: Path | None = None,
        video_manifest_root: Path | None = None,
        video_root: Path | None = None,
    ):
        self.lab_root = Path(lab_root or INCLUDE50_LAB_ROOT)
        self.video_manifest_root = Path(video_manifest_root or INCLUDE50_VIDEO_MANIFEST_ROOT)
        self.video_root = Path(video_root or INCLUDE50_VIDEO_ROOT)
        self.cache_dir = self.lab_root / "cache"
        self.wholebody_dir = self.cache_dir / "wholebody"
        self.landmarks_dir = self.cache_dir / "landmarks"
        self.body_dir = self.cache_dir / "mspt_body"
        self.face_dir = self.cache_dir / "landmarks_face"
        self.skeleton_backend = (
            "rtmlib" if self.wholebody_dir.is_dir() else "mediapipe"
        )
        self._clips: list[ClipRecord] | None = None
        self._video_by_word_stem: dict[tuple[str, str], Path] | None = None
        self._video_by_stem: dict[str, Path] | None = None
        self._stem_video_index: dict[str, Path] | None = None
        self._lab_manifest_paths: dict[tuple[str, str], str] | None = None

    def _load_lab_manifest_paths(self) -> dict[tuple[str, str], str]:
        if self._lab_manifest_paths is not None:
            return self._lab_manifest_paths
        paths: dict[tuple[str, str], str] = {}
        man = self.lab_root / "manifest.csv"
        if man.is_file():
            with man.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    word = row["label"].strip()
                    stem = row.get("stem", "").strip() or Path(row["path"]).stem
                    paths[(word, stem)] = row["path"].strip()
        self._lab_manifest_paths = paths
        return paths

    def _resolve_vol_input_path(self, manifest_path: str) -> Path | None:
        """Map Modal ``/vol/input/...`` paths to local INCLUDE-50 tree."""
        p = Path(manifest_path)
        if not manifest_path.startswith("/vol/input/"):
            return None
        rel = manifest_path[len("/vol/input/") :].lstrip("/")
        stem = p.stem

        candidates = [
            self.video_root / rel,
        ]
        # e.g. Places/35. Bank/file -> Places_4of4/Places/35. Bank/file
        parts = rel.split("/", 1)
        if len(parts) == 2:
            category, rest = parts
            candidates.extend(
                self.video_root / f"{category}_{suffix_part}" / category / rest
                for suffix_part in ("1of2", "2of2", "3of3", "4of4", "5of5", "6of6", "7of7", "8of8")
            )
            candidates.append(self.video_root / category / rest)

        for c in candidates:
            if c.is_file():
                return c

        stem_hit = self._build_stem_video_index().get(stem)
        if stem_hit is not None and stem_hit.is_file():
            return stem_hit
        return None

    def _manifest_dirs(self) -> list[Path]:
        dirs: list[Path] = []
        for root in (self.video_manifest_root, MODAL_MANIFEST_ROOT):
            manifest_dir = root / "manifests"
            if manifest_dir.is_dir() and manifest_dir not in dirs:
                dirs.append(manifest_dir)
        return dirs

    def _build_video_index(self) -> None:
        if self._video_by_word_stem is not None:
            return
        by_word_stem: dict[tuple[str, str], Path] = {}
        by_stem: dict[str, Path] = {}
        for manifest_dir in self._manifest_dirs():
            for csv_path in sorted(manifest_dir.glob("*.csv")):
                if csv_path.name == "all.csv":
                    continue
                with csv_path.open(newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        word = row["label"].strip()
                        manifest_path = row["path"].strip()
                        if Path(manifest_path).suffix.lower() not in _VIDEO_SUFFIXES:
                            continue
                        stem = row.get("stem", "").strip() or Path(manifest_path).stem
                        video = C.resolve_video_path(manifest_path)
                        by_word_stem[(word, stem)] = video
                        by_stem.setdefault(stem, video)
        self._video_by_word_stem = by_word_stem
        self._video_by_stem = by_stem

    def _build_stem_video_index(self) -> dict[str, Path]:
        if self._stem_video_index is not None:
            return self._stem_video_index
        idx: dict[str, Path] = {}
        root = self.video_root
        if root.is_dir():
            for pattern in ("MVI_*.MOV", "MVI_*.mov", "MVI_*.mp4", "MVI_*.MP4"):
                for p in root.rglob(pattern):
                    if p.is_file():
                        idx.setdefault(p.stem, p)
        self._stem_video_index = idx
        return idx

    def _resolve_video_path(self, word: str, stem: str, manifest_path: str) -> Path:
        path = Path(manifest_path)
        if path.suffix.lower() in _VIDEO_SUFFIXES and path.is_file():
            return C.resolve_video_path(manifest_path)

        lab_path = self._load_lab_manifest_paths().get((word, stem))
        if lab_path:
            vol_hit = self._resolve_vol_input_path(lab_path)
            if vol_hit is not None:
                return vol_hit
            resolved = C.resolve_video_path(lab_path)
            if resolved.is_file():
                return resolved

        self._build_video_index()
        assert self._video_by_word_stem is not None
        assert self._video_by_stem is not None

        for key in ((word, stem),):
            hit = self._video_by_word_stem.get(key)
            if hit is not None and hit.is_file():
                return hit

        hit = self._video_by_stem.get(stem)
        if hit is not None and hit.is_file():
            return hit

        stem_hit = self._build_stem_video_index().get(stem)
        if stem_hit is not None and stem_hit.is_file():
            return stem_hit

        meta_p = self.cache_dir / "meta" / word / f"{stem}.json"
        if meta_p.is_file():
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                meta_stem = Path(str(meta.get("video", ""))).stem
                if meta_stem:
                    meta_hit = self._build_stem_video_index().get(meta_stem)
                    if meta_hit is not None and meta_hit.is_file():
                        return meta_hit
            except (json.JSONDecodeError, OSError):
                pass

        return path

    def _load_clips(self) -> list[ClipRecord]:
        if self._clips is not None:
            return self._clips
        clips: list[ClipRecord] = []
        manifest_dir = self.lab_root / "manifests"
        for csv_path in sorted(manifest_dir.glob("*.csv")):
            if csv_path.name == "all.csv":
                continue
            split = csv_path.stem
            with csv_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    word = row["label"].strip()
                    manifest_path = row["path"].strip()
                    stem = row.get("stem", "").strip() or Path(manifest_path).stem
                    src = self._resolve_video_path(word, stem, manifest_path)
                    clips.append(
                        ClipRecord(
                            word=word,
                            stem=stem,
                            label_id=int(row.get("label_id", 0)),
                            split=row.get("split", split),
                            source_path=src,
                        )
                    )
        self._clips = clips
        return clips

    def list_words(self) -> list[str]:
        words = sorted({c.word for c in self._load_clips()})
        return words

    def corpus_summary(self) -> dict:
        clips = self._load_clips()
        with_video = sum(1 for c in clips if c.source_path.is_file())
        with_skel = sum(
            1 for c in clips if self._primary_cache_path(c.word, c.stem).is_file()
        )
        by_word: dict[str, int] = {}
        for c in clips:
            by_word[c.word] = by_word.get(c.word, 0) + 1
        return {
            "num_words": len(by_word),
            "num_clips": len(clips),
            "clips_with_video": with_video,
            "clips_with_skeleton": with_skel,
            "skeleton_backend": self.skeleton_backend,
            "lab_root": str(self.lab_root),
            "video_root": str(self.video_root),
        }

    def corpus_glosses(self) -> list[dict]:
        from collection_dashboard.config import load_corpus_vocabulary

        counts: dict[str, int] = {}
        label_ids: dict[str, int] = {}
        for c in self._load_clips():
            counts[c.word] = counts.get(c.word, 0) + 1
            label_ids.setdefault(c.word, c.label_id)
        vocab = {v["word"]: v for v in load_corpus_vocabulary()}
        glosses: list[dict] = []
        for word in sorted(counts, key=lambda w: label_ids.get(w, 999)):
            v = vocab.get(word, {})
            glosses.append(
                {
                    "label_id": label_ids.get(word, v.get("label_id", -1)),
                    "word": word,
                    "display_name": v.get("display_name", word.replace("_", " ").title()),
                    "clip_count": counts[word],
                    "in_include50": v.get("in_include50", label_ids.get(word, 999) < 50),
                }
            )
        glosses.sort(key=lambda g: g["label_id"])
        return glosses

    def clips_for_word(self, word: str) -> list[ClipRecord]:
        return [c for c in self._load_clips() if c.word == word]

    def get_clip(self, word: str, stem: str) -> ClipRecord | None:
        for c in self._load_clips():
            if c.word == word and c.stem == stem:
                return c
        return None

    def _primary_cache_path(self, word: str, stem: str) -> Path:
        if self.skeleton_backend == "rtmlib":
            return self.wholebody_dir / word / f"{stem}.npy"
        return self.landmarks_dir / word / f"{stem}.npy"

    def _cache_paths(self, word: str, stem: str) -> tuple[Path, Path, Path]:
        return (
            self._primary_cache_path(word, stem),
            self.body_dir / word / f"{stem}.npy",
            self.face_dir / word / f"{stem}.npy",
        )

    def clip_meta(self, word: str, stem: str) -> dict:
        clip = self.get_clip(word, stem)
        if clip is None:
            raise KeyError(f"Unknown clip: {word}/{stem}")
        lm_p, body_p, face_p = self._cache_paths(word, stem)
        frame_count = 0
        fps = 30.0
        if lm_p.is_file():
            arr = np.load(lm_p, mmap_mode="r")
            frame_count = int(len(arr))
            del arr
        src = clip.source_path
        has_video = src.is_file()
        if has_video:
            cap = cv2.VideoCapture(str(src))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                if frame_count == 0:
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                cap.release()
        return {
            "word": word,
            "stem": stem,
            "label_id": clip.label_id,
            "split": clip.split,
            "source": str(src),
            "video_url": f"/api/include50/clips/{word}/{stem}/video",
            "frame_count": frame_count,
            "fps": float(fps),
            "has_landmarks": lm_p.is_file(),
            "has_body": body_p.is_file(),
            "has_face": face_p.is_file(),
            "has_video": has_video,
            "skeleton_backend": self.skeleton_backend,
            "keypoint_root": str(self.lab_root),
            "video_root": str(self.video_root),
        }

    def playable_video(self, word: str, stem: str) -> Path:
        clip = self.get_clip(word, stem)
        if clip is None:
            raise KeyError(f"Unknown clip: {word}/{stem}")
        src = clip.source_path
        if not src.is_file():
            raise FileNotFoundError(
                f"RGB video missing for {word}/{stem}. "
                f"Expected under {self.video_root} (mounted INCLUDE-50). "
                f"Set INCLUDE50_VIDEO_ROOT and INCLUDE_ML_ROOT."
            )
        if src.suffix.lower() == ".mp4" and src.is_file():
            from collection_dashboard.server.transcode import is_browser_playable

            if is_browser_playable(src):
                return src
        from collection_dashboard.server.transcode import ensure_mp4

        CORPUS_TRANSCODE_DIR.mkdir(parents=True, exist_ok=True)
        out_dir = CORPUS_TRANSCODE_DIR / word
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{stem}.mp4"
        if out.is_file() and out.stat().st_mtime >= src.stat().st_mtime:
            from collection_dashboard.server.transcode import is_browser_playable

            if is_browser_playable(out):
                return out
        result = ensure_mp4(src)
        if result.suffix.lower() == ".mp4" and result.is_file():
            from collection_dashboard.server.transcode import is_browser_playable

            if is_browser_playable(result):
                return result
        if out.is_file():
            return out
        raise FileNotFoundError(f"Could not prepare browser video for {word}/{stem}")

    @lru_cache(maxsize=256)
    def _load_rtmlib_sequence(self, word: str, stem: str) -> np.ndarray | None:
        wb_p = self.wholebody_dir / word / f"{stem}.npy"
        if not wb_p.is_file():
            return None
        return np.load(wb_p, mmap_mode="r")

    @lru_cache(maxsize=256)
    def _load_mediapipe_sequence(
        self, word: str, stem: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        lm_p, body_p, face_p = self._cache_paths(word, stem)
        if not lm_p.is_file():
            return None
        lm = np.load(lm_p, mmap_mode="r")
        hands = np.array(lm[:, 12:54, :2], dtype=np.float32)
        t = len(hands)
        if body_p.is_file():
            body = np.array(np.load(body_p, mmap_mode="r")[:, :33, :2], dtype=np.float32)
        else:
            body = np.zeros((t, 33, 2), dtype=np.float32)
            body[:, :12] = np.array(lm[:, :12, :2], dtype=np.float32)
        if face_p.is_file():
            face = np.array(np.load(face_p, mmap_mode="r")[..., :2], dtype=np.float32)
        else:
            face = np.zeros((t, 72, 2), dtype=np.float32)
        del lm
        return hands, body, face

    def skeleton_png(self, word: str, stem: str, frame_idx: int, panel_size: int = 480) -> bytes:
        if self.skeleton_backend == "rtmlib":
            seq = self._load_rtmlib_sequence(word, stem)
            if seq is None:
                raise FileNotFoundError(f"No wholebody cache for {word}/{stem}")
            fi = max(0, min(frame_idx, len(seq) - 1))
            panel = render_rtmlib_skeleton_panel(
                seq[fi], panel_size=panel_size,
            )
        else:
            seq = self._load_mediapipe_sequence(word, stem)
            if seq is None:
                raise FileNotFoundError(f"No landmark cache for {word}/{stem}")
            hands, body, face = seq
            fi = max(0, min(frame_idx, len(hands) - 1))
            panel = render_skeleton_panel(
                hands[fi], body[fi], face[fi], panel_size=panel_size,
            )
        ok, buf = cv2.imencode(".png", panel)
        if not ok:
            raise RuntimeError("Failed to encode skeleton PNG")
        return buf.tobytes()

    def eval_analysis(self) -> dict | None:
        path = EVALS_DIR / "eval_analysis.json"
        if not path.is_file():
            return None
        import json

        return json.loads(path.read_text(encoding="utf-8"))

    def confusion_matrix_png(self) -> Path | None:
        for name in ("confusion_matrix_all.png", "confusion_matrix_test.png"):
            p = EVALS_DIR / name
            if p.is_file():
                return p
        return None


corpus = Include50Corpus()
