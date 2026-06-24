import { useCallback, useEffect, useRef, useState } from "react";
import {
  getInclude50Eval,
  getInclude50Clips,
  include50ClipUrl,
  include50SkeletonUrl,
  type CorpusGlossInfo,
  type CorpusSummary,
  type Include50Clip,
  type Include50Eval,
} from "./api";

type Props = {
  glosses: CorpusGlossInfo[];
  summary: CorpusSummary | null;
  selected: CorpusGlossInfo | null;
  onSelect: (g: CorpusGlossInfo) => void;
};

export default function ExploreView({ glosses, summary, selected, onSelect }: Props) {
  const [evalData, setEvalData] = useState<Include50Eval | null>(null);
  const [clips, setClips] = useState<Include50Clip[]>([]);
  const [activeClip, setActiveClip] = useState<Include50Clip | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const skelRef = useRef<HTMLImageElement>(null);

  const word = selected?.word ?? glosses[0]?.word ?? null;

  useEffect(() => {
    getInclude50Eval()
      .then(setEvalData)
      .catch((e) => setError(String(e)));
  }, []);

  const loadClips = useCallback(async () => {
    if (!word) return;
    setLoading(true);
    setError(null);
    try {
      const res = await getInclude50Clips(word);
      setClips(res.clips);
      setActiveClip((prev) => {
        if (prev && prev.word === word) {
          const still = res.clips.find((c) => c.stem === prev.stem);
          if (still) return still;
        }
        return res.clips[0] ?? null;
      });
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [word]);

  useEffect(() => {
    loadClips();
  }, [loadClips]);

  const syncSkeleton = useCallback(() => {
    const video = videoRef.current;
    const skel = skelRef.current;
    if (!video || !skel || !activeClip || !activeClip.has_landmarks || !activeClip.has_video) return;
    const fps = activeClip.fps > 0 ? activeClip.fps : 30;
    const frameIdx = Math.min(
      Math.max(0, Math.floor(video.currentTime * fps)),
      Math.max(0, activeClip.frame_count - 1)
    );
    skel.src = `${include50SkeletonUrl(activeClip.word, activeClip.stem, frameIdx)}?t=${frameIdx}`;
  }, [activeClip]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const onTime = () => syncSkeleton();
    video.addEventListener("timeupdate", onTime);
    video.addEventListener("seeked", onTime);
    return () => {
      video.removeEventListener("timeupdate", onTime);
      video.removeEventListener("seeked", onTime);
    };
  }, [syncSkeleton, activeClip]);

  useEffect(() => {
    syncSkeleton();
  }, [activeClip, syncSkeleton]);

  const analysis = evalData?.analysis;
  const videoOk = clips.filter((c) => c.has_video).length;
  const skelOk = clips.filter((c) => c.has_landmarks).length;

  return (
    <div className="explore-panel">
      <section className="explore-eval card">
        <h2>
          Model diagnostics
          {analysis?.checkpoint && (
            <span className="muted eval-ckpt">
              {" "}
              — {analysis.checkpoint.split("/").pop()}
            </span>
          )}
        </h2>
        <p className="muted eval-split">
          {analysis?.split === "all"
            ? `Full corpus — ${summary?.num_words ?? 263} glosses, ${summary?.num_clips ?? 4258} clips`
            : `${analysis?.split ?? "test"} split`}
          {analysis?.n_clips ? ` · eval on ${analysis.n_clips} clips` : ""}
          {summary
            ? ` · ${summary.clips_with_video} videos · ${summary.clips_with_skeleton} skeletons`
            : ""}
        </p>
        {!evalData?.ready && (
          <p className="muted">
            Run{" "}
            <code>python notebooks/eval_confusion_matrix.py --rtmlib --split all</code> to
            generate the confusion matrix.
          </p>
        )}
        {analysis && (
          <div className="eval-grid">
            <div className="eval-stats">
              <p>
                <strong>Accuracy:</strong>{" "}
                {((analysis.accuracy ?? analysis.test_accuracy) * 100).toFixed(2)}%
                {analysis.n_clips ? ` (${analysis.n_clips} clips)` : ""}
              </p>
              {analysis.best_classes[0] && (
                <p>
                  <strong>Best class:</strong>{" "}
                  {analysis.best_classes[0].display_name} (
                  {(analysis.best_classes[0].accuracy * 100).toFixed(1)}%)
                </p>
              )}
              {analysis.worst_classes[0] && (
                <p>
                  <strong>Worst class:</strong>{" "}
                  {analysis.worst_classes[0].display_name} (
                  {(analysis.worst_classes[0].accuracy * 100).toFixed(1)}%)
                </p>
              )}
              <h3>Worst classes</h3>
              <ul className="check-list">
                {analysis.worst_classes.slice(0, 8).map((c) => (
                  <li key={c.word}>
                    <button
                      type="button"
                      className="linkish"
                      onClick={() => {
                        const g = glosses.find((x) => x.word === c.word);
                        if (g) onSelect(g);
                      }}
                    >
                      {c.display_name}
                    </button>
                    <span className="muted">
                      {" "}
                      — {(c.accuracy * 100).toFixed(1)}%
                      {c.top_confusions?.[0]
                        ? ` · confused with ${c.top_confusions[0].word.replace(/_/g, " ")} (${c.top_confusions[0].count}×)`
                        : ""}
                    </span>
                  </li>
                ))}
              </ul>
              <h3>Check skeleton quality for these classes</h3>
              <ul className="check-list">
                {analysis.check_skeleton_classes.map((c) => (
                  <li key={c.label_id}>
                    <button
                      type="button"
                      className="linkish"
                      onClick={() => {
                        const g = glosses.find((x) => x.word === c.word);
                        if (g) onSelect(g);
                      }}
                    >
                      {c.word.replace(/_/g, " ")}
                    </button>
                    <span className="muted"> — {c.reason}</span>
                  </li>
                ))}
              </ul>
            </div>
            {evalData.confusion_matrix_url && (
              <a
                href={evalData.confusion_matrix_url}
                target="_blank"
                rel="noreferrer"
                className="cm-thumb"
              >
                <img src={evalData.confusion_matrix_url} alt="Confusion matrix" />
                <span>Open full PNG</span>
              </a>
            )}
          </div>
        )}
      </section>

      <div className="explore-toolbar">
        <label>
          Class{" "}
          <select
            value={word ?? ""}
            onChange={(e) => {
              const g = glosses.find((x) => x.word === e.target.value);
              if (g) onSelect(g);
            }}
          >
            {glosses.map((g) => (
              <option key={g.word} value={g.word}>
                {g.display_name}
                {!g.in_include50 ? " (INCLUDE-263)" : ""}
              </option>
            ))}
          </select>
        </label>
        <span className="muted">
          {clips.length} clips · {videoOk} with RGB · {skelOk} with skeleton
        </span>
      </div>

      {error && <p className="error">{error}</p>}
      {loading && <p className="muted">Loading clips…</p>}

      {activeClip && (
        <div className="side-by-side-viewer card">
          <h3>
            {activeClip.stem}{" "}
            <span className="badge">{activeClip.split}</span>
          </h3>
          <div className="side-by-side-row">
            <div className="side-pane">
              <div className="pane-label">RGB video</div>
              {activeClip.has_video ? (
                <video
                  ref={videoRef}
                  key={activeClip.video_url}
                  src={include50ClipUrl(activeClip.word, activeClip.stem)}
                  controls
                  playsInline
                  className="explore-video"
                />
              ) : (
                <p className="muted">
                  RGB video not found. Mount INCLUDE videos at{" "}
                  <code>INCLUDE50_VIDEO_ROOT</code> (default:{" "}
                  <code>$INCLUDE_ML_ROOT/include-50</code>).
                </p>
              )}
            </div>
            <div className="side-pane">
              <div className="pane-label">
                {activeClip.skeleton_backend === "rtmlib"
                  ? "Skeleton (rtmlib COCO-WholeBody)"
                  : "Skeleton (from cache)"}
              </div>
              {activeClip.has_landmarks ? (
                <img
                  ref={skelRef}
                  alt="Skeleton"
                  className="explore-skeleton"
                />
              ) : (
                <p className="muted">No keypoint cache for this clip.</p>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="clip-grid card">
        <h3>All clips in class</h3>
        <div className="clip-tiles">
          {clips.map((c) => (
            <button
              key={c.stem}
              type="button"
              className={`clip-tile ${activeClip?.stem === c.stem ? "active" : ""}`}
              onClick={() => setActiveClip(c)}
            >
              <span className="clip-stem">{c.stem}</span>
              <span className="badge">{c.split}</span>
              <span className="clip-meta">
                {c.frame_count} frames ·{" "}
                {c.has_landmarks ? "skeleton ok" : "no skeleton"} ·{" "}
                {c.has_video ? "video ok" : "no video"}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
