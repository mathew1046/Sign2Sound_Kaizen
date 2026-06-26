import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import SkeletonPlayer from "./SkeletonPlayer";
import {
  analyzeOrientation,
  getOrientationProgress,
  getSign,
  type AnalyzeResponse,
  type LearnGlossInfo,
  type OrientationProgress,
} from "./api";

const SPEEDS = [0.5, 0.75, 1, 1.25] as const;
const COUNTDOWN_START = 3;
const RECORD_MS = 2000;

type CapturePhase = "idle" | "countdown" | "recording" | "analyzing";

type Props = {
  glosses: LearnGlossInfo[];
  selected: LearnGlossInfo | null;
  onSelect: (g: LearnGlossInfo) => void;
  filter: string;
  onFilterChange: (v: string) => void;
  vocabMeta: { total: number; withData: number };
};

export default function OrientationCoachPanel({
  glosses,
  selected,
  onSelect,
  filter,
  onFilterChange,
  vocabMeta,
}: Props) {
  const [exemplarId, setExemplarId] = useState<string | null>(null);
  const [numFrames, setNumFrames] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [videoBlob, setVideoBlob] = useState<Blob | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [capturePhase, setCapturePhase] = useState<CapturePhase>("idle");
  const [countdown, setCountdown] = useState<number | null>(null);
  const [useGemma, setUseGemma] = useState(true);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [progress, setProgress] = useState<OrientationProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const liveVideoRef = useRef<HTMLVideoElement | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const countdownTimerRef = useRef<number | null>(null);
  const recordTimerRef = useRef<number | null>(null);
  const selectedRef = useRef(selected);
  const useGemmaRef = useRef(useGemma);

  selectedRef.current = selected;
  useGemmaRef.current = useGemma;

  const loadSign = useCallback(async (g: LearnGlossInfo) => {
    if (!g.default_exemplar_id) {
      setExemplarId(null);
      setNumFrames(0);
      return;
    }
    const detail = await getSign(g.gloss);
    const ex = detail.default_exemplar_id ?? g.default_exemplar_id;
    const variant = detail.variants.find((v) => v.exemplar_id === ex) ?? detail.variants[0];
    setExemplarId(ex);
    setNumFrames(variant?.num_frames ?? 0);
  }, []);

  const clearTimers = () => {
    if (countdownTimerRef.current !== null) {
      window.clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = null;
    }
    if (recordTimerRef.current !== null) {
      window.clearTimeout(recordTimerRef.current);
      recordTimerRef.current = null;
    }
  };

  const stopCamera = useCallback(() => {
    clearTimers();
    if (mediaRecorderRef.current?.state !== "inactive") {
      try {
        mediaRecorderRef.current?.stop();
      } catch {
        /* ignore */
      }
    }
    mediaRecorderRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (liveVideoRef.current) {
      liveVideoRef.current.srcObject = null;
    }
    setCountdown(null);
    if (capturePhase !== "analyzing") {
      setCapturePhase("idle");
    }
  }, [capturePhase]);

  useEffect(() => {
    if (!selected) return;
    loadSign(selected).catch((e) => setError(String(e)));
    getOrientationProgress(selected.gloss)
      .then(setProgress)
      .catch(() => setProgress(null));
    setResult(null);
    setError(null);
  }, [selected, loadSign]);

  useEffect(() => {
    return () => {
      clearTimers();
      if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [videoPreviewUrl]);

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    const list = [...glosses].sort((a, b) => a.display_name.localeCompare(b.display_name));
    if (!q) return list;
    return list.filter(
      (g) => g.gloss.includes(q) || g.display_name.toLowerCase().includes(q)
    );
  }, [glosses, filter]);

  const playerSource = useMemo(() => {
    if (selected?.default_exemplar_id && exemplarId && numFrames > 0) {
      return {
        mode: "url" as const,
        gloss: selected.gloss,
        exemplarId,
        numFrames,
      };
    }
    return null;
  }, [selected, exemplarId, numFrames]);

  const setVideo = (blob: Blob) => {
    if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
    setVideoBlob(blob);
    setVideoPreviewUrl(URL.createObjectURL(blob));
    setResult(null);
  };

  const runAnalyze = async (blob: Blob, gloss: string) => {
    setCapturePhase("analyzing");
    setError(null);
    try {
      const res = await analyzeOrientation(gloss, blob, useGemmaRef.current);
      setResult(res);
      setProgress(res.progress);
    } catch (e) {
      setError(String(e));
    } finally {
      setCapturePhase("idle");
    }
  };

  const startRecording = async () => {
    if (!selected) {
      setError("Choose a sign first.");
      return;
    }
    setError(null);
    setResult(null);
    clearTimers();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (liveVideoRef.current) {
        liveVideoRef.current.srcObject = stream;
        await liveVideoRef.current.play().catch(() => undefined);
      }

      setCapturePhase("countdown");
      let remaining = COUNTDOWN_START;
      setCountdown(remaining);

      countdownTimerRef.current = window.setInterval(() => {
        remaining -= 1;
        if (remaining > 0) {
          setCountdown(remaining);
          return;
        }

        clearTimers();
        setCountdown(null);
        setCapturePhase("recording");

        chunksRef.current = [];
        const recorder = new MediaRecorder(stream, { mimeType: getSupportedMimeType() });
        mediaRecorderRef.current = recorder;

        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) chunksRef.current.push(e.data);
        };

        recorder.onstop = async () => {
          const blob = new Blob(chunksRef.current, { type: recorder.mimeType });
          stream.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
          if (liveVideoRef.current) {
            liveVideoRef.current.srcObject = null;
          }
          mediaRecorderRef.current = null;
          setVideo(blob);

          const gloss = selectedRef.current?.gloss;
          if (gloss) {
            await runAnalyze(blob, gloss);
          } else {
            setCapturePhase("idle");
          }
        };

        recorder.start();
        recordTimerRef.current = window.setTimeout(() => {
          if (recorder.state === "recording") recorder.stop();
        }, RECORD_MS);
      }, 1000);
    } catch (e) {
      setError(String(e));
      stopCamera();
    }
  };

  const onFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file || !selected) return;
    setVideo(file);
    await runAnalyze(file, selected.gloss);
  };

  const busy = capturePhase !== "idle";
  const resultBadgeClass =
    result?.comparison.overall_result === "pass"
      ? "pass"
      : result?.comparison.overall_result === "unusable"
        ? "unusable"
        : "needs";

  return (
    <div className="coach-layout">
      <section
        className={`stage-card coach-stage ${capturePhase === "analyzing" ? "loading-overlay" : ""}`}
      >
        <div className="sign-label">
          <span className="eyebrow">Reference sign</span>
          <h2>{selected?.display_name ?? "Choose a sign"} {selected?.display_name_ml && <span className="ml-name">{selected.display_name_ml}</span>}</h2>
          {selected && (
            <p className="slug">Gloss: {selected.gloss.replace(/_/g, " ")}</p>
          )}
          {progress?.mastered && <span className="mastered-badge">Mastered</span>}
        </div>

        <SkeletonPlayer source={playerSource} loop speed={speed} displaySize={480} />

        <div className="speed-row">
          <label>Reference speed</label>
          <div className="speed-btns">
            {SPEEDS.map((s) => (
              <button
                key={s}
                type="button"
                className={speed === s ? "active" : ""}
                onClick={() => setSpeed(s)}
              >
                {s}×
              </button>
            ))}
          </div>
        </div>

        <div className="coach-capture-card">
          <h3>Your practice clip</h3>
          <p className="subtitle">
            Record a 2-second clip after a short countdown, or upload a video.
            Analysis runs automatically when recording finishes.
          </p>

          <div className="coach-capture-actions">
            <button
              type="button"
              className="btn-secondary"
              disabled={busy || !selected}
              onClick={startRecording}
            >
              {capturePhase === "countdown"
                ? `Starting in ${countdown}…`
                : capturePhase === "recording"
                  ? "Recording…"
                  : capturePhase === "analyzing"
                    ? "Analyzing…"
                    : "Record from webcam"}
            </button>
            <button
              type="button"
              className="btn-secondary"
              disabled={busy}
              onClick={() => fileInputRef.current?.click()}
            >
              Upload video
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              hidden
              onChange={onFileChange}
            />
            <label className="checkbox-label coach-gemma-toggle">
              <input
                type="checkbox"
                checked={useGemma}
                disabled={busy}
                onChange={(e) => setUseGemma(e.target.checked)}
              />
              Use Gemma 4 for coaching feedback
            </label>
          </div>

          <div className="coach-live-wrap">
            <video
              ref={liveVideoRef}
              className={`coach-live ${capturePhase === "countdown" || capturePhase === "recording" ? "visible" : ""}`}
              muted
              playsInline
            />
            {capturePhase === "countdown" && countdown !== null && (
              <div className="coach-countdown" aria-live="polite">
                <span className="coach-countdown-num">{countdown}</span>
                <span className="coach-countdown-label">Get ready to sign</span>
              </div>
            )}
            {capturePhase === "recording" && (
              <div className="coach-recording-badge" aria-live="polite">
                Recording
              </div>
            )}
          </div>

          {videoPreviewUrl && capturePhase === "idle" && (
            <video className="coach-preview" src={videoPreviewUrl} controls playsInline />
          )}
        </div>

        {result && (
          <div className="feedback-card">
            <div className={`result-badge ${resultBadgeClass}`}>
              {result.comparison.overall_result === "pass"
                ? "Within tolerance"
                : result.comparison.overall_result === "unusable"
                  ? "Try again"
                  : "Needs correction"}
            </div>
            <p className="feedback-text">{result.feedback_text}</p>
            {result.comparison.errors.length > 0 && (
              <div className="error-chips" aria-label="Orientation errors">
                {result.comparison.errors.map((err, i) => (
                  <span key={`${err.feature}-${i}`} className={`error-chip severity-${err.severity}`}>
                    <strong>{err.feature.replace(/_/g, " ")}</strong>
                    <span>{err.direction}</span>
                    <span className="deviation">{err.deviation_deg}°</span>
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {progress && progress.attempt_count > 0 && (
          <div className="coach-history">
            <h4>Recent attempts ({progress.attempt_count})</h4>
            <ul>
              {progress.attempts.slice(-5).reverse().map((a, i) => (
                <li key={`${a.timestamp}-${i}`}>
                  <span className={`hist-result ${a.overall_result}`}>{a.overall_result}</span>
                  <span className="hist-time">{new Date(a.timestamp).toLocaleString()}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {error && <p className="message error">{error}</p>}
      </section>

      <aside className="sidebar">
        <div className="side-card">
          <h3>Practice signs ({vocabMeta.withData}/{vocabMeta.total})</h3>
          <input
            className="search"
            placeholder="Search signs…"
            value={filter}
            onChange={(e) => onFilterChange(e.target.value)}
            aria-label="Search vocabulary"
          />
          <div className="word-list" role="listbox" aria-label="Sign list">
            {filtered.map((g) => (
              <button
                key={g.gloss}
                type="button"
                role="option"
                aria-selected={selected?.gloss === g.gloss}
                className={`word-item ${selected?.gloss === g.gloss ? "active" : ""} ${!g.has_sign ? "missing" : ""}`}
                disabled={!g.has_sign || busy}
                onClick={() => onSelect(g)}
              >
                <span>{g.display_name}{g.display_name_ml && <span className="ml-name"> {g.display_name_ml}</span>}</span>
                {g.has_sign && <span className="badge">{g.variant_count} clips</span>}
              </button>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}

function getSupportedMimeType(): string {
  const types = ["video/webm;codecs=vp9", "video/webm", "video/mp4"];
  for (const t of types) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return "video/webm";
}
