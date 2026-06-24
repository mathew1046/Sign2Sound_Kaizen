import { useCallback, useEffect, useRef, useState } from "react";
import type { EngineStatus, GlossInfo } from "./api";
import { getReferences, getHealth, snapshotUrl, referenceUrl } from "./api";
import RecordedClipsPanel from "./RecordedClipsPanel";

type Props = {
  engine: EngineStatus | null;
  selected: GlossInfo | null;
  cameraEnabled: boolean;
  onRefresh: () => void;
  onOpenReview: () => void;
};

export default function CollectionView({
  engine,
  selected,
  cameraEnabled,
  onRefresh,
  onOpenReview,
}: Props) {
  const collectWord = engine?.current_word ?? null;
  const viewWord = selected?.word ?? collectWord;
  const phase = engine?.phase ?? "idle";

  const [refCount, setRefCount] = useState(0);
  const [activeRef, setActiveRef] = useState(0);
  const [refToken, setRefToken] = useState(0);
  const [playError, setPlayError] = useState<string | null>(null);
  const [frameTs, setFrameTs] = useState(0);
  const [cameraOk, setCameraOk] = useState(true);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const retryCount = useRef(0);

  useEffect(() => {
    if (!cameraEnabled) {
      setCameraOk(false);
      setCameraError("Camera disabled");
      return;
    }
    const poll = () =>
      getHealth()
        .then((h) => {
          setCameraOk(h.camera_enabled && h.camera_ok);
          setCameraError(h.camera_error);
        })
        .catch(() => {
          setCameraOk(false);
          setCameraError("Cannot reach server");
        });
    poll();
    const id = setInterval(poll, 3000);
    return () => clearInterval(id);
  }, [cameraEnabled]);

  useEffect(() => {
    if (!cameraOk || !cameraEnabled) return;
    const id = setInterval(() => setFrameTs(Date.now()), 100);
    return () => clearInterval(id);
  }, [cameraOk, cameraEnabled]);

  useEffect(() => {
    if (!viewWord) {
      setRefCount(0);
      return;
    }
    setActiveRef(0);
    setRefToken(0);
    retryCount.current = 0;
    setPlayError(null);
    getReferences(viewWord)
      .then((r) => setRefCount(r.references.length))
      .catch((e) => setPlayError(String(e)));
  }, [viewWord]);

  const displayWord = viewWord?.replace(/_/g, " ") ?? "—";
  const engineWord = collectWord?.replace(/_/g, " ") ?? null;

  const engineRefMode =
    Boolean(
      collectWord &&
        collectWord === viewWord &&
        phase === "reference" &&
        !engine?.paused
    );

  useEffect(() => {
    if (engineRefMode && engine) {
      setActiveRef(engine.ref_index);
    }
  }, [engineRefMode, engine?.ref_index, engine]);

  const bumpRef = useCallback(() => {
    retryCount.current = 0;
    setRefToken((t) => t + 1);
  }, []);

  useEffect(() => {
    if (!viewWord || refCount === 0) return;
    bumpRef();
  }, [viewWord, activeRef, refCount, engine?.phase_started_at, bumpRef]);

  const refSrc =
    viewWord && refCount > 0
      ? `${referenceUrl(viewWord, activeRef)}?t=${refToken}`
      : undefined;

  const onRefEnded = () => {
    if (refCount <= 1) return;
    if (engineRefMode) {
      setActiveRef((i) => (i + 1 < refCount ? i + 1 : i));
      return;
    }
    setActiveRef((i) => (i + 1) % refCount);
  };

  const onRefError = () => {
    if (retryCount.current < 4) {
      retryCount.current += 1;
      setPlayError(`Reference ${activeRef + 1} failed to load — retrying…`);
      window.setTimeout(() => setRefToken((t) => t + 1), 1500);
      return;
    }
    setPlayError(
      `Reference ${activeRef + 1} could not be loaded. The server may still be preparing this clip — wait a moment and refresh.`
    );
  };

  const onRefPlaying = () => {
    retryCount.current = 0;
    setPlayError(null);
  };

  const collecting =
    !engine?.paused &&
    collectWord === viewWord &&
    (phase === "recording" || phase === "cooldown");

  return (
    <div className="collect-panel">
      <div className="phase-banner">
        <span className={`phase-badge phase-${phase}`}>{phase.replace(/_/g, " ")}</span>
        <p className="phase-message">
          {engine?.paused
            ? "Collection paused — press Resume to continue"
            : engine?.message || "Collecting automatically…"}
        </p>
        {engineWord && viewWord !== collectWord && (
          <p className="phase-message muted">
            Engine is collecting <strong>{engineWord}</strong> — sidebar shows{" "}
            <strong>{displayWord}</strong>
          </p>
        )}
      </div>

      <div className="video-grid">
        <div className="video-card">
          <h3>Reference sample {engineRefMode ? "(watch all 3)" : "(looping)"}</h3>
          {viewWord ? (
            <>
              <video
                className="ref-video"
                src={refSrc}
                autoPlay
                muted
                playsInline
                loop={!engineRefMode && refCount === 1}
                preload="auto"
                onEnded={refCount > 1 ? onRefEnded : undefined}
                onError={onRefError}
                onPlaying={onRefPlaying}
              />
              <p className="video-caption">
                Reference {activeRef + 1}/{refCount || 3} — {displayWord}
                {engineRefMode
                  ? " (watch once before signing…)"
                  : collecting
                    ? " (looping while you record)"
                    : " (looping)"}
              </p>
              {playError && <p className="error">{playError}</p>}
            </>
          ) : (
            <div className="video-placeholder">Waiting for next word…</div>
          )}
        </div>

        <div className="video-card">
          <h3>Live webcam</h3>
          {!cameraEnabled ? (
            <div className="video-placeholder camera-error">
              Camera off — use the header toggle to enable for collection
            </div>
          ) : cameraOk ? (
            <img
              className="live-feed"
              src={`${snapshotUrl()}?t=${frameTs}`}
              alt="Live webcam feed"
            />
          ) : (
            <div className="video-placeholder camera-error">
              Camera unavailable
              {cameraError ? `: ${cameraError}` : ""}
            </div>
          )}
          <div className="live-meta">
            <span>
              Word: <strong>{displayWord}</strong>
            </span>
            {engine && collectWord === viewWord && (
              <span>
                Slot: {engine.current_slot + 1}/10
                {phase === "recording" && !engine.paused ? " — REC" : ""}
              </span>
            )}
          </div>
        </div>
      </div>

      <RecordedClipsPanel word={viewWord} onRefresh={onRefresh} onOpenReview={onOpenReview} />
    </div>
  );
}
