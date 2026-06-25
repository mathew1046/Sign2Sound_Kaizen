import { useCallback, useEffect, useRef, useState } from "react";
import { frameUrl, type Timeline } from "./api";

type FrameSource =
  | { mode: "url"; gloss: string; exemplarId: string; numFrames: number }
  | { mode: "timeline"; timeline: Timeline };

type Props = {
  source: FrameSource | null;
  loop?: boolean;
  autoplay?: boolean;
  fps?: number;
  speed?: number;
  displaySize?: number;
  onFrameChange?: (index: number, total: number) => void;
};

export default function SkeletonPlayer({
  source,
  loop = true,
  autoplay = true,
  fps: fpsProp,
  speed = 1,
  displaySize = 480,
  onFrameChange,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [playing, setPlaying] = useState(autoplay);
  const [frameIdx, setFrameIdx] = useState(0);
  const rafRef = useRef<number>(0);
  const lastRef = useRef<number>(0);
  const cacheRef = useRef<Map<number, HTMLImageElement>>(new Map());

  const numFrames =
    source?.mode === "url"
      ? source.numFrames
      : source?.mode === "timeline"
        ? source.timeline.num_frames
        : 0;

  const baseFps =
    fpsProp ?? (source?.mode === "timeline" ? source.timeline.fps : 25);
  const fps = baseFps * Math.max(0.25, speed);

  const loadFrame = useCallback(
    async (idx: number): Promise<HTMLImageElement | null> => {
      if (!source || idx < 0 || idx >= numFrames) return null;
      if (cacheRef.current.has(idx)) return cacheRef.current.get(idx)!;

      const img = new Image();
      if (source.mode === "url") {
        img.src = frameUrl(source.gloss, source.exemplarId, idx);
      } else {
        const fr = source.timeline.frames[idx];
        if (fr?.frame_b64) {
          img.src = `data:image/png;base64,${fr.frame_b64}`;
        } else {
          return null;
        }
      }
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error(`frame ${idx}`));
      });
      cacheRef.current.set(idx, img);
      return img;
    },
    [source, numFrames]
  );

  useEffect(() => {
    cacheRef.current.clear();
    setFrameIdx(0);
    setPlaying(autoplay);
    lastRef.current = 0;
  }, [source, autoplay]);

  useEffect(() => {
    onFrameChange?.(frameIdx, numFrames);
  }, [frameIdx, numFrames, onFrameChange]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !source || numFrames === 0) return;

    let cancelled = false;

    const draw = async () => {
      const img = await loadFrame(frameIdx);
      if (cancelled || !img) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    draw();
    return () => {
      cancelled = true;
    };
  }, [source, frameIdx, numFrames, loadFrame]);

  useEffect(() => {
    if (!playing || numFrames === 0) return;
    const step = (ts: number) => {
      if (!lastRef.current) lastRef.current = ts;
      const dt = ts - lastRef.current;
      if (dt >= 1000 / fps) {
        lastRef.current = ts;
        setFrameIdx((i) => {
          const next = i + 1;
          if (next >= numFrames) return loop ? 0 : i;
          return next;
        });
      }
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing, numFrames, fps, loop]);

  const goTo = (idx: number) => {
    if (numFrames === 0) return;
    setFrameIdx(Math.max(0, Math.min(numFrames - 1, idx)));
    lastRef.current = 0;
  };

  const stepFrame = (delta: number) => {
    goTo(frameIdx + delta);
    setPlaying(false);
  };

  const progress = numFrames > 1 ? (frameIdx / (numFrames - 1)) * 100 : 0;

  return (
    <div className="player-stage">
      <div
        className="player-frame"
        style={{ width: displaySize, height: displaySize }}
      >
        <canvas
          ref={canvasRef}
          className="player-canvas"
          width={224}
          height={224}
        />
        {numFrames === 0 && (
          <div className="player-empty">
            <span>Select a sign to begin</span>
          </div>
        )}
      </div>

      <input
        type="range"
        className="player-scrub"
        min={0}
        max={Math.max(0, numFrames - 1)}
        value={frameIdx}
        disabled={numFrames === 0}
        onChange={(e) => goTo(Number(e.target.value))}
        aria-label="Scrub through sign frames"
      />

      <div className="player-toolbar">
        <button
          type="button"
          className="btn-icon"
          disabled={numFrames === 0}
          onClick={() => stepFrame(-1)}
          aria-label="Previous frame"
        >
          ‹
        </button>
        <button
          type="button"
          className="btn-primary btn-play"
          disabled={numFrames === 0}
          onClick={() => setPlaying((p) => !p)}
        >
          {playing ? "Pause" : "Play"}
        </button>
        <button
          type="button"
          className="btn-icon"
          disabled={numFrames === 0}
          onClick={() => stepFrame(1)}
          aria-label="Next frame"
        >
          ›
        </button>
        <button
          type="button"
          className="btn-ghost"
          disabled={numFrames === 0}
          onClick={() => {
            goTo(0);
            setPlaying(true);
          }}
        >
          Restart
        </button>
        <span className="player-counter">
          {numFrames > 0 ? `${frameIdx + 1} / ${numFrames}` : "—"}
        </span>
      </div>
      <div
        className="player-progress-track"
        role="progressbar"
        aria-valuenow={progress}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div className="player-progress-fill" style={{ width: `${progress}%` }} />
      </div>
    </div>
  );
}
