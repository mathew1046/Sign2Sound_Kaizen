import { useEffect, useState } from "react";
import type { EngineStatus } from "./api";

type Props = {
  engine: EngineStatus | null;
};

export default function RecordingBar({ engine }: Props) {
  const [progress, setProgress] = useState(0);

  const phase = engine?.phase ?? "idle";
  const active = phase === "recording" || phase === "cooldown";
  const startedAt = engine?.phase_started_at ?? 0;
  const duration = engine?.phase_duration_sec ?? 1;

  useEffect(() => {
    if (!active || !startedAt) {
      setProgress(0);
      return;
    }

    const tick = () => {
      const elapsed = Date.now() / 1000 - startedAt;
      setProgress(Math.min(1, Math.max(0, elapsed / duration)));
    };

    tick();
    const id = setInterval(tick, 50);
    return () => clearInterval(id);
  }, [active, startedAt, duration, phase]);

  if (!active || engine?.paused) return null;

  const isRecording = phase === "recording";
  const word = engine?.current_word?.replace(/_/g, " ") ?? "";
  const slot = (engine?.current_slot ?? 0) + 1;
  const pct = Math.round(progress * 100);

  return (
    <div
      className={`recording-bar ${isRecording ? "recording" : "cooldown"}`}
      role="progressbar"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div className="recording-bar-fill" style={{ width: `${pct}%` }} />
      <div className="recording-bar-label">
        <span className="recording-bar-title">
          {isRecording ? "SIGN NOW" : "WAIT"}
        </span>
        <span className="recording-bar-detail">
          {word} · sample {slot}/10 · {pct}%
        </span>
      </div>
    </div>
  );
}
