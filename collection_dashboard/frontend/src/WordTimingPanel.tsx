import { useEffect, useState } from "react";
import { updateWordTiming, type GlossInfo } from "./api";

type Props = {
  word: GlossInfo | null;
  onUpdated: () => void;
};

const PRESETS = [
  { label: "None", cooldown: 0, ref: 0 },
  { label: "0.5 s", cooldown: 0.5, ref: 0.5 },
  { label: "1 s", cooldown: 1, ref: 1 },
  { label: "Default (2 s)", cooldown: 2, ref: 2 },
];

export default function WordTimingPanel({ word, onUpdated }: Props) {
  const [cooldown, setCooldown] = useState(2);
  const [refCountdown, setRefCountdown] = useState(2);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!word) return;
    setCooldown(word.cooldown_sec);
    setRefCountdown(word.ref_countdown_sec);
    setError(null);
  }, [word?.word, word?.cooldown_sec, word?.ref_countdown_sec]);

  if (!word) return null;

  const save = async (nextCooldown: number, nextRef: number) => {
    setSaving(true);
    setError(null);
    try {
      await updateWordTiming(word.word, {
        cooldown_sec: nextCooldown,
        ref_countdown_sec: nextRef,
      });
      onUpdated();
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  const applyPreset = (cooldownSec: number, refSec: number) => {
    setCooldown(cooldownSec);
    setRefCountdown(refSec);
    void save(cooldownSec, refSec);
  };

  const commitField = () => {
    void save(cooldown, refCountdown);
  };

  return (
    <section className="word-timing-panel">
      <h3>Wait times — {word.display_name}{word.display_name_ml ? <span className="ml-name"> {word.display_name_ml}</span> : ""}</h3>
      <p className="muted timing-hint">
        Per-word gaps between clips and after reference videos. Use 0 for no wait.
      </p>

      <div className="timing-grid">
        <label className="timing-field">
          <span>Between clips (WAIT bar)</span>
          <input
            type="number"
            min={0}
            max={30}
            step={0.5}
            value={cooldown}
            disabled={saving}
            onChange={(e) => setCooldown(Number(e.target.value))}
            onBlur={commitField}
          />
          <span className="unit">seconds</span>
        </label>

        <label className="timing-field">
          <span>After 3 references</span>
          <input
            type="number"
            min={0}
            max={30}
            step={0.5}
            value={refCountdown}
            disabled={saving}
            onChange={(e) => setRefCountdown(Number(e.target.value))}
            onBlur={commitField}
          />
          <span className="unit">seconds</span>
        </label>
      </div>

      <div className="timing-presets">
        {PRESETS.map((p) => (
          <button
            key={p.label}
            type="button"
            className="btn secondary btn-sm"
            disabled={saving}
            onClick={() => applyPreset(p.cooldown, p.ref)}
          >
            {p.label}
          </button>
        ))}
      </div>

      {error && <p className="error">{error}</p>}
    </section>
  );
}
