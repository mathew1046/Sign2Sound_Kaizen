import { useCallback, useEffect, useState } from "react";
import {
  collectedUrl,
  deleteSlot,
  getCollected,
  rerecordSlot,
  type CollectedWord,
} from "./api";

type Props = {
  word: string | null;
  onRefresh: () => void;
  onOpenReview?: () => void;
};

export default function RecordedClipsPanel({ word, onRefresh, onOpenReview }: Props) {
  const [data, setData] = useState<CollectedWord | null>(null);
  const [playing, setPlaying] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!word) {
      setData(null);
      return;
    }
    try {
      const d = await getCollected(word);
      setData(d);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, [word]);

  useEffect(() => {
    load();
    const id = setInterval(load, 3000);
    return () => clearInterval(id);
  }, [load]);

  const handleDelete = async (slot: number) => {
    if (!word || !confirm(`Delete recording #${slot + 1}?`)) return;
    await deleteSlot(word, slot);
    if (playing === slot) setPlaying(null);
    await load();
    onRefresh();
  };

  const handleRetake = async (slot: number) => {
    if (!word || !confirm(`Re-record slot #${slot + 1}? The engine will capture it again.`)) return;
    await rerecordSlot(word, slot);
    if (playing === slot) setPlaying(null);
    await load();
    onRefresh();
  };

  if (!word) return null;

  return (
    <section className="recorded-panel">
      <div className="recorded-header">
        <h3>
          Recorded clips — {data?.display_name ?? word.replace(/_/g, " ")}{data?.display_name_ml ? <span className="ml-name"> {data.display_name_ml}</span> : ""} (
          {data?.completed_count ?? 0}/10)
        </h3>
        {onOpenReview && (
          <button type="button" className="btn secondary btn-sm" onClick={onOpenReview}>
            Open full review
          </button>
        )}
      </div>

      {error && <p className="error">{error}</p>}

      <div className="slot-grid recorded-grid">
        {Array.from({ length: 10 }, (_, i) => {
          const slot = data?.slots.find((s) => s.index === i);
          const hasVideo = Boolean(slot?.url);
          return (
            <div
              key={i}
              className={`slot-card recorded ${hasVideo ? "complete" : "empty"} ${playing === i ? "active" : ""}`}
            >
              {hasVideo ? (
                <video
                  src={`${collectedUrl(word, i)}?t=${slot?.recorded_at ?? i}`}
                  muted
                  playsInline
                  preload="metadata"
                  onClick={() => setPlaying(playing === i ? null : i)}
                />
              ) : (
                <div className="slot-empty">
                  {slot?.status === "pending_rerecord" ? "Queued" : "Empty"}
                </div>
              )}
              <div className="slot-footer recorded-footer">
                <span>#{i + 1}</span>
                <div className="slot-actions">
                  {hasVideo && (
                    <button type="button" className="link-btn" onClick={() => setPlaying(i)}>
                      View
                    </button>
                  )}
                  {hasVideo && (
                    <button type="button" className="link-btn danger" onClick={() => handleDelete(i)}>
                      Delete
                    </button>
                  )}
                  <button type="button" className="link-btn" onClick={() => handleRetake(i)}>
                    {hasVideo ? "Retake" : "Record"}
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {playing !== null && data?.slots[playing]?.url && (
        <div className="review-player inline-player">
          <h4>Sample #{playing + 1}</h4>
          <video
            key={`play-${word}-${playing}-${data.slots[playing]?.recorded_at}`}
            src={`${collectedUrl(word, playing)}?t=${data.slots[playing]?.recorded_at}`}
            controls
            autoPlay
            className="main-player"
          />
        </div>
      )}
    </section>
  );
}
