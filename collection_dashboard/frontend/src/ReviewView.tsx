import { useCallback, useEffect, useState } from "react";
import {
  collectedUrl,
  deleteSlot,
  exportManifestUrl,
  getCollected,
  referenceUrl,
  rerecordSlot,
  type CollectedWord,
  type GlossInfo,
} from "./api";

type Props = {
  glosses: GlossInfo[];
  selected: GlossInfo | null;
  onSelect: (g: GlossInfo) => void;
  onRefresh: () => void;
};

export default function ReviewView({ glosses, selected, onSelect, onRefresh }: Props) {
  const [data, setData] = useState<CollectedWord | null>(null);
  const [activeSlot, setActiveSlot] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const word = selected?.word ?? glosses[0]?.word ?? null;

  const load = useCallback(async () => {
    if (!word) return;
    setLoading(true);
    setError(null);
    try {
      const d = await getCollected(word);
      setData(d);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [word]);

  useEffect(() => {
    load();
    const id = setInterval(load, 3000);
    return () => clearInterval(id);
  }, [load]);

  const handleDelete = async (slot: number) => {
    if (!word || !confirm(`Delete sample ${slot + 1}?`)) return;
    await deleteSlot(word, slot);
    if (activeSlot === slot) setActiveSlot(null);
    await load();
    onRefresh();
  };

  const handleRerecord = async (slot: number) => {
    if (!word || !confirm(`Re-record sample ${slot + 1}? Collection engine will capture it again.`)) return;
    await rerecordSlot(word, slot);
    if (activeSlot === slot) setActiveSlot(null);
    await load();
    onRefresh();
  };

  return (
    <div className="review-panel">
      <div className="review-toolbar">
        <label>
          Word{" "}
          <select
            value={word ?? ""}
            onChange={(e) => {
              const g = glosses.find((x) => x.word === e.target.value);
              if (g) onSelect(g);
            }}
          >
            {glosses.map((g) => (
              <option key={g.word} value={g.word}>
                {g.display_name}{g.display_name_ml ? <span className="ml-name"> {g.display_name_ml}</span> : ""} ({g.completed_count}/10)
              </option>
            ))}
          </select>
        </label>
        <a className="btn secondary" href={exportManifestUrl()} download>
          Export manifest CSV
        </a>
      </div>

      <p className="review-hint">
        Click <strong>View</strong> to play a clip, <strong>Delete</strong> to remove it, or{" "}
        <strong>Retake</strong> to queue a new recording for that slot.
      </p>

      {error && <p className="error">{error}</p>}
      {loading && !data && <p className="muted">Loading…</p>}

      {data && (
        <>
          <section className="review-section">
            <h3>Reference samples</h3>
            <div className="slot-grid refs">
              {data.references.map((r) => (
                <div key={r.index} className="slot-card ref">
                  <video
                    key={`${data.word}-ref-${r.index}`}
                    src={`${referenceUrl(data.word, r.index)}?t=${data.word}-${r.index}`}
                    controls
                    muted
                    playsInline
                    preload="metadata"
                    onError={(e) => {
                      const el = e.currentTarget;
                      el.src = `${referenceUrl(data.word, r.index)}?retry=${Date.now()}`;
                      el.load();
                    }}
                  />
                  <span>Ref {r.index + 1}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="review-section">
            <h3>Your recordings ({data.completed_count}/10)</h3>
            <div className="slot-grid">
              {data.slots.map((slot) => (
                <div
                  key={slot.index}
                  className={`slot-card ${slot.status} ${activeSlot === slot.index ? "active" : ""}`}
                >
                  {slot.url ? (
                    <video
                      src={`${collectedUrl(data.word, slot.index)}?t=${slot.recorded_at ?? slot.index}`}
                      muted
                      playsInline
                      preload="metadata"
                      onClick={() => setActiveSlot(slot.index)}
                    />
                  ) : (
                    <div className="slot-empty">
                      {slot.status === "pending_rerecord" ? "Re-record queued" : "Empty"}
                    </div>
                  )}
                  <div className="slot-footer">
                    <span>#{slot.index + 1}</span>
                    <div className="slot-actions">
                      {slot.url && (
                        <button
                          type="button"
                          className="action-btn"
                          onClick={() => setActiveSlot(slot.index)}
                        >
                          View
                        </button>
                      )}
                      {slot.url && (
                        <a
                          className="action-btn"
                          href={collectedUrl(data.word, slot.index)}
                          download
                        >
                          Download
                        </a>
                      )}
                      {slot.url && (
                        <button
                          type="button"
                          className="action-btn danger"
                          onClick={() => handleDelete(slot.index)}
                        >
                          Delete
                        </button>
                      )}
                      <button
                        type="button"
                        className="action-btn primary"
                        onClick={() => handleRerecord(slot.index)}
                      >
                        {slot.url ? "Retake" : "Record"}
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {activeSlot !== null && data.slots[activeSlot]?.url && (
            <section className="review-player">
              <h3>Playing sample #{activeSlot + 1}</h3>
              <video
                key={`${data.word}-${activeSlot}-${data.slots[activeSlot]?.recorded_at}`}
                src={`${collectedUrl(data.word, activeSlot)}?t=${data.slots[activeSlot]?.recorded_at}`}
                controls
                autoPlay
                className="main-player"
              />
            </section>
          )}
        </>
      )}
    </div>
  );
}
