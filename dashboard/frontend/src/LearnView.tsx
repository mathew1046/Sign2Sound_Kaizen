import { useCallback, useEffect, useMemo, useState } from "react";
import SkeletonPlayer from "./SkeletonPlayer";
import OrientationCoachPanel from "./OrientationCoachPanel";
import {
  getLearnVocab,
  sentenceToTimeline,
  type LearnGlossInfo,
  type Timeline,
  type TranslateResult,
} from "./api";

const SPEEDS = [0.5, 0.75, 1, 1.25] as const;

type LearnTab = "coach" | "sentences";

export default function LearnView() {
  const [glosses, setGlosses] = useState<LearnGlossInfo[]>([]);
  const [vocabMeta, setVocabMeta] = useState({ total: 263, withData: 0 });
  const [filter, setFilter] = useState("");
  const [selected, setSelected] = useState<LearnGlossInfo | null>(null);
  const [sentence, setSentence] = useState("good morning thank you");
  const [translateResult, setTranslateResult] = useState<TranslateResult | null>(null);
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [tab, setTab] = useState<LearnTab>("coach");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useGemini, setUseGemini] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [currentSegmentGloss, setCurrentSegmentGloss] = useState<string | null>(null);

  useEffect(() => {
    getLearnVocab()
      .then((v) => {
        setGlosses(v.glosses);
        setVocabMeta({ total: v.num_glosses, withData: v.glosses_with_data });
        const first = v.glosses.find((g) => g.has_sign) ?? v.glosses[0];
        if (first) setSelected(first);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const playerSource = useMemo(() => {
    if (tab === "sentences" && timeline) {
      return { mode: "timeline" as const, timeline };
    }
    return null;
  }, [tab, timeline]);

  const displayTitle = useMemo(() => {
    if (tab === "sentences" && timeline) {
      if (currentSegmentGloss) {
        const g = glosses.find((x) => x.gloss === currentSegmentGloss);
        return g?.display_name ?? currentSegmentGloss.replace(/_/g, " ");
      }
      return "Full sentence";
    }
    return "Full sentence";
  }, [tab, timeline, currentSegmentGloss, glosses]);

  const onFrameChange = useCallback(
    (index: number, _total: number) => {
      if (tab !== "sentences" || !timeline?.segments) return;
      const seg = timeline.segments.find(
        (s) => index >= s.start_frame && index < s.end_frame
      );
      setCurrentSegmentGloss(seg?.gloss ?? null);
    },
    [tab, timeline]
  );

  const onSentencePlay = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await sentenceToTimeline(sentence, useGemini);
      setTranslateResult(res.translate);
      setTimeline(res.timeline);
      setTab("sentences");
      setCurrentSegmentGloss(res.timeline.segments[0]?.gloss ?? null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const selectCoachWord = (g: LearnGlossInfo) => {
    setSelected(g);
    setTab("coach");
    setTimeline(null);
    setTranslateResult(null);
    setError(null);
  };

  return (
    <>
      <nav className="sub-tabs" aria-label="Learning mode">
        <button
          type="button"
          className={`tab ${tab === "coach" ? "active" : ""}`}
          onClick={() => setTab("coach")}
        >
          Practice signs
        </button>
        <button
          type="button"
          className={`tab ${tab === "sentences" ? "active" : ""}`}
          onClick={() => setTab("sentences")}
        >
          Practice sentences
        </button>
      </nav>

      {tab === "coach" ? (
        <OrientationCoachPanel
          glosses={glosses}
          selected={selected}
          onSelect={selectCoachWord}
          filter={filter}
          onFilterChange={setFilter}
          vocabMeta={vocabMeta}
        />
      ) : (
        <div className={`sentence-layout ${loading ? "loading-overlay" : ""}`}>
          <section className="sentence-card">
            <h2>Build a sentence</h2>
            <p className="subtitle">
              Type English text. We convert it to ISL glosses and play the
              continuous sign sequence.
            </p>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={useGemini}
                onChange={(e) => setUseGemini(e.target.checked)}
              />
              Use Gemini for better gloss ordering (optional API key)
            </label>
            <div className="sentence-row">
              <textarea
                value={sentence}
                onChange={(e) => setSentence(e.target.value)}
                placeholder="e.g. Good morning, thank you"
                aria-label="Sentence to translate"
              />
              <div className="action-stack">
                <button
                  type="button"
                  className="btn-primary"
                  disabled={loading}
                  onClick={onSentencePlay}
                >
                  {loading ? "Loading…" : "Learn this sentence"}
                </button>
              </div>
            </div>

            {translateResult && translateResult.glosses.length > 0 && (
              <div className="gloss-trail" aria-label="Gloss sequence">
                {translateResult.glosses.map((g, i) => (
                  <span key={`${g}-${i}`} className="gloss-step">
                    {i > 0 && <span className="arrow">→</span>}
                    <span className="chip">
                      {glosses.find((x) => x.gloss === g)?.display_name ??
                        g.replace(/_/g, " ")}
                    </span>
                  </span>
                ))}
              </div>
            )}

            {translateResult?.unknown && translateResult.unknown.length > 0 && (
              <p className="message info">
                Could not map: {translateResult.unknown.join(", ")}
              </p>
            )}
            {error && <p className="message error">{error}</p>}
          </section>

          <section className="stage-card">
            <div className="sign-label">
              <span className="eyebrow">
                {currentSegmentGloss ? "Now signing" : "Sentence practice"}
              </span>
              <h2>{displayTitle}</h2>
            </div>

            <SkeletonPlayer
              source={playerSource}
              loop
              speed={speed}
              displaySize={520}
              onFrameChange={onFrameChange}
            />

            {timeline && (
              <p className="segment-hint">
                {timeline.num_frames} frames · glosses highlight as playback moves
              </p>
            )}

            <div className="speed-row">
              <label>Playback speed</label>
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
          </section>
        </div>
      )}
    </>
  );
}
