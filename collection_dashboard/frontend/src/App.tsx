import { useCallback, useEffect, useRef, useState } from "react";
import CollectionView from "./CollectionView";
import ExploreView from "./ExploreView";
import OverviewView from "./OverviewView";
import ProgressSidebar from "./ProgressSidebar";
import RecordingBar from "./RecordingBar";
import ReviewView from "./ReviewView";
import WordTimingPanel from "./WordTimingPanel";
import {
  disableCamera,
  enableCamera,
  getCameraStatus,
  getCorpusVocab,
  getEngineStatus,
  getVocab,
  pauseEngine,
  resetAllData,
  resumeEngine,
  type CorpusGlossInfo,
  type EngineStatus,
  type GlossInfo,
} from "./api";

type Tab = "collect" | "review" | "overview" | "explore";

export default function App() {
  const [tab, setTab] = useState<Tab>("collect");
  const [glosses, setGlosses] = useState<GlossInfo[]>([]);
  const [corpusGlosses, setCorpusGlosses] = useState<CorpusGlossInfo[]>([]);
  const [corpusSummary, setCorpusSummary] = useState<{
    num_words: number;
    num_clips: number;
    clips_with_video: number;
    clips_with_skeleton: number;
  } | null>(null);
  const [selected, setSelected] = useState<GlossInfo | null>(null);
  const [exploreSelected, setExploreSelected] = useState<CorpusGlossInfo | null>(null);
  const [filter, setFilter] = useState("");
  const [engine, setEngine] = useState<EngineStatus | null>(null);
  const [totalCompleted, setTotalCompleted] = useState(0);
  const [totalTarget, setTotalTarget] = useState(500);
  const [error, setError] = useState<string | null>(null);
  const [cameraEnabled, setCameraEnabled] = useState(true);
  const followEngineRef = useRef(true);

  useEffect(() => {
    getCameraStatus()
      .then((s) => setCameraEnabled(s.enabled))
      .catch(() => {});
  }, []);

  useEffect(() => {
    getCorpusVocab()
      .then((v) => {
        setCorpusGlosses(v.glosses);
        setCorpusSummary(v.summary);
        setExploreSelected((prev) => prev ?? v.glosses[0] ?? null);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const refreshVocab = useCallback(() => {
    getVocab()
      .then((v) => {
        setGlosses(v.glosses);
        setTotalCompleted(v.total_completed);
        setTotalTarget(v.total_target);
        setSelected((prev) => {
          if (prev) return prev;
          const current = v.glosses.find((g) => !g.is_complete) ?? v.glosses[0];
          return current ?? null;
        });
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    refreshVocab();
    const poll = () => {
      getEngineStatus()
        .then((s) => {
          setEngine(s);
          setTotalCompleted(s.total_completed);
          setTotalTarget(s.total_target);
          setGlosses((prev) =>
            prev.map((g) =>
              g.word === s.current_word
                ? { ...g }
                : g
            )
          );
        })
        .catch(() => {});
    };
    poll();
    const id = setInterval(poll, 500);
    const vocabId = setInterval(refreshVocab, 5000);
    return () => {
      clearInterval(id);
      clearInterval(vocabId);
    };
  }, [refreshVocab]);

  useEffect(() => {
    if (tab !== "collect" || !followEngineRef.current) return;
    if (engine?.current_word) {
      const g = glosses.find((x) => x.word === engine.current_word);
      if (g) setSelected(g);
    }
  }, [engine?.current_word, glosses, tab]);

  const handleSelectWord = (g: GlossInfo) => {
    followEngineRef.current = false;
    setSelected(g);
  };

  const handleTabChange = (next: Tab) => {
    if (next === "collect") {
      followEngineRef.current = true;
      if (engine?.current_word) {
        const g = glosses.find((x) => x.word === engine.current_word);
        if (g) setSelected(g);
      }
    }
    setTab(next);
  };

  const togglePause = async () => {
    try {
      if (engine?.paused) {
        await resumeEngine();
      } else {
        await pauseEngine();
      }
      const s = await getEngineStatus();
      setEngine(s);
    } catch (e) {
      setError(String(e));
    }
  };

  const toggleCamera = async () => {
    try {
      if (cameraEnabled) {
        await disableCamera();
        setCameraEnabled(false);
        const s = await getEngineStatus();
        setEngine(s);
      } else {
        const res = await enableCamera();
        setCameraEnabled(res.enabled);
        if (!res.active && res.error) {
          setError(res.error);
        }
      }
    } catch (e) {
      setError(String(e));
    }
  };

  const handleReset = async () => {
    if (
      !confirm(
        "Delete ALL recorded videos and start collection from scratch? This cannot be undone."
      )
    ) {
      return;
    }
    try {
      await resetAllData();
      refreshVocab();
      const s = await getEngineStatus();
      setEngine(s);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  };

  return (
    <div className="app">
      <RecordingBar engine={engine} />
      <header className="site-header">
        <div className="brand">
          <h1>INCLUDE-50 Data Collection</h1>
          <p>Autonomous collection for MSPT finetuning — 10 clips × 50 words (2.5 s each, 2 s gap)</p>
        </div>
        <div className="header-actions">
          <div className="stat-pill">
            <span className="stat-num">{totalCompleted}</span>
            <span className="stat-lbl">/ {totalTarget} clips</span>
          </div>
          <div className={`engine-badge state-${engine?.paused ? "paused" : engine?.state ?? "idle"}`}>
            {engine?.paused ? "Paused" : engine?.state ?? "Starting…"}
          </div>
          <button
            type="button"
            className={`btn ${cameraEnabled ? "" : "btn-muted"}`}
            onClick={toggleCamera}
          >
            {cameraEnabled ? "Camera on" : "Camera off"}
          </button>
          <button type="button" className="btn" onClick={togglePause}>
            {engine?.paused ? "Resume" : "Pause"}
          </button>
          <button type="button" className="btn btn-danger" onClick={handleReset}>
            Reset all data
          </button>
        </div>
      </header>

      <nav className="tabs">
        {(["collect", "review", "explore", "overview"] as Tab[]).map((t) => (
          <button
            key={t}
            type="button"
            className={`tab ${tab === t ? "active" : ""}`}
            onClick={() => handleTabChange(t)}
          >
            {t === "review" && totalCompleted > 0
              ? `Review (${totalCompleted})`
              : t === "explore"
                ? `Explore (${corpusSummary?.num_words ?? 263} glosses)`
                : t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </nav>

      {error && <p className="error banner">{error}</p>}

      <div className="layout">
        {(tab === "collect" || tab === "review" || tab === "explore") && (
          <ProgressSidebar
            glosses={tab === "explore" ? corpusGlosses : glosses}
            selectedWord={
              tab === "explore" ? exploreSelected?.word ?? null : selected?.word ?? null
            }
            filter={filter}
            onFilterChange={setFilter}
            onSelect={(g) => {
              if (tab === "explore") {
                setExploreSelected(g as CorpusGlossInfo);
              } else {
                handleSelectWord(g as GlossInfo);
              }
            }}
            mode={tab === "explore" ? "explore" : "collect"}
          />
        )}

        <main className="main-panel">
          {(tab === "collect" || tab === "review") && (
            <WordTimingPanel word={selected} onUpdated={refreshVocab} />
          )}
          {tab === "collect" && (
            <CollectionView
              engine={engine}
              selected={selected}
              cameraEnabled={cameraEnabled}
              onRefresh={refreshVocab}
              onOpenReview={() => setTab("review")}
            />
          )}
          {tab === "review" && (
            <ReviewView
              glosses={glosses}
              selected={selected}
              onSelect={handleSelectWord}
              onRefresh={refreshVocab}
            />
          )}
          {tab === "overview" && <OverviewView />}
          {tab === "explore" && (
            <ExploreView
              glosses={corpusGlosses}
              summary={corpusSummary}
              selected={exploreSelected}
              onSelect={setExploreSelected}
            />
          )}
        </main>
      </div>
    </div>
  );
}
