import { useCallback, useEffect, useRef, useState } from "react";
import CollectionView from "./CollectionView";
import ConsentModal from "./ConsentModal";
import ExploreView from "./ExploreView";
import LearnView from "./LearnView";
import OverviewView from "./OverviewView";
import ProgressSidebar from "./ProgressSidebar";
import RecordingBar from "./RecordingBar";
import ReviewView from "./ReviewView";
import WordTimingPanel from "./WordTimingPanel";
import {
  disableCamera,
  enableCamera,
  getCameraStatus,
  getCollectionVocab,
  getConsentStatus,
  getCorpusVocab,
  getEngineStatus,
  pauseEngine,
  recordConsent,
  resetAllData,
  resumeEngine,
  startEngine,
  stopEngine,
  type CorpusGlossInfo,
  type CorpusSummary,
  type EngineStatus,
  type GlossInfo,
} from "./api";

type Tab = "learn" | "collect" | "review" | "explore" | "overview";

export default function App() {
  const [tab, setTab] = useState<Tab>("learn");
  const [glosses, setGlosses] = useState<GlossInfo[]>([]);
  const [corpusGlosses, setCorpusGlosses] = useState<CorpusGlossInfo[]>([]);
  const [corpusSummary, setCorpusSummary] = useState<CorpusSummary | null>(null);
  const [selected, setSelected] = useState<GlossInfo | null>(null);
  const [exploreSelected, setExploreSelected] = useState<CorpusGlossInfo | null>(null);
  const [filter, setFilter] = useState("");
  const [engine, setEngine] = useState<EngineStatus | null>(null);
  const [totalCompleted, setTotalCompleted] = useState(0);
  const [totalTarget, setTotalTarget] = useState(500);
  const [error, setError] = useState<string | null>(null);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [showConsent, setShowConsent] = useState(false);
  const [consented, setConsented] = useState(false);
  const followEngineRef = useRef(true);

  const collectionActive =
    engine?.state === "collecting" || (engine?.paused === true && engine?.current_word != null);

  useEffect(() => {
    getCameraStatus()
      .then((s) => setCameraEnabled(s.enabled))
      .catch(() => {});
    getCorpusVocab()
      .then((v) => {
        setCorpusGlosses(v.glosses);
        setCorpusSummary(v.summary);
        setExploreSelected((prev) => prev ?? v.glosses[0] ?? null);
      })
      .catch((e) => setError(String(e)));
    getConsentStatus()
      .then((s) => setConsented(s.consented))
      .catch(() => {});
  }, []);

  const refreshVocab = useCallback(() => {
    getCollectionVocab()
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
    if (tab !== "collect" && tab !== "review") return;
    const poll = () => {
      getEngineStatus()
        .then((s) => {
          setEngine(s);
          setTotalCompleted(s.total_completed);
          setTotalTarget(s.total_target);
        })
        .catch(() => {});
    };
    poll();
    const id = setInterval(poll, collectionActive ? 500 : 3000);
    const vocabId = setInterval(refreshVocab, 5000);
    return () => {
      clearInterval(id);
      clearInterval(vocabId);
    };
  }, [refreshVocab, tab, collectionActive]);

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

  const startCollection = async () => {
    if (!consented) {
      setShowConsent(true);
      return;
    }
    try {
      setError(null);
      const cam = await enableCamera();
      setCameraEnabled(cam.enabled);
      if (!cam.active) {
        setError(cam.error || "Could not open camera");
        return;
      }
      await startEngine();
      const s = await getEngineStatus();
      setEngine(s);
    } catch (e) {
      setError(String(e));
    }
  };

  const stopCollection = async () => {
    try {
      await stopEngine();
      await disableCamera();
      setCameraEnabled(false);
      const s = await getEngineStatus();
      setEngine(s);
    } catch (e) {
      setError(String(e));
    }
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
        if (collectionActive) {
          await stopCollection();
        } else {
          await disableCamera();
          setCameraEnabled(false);
        }
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

  const handleConsentAccept = async () => {
    try {
      await recordConsent(true);
      setConsented(true);
      setShowConsent(false);
      startCollection();
    } catch (e) {
      setError(String(e));
    }
  };

  const handleConsentDecline = () => {
    setShowConsent(false);
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

  const showCollectionChrome = tab === "collect" || tab === "review";

  return (
    <div className="app">
      {showCollectionChrome && collectionActive && <RecordingBar engine={engine} />}
      <header className="site-header">
        <div className="brand">
          <h1>Sign2Sound ISL Dashboard</h1>
          <p>
            Learn signs, compose sentences, collect training data, and explore the
            263-gloss rtmlib corpus — all from one place.
          </p>
        </div>
        <div className="header-actions">
          {showCollectionChrome && (
            <>
              <div className="stat-pill">
                <span className="stat-num">{totalCompleted}</span>
                <span className="stat-lbl">/ {totalTarget} clips</span>
              </div>
              <div
                className={`engine-badge state-${engine?.paused ? "paused" : engine?.state ?? "idle"}`}
              >
                {engine?.paused ? "Paused" : engine?.state ?? "Idle"}
              </div>
              {!collectionActive ? (
                <button type="button" className="btn btn-primary" onClick={startCollection}>
                  Start collection
                </button>
              ) : (
                <>
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
                  <button type="button" className="btn" onClick={stopCollection}>
                    Stop
                  </button>
                </>
              )}
              <button type="button" className="btn btn-danger" onClick={handleReset}>
                Reset data
              </button>
            </>
          )}
          {corpusSummary && (
            <div className="stat-pill">
              <strong>{corpusSummary.num_words}</strong> glosses ·{" "}
              <strong>{corpusSummary.clips_with_skeleton}</strong> skeleton clips
            </div>
          )}
        </div>
      </header>

      <nav className="tabs">
        {(
          [
            ["learn", "Learn"],
            ["collect", "Collect"],
            ["review", totalCompleted > 0 ? `Review (${totalCompleted})` : "Review"],
            ["explore", `Explore (${corpusSummary?.num_words ?? 263})`],
            ["overview", "Overview"],
          ] as const
        ).map(([t, label]) => (
          <button
            key={t}
            type="button"
            className={`tab ${tab === t ? "active" : ""}`}
            onClick={() => handleTabChange(t)}
          >
            {label}
          </button>
        ))}
      </nav>

      {error && <p className="error banner">{error}</p>}

      {showConsent && (
        <ConsentModal
          onAccept={handleConsentAccept}
          onDecline={handleConsentDecline}
        />
      )}

      {tab === "learn" ? (
        <LearnView />
      ) : (
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
                collectionActive={collectionActive}
                onStart={startCollection}
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
      )}
    </div>
  );
}
