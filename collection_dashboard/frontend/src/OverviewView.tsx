import { useEffect, useState } from "react";
import { getOverview, type Overview } from "./api";

export default function OverviewView() {
  const [data, setData] = useState<Overview | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = () =>
      getOverview()
        .then(setData)
        .catch((e) => setError(String(e)));
    load();
    const id = setInterval(load, 3000);
    return () => clearInterval(id);
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading overview…</p>;

  const pct = Math.round((data.total_completed / data.total_target) * 100);

  return (
    <div className="overview-panel">
      <div className="stats-grid">
        <div className="stat-card">
          <span className="stat-value">{data.total_completed}</span>
          <span className="stat-label">Clips collected</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">{data.complete_words}/{data.total_words}</span>
          <span className="stat-label">Words complete</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">{pct}%</span>
          <span className="stat-label">Overall progress</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">{data.engine_state}</span>
          <span className="stat-label">Engine state</span>
        </div>
      </div>

      <div className="progress-bar-lg">
        <div className="progress-bar-lg-fill" style={{ width: `${pct}%` }} />
      </div>

      <h3>Incomplete words ({data.incomplete_words.length})</h3>
      <ul className="incomplete-list">
        {data.incomplete_words.map((w) => (
          <li key={w.word}>
            <strong>{w.display_name}{w.display_name_ml ? <span className="ml-name"> {w.display_name_ml}</span> : ""}</strong>
            <span>
              {w.completed_count}/10 — {w.remaining} remaining
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
