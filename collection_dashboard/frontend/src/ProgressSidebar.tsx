import type { CorpusGlossInfo, GlossInfo } from "./api";

type Props = {
  glosses: GlossInfo[] | CorpusGlossInfo[];
  selectedWord: string | null;
  filter: string;
  onFilterChange: (v: string) => void;
  onSelect: (g: GlossInfo | CorpusGlossInfo) => void;
  mode?: "collect" | "explore";
};

function isCorpusGloss(g: GlossInfo | CorpusGlossInfo): g is CorpusGlossInfo {
  return "clip_count" in g;
}

export default function ProgressSidebar({
  glosses,
  selectedWord,
  filter,
  onFilterChange,
  onSelect,
  mode = "collect",
}: Props) {
  const q = filter.trim().toLowerCase();
  const list = [...glosses]
    .filter(
      (g) =>
        !q ||
        g.word.includes(q) ||
        g.display_name.toLowerCase().includes(q)
    )
    .sort((a, b) => a.label_id - b.label_id);

  return (
    <aside className="sidebar">
      <input
        className="search"
        placeholder={mode === "explore" ? "Filter 263 glosses…" : "Filter words…"}
        value={filter}
        onChange={(e) => onFilterChange(e.target.value)}
      />
      <ul className="word-list">
        {list.map((g) => {
          const active = g.word === selectedWord;
          const explore = mode === "explore" && isCorpusGloss(g);
          const done = !explore && "is_complete" in g && g.is_complete;
          const pct = !explore && "completed_count" in g && "samples_per_word" in g
            ? (g.completed_count / g.samples_per_word) * 100
            : 0;
          return (
            <li key={g.word}>
              <button
                type="button"
                className={`word-btn ${active ? "active" : ""} ${done ? "done" : ""}`}
                onClick={() => onSelect(g)}
              >
                <span className="word-name">{g.display_name}</span>
                {explore ? (
                  <span className="word-count">{g.clip_count} clips</span>
                ) : (
                  "completed_count" in g && "samples_per_word" in g && (
                    <span className="word-count">
                      {g.completed_count}/{g.samples_per_word}
                    </span>
                  )
                )}
                {!explore && (
                  <span className="word-bar">
                    <span className="word-bar-fill" style={{ width: `${pct}%` }} />
                  </span>
                )}
              </button>
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
