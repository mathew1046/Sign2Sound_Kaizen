const API = import.meta.env.VITE_API_URL ?? "";

export type GlossInfo = {
  label_id: number;
  word: string;
  display_name: string;
  display_name_ml: string | null;
  completed_count: number;
  samples_per_word: number;
  is_complete: boolean;
  cooldown_sec: number;
  ref_countdown_sec: number;
};

export type EngineStatus = {
  state: string;
  current_word: string | null;
  current_word_index: number;
  current_slot: number;
  phase: string;
  ref_index: number;
  phase_started_at: number | null;
  phase_duration_sec: number;
  motion: number;
  total_completed: number;
  total_target: number;
  paused: boolean;
  message: string;
};

export type SlotInfo = {
  index: number;
  status: string;
  file: string | null;
  url: string | null;
  duration_ms?: number;
  recorded_at?: string;
};

export type CollectedWord = {
  word: string;
  label_id: number;
  display_name: string;
  display_name_ml: string | null;
  completed_count: number;
  slots: SlotInfo[];
  references: Array<{ index: number; url: string }>;
};

export type Overview = {
  total_words: number;
  complete_words: number;
  total_completed: number;
  total_target: number;
  engine_state: string;
  incomplete_words: Array<{
    word: string;
    display_name: string;
    display_name_ml: string | null;
    completed_count: number;
    remaining: number;
  }>;
};

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${url}`, init);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof err.detail === "string" ? err.detail : JSON.stringify(err));
  }
  return res.json();
}

export function getVocab() {
  return fetchJson<{
    num_glosses: number;
    glosses: GlossInfo[];
    total_completed: number;
    total_target: number;
    default_cooldown_sec: number;
    default_ref_countdown_sec: number;
  }>("/api/vocab");
}

export function getEngineStatus() {
  return fetchJson<EngineStatus>("/api/engine/status");
}

export function pauseEngine() {
  return fetchJson<{ ok: boolean }>("/api/engine/pause", { method: "POST" });
}

export function resumeEngine() {
  return fetchJson<{ ok: boolean }>("/api/engine/resume", { method: "POST" });
}

export function getCollected(word: string) {
  return fetchJson<CollectedWord>(`/api/collected/${encodeURIComponent(word)}`);
}

export function deleteSlot(word: string, slot: number) {
  return fetchJson<{ ok: boolean }>(
    `/api/collected/${encodeURIComponent(word)}/${slot}`,
    { method: "DELETE" }
  );
}

export function rerecordSlot(word: string, slot: number) {
  return fetchJson<{ ok: boolean }>(
    `/api/collected/${encodeURIComponent(word)}/${slot}/rerecord`,
    { method: "POST" }
  );
}

export function getOverview() {
  return fetchJson<Overview>("/api/overview");
}

export function getReferences(word: string) {
  return fetchJson<{
    word: string;
    references: Array<{ index: number; source: string; url: string }>;
  }>(`/api/references/${encodeURIComponent(word)}`);
}

export function referenceUrl(word: string, idx: number) {
  return `${API}/api/references/${encodeURIComponent(word)}/${idx}`;
}

export function collectedUrl(word: string, slot: number) {
  return `${API}/api/collected/${encodeURIComponent(word)}/${slot}`;
}

export function liveStreamUrl() {
  return `${API}/api/stream/live.mjpg`;
}

export function snapshotUrl() {
  return `${API}/api/stream/snapshot.jpg`;
}

export function getHealth() {
  return fetchJson<{
    status: string;
    camera_enabled: boolean;
    camera_ok: boolean;
    camera_error: string | null;
    engine: EngineStatus;
    output_dir: string;
  }>("/api/health");
}

export function getCameraStatus() {
  return fetchJson<{ enabled: boolean; active: boolean; error: string | null }>(
    "/api/camera/status"
  );
}

export function enableCamera() {
  return fetchJson<{ ok: boolean; enabled: boolean; active: boolean; error: string | null }>(
    "/api/camera/enable",
    { method: "POST" }
  );
}

export function disableCamera() {
  return fetchJson<{ ok: boolean; enabled: boolean; active: boolean; paused_engine?: boolean }>(
    "/api/camera/disable",
    { method: "POST" }
  );
}

export function resetAllData() {
  return fetchJson<{ ok: boolean; message: string }>("/api/data/reset", {
    method: "POST",
  });
}

export function exportManifestUrl() {
  return `${API}/api/export/manifest.csv`;
}

export function updateWordTiming(
  word: string,
  timing: { cooldown_sec?: number; ref_countdown_sec?: number }
) {
  return fetchJson<{
    ok: boolean;
    word: string;
    display_name: string;
    display_name_ml: string | null;
    cooldown_sec: number;
    ref_countdown_sec: number;
  }>(`/api/words/${encodeURIComponent(word)}/timing`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(timing),
  });
}

export type Include50Clip = {
  word: string;
  stem: string;
  label_id: number;
  split: string;
  source: string;
  video_url: string;
  frame_count: number;
  fps: number;
  has_landmarks: boolean;
  has_body: boolean;
  has_face: boolean;
  has_video: boolean;
  skeleton_backend?: "rtmlib" | "mediapipe";
};

export type Include50Eval = {
  ready: boolean;
  analysis: {
    accuracy: number;
    test_accuracy?: number;
    n_clips?: number;
    checkpoint?: string;
    split?: string;
    best_classes: Array<{
      word: string;
      display_name: string;
      display_name_ml: string | null;
      accuracy: number;
      n_test: number;
    }>;
    worst_classes: Array<{
      word: string;
      display_name: string;
      display_name_ml: string | null;
      accuracy: number;
      n_test: number;
      top_confusions?: Array<{ word: string; count: number }>;
    }>;
    check_skeleton_classes: Array<{
      label_id: number;
      word: string;
      display_name_ml: string | null;
      reason: string;
      accuracy: number;
    }>;
  } | null;
  confusion_matrix_url: string | null;
};

export type CorpusGlossInfo = {
  label_id: number;
  word: string;
  display_name: string;
  display_name_ml: string | null;
  clip_count: number;
  in_include50: boolean;
};

export type CorpusSummary = {
  num_words: number;
  num_clips: number;
  clips_with_video: number;
  clips_with_skeleton: number;
  skeleton_backend: string;
  lab_root: string;
  video_root: string;
};

export function getCorpusVocab() {
  return fetchJson<{ glosses: CorpusGlossInfo[]; summary: CorpusSummary }>(
    "/api/include50/corpus/vocab"
  );
}

export function getInclude50Eval() {
  return fetchJson<Include50Eval>("/api/include50/eval");
}

export function getInclude50Clips(word: string) {
  return fetchJson<{ word: string; clips: Include50Clip[]; count: number }>(
    `/api/include50/${encodeURIComponent(word)}/clips`
  );
}

export function include50ClipUrl(word: string, stem: string) {
  return `${API}/api/include50/clips/${encodeURIComponent(word)}/${encodeURIComponent(stem)}/video`;
}

export function include50SkeletonUrl(word: string, stem: string, frameIdx: number) {
  return `${API}/api/include50/clips/${encodeURIComponent(word)}/${encodeURIComponent(stem)}/skeleton/${frameIdx}.png`;
}
