const API = import.meta.env.VITE_API_URL ?? "";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${url}`, init);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof err.detail === "string" ? err.detail : JSON.stringify(err));
  }
  return res.json();
}

// --- Learn / skeleton browser ---

export type LearnGlossInfo = {
  gloss: string;
  display_name: string;
  display_name_ml: string | null;
  label_id: number;
  variant_count: number;
  default_exemplar_id: string | null;
  has_sign: boolean;
};

export type SignDetail = {
  gloss: string;
  display_name: string;
  display_name_ml: string | null;
  default_exemplar_id: string | null;
  variants: Array<{
    exemplar_id: string;
    signer_id: string;
    num_frames: number;
    visibility_score: number;
  }>;
};

export type TranslateResult = {
  sentence: string;
  glosses: string[];
  unknown: string[];
  model: string;
  warning?: string;
};

export type TimelineFrame = {
  t: number;
  index: number;
  frame_b64?: string;
};

export type Timeline = {
  fps: number;
  num_frames: number;
  segments: Array<{
    gloss: string;
    exemplar_id: string;
    start_frame: number;
    end_frame: number;
  }>;
  frames: TimelineFrame[];
};

export type LearnVocabResponse = {
  vocab_version: number;
  num_glosses: number;
  glosses_with_data: number;
  glosses: LearnGlossInfo[];
};

export function getLearnVocab() {
  return fetchJson<LearnVocabResponse>("/api/vocab");
}

export function getSign(gloss: string) {
  return fetchJson<SignDetail>(`/api/signs/${encodeURIComponent(gloss)}`);
}

export function frameUrl(gloss: string, exemplarId: string, frameIdx: number) {
  return `${API}/api/signs/${encodeURIComponent(gloss)}/${encodeURIComponent(exemplarId)}/frames/${frameIdx}`;
}

export function translateSentence(sentence: string, useGemini = true) {
  return fetchJson<TranslateResult>("/api/translate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sentence, use_gemini: useGemini }),
  });
}

export function composeGlosses(glosses: string[]) {
  return fetchJson<Timeline>("/api/compose", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ glosses, encode_frames: true }),
  });
}

export function sentenceToTimeline(sentence: string, useGemini = true) {
  return fetchJson<{ translate: TranslateResult; timeline: Timeline }>("/api/sentence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sentence, use_gemini: useGemini }),
  });
}

// --- Orientation coach ---

export type OrientationError = {
  feature: string;
  deviation_deg: number;
  direction: string;
  severity: "low" | "medium" | "high";
};

export type ComparisonResult = {
  sign_id: string;
  overall_result: "pass" | "needs_correction" | "unusable";
  errors: OrientationError[];
  message?: string | null;
  usable_frame_ratio: number;
};

export type OrientationAttempt = {
  timestamp: string;
  overall_result: string;
  error_count: number;
  feedback_text: string;
};

export type OrientationProgress = {
  gloss: string;
  attempts: OrientationAttempt[];
  mastered: boolean;
  attempt_count: number;
};

export type AnalyzeResponse = {
  comparison: ComparisonResult;
  feedback_text: string;
  progress: OrientationProgress;
  display_name: string;
};

export type OrientationReferenceMeta = {
  sign_id: string;
  display_name: string;
  display_name_ml: string | null;
  sign_type: "static" | "dynamic";
  active_hand: "left" | "right";
  critical_features: string[];
  tolerance: Record<string, number>;
  num_reference_frames: number;
};

export function getOrientationReference(gloss: string) {
  return fetchJson<OrientationReferenceMeta>(
    `/api/orientation/reference/${encodeURIComponent(gloss)}`
  );
}

export function getOrientationProgress(gloss: string) {
  return fetchJson<OrientationProgress>(
    `/api/orientation/progress/${encodeURIComponent(gloss)}`
  );
}

export async function analyzeOrientation(gloss: string, video: Blob, useGemma = true) {
  const form = new FormData();
  form.append("gloss", gloss);
  form.append("use_gemma", useGemma ? "true" : "false");
  const name = video instanceof File ? video.name : "practice.webm";
  form.append("video", video, name);
  const res = await fetch(`${API}/api/orientation/analyze`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof err.detail === "string" ? err.detail : JSON.stringify(err));
  }
  return res.json() as Promise<AnalyzeResponse>;
}

// --- Data collection ---

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

export function getCollectionVocab() {
  return fetchJson<{
    num_glosses: number;
    glosses: GlossInfo[];
    total_completed: number;
    total_target: number;
    default_cooldown_sec: number;
    default_ref_countdown_sec: number;
  }>("/api/collection/vocab");
}

export function getEngineStatus() {
  return fetchJson<EngineStatus>("/api/engine/status");
}

export function pauseEngine() {
  return fetchJson<{ ok: boolean }>("/api/engine/pause", { method: "POST" });
}

export function startEngine() {
  return fetchJson<{ ok: boolean; running: boolean }>("/api/engine/start", { method: "POST" });
}

export function stopEngine() {
  return fetchJson<{ ok: boolean; was_running: boolean }>("/api/engine/stop", { method: "POST" });
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

// --- Corpus explore ---

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

// --- Data collection consent ---

export type ConsentStatus = {
  ok: boolean;
  consented: boolean;
  record?: {
    consented: boolean;
    consent_version: number;
    recorded_at: string;
    rgb_not_public: boolean;
    skeleton_may_be_public: boolean;
  } | null;
};

export function getConsentStatus() {
  return fetchJson<ConsentStatus>("/api/consent");
}

export function recordConsent(agreed: boolean, consentVersion = 1) {
  return fetchJson<{ ok: boolean; consented: boolean }>("/api/consent", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ agreed, consent_version: consentVersion }),
  });
}

export function withdrawConsent() {
  return fetchJson<{ ok: boolean; consented: boolean }>("/api/consent", {
    method: "DELETE",
  });
}
