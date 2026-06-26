"""ISL gloss sequences → natural English for live MSPT TTS."""

from __future__ import annotations

import csv
import json
import os
import queue
import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"
DEFAULT_VOCAB_CSV = REPO_ROOT / "scripts" / "mspt" / "include50_mspt_and_include263_vocabulary.csv"

GEMINI_SYSTEM_INSTRUCTION = """You are a real-time Indian Sign Language (ISL) to spoken-English converter for the Sign2Sound accessibility pipeline.

Your job: convert a short sequence of ISL gloss tokens from continuous sign recognition into exactly one natural English sentence for text-to-speech.

Context:
- Glosses come from the INCLUDE-263 lexicon used by an MSPT pose-based sign classifier.
- Each gloss is one signed concept. Underscores mark multi-word glosses (e.g. good_morning, thank_you, train_ticket).
- Gloss order follows ISL signing order, which may differ from English (time/topic comments may come first).

Hard constraints:
1. Use ONLY meaning present in the input gloss list. Never invent topics, names, places, or events.
2. Preserve every gloss's meaning. Do not drop glosses unless they are exact consecutive duplicates.
3. Never split compound gloss tokens (good_morning stays "good morning"; thank_you stays "thank you").
4. Make minimal edits: insert copulas (am/is/are), articles (a/the), prepositions (to/at/in), and lightly reorder for natural English.
5. One short speakable sentence only (typically 3–12 words).
6. If glosses already form natural English (e.g. ["thank_you"]), return them with light punctuation/casing only.
7. Use sentence case. End with . ! or ?

Output: JSON only, no markdown, no extra keys:
{"speak": "<one English sentence>", "changed": <true if edited beyond punctuation/casing>}

Examples:
- ["i", "happy"] → {"speak": "I am happy.", "changed": true}
- ["good_morning"] → {"speak": "Good morning.", "changed": false}
- ["i", "go", "bank"] → {"speak": "I am going to the bank.", "changed": true}
- ["thank_you"] → {"speak": "Thank you.", "changed": false}
- ["hello", "how_are_you"] → {"speak": "Hello, how are you?", "changed": true}
"""

_env_loaded = False

PRONOUNS = frozenset({"i", "you", "he", "she", "it", "we", "they", "you_plural"})
PRONOUN_COPULA = {
    "i": "am",
    "you": "are",
    "he": "is",
    "she": "is",
    "it": "is",
    "we": "are",
    "they": "are",
    "you_plural": "are",
}

# Common ISL verbs for rule-based grammar
VERBS = frozenset({
    "go", "want", "need", "like", "eat", "drink", "sleep", "come",
    "give", "take", "help", "know", "think", "see", "hear", "learn",
    "teach", "play", "work", "buy", "make", "sit", "stand", "walk",
    "run", "read", "write", "speak", "tell", "ask", "live",
})

# Greetings that can open a phrase
GREETINGS = frozenset({
    "hello", "good_morning", "good_night", "good_afternoon", "good_evening",
    "how_are_you", "namaste",
})


def _verb_form(verb: str, subject: str) -> str:
    """Basic English verb conjugation for present continuous / simple present."""
    v = verb.replace("_", " ").lower()
    copula = PRONOUN_COPULA.get(subject.lower(), "is")
    # Use present continuous for action verbs
    if v.endswith("e"):
        v_ing = v[:-1] + "ing"
    elif len(v) >= 3 and v[-1] not in "aeiou" and v[-2] in "aeiou" and v[-3] not in "aeiou":
        v_ing = v + v[-1] + "ing"
    else:
        v_ing = v + "ing"
    return f"{copula} {v_ing}"


def _needs_article(noun: str) -> str:
    """Prepend 'the' to common nouns (not pronouns, not compounds)."""
    n = noun.replace("_", " ").lower()
    if n in PRONOUNS or "_" in noun:
        return n
    if n[0] in "aeiou":
        return f"an {n}"
    return f"the {n}"


ComposeSource = Literal["rules", "gemini", "fallback_join"]


@dataclass(frozen=True)
class ComposeResult:
    speak: str
    source: ComposeSource
    changed: bool
    needs_llm: bool = False


def load_project_env(*, force: bool = False) -> bool:
    """Load repo-root ``.env`` into ``os.environ`` (idempotent)."""
    global _env_loaded
    if _env_loaded and not force:
        return DEFAULT_ENV_PATH.is_file()
    if not DEFAULT_ENV_PATH.is_file():
        _env_loaded = True
        return False
    try:
        from dotenv import load_dotenv

        load_dotenv(DEFAULT_ENV_PATH, override=False)
        _env_loaded = True
        return True
    except ImportError:
        _env_loaded = True
        return False


def ensure_project_env(env_path: Path | None = None) -> Path | None:
    """Load repo ``.env`` once; return path when the file exists."""
    path = env_path or DEFAULT_ENV_PATH
    if path != DEFAULT_ENV_PATH:
        if path.is_file():
            try:
                from dotenv import load_dotenv

                load_dotenv(path, override=False)
            except ImportError:
                pass
        return path if path.is_file() else None
    load_project_env()
    return path if path.is_file() else None


def env_status() -> dict[str, object]:
    """Diagnostic snapshot for smoke tests (never exposes full secrets)."""
    load_project_env()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return {
        "repo_root": str(REPO_ROOT),
        "env_path": str(DEFAULT_ENV_PATH),
        "env_file_exists": DEFAULT_ENV_PATH.is_file(),
        "dotenv_loaded": _env_loaded,
        "gemini_key_set": bool(key),
        "gemini_key_prefix": (key[:8] + "…") if key and len(key) >= 8 else None,
        "gemini_model": os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
    }


def display_gloss(label: str) -> str:
    """Title-case label for on-screen display."""
    return label.replace("_", " ").title()


def gloss_to_speech_text(label: str, *, sentence_start: bool = False) -> str:
    """Natural casing for TTS (sentence-initial cap only)."""
    text = label.replace("_", " ").lower()
    if not text:
        return text
    if sentence_start:
        if text == "i":
            return "I"
        return text[0].upper() + text[1:]
    if text == "i":
        return "I"
    return text


def ensure_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text[-1] not in ".!?":
        text += "."
    return text


@lru_cache(maxsize=1)
def _load_vocab_meta(vocab_csv: str) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    path = Path(vocab_csv)
    slugs: set[str] = set()
    adjectives: set[str] = set()
    compounds: set[str] = set()
    if not path.is_file():
        return frozenset(), frozenset(), frozenset()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gloss = row["canonical_gloss"].strip().lower()
            slugs.add(gloss)
            category = (row.get("category") or "").strip()
            if category.startswith("Adjectives") or category == "Colours":
                adjectives.add(gloss)
            if "_" in gloss:
                compounds.add(gloss)
    return frozenset(slugs), frozenset(adjectives), frozenset(compounds)


def vocab_slugs(vocab_csv: Path | None = None) -> list[str]:
    path = str(vocab_csv or DEFAULT_VOCAB_CSV)
    slugs, _, _ = _load_vocab_meta(path)
    return sorted(slugs)


def is_adjective_gloss(gloss: str, vocab_csv: Path | None = None) -> bool:
    _, adjectives, _ = _load_vocab_meta(str(vocab_csv or DEFAULT_VOCAB_CSV))
    return gloss.lower() in adjectives


def is_compound_gloss(gloss: str, vocab_csv: Path | None = None) -> bool:
    g = gloss.lower()
    if "_" in g:
        return True
    _, _, compounds = _load_vocab_meta(str(vocab_csv or DEFAULT_VOCAB_CSV))
    return g in compounds


def deduplicate_glosses(glosses: list[str]) -> list[str]:
    """Remove consecutive duplicate glosses.

    e.g. ["hello", "hello", "hello", "i", "happy"] → ["hello", "i", "happy"]
    """
    if not glosses:
        return []
    result = [glosses[0]]
    for g in glosses[1:]:
        if g.lower() != result[-1].lower():
            result.append(g)
    return result


def fallback_join(glosses: list[str]) -> ComposeResult:
    if not glosses:
        return ComposeResult(speak="", source="fallback_join", changed=False)
    parts: list[str] = []
    for i, g in enumerate(glosses):
        parts.append(gloss_to_speech_text(g, sentence_start=(i == 0)))
    text = " ".join(parts)
    return ComposeResult(
        speak=ensure_sentence(text),
        source="fallback_join",
        changed=len(glosses) > 1,
    )


def compose_rules(glosses: list[str], vocab_csv: Path | None = None) -> ComposeResult:
    """Rule-based ISL gloss → English sentence composition.

    Handles common ISL patterns without needing the Gemini LLM:
    - Single gloss: direct conversion
    - Pronoun + adjective: "I happy" → "I am happy."
    - Pronoun + verb: "I go" → "I am going."
    - Greeting + pronoun + adjective: "hello I happy" → "Hello, I am happy."
    - Pronoun + verb + noun: "I go bank" → "I am going to the bank."
    - 4+ glosses or unrecognized patterns → fall through to LLM
    """
    if not glosses:
        return ComposeResult(speak="", source="rules", changed=False)

    # Deduplicate consecutive repeats first
    glosses = deduplicate_glosses(glosses)

    if len(glosses) == 1:
        g = glosses[0].lower()
        return ComposeResult(
            speak=ensure_sentence(gloss_to_speech_text(g, sentence_start=True)),
            source="rules",
            changed=False,
        )

    if len(glosses) == 2:
        a, b = glosses[0].lower(), glosses[1].lower()

        # Pronoun + adjective: "I happy" → "I am happy."
        if a in PRONOUNS and is_adjective_gloss(b, vocab_csv):
            copula = PRONOUN_COPULA[a]
            return ComposeResult(
                speak=ensure_sentence(
                    f"{gloss_to_speech_text(a, sentence_start=True)} {copula} {gloss_to_speech_text(b)}"
                ),
                source="rules",
                changed=True,
            )

        # Pronoun + verb: "I go" → "I am going."
        if a in PRONOUNS and b in VERBS:
            return ComposeResult(
                speak=ensure_sentence(
                    f"{gloss_to_speech_text(a, sentence_start=True)} {_verb_form(b, a)}"
                ),
                source="rules",
                changed=True,
            )

        # Greeting + pronoun/greeting: "hello good_morning" → "Hello, good morning."
        if a in GREETINGS and b in GREETINGS:
            return ComposeResult(
                speak=ensure_sentence(
                    f"{gloss_to_speech_text(a, sentence_start=True)}, {gloss_to_speech_text(b)}"
                ),
                source="rules",
                changed=True,
            )

        # Greeting + noun/word: "hello teacher" → "Hello, teacher."
        if a in GREETINGS:
            return ComposeResult(
                speak=ensure_sentence(
                    f"{gloss_to_speech_text(a, sentence_start=True)}, {gloss_to_speech_text(b)}"
                ),
                source="rules",
                changed=True,
            )

        # Pronoun + noun: "i teacher" → "I am a teacher."
        if a in PRONOUNS and b not in VERBS and not is_adjective_gloss(b, vocab_csv):
            copula = PRONOUN_COPULA[a]
            noun_phrase = _needs_article(b)
            return ComposeResult(
                speak=ensure_sentence(
                    f"{gloss_to_speech_text(a, sentence_start=True)} {copula} {noun_phrase}"
                ),
                source="rules",
                changed=True,
            )

        # Greeting + pronoun: let LLM handle ("hello I" needs more context)

    if len(glosses) == 3:
        a, b, c = glosses[0].lower(), glosses[1].lower(), glosses[2].lower()

        # Greeting + pronoun + adjective: "hello I happy" → "Hello, I am happy."
        if a in GREETINGS and b in PRONOUNS and is_adjective_gloss(c, vocab_csv):
            copula = PRONOUN_COPULA[b]
            return ComposeResult(
                speak=ensure_sentence(
                    f"{gloss_to_speech_text(a, sentence_start=True)}, "
                    f"{gloss_to_speech_text(b)} {copula} {gloss_to_speech_text(c)}"
                ),
                source="rules",
                changed=True,
            )

        # Pronoun + verb + noun (SVO): "I go house" → "I am going to the house."
        if a in PRONOUNS and b in VERBS and c not in PRONOUNS:
            noun_phrase = _needs_article(c) if not is_compound_gloss(c) else gloss_to_speech_text(c)
            return ComposeResult(
                speak=ensure_sentence(
                    f"{gloss_to_speech_text(a, sentence_start=True)} {_verb_form(b, a)} to {noun_phrase}"
                ),
                source="rules",
                changed=True,
            )

        # Pronoun + noun + verb (SOV): "I house go" → "I am going to the house."
        if a in PRONOUNS and b not in PRONOUNS and b not in VERBS and c in VERBS:
            noun_phrase = _needs_article(b) if not is_compound_gloss(b) else gloss_to_speech_text(b)
            return ComposeResult(
                speak=ensure_sentence(
                    f"{gloss_to_speech_text(a, sentence_start=True)} {_verb_form(c, a)} to {noun_phrase}"
                ),
                source="rules",
                changed=True,
            )

        # Pronoun + verb + adjective: "I feel happy" → "I am feeling happy."
        if a in PRONOUNS and b in VERBS and is_adjective_gloss(c, vocab_csv):
            return ComposeResult(
                speak=ensure_sentence(
                    f"{gloss_to_speech_text(a, sentence_start=True)} {_verb_form(b, a)} {gloss_to_speech_text(c)}"
                ),
                source="rules",
                changed=True,
            )

        # Greeting + pronoun + verb: "hello I go" → "Hello, I am going."
        if a in GREETINGS and b in PRONOUNS and c in VERBS:
            return ComposeResult(
                speak=ensure_sentence(
                    f"{gloss_to_speech_text(a, sentence_start=True)}, "
                    f"{gloss_to_speech_text(b)} {_verb_form(c, b)}"
                ),
                source="rules",
                changed=True,
            )

    return ComposeResult(speak="", source="rules", changed=False, needs_llm=True)


def build_compose_user_message(glosses: list[str]) -> str:
    return json.dumps({"glosses": glosses}, ensure_ascii=False)


def build_compose_prompt(glosses: list[str], vocab: list[str]) -> str:
    """Legacy combined prompt; prefer system_instruction + user message."""
    gloss_json = json.dumps(glosses)
    return (
        f"{GEMINI_SYSTEM_INSTRUCTION}\n\n"
        f"Input glosses (signing order): {gloss_json}\n"
        f"Vocabulary reference ({len(vocab)} slugs): comma-separated lexicon tokens from INCLUDE-263."
    )


def compose_with_gemini(
    glosses: list[str],
    vocab_csv: Path | None = None,
    model: str | None = None,
) -> ComposeResult:
    load_project_env()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY for gloss composition")

    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError("Install google-generativeai: pip install google-generativeai") from exc

    genai.configure(api_key=api_key)
    model_name = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    gm = genai.GenerativeModel(
        model_name,
        system_instruction=GEMINI_SYSTEM_INSTRUCTION,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.1,
        },
    )
    response = gm.generate_content(build_compose_user_message(glosses))
    text = response.text or "{}"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0)) if m else {"speak": ""}

    speak = str(data.get("speak") or "").strip()
    if not speak:
        raise ValueError("Gemini returned empty speak field")
    changed = bool(data.get("changed", True))
    return ComposeResult(
        speak=ensure_sentence(speak),
        source="gemini",
        changed=changed,
    )


def compose_glosses(
    glosses: list[str],
    *,
    use_gemini: bool = True,
    vocab_csv: Path | None = None,
    model: str | None = None,
) -> ComposeResult:
    if not glosses:
        return ComposeResult(speak="", source="rules", changed=False)

    ruled = compose_rules(glosses, vocab_csv)
    if not ruled.needs_llm:
        return ruled

    load_project_env()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if use_gemini and api_key:
        try:
            return compose_with_gemini(glosses, vocab_csv=vocab_csv, model=model)
        except Exception:
            return fallback_join(glosses)

    return fallback_join(glosses)


class AsyncComposeWorker:
    """Background gloss→sentence worker; invokes callback with ComposeResult."""

    def __init__(
        self,
        on_result: Callable[[ComposeResult, list[str]], None],
        *,
        use_gemini: bool = True,
        vocab_csv: Path | None = None,
    ) -> None:
        self._on_result = on_result
        self._use_gemini = use_gemini
        self._vocab_csv = vocab_csv
        self._queue: queue.Queue[list[str] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gloss-compose")
        self._thread.start()

    def _run(self) -> None:
        while True:
            glosses = self._queue.get()
            if glosses is None:
                break
            try:
                result = compose_glosses(
                    glosses,
                    use_gemini=self._use_gemini,
                    vocab_csv=self._vocab_csv,
                )
            except Exception:
                result = fallback_join(glosses)
            self._on_result(result, glosses)

    def submit(self, glosses: list[str]) -> None:
        if glosses:
            self._queue.put(list(glosses))

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=10.0)


class GlossComposer:
    """Buffers MSPT gloss predictions and flushes composed English on utterance pause."""

    def __init__(
        self,
        *,
        utterance_pause_sec: float = 2.5,
        max_buffer: int = 8,
        early_flush_min: int = 3,
        use_gemini: bool = True,
        vocab_csv: Path | None = None,
        on_composed: Callable[[ComposeResult], None] | None = None,
        speak_enabled: bool = True,
    ) -> None:
        self.utterance_pause_sec = utterance_pause_sec
        self.max_buffer = max_buffer
        self.early_flush_min = early_flush_min
        self.use_gemini = use_gemini
        self.speak_enabled = speak_enabled
        self._buffer: list[str] = []
        self._last_gloss_time = 0.0
        self._last_spoken = ""
        self._state: Literal["idle", "buffering", "composing"] = "idle"
        self._on_composed = on_composed
        self._worker: AsyncComposeWorker | None = None
        if on_composed is not None:
            self._worker = AsyncComposeWorker(
                self._handle_compose_result,
                use_gemini=use_gemini,
                vocab_csv=vocab_csv,
            )

    def _handle_compose_result(self, result: ComposeResult, _glosses: list[str]) -> None:
        self._state = "idle"
        if result.speak:
            self._last_spoken = result.speak
            if self._on_composed is not None:
                self._on_composed(result)

    @property
    def buffer(self) -> list[str]:
        return list(self._buffer)

    @property
    def pending_display(self) -> str:
        if not self._buffer:
            return ""
        return " · ".join(display_gloss(g) for g in self._buffer)

    @property
    def last_spoken(self) -> str:
        return self._last_spoken

    @property
    def state(self) -> str:
        return self._state

    def add_gloss(self, gloss: str, confidence: float, now: float) -> None:
        if not gloss or gloss == "uncertain":
            return
        # Suppress consecutive duplicate glosses
        if self._buffer and gloss.lower() == self._buffer[-1].lower():
            return
        self._buffer.append(gloss)
        self._last_gloss_time = now
        self._state = "buffering"
        if len(self._buffer) >= self.max_buffer:
            self.flush()

    def tick(self, now: float) -> None:
        if not self._buffer or self._state == "composing":
            return
        if now - self._last_gloss_time >= self.utterance_pause_sec:
            self.flush()

    def flush(self) -> None:
        if not self._buffer or self._worker is None:
            if not self._buffer:
                return
            glosses = list(self._buffer)
            self._buffer.clear()
            self._state = "composing"
            result = compose_glosses(glosses, use_gemini=self.use_gemini)
            self._handle_compose_result(result, glosses)
            return

        glosses = list(self._buffer)
        self._buffer.clear()
        self._state = "composing"
        self._worker.submit(glosses)

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()


def gemini_available() -> bool:
    load_project_env()
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
