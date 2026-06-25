"""Sentence → ISL glosses via Gemini API."""

from __future__ import annotations

import json
import os
import re
from difflib import get_close_matches
from typing import Any

from dashboard.config import DEFAULT_GEMINI_MODEL, LABEL_MAP_PATH
from dashboard.label_utils import ALIASES, canonical_label, load_label_map, slugify


def _vocab_slugs(label_map_path) -> list[str]:
    label_map, _ = load_label_map(label_map_path)
    return sorted(label_map.keys())


def normalize_gloss_token(token: str, valid: set[str]) -> str | None:
    slug = slugify(token)
    slug = ALIASES.get(slug, slug)
    c = canonical_label(slug, valid)
    if c:
        return c
    return None


def fuzzy_fix(token: str, valid: set[str], cutoff: float = 0.72) -> str | None:
    matches = get_close_matches(token, list(valid), n=1, cutoff=cutoff)
    return matches[0] if matches else None


def build_prompt(sentence: str, vocab: list[str]) -> str:
    vocab_str = ", ".join(vocab)
    return f"""You convert English sentences into Indian Sign Language (ISL) gloss sequences for the INCLUDE-50 lexicon.

Rules:
- Output ONLY gloss tokens from this closed vocabulary (use exact slugs): {vocab_str}
- Use ISL gloss order (not English word order). Time/topic comments often come first.
- One concept per gloss. Split compounds when needed (e.g. thank_you, good_morning, train_ticket).
- Do not invent tokens. If a concept is missing, omit it or use the closest available gloss.
- Return JSON only: {{"glosses": ["gloss_one", "gloss_two"]}}

Sentence: {sentence.strip()}
"""


def translate_with_gemini(
    sentence: str,
    label_map_path=LABEL_MAP_PATH,
    model: str | None = None,
) -> dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY for gloss translation")

    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError("Install google-generativeai: pip install google-generativeai") from exc

    vocab = _vocab_slugs(label_map_path)
    valid = set(vocab)
    genai.configure(api_key=api_key)
    model_name = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    gm = genai.GenerativeModel(
        model_name,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.2,
        },
    )
    response = gm.generate_content(build_prompt(sentence, vocab))
    text = response.text or "{}"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0)) if m else {"glosses": []}

    raw = data.get("glosses") or data.get("gloss") or []
    if isinstance(raw, str):
        raw = [g.strip() for g in raw.replace(",", " ").split() if g.strip()]

    normalized: list[str] = []
    unknown: list[str] = []
    for tok in raw:
        if isinstance(tok, dict):
            tok = tok.get("gloss") or tok.get("word") or ""
        tok = str(tok).strip().lower().replace(" ", "_")
        g = normalize_gloss_token(tok, valid) or fuzzy_fix(tok, valid)
        if g and g not in normalized:
            normalized.append(g)
        elif tok:
            unknown.append(tok)

    return {
        "sentence": sentence,
        "glosses": normalized,
        "unknown": unknown,
        "model": model_name,
        "raw": raw,
    }


def translate_offline_rules(sentence: str, label_map_path=LABEL_MAP_PATH) -> dict[str, Any]:
    """Fallback when no API key: longest-match n-grams against vocab."""
    vocab = _vocab_slugs(label_map_path)
    valid = set(vocab)
    text = sentence.lower().replace("-", " ")
    tokens = re.findall(r"[a-z]+", text)
    normalized: list[str] = []
    unknown: list[str] = []
    i = 0
    while i < len(tokens):
        matched = None
        for size in range(min(4, len(tokens) - i), 0, -1):
            phrase = "_".join(tokens[i : i + size])
            g = normalize_gloss_token(phrase, valid)
            if g is None and size == 1:
                g = fuzzy_fix(phrase, valid)
            if g:
                matched = g
                i += size
                break
        if matched:
            if matched not in normalized:
                normalized.append(matched)
        else:
            unknown.append(tokens[i])
            i += 1
    return {
        "sentence": sentence,
        "glosses": normalized,
        "unknown": unknown,
        "model": "offline",
        "raw": tokens,
    }


def translate_sentence(sentence: str, use_gemini: bool = True, **kwargs) -> dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if use_gemini and api_key:
        try:
            return translate_with_gemini(sentence, **kwargs)
        except Exception as exc:
            return {
                **translate_offline_rules(sentence, kwargs.get("label_map_path", LABEL_MAP_PATH)),
                "warning": f"Gemini failed, used offline matcher: {exc}",
            }
    return translate_offline_rules(sentence, **kwargs)
