"""Gemma 4 API feedback for orientation coaching."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from dashboard.config import DEFAULT_GEMMA_MODEL, ORIENTATION_FEEDBACK_CACHE_DIR
from dashboard.server.orientation.schemas import ComparisonResult, OrientationError


def _error_hash(sign_id: str, errors: list[OrientationError]) -> str:
    payload = json.dumps(
        {"sign_id": sign_id, "errors": [e.model_dump() for e in errors]},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cache_path(sign_id: str, err_hash: str) -> Path:
    return ORIENTATION_FEEDBACK_CACHE_DIR / sign_id / f"{err_hash}.json"


def _read_cache(sign_id: str, err_hash: str) -> str | None:
    p = _cache_path(sign_id, err_hash)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("feedback_text")
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(sign_id: str, err_hash: str, text: str) -> None:
    p = _cache_path(sign_id, err_hash)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"feedback_text": text}, indent=2), encoding="utf-8")


def template_feedback(display_name: str, comparison: ComparisonResult) -> str:
    if comparison.overall_result == "unusable":
        return comparison.message or "Reposition your hand and upper body so the camera can see them clearly, then try again."

    if comparison.overall_result == "pass":
        return f"Great work on “{display_name}”! Your hand orientation and shape look within the expected range. Keep practicing for consistency."

    parts = [f"For “{display_name}”, try adjusting:"]
    for err in comparison.errors[:4]:
        feat = err.feature.replace("_", " ").replace("finger curl", "finger")
        parts.append(f"• {feat}: {err.direction} ({err.severity} priority).")
    if len(comparison.errors) > 4:
        parts.append(f"• Plus {len(comparison.errors) - 4} more small adjustments.")
    parts.append("You're close — small corrections make a big difference.")
    return " ".join(parts)


def build_gemma_prompt(display_name: str, comparison: ComparisonResult) -> str:
    errors_json = json.dumps([e.model_dump() for e in comparison.errors], indent=2)
    return f"""You are a friendly Indian Sign Language (ISL) orientation coach.

The learner practiced the sign: {display_name}
Overall result: {comparison.overall_result}

Structured errors (geometry-based, trust these):
{errors_json}

Write 1-2 short, encouraging sentences that:
- Name the body part and what to change (direction field)
- Stay brief and supportive
- Do not invent errors not in the list
- If result is pass, congratulate them briefly

Return plain text only, no JSON."""


def feedback_with_gemma(
    display_name: str,
    comparison: ComparisonResult,
    *,
    model: str | None = None,
    use_api: bool = True,
) -> str:
    if comparison.overall_result == "unusable":
        return template_feedback(display_name, comparison)

    if not comparison.errors and comparison.overall_result == "pass":
        return template_feedback(display_name, comparison)

    err_hash = _error_hash(comparison.sign_id, comparison.errors)
    cached = _read_cache(comparison.sign_id, err_hash)
    if cached:
        return cached

    if not use_api:
        text = template_feedback(display_name, comparison)
        _write_cache(comparison.sign_id, err_hash, text)
        return text

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return template_feedback(display_name, comparison)

    model_name = model or os.environ.get("GEMMA_MODEL", DEFAULT_GEMMA_MODEL)
    prompt = build_gemma_prompt(display_name, comparison)

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        text = (response.text or "").strip()
        if not text:
            text = template_feedback(display_name, comparison)
    except Exception:
        text = template_feedback(display_name, comparison)

    _write_cache(comparison.sign_id, err_hash, text)
    return text
