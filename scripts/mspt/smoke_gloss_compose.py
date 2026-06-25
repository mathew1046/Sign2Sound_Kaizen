#!/usr/bin/env python3
"""Smoke test: .env loading + Gemini gloss-compose latency."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "mspt"))

from mspt.gloss_compose import (  # noqa: E402
    compose_rules,
    compose_with_gemini,
    env_status,
    load_project_env,
)

CASES = [
    ["i", "go", "bank"],
    ["hello", "how_are_you"],
    ["i", "want", "water"],
    ["teacher", "good", "morning"],
]


def main() -> int:
    # Simulate fresh process: clear key from env before load (if not already exported)
    status_before = {
        "env_file_exists": (REPO_ROOT / ".env").is_file(),
        "repo_root": str(REPO_ROOT),
    }
    loaded = load_project_env()
    status = env_status()

    print("=== Env smoke test ===")
    print(f"repo_root:          {status['repo_root']}")
    print(f"env_path:           {status['env_path']}")
    print(f".env exists:        {status['env_file_exists']} (pre-check: {status_before['env_file_exists']})")
    print(f"dotenv loaded:      {status['dotenv_loaded']} (load_project_env returned {loaded})")
    print(f"GEMINI_API_KEY set: {status['gemini_key_set']}")
    if status["gemini_key_prefix"]:
        print(f"key prefix:         {status['gemini_key_prefix']}")
    print(f"model:              {status['gemini_model']}")

    if not status["gemini_key_set"]:
        print("\nSkipping Gemini latency (no API key). Rules-only spot check:")
        for glosses in CASES[:2]:
            ruled = compose_rules(glosses)
            print(f"  {glosses!r} -> rules needs_llm={ruled.needs_llm} speak={ruled.speak!r}")
        return 1

    print("\n=== Gemini latency (compose_with_gemini, system_instruction) ===")
    latencies_ms: list[float] = []
    failures = 0
    for i, glosses in enumerate(CASES):
        t0 = time.perf_counter()
        try:
            result = compose_with_gemini(glosses)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            print(
                f"  [{i + 1}] {glosses!r}\n"
                f"       -> {result.speak!r} ({result.source}, {elapsed_ms:.0f} ms)"
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            failures += 1
            err = str(exc).split("\n")[0]
            print(f"  [{i + 1}] {glosses!r} FAILED after {elapsed_ms:.0f} ms: {err}")
            if "429" in str(exc) or "quota" in str(exc).lower():
                print("       (API quota/rate limit — env + auth are OK; retry later or check billing)")
                break

    if not latencies_ms:
        print("\nNo successful Gemini requests (see errors above).")
        return 2 if failures else 0

    if len(latencies_ms) >= 1:
        gaps = [latencies_ms[i] - latencies_ms[i - 1] for i in range(1, len(latencies_ms))]
        print("\n=== Summary ===")
        print(f"requests:     {len(latencies_ms)}")
        print(f"latency min:  {min(latencies_ms):.0f} ms")
        print(f"latency max:  {max(latencies_ms):.0f} ms")
        print(f"latency mean: {statistics.mean(latencies_ms):.0f} ms")
        print(f"latency p50:  {statistics.median(latencies_ms):.0f} ms")
        if len(latencies_ms) >= 3:
            print(f"latency p95:  {sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))]:.0f} ms")
        if gaps:
            print(f"gap mean:     {statistics.mean(gaps):.0f} ms (sequential, not parallel)")

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
