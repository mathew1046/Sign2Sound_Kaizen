"""Standalone glove feed TCP check — no models, no video.

Usage:
    python scripts/fusion/test_glove_feed.py
    python scripts/fusion/test_glove_feed.py --host 10.43.206.118 --port 8081 --duration 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from fusion.glove_worker import DEFAULT_GLOVE_FEED_PORT, DEFAULT_GLOVE_HOST, GloveWorker


def main() -> int:
    ap = argparse.ArgumentParser(description="Glove feed TCP liveness check")
    ap.add_argument("--host", type=str, default=DEFAULT_GLOVE_HOST,
                    help=f"GloveTalk feed host (default: {DEFAULT_GLOVE_HOST})")
    ap.add_argument("--port", type=int, default=DEFAULT_GLOVE_FEED_PORT,
                    help=f"GloveTalk feed port (default: {DEFAULT_GLOVE_FEED_PORT})")
    ap.add_argument("--connect-timeout", type=float, default=15.0,
                    help="TCP connect timeout in seconds (default: 15)")
    ap.add_argument("--duration", type=float, default=30.0,
                    help="How long to listen for tokens in seconds (default: 30)")
    args = ap.parse_args()

    glove = GloveWorker(
        host=args.host,
        feed_port=args.port,
        connect_timeout_sec=args.connect_timeout,
    )
    print(f"[test] connecting to glove feed {args.host}:{args.port} ...")
    if not glove.start():
        print(f"[test] FAILED to connect: {glove.error}")
        glove.close()
        return 1

    print(f"[test] connected OK — health={glove.health()}")
    token_count = 0
    deadline = time.monotonic() + args.duration
    try:
        while time.monotonic() < deadline:
            tokens = glove.poll()
            for t in tokens:
                token_count += 1
                print(f"[test] token #{token_count}: gloss={t.gloss!r} "
                      f"conf={t.confidence:.3f} meta={t.meta}")
            time.sleep(0.05)
    finally:
        print(f"[test] done — received {token_count} token(s), "
              f"final health={glove.health()}")
        glove.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())