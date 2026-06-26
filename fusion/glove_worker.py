"""Background glove prediction feed TCP client.

Consumes newline-delimited JSON prediction events from a remote
GloveTalk live_translator.py server and emits SignToken objects.
"""

from __future__ import annotations

import json
import queue
import socket
import threading
import time

from fusion.tokens import SignToken

DEFAULT_GLOVE_HOST = "10.43.206.118"
DEFAULT_GLOVE_FEED_PORT = 8081


class GloveWorker:
    """TCP client consuming JSON prediction events from a remote GloveTalk feed."""

    def __init__(
        self,
        host: str = DEFAULT_GLOVE_HOST,
        feed_port: int = DEFAULT_GLOVE_FEED_PORT,
        *,
        connect_timeout_sec: float = 15.0,
        reconnect_delay_sec: float = 3.0,
    ):
        self.host = host
        self.feed_port = feed_port
        self.connect_timeout_sec = connect_timeout_sec
        self.reconnect_delay_sec = reconnect_delay_sec

        self._queue: queue.Queue[SignToken] = queue.Queue(maxsize=32)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._available = False
        self._error: str | None = None

    @property
    def available(self) -> bool:
        return self._available

    @property
    def error(self) -> str | None:
        return self._error

    def start(self) -> bool:
        self._thread = threading.Thread(target=self._run, daemon=True, name="glove-worker")
        self._thread.start()
        deadline = time.monotonic() + self.connect_timeout_sec
        while time.monotonic() < deadline:
            if self._available:
                return True
            if self._error:
                return False
            time.sleep(0.05)
        return self._available

    def poll(self) -> list[SignToken]:
        tokens: list[SignToken] = []
        while True:
            try:
                tokens.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return tokens

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def health(self) -> dict:
        return {
            "available": self._available,
            "error": self._error,
            "queue_depth": self._queue.qsize(),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._connect_and_consume()
            except Exception as exc:
                self._error = str(exc)
                self._available = False
                print(f"[glove] feed error: {exc}; reconnecting in {self.reconnect_delay_sec}s")
                if not self._stop.is_set():
                    time.sleep(self.reconnect_delay_sec)

    def _connect_and_consume(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.connect_timeout_sec)
        try:
            sock.connect((self.host, self.feed_port))
        except (socket.timeout, OSError) as exc:
            sock.close()
            self._error = str(exc)
            self._available = False
            return

        sock.settimeout(0.5)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._available = True
        self._error = None
        print(f"[glove] connected to prediction feed at {self.host}:{self.feed_port}")

        pending = b""
        try:
            while not self._stop.is_set():
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break

                if not chunk:
                    print("[glove] feed connection closed by server")
                    break

                pending += chunk
                while b"\n" in pending:
                    raw_line, pending = pending.split(b"\n", 1)
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    self._process_line(line)
        finally:
            sock.close()
            self._available = False

    def _process_line(self, line: str) -> None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return

        if event.get("type") != "prediction":
            return

        stable_label = event.get("stable_label")
        if not stable_label or stable_label == "rest":
            return

        now = time.monotonic()
        token = SignToken(
            gloss=stable_label,
            source="glove",
            confidence=float(event.get("confidence", 0.0)),
            timestamp=now,
            meta={
                "raw_label": event.get("label", ""),
                "raw_confidence": float(event.get("confidence", 0.0)),
                "sequence": event.get("sequence", 0),
                "remote_timestamp": event.get("timestamp", 0.0),
            },
        )
        try:
            self._queue.put_nowait(token)
        except queue.Full:
            pass
