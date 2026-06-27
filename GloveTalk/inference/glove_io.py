"""Shared glove data I/O: TCP Wi-Fi stream and line parsing.

Glove TCP server defaults to port 8080. Use a different host port for video, e.g.
``adb forward tcp:8090 tcp:8080`` → ``http://localhost:8090/video``.
"""

from __future__ import annotations

import socket
import threading
import queue

NUM_RAW_FEATURES = 18
DEFAULT_GLOVE_TCP_PORT = 8080


class TCPLineReceiver:
    """Accept one hardware TCP client at a time and queue incoming lines."""

    def __init__(self, host: str = "0.0.0.0", port: int = DEFAULT_GLOVE_TCP_PORT):
        self.host = host
        self.port = port
        self.lines: queue.Queue[str] = queue.Queue(maxsize=5000)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.stop()

    def readline(self, timeout: float = 0.1) -> str | None:
        try:
            return self.lines.get(timeout=timeout)
        except queue.Empty:
            return None

    def reset_input_buffer(self) -> None:
        while True:
            try:
                self.lines.get_nowait()
            except queue.Empty:
                return

    def _serve(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            server.settimeout(0.5)
            print(f"Hardware input listening on {self.host}:{self.port}")

            while not self._stop.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue

                print(f"Hardware connected from {addr[0]}:{addr[1]}")
                self.reset_input_buffer()
                with conn:
                    conn.settimeout(0.5)
                    pending = b""
                    while not self._stop.is_set():
                        try:
                            chunk = conn.recv(4096)
                        except socket.timeout:
                            continue
                        except OSError:
                            break
                        if not chunk:
                            break

                        pending += chunk
                        while b"\n" in pending:
                            raw_line, pending = pending.split(b"\n", 1)
                            line = raw_line.decode("utf-8", errors="ignore").strip()
                            if not line:
                                continue
                            try:
                                self.lines.put_nowait(line)
                            except queue.Full:
                                self.reset_input_buffer()
                                self.lines.put_nowait(line)

                print("Hardware disconnected; waiting for reconnect...")


class TCPSerial:
    """TCP server that mimics pyserial readline/in_waiting for glove text lines."""

    def __init__(self, port: int = DEFAULT_GLOVE_TCP_PORT):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(("0.0.0.0", port))
        self.server_socket.listen(1)
        self.server_socket.settimeout(1.0)

        self.client_socket = None
        self.buffer = b""
        self._is_connected = False
        self.port = port

        print(f"Waiting for glove to connect on TCP port {port}...")

        self.thread = threading.Thread(target=self._accept_connections, daemon=True)
        self.thread.start()

    def _accept_connections(self) -> None:
        while True:
            try:
                conn, addr = self.server_socket.accept()
                print(f"Glove connected from {addr[0]}")
                self.client_socket = conn
                self.client_socket.settimeout(0.1)
                self._is_connected = True
                self._receive_data()
            except socket.timeout:
                continue

    def _receive_data(self) -> None:
        while self._is_connected:
            try:
                data = self.client_socket.recv(1024)
                if not data:
                    print("Glove disconnected.")
                    self._is_connected = False
                    self.client_socket = None
                    break
                self.buffer += data
            except socket.timeout:
                pass
            except Exception:
                self._is_connected = False

    @property
    def in_waiting(self) -> bool:
        return b"\n" in self.buffer

    def readline(self) -> bytes:
        if b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            return line + b"\n"
        return b""

    def reset_input_buffer(self) -> None:
        self.buffer = b""

    def close(self) -> None:
        self._is_connected = False
        if self.client_socket:
            self.client_socket.close()
        self.server_socket.close()


def parse_glove_line(line: str | bytes) -> list[float] | None:
    try:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="ignore")
        line = line.strip()
        if not line or "|" not in line:
            return None
        parts = line.split("|")
        if len(parts) < 2:
            return None
        l_data = parts[0].strip().split(",")[1:]
        r_data = parts[1].strip().split(",")[1:]
        values = [float(x) for x in l_data] + [float(x) for x in r_data]
        if len(values) != NUM_RAW_FEATURES:
            return None
        return values
    except Exception:
        return None
