import socket
import time


class TCPSerial:
    def __init__(self, port=8080, host="0.0.0.0", timeout=0.1):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.settimeout(timeout)
        self._socket.bind((host, port))
        self._socket.listen(1)
        self._conn, self._addr = self._socket.accept()
        self._conn.settimeout(timeout)
        self.in_waiting = 0

    def readline(self):
        data = b""
        while not data.endswith(b"\n"):
            chunk = self._conn.recv(1)
            if not chunk:
                break
            data += chunk
        self.in_waiting = 0
        return data

    def reset_input_buffer(self):
        self.in_waiting = 0

    def close(self):
        if self._conn:
            self._conn.close()
        if self._socket:
            self._socket.close()
