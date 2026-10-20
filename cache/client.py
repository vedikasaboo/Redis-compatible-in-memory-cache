"""
Minimal blocking client.

Deliberately small — Phase 3 grows this into the real client library and
Phase 4 adds consistent-hash routing on top. It exists now because the
crash-test rig needs to issue thousands of writes per run, which is far
too slow through `redis-cli` subprocesses.

The request side reuses resp.encode(); no second encoder. The reply side
needs a decoder that resp.parse() does not provide — parse() decodes
REQUESTS (always an array of bulk strings), while replies can be any of
the five RESP types, so read_reply() below handles those.
"""
import socket
import ssl

from cache.resp import ProtocolError, encode


class CacheError(Exception):
    """An error reply (RESP `-`) returned by the server."""


class Client:
    def __init__(self, host: str = "127.0.0.1", port: int = 6390, timeout: float = 5.0,
                 password: str | None = None, tls: "ssl.SSLContext | bool | None" = None,
                 name: str | None = None):
        """
        password — sent with AUTH right after connecting.
        tls      — True for a default verifying context, or an SSLContext
                   (e.g. one trusting a self-signed CA via load_verify_locations).
        name     — sent with CLIENT SETNAME, so the server's logs and the
                   guard can tell connections apart.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.password = password
        self.tls = ssl.create_default_context() if tls is True else tls
        self.name = name
        self._sock = None
        self._f = None

    def connect(self) -> "Client":
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        if self.tls:
            self._sock = self.tls.wrap_socket(self._sock, server_hostname=self.host)
        # A buffered file object gives readline()/readexactly-style reads,
        # which is what makes reply framing straightforward here.
        self._f = self._sock.makefile("rb")
        if self.password is not None:
            self.command("AUTH", self.password)
        if self.name:
            self.command("CLIENT", "SETNAME", self.name)
        return self

    def close(self) -> None:
        for closeable in (self._f, self._sock):
            if closeable is not None:
                try:
                    closeable.close()
                except OSError:
                    pass
        self._f = self._sock = None

    def __enter__(self) -> "Client":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- protocol --------------------------------------------------------

    def _read_line(self) -> bytes:
        line = self._f.readline()
        if not line:
            raise ConnectionError("server closed the connection")
        if not line.endswith(b"\r\n"):
            raise ProtocolError("reply line not terminated by CRLF")
        return line[:-2]

    def read_reply(self):
        """Decode one reply. Returns str / int / bytes / None / list."""
        line = self._read_line()
        kind, rest = line[:1], line[1:]
        if kind == b"+":
            return rest.decode()
        if kind == b"-":
            raise CacheError(rest.decode())
        if kind == b":":
            return int(rest)
        if kind == b"$":
            length = int(rest)
            if length == -1:
                return None
            payload = self._f.read(length + 2)  # payload plus its CRLF
            if len(payload) < length + 2:
                raise ConnectionError("truncated bulk string reply")
            return payload[:-2]
        if kind == b"*":
            count = int(rest)
            if count == -1:
                return None
            return [self.read_reply() for _ in range(count)]
        raise ProtocolError(f"unknown reply type {kind!r}")

    def send(self, *args) -> None:
        """Write a command without reading its reply (for pipelining)."""
        parts = [a if isinstance(a, bytes) else str(a).encode() for a in args]
        self._sock.sendall(encode(parts))

    def command(self, *args):
        self.send(*args)
        return self.read_reply()

    def pipeline(self, commands: list[tuple]) -> list:
        """
        Send every command in ONE write, then collect all replies.

        This is the throughput lever measured in NOTES.md: it pays a single
        network round trip for the whole batch instead of one per command.
        Errors are returned in place rather than raised, so one failing
        command does not discard the rest of the batch's replies.
        """
        payload = b"".join(
            encode([a if isinstance(a, bytes) else str(a).encode() for a in cmd])
            for cmd in commands
        )
        self._sock.sendall(payload)
        replies = []
        for _ in commands:
            try:
                replies.append(self.read_reply())
            except CacheError as exc:
                replies.append(exc)
        return replies

    # -- conveniences ----------------------------------------------------

    def ping(self):
        return self.command("PING")

    def get(self, key):
        return self.command("GET", key)

    def set(self, key, value, ex=None, px=None, exat=None, pxat=None):
        for name, value_ in (("EX", ex), ("PX", px), ("EXAT", exat), ("PXAT", pxat)):
            if value_ is not None:
                return self.command("SET", key, value, name, value_)
        return self.command("SET", key, value)

    def delete(self, *keys):
        return self.command("DEL", *keys)

    def exists(self, *keys):
        return self.command("EXISTS", *keys)

    def ttl(self, key):
        """Seconds remaining; -1 if no expiry, -2 if the key is gone."""
        return self.command("TTL", key)

    def pttl(self, key):
        """Milliseconds remaining; -1 if no expiry, -2 if the key is gone."""
        return self.command("PTTL", key)
