"""
End-to-end tests: real TCP socket <-> real server subprocess.

These exercise the FULL path (parse -> dispatch -> LRUCache -> encode), so
they will fail until both cache.resp.parse and cache.lru.LRUCache are
implemented. That's expected — they're the validation for those stubs.
"""
import socket
import time

import pytest


def _send_recv(host, port, payload: bytes, bufsize: int = 65536) -> bytes:
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(payload)
        return sock.recv(bufsize)


def _cmd(*args: str) -> bytes:
    parts = [f"*{len(args)}\r\n".encode()]
    for a in args:
        b = a.encode()
        parts.append(f"${len(b)}\r\n".encode() + b + b"\r\n")
    return b"".join(parts)


def test_ping(server):
    host, port = server
    reply = _send_recv(host, port, _cmd("PING"))
    assert reply == b"+PONG\r\n"


def test_set_then_get(server):
    host, port = server
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(_cmd("SET", "foo", "bar"))
        assert sock.recv(65536) == b"+OK\r\n"
        sock.sendall(_cmd("GET", "foo"))
        assert sock.recv(65536) == b"$3\r\nbar\r\n"


def test_get_missing_key_returns_nil(server):
    host, port = server
    reply = _send_recv(host, port, _cmd("GET", "does-not-exist"))
    assert reply == b"$-1\r\n"


def test_del(server):
    host, port = server
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(_cmd("SET", "k", "v"))
        sock.recv(65536)
        sock.sendall(_cmd("DEL", "k"))
        assert sock.recv(65536) == b":1\r\n"
        sock.sendall(_cmd("GET", "k"))
        assert sock.recv(65536) == b"$-1\r\n"


def test_unknown_command_returns_error(server):
    host, port = server
    reply = _send_recv(host, port, _cmd("NOTACOMMAND"))
    assert reply.startswith(b"-ERR")


def test_pipelined_commands_get_pipelined_replies(server):
    host, port = server
    payload = _cmd("SET", "a", "1") + _cmd("SET", "b", "2") + _cmd("GET", "a") + _cmd("GET", "b")
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(payload)
        time.sleep(0.2)  # let the server process and reply to all four
        reply = sock.recv(65536)
    assert reply == b"+OK\r\n+OK\r\n$1\r\n1\r\n$1\r\n2\r\n"


def test_partial_send_split_mid_command(server):
    """Bytes of a single command arrive in two separate writes."""
    host, port = server
    full = _cmd("SET", "split", "works")
    midpoint = len(full) // 2
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(full[:midpoint])
        time.sleep(0.1)
        sock.sendall(full[midpoint:])
        assert sock.recv(65536) == b"+OK\r\n"
        sock.sendall(_cmd("GET", "split"))
        assert sock.recv(65536) == b"$5\r\nworks\r\n"


def test_partial_send_byte_by_byte(server):
    """Extreme case: one byte per send() call."""
    host, port = server
    full = _cmd("PING")
    with socket.create_connection((host, port), timeout=5) as sock:
        for b in full:
            sock.sendall(bytes([b]))
        assert sock.recv(65536) == b"+PONG\r\n"


def test_oversized_bulk_string_rejected_and_connection_closed(server):
    host, port = server
    huge_len = 8 * 1024 * 1024 + 1
    payload = f"*3\r\n$3\r\nSET\r\n$3\r\nfoo\r\n${huge_len}\r\n".encode()
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(payload)
        reply = sock.recv(65536)
        assert reply.startswith(b"-ERR")
        # server must close the connection after a protocol error
        assert sock.recv(65536) == b""


@pytest.mark.capacity(2)
def test_eviction_under_a_real_connection(server):
    host, port = server
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(_cmd("SET", "a", "1"))
        sock.recv(65536)
        sock.sendall(_cmd("SET", "b", "2"))
        sock.recv(65536)
        sock.sendall(_cmd("SET", "c", "3"))  # capacity 2 -> evicts "a"
        sock.recv(65536)
        sock.sendall(_cmd("GET", "a"))
        assert sock.recv(65536) == b"$-1\r\n"
        sock.sendall(_cmd("GET", "c"))
        assert sock.recv(65536) == b"$1\r\n3\r\n"
