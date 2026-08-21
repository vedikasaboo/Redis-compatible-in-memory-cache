"""
Hardening tests: misbehaving clients must not stall well-behaved ones and
must not take the server down.

Every test here follows the same shape — start a pathological client, then
measure a NORMAL client's latency while it misbehaves. The assertion is
about the healthy client, because "the server survived" is not the
interesting property; "the server stayed responsive to everyone else" is.
"""
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cache.client import CacheError, Client  # noqa: E402

# Generous enough not to be flaky on a loaded machine, tight enough that a
# genuine head-of-line stall (which would be seconds) still fails.
MAX_ACCEPTABLE_LATENCY = 0.5


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(**extra) -> tuple[subprocess.Popen, int]:
    port = _free_port()
    cmd = [sys.executable, "-m", "cache.server", "--port", str(port), "--capacity", "10000"]
    for key, value in extra.items():
        cmd += [f"--{key.replace('_', '-')}", str(value)]
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return proc, port
        except OSError:
            time.sleep(0.02)
    proc.kill()
    raise TimeoutError("server did not start")


@pytest.fixture
def server_proc():
    started = []

    def _start(**extra):
        proc, port = _start_server(**extra)
        started.append(proc)
        return port

    yield _start
    for proc in started:
        proc.kill()
        proc.wait()


def worst_latency(port: int, samples: int = 20, gap: float = 0.02) -> float:
    """Worst round-trip a healthy client sees, in seconds."""
    worst = 0.0
    with Client(port=port, timeout=5.0) as c:
        for _ in range(samples):
            started = time.time()
            assert c.ping() == "PONG"
            worst = max(worst, time.time() - started)
            time.sleep(gap)
    return worst


def test_slow_client_one_byte_per_second_does_not_stall_others(server_proc):
    """
    A client dribbling a single command one byte per second holds an
    incomplete command in the server's buffer the whole time. Because the
    handler awaits on read() rather than spinning, that costs one idle
    coroutine and nothing else.
    """
    port = server_proc()
    slow = socket.create_connection(("127.0.0.1", port), timeout=10)
    payload = b"*1\r\n$4\r\nPING\r\n"
    try:
        for byte in payload[:3]:
            slow.sendall(bytes([byte]))
            time.sleep(1.0)  # genuinely one byte per second
            assert worst_latency(port, samples=3) < MAX_ACCEPTABLE_LATENCY
        # finish the command; the slow client's own request still works
        slow.sendall(payload[3:])
        slow.settimeout(5)
        assert slow.recv(100) == b"+PONG\r\n"
    finally:
        slow.close()

    assert worst_latency(port) < MAX_ACCEPTABLE_LATENCY


def test_malformed_resp_is_rejected_and_isolated(server_proc):
    port = server_proc()
    bad = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        bad.sendall(b"+not a request\r\n")  # a reply type, invalid as a request
        reply = bad.recv(1000)
        assert reply.startswith(b"-ERR Protocol error")
        assert bad.recv(1000) == b""  # server closed this connection
    finally:
        bad.close()

    assert worst_latency(port) < MAX_ACCEPTABLE_LATENCY


def test_garbage_bytes_do_not_crash_server(server_proc):
    port = server_proc()
    for junk in (b"\x00\x01\x02\x03", b"GET foo\r\n", b"*abc\r\n", b"*2\r\n$xx\r\n"):
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        try:
            s.sendall(junk)
            s.recv(1000)  # error reply or close; either is acceptable
        finally:
            s.close()
    assert worst_latency(port) < MAX_ACCEPTABLE_LATENCY


def test_client_that_never_reads_replies_does_not_stall_others(server_proc):
    """
    The classic head-of-line hazard. The server writes replies this client
    never drains; once the socket buffers fill, `await writer.drain()`
    suspends THAT handler only. If drain() were not awaited — or if writes
    were unbounded — this client would either block the loop or grow the
    server's memory without limit.
    """
    port = server_proc()
    silent = socket.create_connection(("127.0.0.1", port), timeout=5)
    silent.setblocking(False)
    payload = b"".join(
        b"*3\r\n$3\r\nSET\r\n$6\r\nk%05d\r\n$100\r\n" % i + b"v" * 100 + b"\r\n"
        for i in range(500)
    )
    try:
        # Push until the kernel refuses more; never read a single reply.
        for _ in range(200):
            try:
                silent.sendall(payload)
            except (BlockingIOError, OSError):
                break
        assert worst_latency(port) < MAX_ACCEPTABLE_LATENCY
    finally:
        silent.close()

    assert worst_latency(port) < MAX_ACCEPTABLE_LATENCY


def test_many_idle_connections_do_not_stall_others(server_proc):
    port = server_proc()
    idle = [socket.create_connection(("127.0.0.1", port), timeout=5) for _ in range(200)]
    try:
        assert worst_latency(port) < MAX_ACCEPTABLE_LATENCY
    finally:
        for s in idle:
            s.close()


def test_oversized_multibulk_header_is_rejected(server_proc):
    """
    Without a cap on element count, `*999999999\\r\\n` followed by a slow
    trickle of arguments would make the server hold a partial command of
    unbounded size while waiting for an end that never arrives.
    """
    port = server_proc()
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        s.sendall(b"*999999999\r\n")
        reply = s.recv(1000)
        assert reply.startswith(b"-ERR Protocol error")
    finally:
        s.close()
    assert worst_latency(port) < MAX_ACCEPTABLE_LATENCY


def test_query_buffer_limit_disconnects_a_buffer_hog(server_proc):
    """
    An incomplete command must not be allowed to pin unbounded memory. The
    server is started with a small limit here so the test stays fast; the
    default is 32MB.
    """
    limit = 1_000_000
    port = server_proc(max_query_buffer=limit)
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        # Declares 8MB of payload it will never finish delivering, so the
        # bytes below can only ever sit in the server's unparsed buffer.
        s.sendall(b"*3\r\n$3\r\nSET\r\n$3\r\nfoo\r\n$8000000\r\n")
        sent = 0
        got = b""
        # Assert the limit fires from BUFFERING, well before the declared
        # payload could complete. Without the cap the server happily
        # accepts all 8MB and only errors later on trailing garbage, so a
        # generic "-ERR appeared eventually" check would pass either way.
        budget = limit * 3
        while sent < budget:
            try:
                s.sendall(b"x" * 100_000)
                sent += 100_000
            except OSError:
                break
            s.settimeout(0.1)
            try:
                chunk = s.recv(1000)
            except (socket.timeout, BlockingIOError):
                continue
            except OSError:
                break
            if not chunk:
                break
            got += chunk
            if b"-ERR" in got:
                break
        assert b"query buffer limit exceeded" in got, (
            f"server buffered {sent} bytes of an incomplete command without "
            f"enforcing its {limit}-byte limit"
        )
    finally:
        s.close()

    assert worst_latency(port) < MAX_ACCEPTABLE_LATENCY


def test_oversized_bulk_string_rejected_without_buffering_payload(server_proc):
    """Rejection must happen on the declared length, not after the bytes arrive."""
    port = server_proc()
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        s.sendall(b"*3\r\n$3\r\nSET\r\n$3\r\nfoo\r\n$99999999\r\n")  # no payload follows
        reply = s.recv(1000)
        assert reply.startswith(b"-ERR Protocol error")
    finally:
        s.close()
    assert worst_latency(port) < MAX_ACCEPTABLE_LATENCY


def test_abrupt_disconnect_mid_pipeline(server_proc):
    port = server_proc()
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(b"*3\r\n$3\r\nSET\r\n$1\r\na\r\n$1\r\n1\r\n" * 100)
    s.close()  # vanish without reading any reply
    time.sleep(0.2)
    assert worst_latency(port) < MAX_ACCEPTABLE_LATENCY


def test_server_still_correct_after_all_the_abuse(server_proc):
    port = server_proc()
    for junk in (b"\xff\xfe", b"*99999999999\r\n", b"+bad\r\n"):
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        s.sendall(junk)
        s.close()
    with Client(port=port) as c:
        assert c.set("k", "v") == "OK"
        assert c.get("k") == b"v"
        assert c.command("TTL", "k") == -1
