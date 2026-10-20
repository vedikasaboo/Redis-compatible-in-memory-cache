"""
Security tests against a real server: AUTH, TLS, per-IP connection caps,
and the guard's detection and mitigation of each attack.

The guard tests use the RULE detector so they do not depend on a trained
model file; the ML detector shares every code path except classify(), which
tests/test_guard.py covers directly.
"""
import socket
import ssl
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from cache.client import CacheError, Client

REPO_ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(*extra) -> tuple[subprocess.Popen, int]:
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "cache.server", "--port", str(port), *extra],
                            cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 10
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return proc, port
        time.sleep(0.05)
    proc.kill()
    raise TimeoutError("server did not start")


@pytest.fixture
def server_with(request):
    procs = []

    def start(*extra):
        proc, port = _start(*extra)
        procs.append(proc)
        return port
    yield start
    for p in procs:
        p.kill()
        p.wait()


# -- AUTH -----------------------------------------------------------------

def test_commands_refused_before_auth(server_with):
    port = server_with("--requirepass", "s3cret")
    with Client(port=port) as c:
        with pytest.raises(CacheError, match="NOAUTH"):
            c.get("k")
        with pytest.raises(CacheError, match="NOAUTH"):
            c.command("CLIENT", "SETNAME", "x")


def test_wrong_password_rejected_and_right_one_accepted(server_with):
    port = server_with("--requirepass", "s3cret")
    with Client(port=port) as c:
        with pytest.raises(CacheError, match="WRONGPASS"):
            c.command("AUTH", "nope")
        assert c.command("AUTH", "s3cret") == "OK"
        assert c.set("k", "v") == "OK"
    with Client(port=port, password="s3cret") as c:
        assert c.get("k") == b"v"


def test_auth_with_username_default(server_with):
    port = server_with("--requirepass", "s3cret")
    with Client(port=port) as c:
        assert c.command("AUTH", "default", "s3cret") == "OK"


def test_auth_without_password_configured(server_with):
    port = server_with()
    with Client(port=port) as c:
        with pytest.raises(CacheError, match="without any password configured"):
            c.command("AUTH", "anything")
        assert c.ping() == "PONG"  # and the connection still works


def test_failed_auth_backs_off_then_disconnects(server_with):
    port = server_with("--requirepass", "s3cret")
    with Client(port=port, timeout=10) as c:
        t0 = time.time()
        for _ in range(3):
            with pytest.raises(CacheError):
                c.command("AUTH", "guess")
        # 0.1 + 0.2 + 0.4 s of backoff
        assert time.time() - t0 >= 0.65
        attempts, closed = 3, False
        while attempts < 20 and not closed:
            try:
                c.command("AUTH", "guess")
            except CacheError:
                attempts += 1
            except (ConnectionError, OSError):
                closed = True
        assert closed and attempts == 10  # MAX_AUTH_FAILURES


# -- TLS ------------------------------------------------------------------

@pytest.fixture(scope="module")
def certs(tmp_path_factory):
    d = tmp_path_factory.mktemp("certs")
    subprocess.run(["bash", str(REPO_ROOT / "tools" / "make_certs.sh"), str(d)], check=True,
                   capture_output=True)
    return d


def test_tls_round_trip_with_verified_certificate(server_with, certs):
    port = server_with("--tls-cert", str(certs / "server.pem"), "--tls-key", str(certs / "server.key"))
    ctx = ssl.create_default_context(cafile=str(certs / "ca.pem"))
    with Client(port=port, tls=ctx) as c:
        assert c.set("k", "secret value") == "OK"
        assert c.get("k") == b"secret value"
        assert c._sock.version() in ("TLSv1.2", "TLSv1.3")


def test_tls_rejects_untrusted_certificate(server_with, certs):
    port = server_with("--tls-cert", str(certs / "server.pem"), "--tls-key", str(certs / "server.key"))
    with pytest.raises(ssl.SSLCertVerificationError):
        Client(port=port, tls=ssl.create_default_context()).connect()


def test_tls_port_refuses_plaintext(server_with, certs):
    port = server_with("--tls-cert", str(certs / "server.pem"), "--tls-key", str(certs / "server.key"))
    with pytest.raises((ConnectionError, OSError)):
        with Client(port=port, timeout=2) as c:
            c.ping()


# -- connection caps --------------------------------------------------------

def test_max_conns_per_ip(server_with):
    port = server_with("--max-conns-per-ip", "3")
    held = [Client(port=port).connect() for _ in range(3)]
    try:
        with Client(port=port, timeout=2) as extra:
            with pytest.raises((CacheError, ConnectionError)):
                extra.ping()
        held[0].close()
        time.sleep(0.2)
        with Client(port=port) as again:
            assert again.ping() == "PONG"
    finally:
        for c in held:
            c.close()


# -- guard ------------------------------------------------------------------

def _guard_status(port, password=None):
    import json
    with Client(port=port, password=password) as c:
        return json.loads(c.command("GUARD", "STATUS"))


def test_guard_rate_limits_penetration_but_not_normal_clients(server_with):
    import os
    port = server_with("--guard", "rules")
    stop = threading.Event()
    refused = {"atk": 0, "ok": 0}

    def attacker():
        with Client(port=port, name="atk") as c:
            while not stop.is_set():
                for r in c.pipeline([("GET", f"u:{os.urandom(6).hex()}") for _ in range(20)]):
                    refused["atk"] += isinstance(r, CacheError)
                time.sleep(0.01)

    def normal():
        with Client(port=port, name="ok") as c:
            c.set("hot", "v")
            while not stop.is_set():
                for r in c.pipeline([("GET", "hot"), ("GET", f"k:{os.urandom(1).hex()}")]):
                    refused["ok"] += isinstance(r, CacheError)
                time.sleep(0.01)

    threads = [threading.Thread(target=attacker), threading.Thread(target=normal)]
    for t in threads:
        t.start()
    time.sleep(3.5)
    status = _guard_status(port)
    stop.set()
    for t in threads:
        t.join()
    verdicts = {c["name"]: c["verdict"] for c in status["connections"]}
    assert verdicts["atk"] == "penetration"
    assert verdicts["ok"] == "normal"
    assert refused["atk"] > 0 and refused["ok"] == 0


def test_guard_closes_slowloris(server_with):
    port = server_with("--guard", "rules")
    s = socket.create_connection(("127.0.0.1", port))
    s.sendall(b"*3\r\n$3\r\nSET\r\n$4\r\nslow\r\n$1000\r\n")
    closed = False
    deadline = time.time() + 8
    while time.time() < deadline and not closed:
        try:
            s.sendall(b"x")
            s.settimeout(0.5)
            closed = s.recv(1) == b""
        except socket.timeout:
            pass
        except OSError:
            closed = True
        time.sleep(0.5)
    s.close()
    assert closed, "a stalled partial command should be closed within a few windows"


def test_guard_off_does_nothing(server_with):
    import os
    port = server_with("--guard", "off")
    with Client(port=port) as c:
        for _ in range(30):
            replies = c.pipeline([("GET", f"u:{os.urandom(6).hex()}") for _ in range(50)])
            assert not any(isinstance(r, CacheError) for r in replies)
            time.sleep(0.05)


def test_guard_mode_switch_at_runtime(server_with):
    port = server_with("--guard", "observe")
    with Client(port=port) as c:
        assert c.command("GUARD", "MODE", "rules") == "OK"
        with pytest.raises(CacheError, match="needs --guard-model"):
            c.command("GUARD", "MODE", "ml")
    assert _guard_status(port)["mode"] == "rules"
