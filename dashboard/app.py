"""
Live dashboard for presenting the project.

Launches a real 3-node cluster of `cache.server` processes and drives them
over real TCP sockets. Nothing in the UI is simulated: every byte shown in
the protocol inspector was actually sent and received, every kill is a real
SIGKILL, and every recovery is a real WAL replay.

Stdlib only (http.server), so it runs offline with no extra installs.

Usage:
    python dashboard/app.py            # then open http://127.0.0.1:8000
    python dashboard/app.py --nodes 3 --base-port 6450 --http-port 8000
"""
import argparse
import json
import os
import random
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cache.client import CacheError, Client  # noqa: E402
from cache.cluster import ClusterClient, ClusterConfig, ClusterError  # noqa: E402
from cache.hashring import stable_hash  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent
RING_SPACE = 2 ** 64
SEED_COUNT = 6000


# --------------------------------------------------------------------------
# Raw RESP helpers: deliberately hand-rolled so the bytes are visible
# --------------------------------------------------------------------------

def encode_request(args: list[str]) -> bytes:
    """A client request is always an array of bulk strings."""
    out = [f"*{len(args)}\r\n".encode()]
    for a in args:
        b = a.encode()
        out.append(f"${len(b)}\r\n".encode() + b + b"\r\n")
    return b"".join(out)


def reply_end(buf: bytes, pos: int = 0) -> int | None:
    """Index just past one complete RESP reply starting at `pos`, or None if incomplete."""
    if pos >= len(buf):
        return None
    line_end = buf.find(b"\r\n", pos)
    if line_end == -1:
        return None
    tag, body = buf[pos:pos + 1], buf[pos + 1:line_end]
    after = line_end + 2
    if tag in (b"+", b"-", b":"):
        return after
    if tag == b"$":
        n = int(body)
        if n < 0:
            return after
        return after + n + 2 if len(buf) >= after + n + 2 else None
    if tag == b"*":
        n = int(body)
        for _ in range(max(n, 0)):
            after = reply_end(buf, after)
            if after is None:
                return None
        return after
    raise ValueError(f"unknown RESP type byte {tag!r}")


def read_replies(sock: socket.socket, count: int, buf: bytes = b"") -> tuple[list[bytes], list[tuple[float, bytes]]]:
    """Read `count` complete replies; also return each recv() chunk with its timestamp."""
    replies, chunks = [], []
    while len(replies) < count:
        end = reply_end(buf)
        if end is not None:
            replies.append(buf[:end])
            buf = buf[end:]
            continue
        data = sock.recv(65536)
        if not data:
            raise ConnectionError("server closed the connection")
        chunks.append((time.perf_counter(), data))
        buf += data
    return replies, chunks


def describe_reply(raw: bytes) -> str:
    tag = raw[:1]
    return {b"+": "simple string", b"-": "error", b":": "integer",
            b"$": "null bulk string" if raw.startswith(b"$-1") else "bulk string",
            b"*": "array"}.get(tag, "?")


def show(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# Cluster process management
# --------------------------------------------------------------------------

class ClusterManager:
    def __init__(self, nodes: int, base_port: int, capacity: int):
        self.config = ClusterConfig.local(nodes, start_port=base_port)
        self.capacity = capacity
        self.data_root = Path(tempfile.mkdtemp(prefix="cache-dashboard-"))
        self.procs: dict[str, subprocess.Popen] = {}
        self.events: list[dict] = []
        self.lock = threading.Lock()

    def log_event(self, text: str, kind: str = "info") -> None:
        with self.lock:
            self.events.append({"t": time.time(), "text": text, "kind": kind})
            self.events = self.events[-200:]

    def _log_path(self, node_id: str) -> Path:
        return self.data_root / f"{node_id}.log"

    def start(self, node_id: str) -> None:
        host, port = self.config.address(node_id)
        proc = self.procs.get(node_id)
        if proc is not None and proc.poll() is None:
            return
        with socket.socket() as probe:
            if probe.connect_ex((host, port)) == 0:
                raise RuntimeError(f"port {port} is already in use by another process")
        node_dir = self.data_root / node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        log = open(self._log_path(node_id), "a")
        cmd = [sys.executable, "-m", "cache.server", "--host", host, "--port", str(port),
               "--capacity", str(self.capacity), "--data-dir", str(node_dir), "--fsync", "everysec"]
        self.procs[node_id] = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT)
        deadline = time.time() + 15
        while time.time() < deadline:
            with socket.socket() as probe:
                if probe.connect_ex((host, port)) == 0:
                    break
            time.sleep(0.05)
        recovered = [ln for ln in self.log_tail(node_id, 20) if "replayed" in ln or "loaded snapshot" in ln]
        msg = f"{node_id} started on :{port} (pid {self.procs[node_id].pid})"
        if recovered:
            msg += ": " + recovered[-1].split(" ", 3)[-1]
        self.log_event(msg, "ok")

    def kill(self, node_id: str) -> None:
        proc = self.procs.get(node_id)
        if proc is None or proc.poll() is not None:
            return
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
        self.log_event(f"kill -9 {proc.pid}  ({node_id}): no graceful shutdown", "bad")

    def alive(self, node_id: str) -> bool:
        proc = self.procs.get(node_id)
        return proc is not None and proc.poll() is None

    def pid(self, node_id: str) -> int | None:
        proc = self.procs.get(node_id)
        return proc.pid if proc is not None and proc.poll() is None else None

    def log_tail(self, node_id: str, n: int = 4) -> list[str]:
        try:
            lines = self._log_path(node_id).read_text().splitlines()
        except OSError:
            return []
        return lines[-n:]

    def info(self, node_id: str) -> dict | None:
        if not self.alive(node_id):
            return None
        host, port = self.config.address(node_id)
        try:
            with Client(host=host, port=port, timeout=0.5) as c:
                text = c.command("INFO")
        except (OSError, CacheError):
            return None
        text = text.decode() if isinstance(text, bytes) else text
        return dict(line.split(":", 1) for line in text.splitlines() if ":" in line)

    def shutdown(self) -> None:
        for proc in self.procs.values():
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        shutil.rmtree(self.data_root, ignore_errors=True)


# --------------------------------------------------------------------------
# Load generator
# --------------------------------------------------------------------------

class LoadGenerator:
    """
    N client threads doing GET-then-SET-on-miss over the seeded keyspace.

    Keys are uniform rather than Zipfian on purpose: a Zipfian workload piles
    the hottest keys onto whichever shard owns them, so killing that one shard
    would look like a cluster-wide outage when it is really a skew artifact.
    """

    def __init__(self, config: ClusterConfig):
        self.config = config
        self.running = False
        self.clients = 0
        self.threads: list[threading.Thread] = []
        self.ok_by_node: dict[str, int] = {n: 0 for n in config.ids}
        self.err_by_node: dict[str, int] = {n: 0 for n in config.ids}
        self.history: list[dict] = []
        self.lock = threading.Lock()
        threading.Thread(target=self._sampler, daemon=True).start()

    def _worker(self, seed: int) -> None:
        rng = random.Random(seed)
        client = ClusterClient(self.config, timeout=1.0)
        try:
            while self.running:
                k = rng.randrange(SEED_COUNT)
                key = f"key:{k}"
                node = client.node_for(key)
                try:
                    if client.get(key) is None:
                        client.set(key, f"value-{k}")
                    with self.lock:
                        self.ok_by_node[node] += 1
                except ClusterError:
                    with self.lock:
                        self.err_by_node[node] += 1
        finally:
            client.close()

    def start(self, clients: int) -> None:
        self.stop()
        self.running = True
        self.clients = clients
        self.threads = [threading.Thread(target=self._worker, args=(i,), daemon=True) for i in range(clients)]
        for t in self.threads:
            t.start()

    def stop(self) -> None:
        self.running = False
        for t in self.threads:
            t.join(timeout=2)
        self.threads = []
        self.clients = 0

    def _sampler(self) -> None:
        interval = 0.5
        prev_ok = dict(self.ok_by_node)
        prev_err = dict(self.err_by_node)
        while True:
            time.sleep(interval)
            with self.lock:
                ok, err = dict(self.ok_by_node), dict(self.err_by_node)
            sample = {
                "t": time.time(),
                "ok": {n: (ok[n] - prev_ok[n]) / interval for n in ok},
                "err": {n: (err[n] - prev_err[n]) / interval for n in err},
            }
            prev_ok, prev_err = ok, err
            with self.lock:
                self.history.append(sample)
                self.history = self.history[-120:]


# --------------------------------------------------------------------------
# Demo actions
# --------------------------------------------------------------------------

class Demo:
    def __init__(self, manager: ClusterManager):
        self.m = manager
        self.load = LoadGenerator(manager.config)
        self.ring = ClusterClient(manager.config).ring
        self.seeded = 0

    def resolve_node(self, node: str, args: list[str]) -> str:
        if node and node != "auto":
            return node
        if len(args) >= 2:
            return self.ring.get_node(args[1].encode())
        return self.m.config.ids[0]

    def state(self) -> dict:
        nodes = []
        with self.load.lock:
            last = self.load.history[-1] if self.load.history else None
        for node_id in self.m.config.ids:
            host, port = self.m.config.address(node_id)
            nodes.append({
                "id": node_id,
                "port": port,
                "alive": self.m.alive(node_id),
                "pid": self.m.pid(node_id),
                "info": self.m.info(node_id),
                "ops": last["ok"][node_id] if last else 0,
                "errs": last["err"][node_id] if last else 0,
                "log": self.m.log_tail(node_id, 3),
            })
        with self.load.lock:
            history = [{"t": s["t"], "ok": s["ok"], "err": sum(s["err"].values())}
                       for s in self.load.history]
        with self.m.lock:
            events = list(self.m.events[-30:])
        return {"nodes": nodes, "load": {"running": self.load.running, "clients": self.load.clients,
                                         "history": history},
                "events": events, "seeded": self.seeded, "capacity": self.m.capacity}

    def ring_points(self) -> dict:
        return {"points": [{"pos": p / RING_SPACE, "node": o}
                           for p, o in zip(self.ring._points, self.ring._owners)],
                "nodes": self.m.config.ids}

    def route(self, key: str) -> dict:
        h = stable_hash(key.encode())
        return {"key": key, "hash": f"{h:016x}", "pos": h / RING_SPACE, "node": self.ring.get_node(key.encode())}

    def raw(self, command: str, node: str) -> dict:
        args = shlex.split(command)
        if not args:
            raise ValueError("empty command")
        node_id = self.resolve_node(node, args)
        host, port = self.m.config.address(node_id)
        req = encode_request(args)
        t0 = time.perf_counter()
        with socket.create_connection((host, port), timeout=2) as s:
            s.sendall(req)
            replies, _ = read_replies(s, 1)
        rtt = (time.perf_counter() - t0) * 1000
        route = self.route(args[1]) if len(args) >= 2 else None
        return {"node": node_id, "port": port, "sent": show(req), "received": show(replies[0]),
                "type": describe_reply(replies[0]), "rtt_ms": rtt, "route": route,
                "sent_len": len(req), "recv_len": len(replies[0])}

    def split_send(self, command: str, chunk: int, delay_ms: int) -> dict:
        args = shlex.split(command)
        node_id = self.resolve_node("auto", args)
        host, port = self.m.config.address(node_id)
        req = encode_request(args)
        chunk = max(1, chunk)
        n_chunks = -(-len(req) // chunk)
        delay = min(delay_ms, 4000 // max(n_chunks, 1)) / 1000
        events = []
        with socket.create_connection((host, port), timeout=10) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # one send() -> one segment
            t0 = time.perf_counter()
            for i in range(0, len(req), chunk):
                piece = req[i:i + chunk]
                s.sendall(piece)
                events.append({"t": (time.perf_counter() - t0) * 1000, "dir": "send", "bytes": show(piece)})
                if i + chunk < len(req):
                    time.sleep(delay)
            replies, chunks = read_replies(s, 1)
            for ts, data in chunks:
                events.append({"t": (ts - t0) * 1000, "dir": "recv", "bytes": show(data)})
        return {"node": node_id, "events": events, "total_bytes": len(req), "sends": n_chunks,
                "reply": show(replies[0])}

    def pipeline(self, count: int) -> dict:
        count = max(1, min(count, 20))
        node_id = self.m.config.ids[0]
        host, port = self.m.config.address(node_id)
        cmds = []
        for i in range(count):
            cmds.append(["SET", f"pipe:{i}", f"v{i}"] if i % 2 == 0 else ["GET", f"pipe:{i - 1}"])
        payload = b"".join(encode_request(c) for c in cmds)
        with socket.create_connection((host, port), timeout=5) as s:
            s.sendall(payload)
            replies, chunks = read_replies(s, count)
        return {"node": node_id, "commands": [" ".join(c) for c in cmds], "payload": show(payload),
                "payload_len": len(payload), "replies": [show(r) for r in replies],
                "recv_calls": len(chunks)}

    def rtt_bench(self, count: int) -> dict:
        count = max(10, min(count, 20000))
        node_id = self.m.config.ids[0]
        host, port = self.m.config.address(node_id)
        ping = encode_request(["PING"])
        with socket.create_connection((host, port), timeout=10) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            t0 = time.perf_counter()
            for _ in range(count):
                s.sendall(ping)
                read_replies(s, 1)
            seq = time.perf_counter() - t0
            t0 = time.perf_counter()
            s.sendall(ping * count)
            read_replies(s, count)
            pipe = time.perf_counter() - t0
        return {"count": count, "sequential_ms": seq * 1000, "pipelined_ms": pipe * 1000,
                "speedup": seq / pipe if pipe else 0}

    def concurrency(self, clients: int) -> dict:
        clients = max(1, min(clients, 500))
        node_id = self.m.config.ids[0]
        host, port = self.m.config.address(node_id)
        socks = []
        try:
            t0 = time.perf_counter()
            for _ in range(clients):
                socks.append(socket.create_connection((host, port), timeout=5))
            open_ms = (time.perf_counter() - t0) * 1000
            # Every socket is open simultaneously before any of them speaks.
            t0 = time.perf_counter()
            for i, s in enumerate(socks):
                s.sendall(encode_request(["SET", f"conn:{i}", str(i)]))
            answered = 0
            for s in socks:
                replies, _ = read_replies(s, 1)
                answered += replies[0] == b"+OK\r\n"
            serve_ms = (time.perf_counter() - t0) * 1000
        finally:
            for s in socks:
                s.close()
        return {"node": node_id, "pid": self.m.pid(node_id), "clients": clients, "answered": answered,
                "open_ms": open_ms, "serve_ms": serve_ms, "threads": thread_count(self.m.pid(node_id))}

    def seed(self, count: int) -> dict:
        count = max(1, min(count, 50000))
        counts = {n: 0 for n in self.m.config.ids}
        failed = 0
        with ClusterClient(self.m.config, timeout=1.0) as c:
            for i in range(count):
                key = f"key:{i}"
                try:
                    c.set(key, f"value-{i}")
                    counts[c.node_for(key)] += 1
                except ClusterError:
                    failed += 1
        self.seeded = max(self.seeded, count)
        self.m.log_event(f"wrote {count - failed} keys across the ring" + (f", {failed} failed" if failed else ""), "info")
        return {"counts": counts, "failed": failed}

    def verify(self) -> dict:
        n = self.seeded or SEED_COUNT
        owned = {node: 0 for node in self.m.config.ids}
        present = {node: 0 for node in self.m.config.ids}
        errors = {node: 0 for node in self.m.config.ids}
        with ClusterClient(self.m.config, timeout=1.0) as c:
            for i in range(n):
                key = f"key:{i}"
                node = c.node_for(key)
                owned[node] += 1
                try:
                    if c.get(key) is not None:
                        present[node] += 1
                except ClusterError:
                    errors[node] += 1
        return {"owned": owned, "present": present, "errors": errors}


def thread_count(pid: int | None) -> int | None:
    if pid is None:
        return None
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["ps", "-M", "-p", str(pid)], capture_output=True, text=True).stdout
            return max(len(out.strip().splitlines()) - 1, 0)
        out = subprocess.run(["ps", "-o", "nlwp=", "-p", str(pid)], capture_output=True, text=True).stdout
        return int(out.strip())
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------
# Security lab: one guarded node, one TLS node, attacks on demand
# --------------------------------------------------------------------------

LAB_PASSWORD = "lab-password"


class SecurityLab:
    """
    A password-protected node running the guard, plus a TLS-only node, with
    the same traffic behaviours the guard was trained and benchmarked on
    (tools/traffic.py). Every attack is real traffic against a real server;
    the guard's verdicts come from the server itself via GUARD STATUS.
    """

    def __init__(self, port: int, data_root: Path):
        from cache.database import SimulatedDatabase
        from tools.traffic import Tally

        self.port, self.tls_port = port, port + 1
        self.data_root = data_root
        self.model = REPO_ROOT / "models" / "guard.txt"
        self.db = SimulatedDatabase(latency_ms=0.5, capacity_qps=1500)
        self.legit_tally, self.attack_tally = Tally(), Tally()
        self.legit_stop: threading.Event | None = None
        self.attacks: dict[str, threading.Event] = {}
        self.history: list[dict] = []
        self.procs: list[subprocess.Popen] = []
        self.certs = None
        self.lock = threading.Lock()

    def _spawn(self, args: list[str], port: int) -> None:
        log = open(self.data_root / f"lab-{port}.log", "a")
        self.procs.append(subprocess.Popen([sys.executable, "-m", "cache.server", "--port", str(port), *args],
                                           cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT))
        deadline = time.time() + 15
        while time.time() < deadline:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    return
            time.sleep(0.05)
        raise RuntimeError(f"lab node on :{port} did not start")

    def start(self) -> None:
        for port in (self.port, self.tls_port):
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(f"port {port} is already in use by another process")
        args = ["--capacity", "200000", "--requirepass", LAB_PASSWORD, "--guard", "rules"]
        if self.model.exists():
            args += ["--guard-model", str(self.model)]
        self._spawn(args, self.port)
        certs = self.data_root / "certs"
        made = subprocess.run(["bash", str(REPO_ROOT / "tools" / "make_certs.sh"), str(certs)],
                              capture_output=True)
        if made.returncode == 0:
            self.certs = certs
            self._spawn(["--tls-cert", str(certs / "server.pem"), "--tls-key", str(certs / "server.key"),
                         "--requirepass", LAB_PASSWORD], self.tls_port)
        threading.Thread(target=self._sampler, daemon=True).start()

    def shutdown(self) -> None:
        for ev in [self.legit_stop, *self.attacks.values()]:
            if ev is not None:
                ev.set()
        for p in self.procs:
            if p.poll() is None:
                p.kill()
                p.wait()

    # -- traffic -----------------------------------------------------------

    def _launch(self, mix, tally, stop) -> None:
        from tools.traffic import run_behaviour
        for i, (kind, rate) in enumerate(mix):
            threading.Thread(target=run_behaviour, daemon=True,
                             args=(kind, self.port, self.db, tally, stop, rate, random.randrange(10**9),
                                   LAB_PASSWORD)).start()

    def legit(self, on: bool) -> dict:
        with self.lock:
            if on and self.legit_stop is None:
                self.legit_stop = threading.Event()
                self._launch([("app", 800), ("app", 400), ("busy", 4000), ("warmer", 500),
                              ("viral", 1200), ("idle", None), ("typo", 300)],
                             self.legit_tally, self.legit_stop)
            elif not on and self.legit_stop is not None:
                self.legit_stop.set()
                self.legit_stop = None
        return {"ok": True}

    ATTACK_RATES = {"penetration": 800, "hot_flood": 10000, "brute_force": 4, "slowloris": 20}

    def attack(self, kind: str, on: bool) -> dict:
        if kind not in self.ATTACK_RATES:
            raise ValueError(f"unknown attack {kind!r}")
        with self.lock:
            if on and kind not in self.attacks:
                self.attacks[kind] = threading.Event()
                self._launch([(kind, self.ATTACK_RATES[kind])], self.attack_tally, self.attacks[kind])
            elif not on and kind in self.attacks:
                self.attacks.pop(kind).set()
        return {"ok": True}

    def set_mode(self, mode: str) -> dict:
        with Client(port=self.port, password=LAB_PASSWORD, timeout=2) as c:
            c.command("GUARD", "MODE", mode)
        return {"ok": True}

    def _sampler(self) -> None:
        interval = 0.5
        prev_legit = prev_attack = 0
        while True:
            time.sleep(interval)
            with self.legit_tally.lock:
                legit = sum(self.legit_tally.db_calls.values())
            with self.attack_tally.lock:
                attack = sum(self.attack_tally.db_calls.values())
            with self.lock:
                self.history.append({"t": time.time(), "legit": (legit - prev_legit) / interval,
                                     "attack": (attack - prev_attack) / interval})
                self.history = self.history[-120:]
            prev_legit, prev_attack = legit, attack

    def state(self) -> dict:
        try:
            with Client(port=self.port, password=LAB_PASSWORD, timeout=1) as c:
                guard = json.loads(c.command("GUARD", "STATUS"))
        except (OSError, CacheError) as exc:
            guard = {"mode": "?", "connections": [], "flags": [], "error": str(exc)}
        with self.lock:
            history = list(self.history)
            running = sorted(self.attacks)
            legit_on = self.legit_stop is not None
        def totals(t):
            with t.lock:
                return {"ops": dict(t.ops), "refused": dict(t.refused), "db_calls": dict(t.db_calls)}
        return {"guard": guard, "history": history, "attacks": running, "legit": legit_on,
                "ml_available": self.model.exists(), "tls_available": self.certs is not None,
                "capacity": self.db.capacity_qps, "legit_totals": totals(self.legit_tally),
                "attack_totals": totals(self.attack_tally)}

    # -- access control demos ----------------------------------------------

    def auth_try(self, how: str) -> dict:
        """Run one real exchange and report exactly what the server answered."""
        c = Client(port=self.port, timeout=3).connect()
        try:
            steps = []
            if how == "none":
                cmds = [("GET", "secret")]
            elif how == "wrong":
                cmds = [("AUTH", "letmein"), ("GET", "secret")]
            else:
                cmds = [("AUTH", LAB_PASSWORD), ("SET", "secret", "only-for-authorized-clients"), ("GET", "secret")]
            for cmd in cmds:
                shown = " ".join(cmd if cmd[0] != "AUTH" else ("AUTH", "••••••••"))  # fixed width: no length leak
                t0 = time.perf_counter()
                try:
                    reply = c.command(*cmd)
                    reply = reply.decode() if isinstance(reply, bytes) else str(reply)
                    ok = True
                except CacheError as exc:
                    reply, ok = str(exc), False
                steps.append({"sent": shown, "reply": reply, "ok": ok,
                              "ms": round((time.perf_counter() - t0) * 1000, 1)})
            return {"steps": steps}
        finally:
            c.close()

    def tls_check(self) -> dict:
        import ssl
        if self.certs is None:
            raise RuntimeError("openssl was not available to make certificates")
        out = {}
        ctx = ssl.create_default_context(cafile=str(self.certs / "ca.pem"))
        with Client(port=self.tls_port, tls=ctx, password=LAB_PASSWORD, timeout=3) as c:
            c.ping()
            out["verified"] = {"version": c._sock.version(), "cipher": c._sock.cipher()[0],
                               "issuer": dict(x[0] for x in c._sock.getpeercert()["issuer"]).get("commonName")}
        try:
            Client(port=self.tls_port, tls=ssl.create_default_context(), timeout=3).connect().close()
            out["untrusted"] = "accepted (unexpected)"
        except ssl.SSLCertVerificationError as exc:
            out["untrusted"] = f"rejected: {exc.reason}"
        try:
            with Client(port=self.tls_port, timeout=2) as c:
                c.ping()
            out["plaintext"] = "accepted (unexpected)"
        except (OSError, ConnectionError, CacheError) as exc:
            out["plaintext"] = f"refused: {type(exc).__name__}"
        return out


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------

def make_handler(demo: Demo, lab: "SecurityLab"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(status, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json(demo.state())
            elif self.path == "/api/ring":
                self._json(demo.ring_points())
            elif self.path == "/api/lab/state":
                self._json(lab.state())
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            routes = {
                "/api/raw": lambda: demo.raw(body.get("command", ""), body.get("node", "auto")),
                "/api/route": lambda: demo.route(body.get("key", "")),
                "/api/split": lambda: demo.split_send(body.get("command", "SET greeting hello"),
                                                      int(body.get("chunk", 1)), int(body.get("delay", 60))),
                "/api/pipeline": lambda: demo.pipeline(int(body.get("count", 6))),
                "/api/rtt": lambda: demo.rtt_bench(int(body.get("count", 2000))),
                "/api/concurrency": lambda: demo.concurrency(int(body.get("clients", 200))),
                "/api/seed": lambda: demo.seed(int(body.get("count", SEED_COUNT))),
                "/api/verify": demo.verify,
                "/api/kill": lambda: demo.m.kill(body["node"]) or {"ok": True},
                "/api/start": lambda: demo.m.start(body["node"]) or {"ok": True},
                "/api/load/start": lambda: demo.load.start(int(body.get("clients", 8))) or {"ok": True},
                "/api/load/stop": lambda: demo.load.stop() or {"ok": True},
                "/api/lab/legit": lambda: lab.legit(bool(body.get("on"))),
                "/api/lab/attack": lambda: lab.attack(body["kind"], bool(body.get("on"))),
                "/api/lab/mode": lambda: lab.set_mode(body["mode"]),
                "/api/lab/auth": lambda: lab.auth_try(body.get("how", "none")),
                "/api/lab/tls": lab.tls_check,
            }
            action = routes.get(self.path)
            if action is None:
                return self._json({"error": "not found"}, 404)
            try:
                self._json(action())
            except (ValueError, KeyError, RuntimeError, OSError, ConnectionError, CacheError) as exc:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description="Live presentation dashboard")
    ap.add_argument("--nodes", type=int, default=3)
    ap.add_argument("--base-port", type=int, default=6450)
    ap.add_argument("--capacity", type=int, default=5000, help="max keys per node")
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--lab-port", type=int, default=6480, help="security lab node (and +1 for its TLS twin)")
    args = ap.parse_args()

    manager = ClusterManager(args.nodes, args.base_port, args.capacity)
    demo = Demo(manager)
    lab = SecurityLab(args.lab_port, manager.data_root)

    # Bind the HTTP port BEFORE launching any node, so a busy port fails
    # with nothing to clean up. Fall forward to the next free port.
    httpd = None
    for port in range(args.http_port, args.http_port + 20):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(demo, lab))
            break
        except OSError:
            continue
    if httpd is None:
        print(f"error: ports {args.http_port}-{args.http_port + 19} are all in use; pass --http-port", file=sys.stderr)
        return 1
    httpd.daemon_threads = True

    # Treat SIGTERM / a closed terminal like Ctrl-C so nodes are never orphaned.
    def _on_term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGHUP, _on_term)

    try:
        for node_id in manager.config.ids:
            manager.start(node_id)
        lab.start()
        print(f"Cluster up: {', '.join(f'{n}:{manager.config.address(n)[1]}' for n in manager.config.ids)}")
        print(f"Dashboard:  http://127.0.0.1:{httpd.server_address[1]}")
        print("Ctrl-C to stop (shuts down every node).")
        httpd.serve_forever()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(f"  (a previous run may have left nodes behind: lsof -iTCP:{args.base_port} -sTCP:LISTEN,"
              f" or pass --base-port)", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        demo.load.stop()
        lab.shutdown()
        httpd.server_close()
        manager.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
