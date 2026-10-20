"""
Traffic behaviours for training and evaluating the guard: legitimate
clients, attacks, and — the part that makes the evaluation mean anything —
legitimate clients that LOOK like attacks.

A detector tested only against obvious attacks and quiet normal traffic
scores perfectly and proves nothing. So each attack has a legitimate twin
that shares its most obvious signal:

  attack        its signature                    legitimate look-alike
  penetration   GETs that all miss               warmer: a cache warm-up job
                                                 (misses too, but fills what it misses)
  hot_flood     one key, very high rate          viral: a popular key read hard,
                                                 busy: a high-rate normal service
  brute_force   repeated AUTH                    (none needed: legitimate clients
                                                 authenticate once)
  slowloris     a command that never finishes    idle: a connection that sends nothing,
                                                 uploader: large values that span reads
  brute_force   error replies                    buggy: a client whose commands often fail
                failed AUTH                      typo: a user who mistypes the password
                                                 once or twice, then logs in

Every behaviour is cache-aside against a shared SimulatedDatabase: a GET
that misses becomes a database query, which is how cache penetration hurts
a real system. A command the guard refuses never reaches the database.

Each connection names itself with CLIENT SETNAME (`ok:<kind>` or
`atk:<kind>`) so logged feature rows can be labelled. The name is never a
feature; the detectors only see traffic.
"""
import os
import random
import socket
import threading
import time
from dataclasses import dataclass, field

from cache.client import CacheError, Client

LEGIT = ("app", "busy", "warmer", "viral", "idle", "buggy", "uploader", "typo")
ATTACKS = ("penetration", "hot_flood", "brute_force", "slowloris")
KEYSPACE = 20000


def label_of(name: str) -> str:
    """Class label for a connection name: 'normal' for every legitimate kind."""
    if name.startswith("atk:"):
        return name[4:].split(":")[0]
    return "normal"


@dataclass
class Tally:
    """Shared, per-kind outcome counters."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    ops: dict = field(default_factory=dict)
    refused: dict = field(default_factory=dict)
    db_calls: dict = field(default_factory=dict)
    latencies: dict = field(default_factory=dict)  # kind -> [(t, seconds per op)]
    events: list = field(default_factory=list)

    def add(self, kind: str, ops=0, refused=0, db_calls=0, latency=None) -> None:
        with self.lock:
            self.ops[kind] = self.ops.get(kind, 0) + ops
            self.refused[kind] = self.refused.get(kind, 0) + refused
            self.db_calls[kind] = self.db_calls.get(kind, 0) + db_calls
            if latency is not None:
                self.latencies.setdefault(kind, []).append((time.time(), latency))

    def event(self, text: str) -> None:
        with self.lock:
            self.events.append((time.time(), text))


def _zipf_sampler(rng: random.Random, n: int, s: float):
    weights = [1 / (r ** s) for r in range(1, n + 1)]
    cum, total = [], 0.0
    for w in weights:
        total += w
        cum.append(total)
    perm = list(range(n))
    random.Random(7).shuffle(perm)
    return lambda: perm[rng.choices(range(n), cum_weights=cum)[0]]


def _paced(rate: float, stop: threading.Event, batch_fn, tick: float = 0.01) -> None:
    """Call batch_fn(n) so that about `rate` operations happen per second."""
    carry = 0.0
    while not stop.is_set():
        start = time.monotonic()
        carry += rate * tick
        n = int(carry)
        carry -= n
        if n:
            batch_fn(n)
        rest = tick - (time.monotonic() - start)
        if rest > 0:
            time.sleep(rest)


def _cache_aside(client: Client, kind: str, keys: list[str], db, tally: Tally, fill: bool) -> None:
    """GET a batch; every miss goes to the database (and is cached if `fill`)."""
    t = time.monotonic()
    replies = client.pipeline([("GET", k) for k in keys])
    misses, refused = [], 0
    for k, r in zip(keys, replies):
        if isinstance(r, CacheError):
            refused += 1
        elif r is None:
            misses.append(k)
    for k in misses:
        value = db.get(k)
        if fill:
            client.set(k, value)
    tally.add(kind, ops=len(keys), refused=refused, db_calls=len(misses),
              latency=(time.monotonic() - t) / len(keys))


def run_behaviour(kind: str, port: int, db, tally: Tally, stop: threading.Event,
                  rate: float | None = None, seed: int = 0, password: str | None = None,
                  tls=None) -> None:
    rng = random.Random(seed)
    tag = "atk" if kind in ATTACKS else "ok"
    name = f"{tag}:{kind}"

    if kind == "slowloris":
        return _slowloris(port, tally, stop, conns=int(rate or 10), password=password)
    if kind == "brute_force":
        return _brute_force(port, tally, stop, rate or 4, rng, tls)
    if kind == "typo":
        return _typo(port, db, tally, stop, rate, rng, password, tls)

    client = Client(port=port, password=password, tls=tls, name=name, timeout=5).connect()
    try:
        if kind == "app":
            zipf = _zipf_sampler(rng, KEYSPACE, 1.1)
            r = rate or rng.uniform(100, 1500)
            _paced(r, stop, lambda n: _cache_aside(client, kind, [f"key:{zipf()}" for _ in range(n)],
                                                   db, tally, fill=True))
        elif kind == "busy":
            zipf = _zipf_sampler(rng, KEYSPACE, 1.1)
            r = rate or rng.uniform(3000, 7000)
            _paced(r, stop, lambda n: _cache_aside(client, kind, [f"key:{zipf()}" for _ in range(n)],
                                                   db, tally, fill=True))
        elif kind == "warmer":
            # Walks a range of keys the cache has not seen yet, filling each
            # miss: high miss ratio and all-distinct keys, like penetration,
            # but it WRITES what it misses.
            nxt = [rng.randrange(10**6)]
            r = rate or rng.uniform(100, 1500)

            def step(n):
                keys = [f"warm:{nxt[0] + i}" for i in range(n)]
                nxt[0] += n
                _cache_aside(client, kind, keys, db, tally, fill=True)
            _paced(r, stop, step)
        elif kind == "viral":
            # A genuinely popular key, read hard but at a human-driven rate.
            zipf = _zipf_sampler(rng, KEYSPACE, 1.1)
            hot = f"key:viral:{rng.randrange(100)}"
            r = rate or rng.uniform(300, 2500)
            _paced(r, stop, lambda n: _cache_aside(
                client, kind, [hot if rng.random() < 0.7 else f"key:{zipf()}" for _ in range(n)],
                db, tally, fill=True))
        elif kind == "buggy":
            # A legitimate client with a bug: a fifth of its commands are
            # malformed or unknown and get error replies, as brute-force
            # guesses do, but it never touches AUTH.
            zipf = _zipf_sampler(rng, KEYSPACE, 1.1)
            r = rate or rng.uniform(50, 800)

            def step(n):
                cmds = [("GETT", f"key:{zipf()}") if rng.random() < 0.2 else ("GET", f"key:{zipf()}")
                        for _ in range(n)]
                t = time.monotonic()
                replies = client.pipeline(cmds)
                misses = [c[1] for c, rep in zip(cmds, replies) if c[0] == "GET" and rep is None]
                for k in misses:
                    client.set(k, db.get(k))
                tally.add(kind, ops=n, db_calls=len(misses),
                          refused=sum(1 for c, rep in zip(cmds, replies)
                                      if c[0] == "GET" and isinstance(rep, CacheError)),
                          latency=(time.monotonic() - t) / n)
            _paced(r, stop, step)
        elif kind == "uploader":
            # Writes large blobs, so a command routinely spans several reads
            # and is still incomplete when a window closes, like slowloris,
            # but it moves megabytes a second.
            size = rng.choice((256_000, 1_000_000, 2_000_000))
            blob = os.urandom(size // 2).hex().encode()
            while not stop.wait(rng.uniform(0.1, 0.6)):
                t = time.monotonic()
                r = client.set(f"blob:{rng.randrange(50)}", blob)
                tally.add(kind, ops=1, refused=int(isinstance(r, CacheError)),
                          latency=time.monotonic() - t)
        elif kind == "idle":
            while not stop.wait(rng.uniform(0.5, 3.0)):
                client.ping()
                tally.add(kind, ops=1)
        elif kind == "penetration":
            # Keys that exist nowhere: every GET misses and lands on the
            # database, and there is nothing to cache.
            r = rate or rng.uniform(20, 2000)
            _paced(r, stop, lambda n: _cache_aside(
                client, kind, [f"user:{os.urandom(6).hex()}" for _ in range(n)], db, tally, fill=False))
        elif kind == "hot_flood":
            targets = [f"key:{rng.randrange(KEYSPACE)}" for _ in range(rng.choice((1, 1, 2, 3)))]
            r = rate or rng.uniform(4000, 15000)
            _paced(r, stop, lambda n: _cache_aside(
                client, kind, [rng.choice(targets) for _ in range(n)], db, tally, fill=True))
        else:
            raise ValueError(f"unknown behaviour {kind!r}")
    except (ConnectionError, OSError, CacheError) as exc:
        tally.event(f"{name} connection ended: {type(exc).__name__}")
    finally:
        client.close()


def _typo(port: int, db, tally: Tally, stop: threading.Event, rate, rng: random.Random,
          password: str | None, tls) -> None:
    """
    A legitimate user who gets the password wrong once or twice, then logs
    in and works normally, reconnecting every few seconds. A detector that
    calls one failed AUTH an attack disconnects real people.
    """
    zipf = _zipf_sampler(rng, KEYSPACE, 1.1)
    r = rate or rng.uniform(100, 800)
    while not stop.is_set():
        try:
            c = Client(port=port, tls=tls, timeout=5).connect()
        except OSError:
            time.sleep(0.2)
            continue
        try:
            for _ in range(rng.choice((1, 1, 2))):
                try:
                    c.command("AUTH", f"{password}-typo")
                except CacheError:
                    pass
            c.command("AUTH", password)
            c.command("CLIENT", "SETNAME", "ok:typo")
            session_end = time.time() + rng.uniform(3, 8)
            sub = threading.Event()

            def until_end(n):
                if time.time() >= session_end:
                    sub.set()
                    return
                _cache_aside(c, "typo", [f"key:{zipf()}" for _ in range(n)], db, tally, fill=True)
            threading.Thread(target=lambda: (stop.wait(), sub.set()), daemon=True).start()
            _paced(r, sub, until_end)
        except (ConnectionError, OSError, CacheError) as exc:
            tally.event(f"ok:typo connection ended: {type(exc).__name__}")
            tally.add("typo", refused=1)
        finally:
            c.close()


def _brute_force(port: int, tally: Tally, stop: threading.Event, conns: float,
                 rng: random.Random, tls) -> None:
    """Guess passwords on several connections; reconnect whenever one is closed."""
    def one(i: int) -> None:
        while not stop.is_set():
            try:
                c = Client(port=port, tls=tls, timeout=5).connect()
                c.command("CLIENT", "SETNAME", "atk:brute_force")
            except CacheError:
                # NOAUTH: CLIENT SETNAME needs auth. Unnamed rows are dropped
                # from training, but the guard still sees them in evaluation.
                pass
            except OSError:
                time.sleep(0.2)
                continue
            try:
                while not stop.is_set():
                    try:
                        c.command("AUTH", f"guess{rng.randrange(10**9)}")
                    except CacheError:
                        tally.add("brute_force", ops=1, refused=1)
            except (ConnectionError, OSError):
                tally.add("brute_force", ops=0)
                tally.event("brute_force connection closed by server")
            finally:
                c.close()
    threads = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(int(conns))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def _slowloris(port: int, tally: Tally, stop: threading.Event, conns: int, password) -> None:
    """Hold many connections, each trickling a command that never finishes."""
    socks = []
    prefix = b""
    if password:
        prefix = f"*2\r\n$4\r\nAUTH\r\n${len(password)}\r\n{password}\r\n".encode()
    name = b"*3\r\n$6\r\nCLIENT\r\n$7\r\nSETNAME\r\n$13\r\natk:slowloris\r\n"
    partial = b"*3\r\n$3\r\nSET\r\n$4\r\nslow\r\n$1000\r\n"
    while not stop.is_set():
        while len(socks) < conns and not stop.is_set():
            try:
                s = socket.create_connection(("127.0.0.1", port), timeout=2)
                s.sendall(prefix + name + partial)
                socks.append([s, time.time()])
            except OSError:
                time.sleep(0.2)
        time.sleep(2.0)
        alive = []
        for s, born in socks:
            try:
                s.sendall(b"x")  # one byte, then wait again
                s.settimeout(0.01)
                try:
                    if s.recv(4096) == b"":
                        raise ConnectionError
                except socket.timeout:
                    pass
                alive.append([s, born])
            except OSError:
                tally.add("slowloris", ops=0, refused=1)
                tally.event(f"slowloris connection closed after {time.time() - born:.1f}s")
                s.close()
        socks = alive
        tally.add("slowloris", ops=len(socks))
    for s, _ in socks:
        s.close()
