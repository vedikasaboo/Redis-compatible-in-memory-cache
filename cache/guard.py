"""
Guard: per-connection attack detection and mitigation for the cache server.

Threat model
------------
A cache port reachable by clients that should not all be trusted: a
multi-tenant cache, a microservice mesh, or an instance exposed to the
internet (unauthenticated Redis ports are scanned and hijacked constantly).
The guard watches each CONNECTION's traffic at the protocol level — the
commands and keys it sends, and how the cache answers — which a packet-level
intrusion detector cannot see.

Attacks it classifies, and what is done about each:

  penetration — GETs for keys that do not exist, so every request misses.
                In cache-aside every miss becomes a database query; the
                cache is bypassed and the database flooded.   -> rate limit
  hot_flood   — one connection hammering one or a few keys at very high
                rate, overloading the node that owns them.    -> rate limit
  brute_force — repeated AUTH guesses.                        -> disconnect
  slowloris   — holding a connection open with a command that never
                completes, to exhaust connection slots.       -> disconnect

Detection
---------
Every WINDOW seconds, each connection's counters become one feature row
(feature_row below — ONE definition, shared with the trainer, so training
and serving cannot drift apart; the same rule cache.features follows). Two
detectors read the same rows, so they can be compared on equal terms:

  rules — hand-set thresholds; the baseline an operator would write.
  ml    — a LightGBM classifier trained on labelled traffic
          (tools/train_guard.py).

A connection is flagged only when the detector is confident, and released
only after RELEASE_WINDOWS consecutive normal windows, so a verdict does not
flap on one noisy second.

Refused commands count toward rate and the key features — what the client
is ATTEMPTING — but not toward the hit, miss and error ratios, which are
computed over commands that actually ran. A refused GET has no answer to
judge; counting it as "not a miss" made a throttled penetration attacker
look calm, so it was released, resumed full rate, was flagged again, and
reached the database three times as often as it should have.

Cost on the request path: a few counter increments and one dict update per
command. All feature work and all model inference happen once per window,
batched over every connection, never per command.
"""
import json
import logging
import time

logger = logging.getLogger("cache.guard")

WINDOW = 1.0
KEY_CAP = 2048          # distinct keys tracked per connection per window
CONFIDENCE = 0.8        # ML probability needed to flag
RELEASE_WINDOWS = 3     # consecutive normal windows before a verdict lifts

FEATURE_NAMES = [
    "rate",            # commands per second
    "get_ratio",       # share of commands that are GET
    "set_ratio",       # share of commands that are SET
    "miss_ratio",      # GETs answered with "no such key"
    "distinct_ratio",  # distinct keys / keyed commands
    "top_key_share",   # share of keyed commands on the single busiest key
    "error_ratio",     # error replies / commands
    "auth_failures",   # failed AUTH attempts this window
    "auth_failures_total",  # failed AUTH attempts over the connection's life
    "bytes_per_cmd",   # request bytes per command
    "pending_bytes",   # bytes of an incomplete command held at window end
    "stalled_windows", # consecutive windows with a partial command and no progress
]
CLASSES = ["normal", "penetration", "hot_flood", "brute_force", "slowloris"]
MODES = ("off", "observe", "rules", "ml")

# label -> (action, commands per second allowed while limited)
MITIGATION = {
    "penetration": ("limit", 20),
    "hot_flood": ("limit", 200),
    "brute_force": ("close", 0),
    "slowloris": ("close", 0),
}


def feature_row(cmds: int, gets: int, sets: int, misses: int, keyed: int, distinct: int,
                top_key: int, errors: int, auth_failures: int, bytes_in: int,
                pending_bytes: int, stalled_windows: int, auth_failures_total: int = 0,
                executed: int | None = None, window: float = WINDOW) -> list[float]:
    """
    One window of one connection's traffic, as the detectors see it.
    `cmds` counts every command attempted; `executed` the ones that ran
    (equal unless the guard refused some). Ratios use `executed`.
    """
    c = max(cmds, 1)
    e = max(cmds if executed is None else executed, 1)
    k = max(keyed, 1)
    return [
        cmds / window,
        gets / e,
        sets / e,
        misses / max(gets, 1),
        distinct / k,
        top_key / k,
        errors / e,
        float(auth_failures),
        float(auth_failures_total),
        bytes_in / c,
        float(pending_bytes),
        float(stalled_windows),
    ]


# Policy applied to every detector's output. One window cannot tell a user's
# typo from a guesser's first attempt: both are a lone failed AUTH. Measured,
# the model called a single failed AUTH brute force at 99.9% confidence. So a
# brute_force verdict, which disconnects, needs at least this many failures
# on the connection; the rules use the same bar.
MIN_AUTH_FAILURES_TO_ACT = 3


def apply_policy(label: str, conf: float, row: list[float]) -> tuple[str, float]:
    if label == "brute_force" and row[FEATURE_NAMES.index("auth_failures_total")] < MIN_AUTH_FAILURES_TO_ACT:
        return "normal", conf
    return label, conf


def is_active(row: list[float]) -> bool:
    """Did this window carry any traffic evidence at all?"""
    f = dict(zip(FEATURE_NAMES, row))
    return f["rate"] > 0 or f["pending_bytes"] > 0 or f["auth_failures"] > 0


class ConnStats:
    """Counters for one connection, reset every window."""

    __slots__ = ("id", "peer", "name", "cmds", "gets", "sets", "misses", "keyed", "keys",
                 "overflow", "errors", "auth_failures", "bytes_in", "pending_bytes",
                 "stalled_windows", "verdict", "confidence", "normal_streak", "tokens",
                 "last_refill", "close_requested", "flagged_at", "last_features", "rejected",
                 "closer", "auth_failures_total", "refused_now")

    def __init__(self, conn_id: int, peer):
        self.id = conn_id
        self.peer = peer
        self.name = ""
        self.verdict = "normal"
        self.confidence = 0.0
        self.normal_streak = 0
        self.tokens = 0.0
        self.last_refill = time.monotonic()
        self.close_requested = False
        self.flagged_at = None
        self.last_features = None
        self.rejected = 0
        self.pending_bytes = 0
        self.stalled_windows = 0
        self.closer = None  # set by the server: closes this connection's transport
        self.auth_failures_total = 0
        self._reset()

    def _reset(self) -> None:
        self.cmds = self.gets = self.sets = self.misses = self.keyed = 0
        self.overflow = self.errors = self.auth_failures = self.bytes_in = 0
        self.refused_now = 0
        self.keys: dict[bytes, int] = {}

    def record(self, name: bytes, key: bytes | None, miss: bool, error: bool) -> None:
        self.cmds += 1
        if name == b"GET":
            self.gets += 1
            if miss:
                self.misses += 1
        elif name == b"SET":
            self.sets += 1
        if error:
            self.errors += 1
        self._track_key(key)

    def record_refused(self, key: bytes | None) -> None:
        """A command the guard refused: it counts as attempted, nothing more."""
        self.cmds += 1
        self.refused_now += 1
        self._track_key(key)

    def _track_key(self, key: bytes | None) -> None:
        if key is not None:
            self.keyed += 1
            n = self.keys.get(key)
            if n is not None:
                self.keys[key] = n + 1
            elif len(self.keys) < KEY_CAP:
                self.keys[key] = 1
            else:
                self.overflow += 1  # assume unseen keys beyond the cap are distinct

    def close_window(self) -> list[float]:
        # A partial command with no progress. A large upload also holds a
        # partial command, but moves real bytes; a trickle of under 64 a
        # second is the slowloris signature.
        stalled = self.pending_bytes > 0 and self.cmds == 0 and self.bytes_in < 64
        self.auth_failures_total += self.auth_failures
        self.stalled_windows = self.stalled_windows + 1 if stalled else 0
        row = feature_row(
            cmds=self.cmds, gets=self.gets, sets=self.sets, misses=self.misses,
            keyed=self.keyed, distinct=len(self.keys) + self.overflow,
            top_key=max(self.keys.values(), default=0), errors=self.errors,
            auth_failures=self.auth_failures, bytes_in=self.bytes_in,
            pending_bytes=self.pending_bytes, stalled_windows=self.stalled_windows,
            auth_failures_total=self.auth_failures_total,
            executed=self.cmds - self.refused_now,
        )
        self.last_features = row
        self._reset()
        return row


class RuleDetector:
    """Hand-set thresholds: what an operator would write without a model."""

    name = "rules"

    def classify(self, rows: list[list[float]]) -> list[tuple[str, float]]:
        out = []
        for r in rows:
            f = dict(zip(FEATURE_NAMES, r))
            # Legitimate clients fail AUTH at most once or twice (a typo, a
            # stale password); three failures on one connection is guessing.
            if f["auth_failures_total"] >= MIN_AUTH_FAILURES_TO_ACT and f["auth_failures"] > 0:
                out.append(("brute_force", 1.0))
            elif f["stalled_windows"] >= 3:
                out.append(("slowloris", 1.0))
            elif f["rate"] >= 50 and f["miss_ratio"] >= 0.9 and f["distinct_ratio"] >= 0.8:
                out.append(("penetration", 1.0))
            elif f["rate"] >= 3000 and f["top_key_share"] >= 0.5:
                out.append(("hot_flood", 1.0))
            else:
                out.append(("normal", 1.0))
        return out


class MLDetector:
    """LightGBM multiclass over the same feature rows."""

    name = "ml"

    def __init__(self, model_path):
        import lightgbm as lgb
        import numpy as np

        self._np = np
        self.booster = lgb.Booster(model_file=str(model_path))

    def classify(self, rows: list[list[float]]) -> list[tuple[str, float]]:
        if not rows:
            return []
        # One thread: a model scored inside the event loop must not spin up
        # an OpenMP pool and add jitter to every other connection. (Passed
        # per call: LightGBM ignores params given when loading a model file.)
        probs = self.booster.predict(self._np.array(rows, dtype=float), num_threads=1)
        out = []
        for p in probs:
            i = int(p.argmax())
            out.append((CLASSES[i], float(p[i])))
        return out


class Guard:
    def __init__(self, mode: str = "off", model_path: str | None = None, log_path: str | None = None):
        if mode not in MODES:
            raise ValueError(f"guard mode must be one of {MODES}")
        self.conns: dict[int, ConnStats] = {}
        self._next_id = 1
        self._ml = MLDetector(model_path) if model_path else None
        self._rules = RuleDetector()
        self.mode = "off"
        self.set_mode(mode)
        self._log = open(log_path, "a") if log_path else None
        self.flag_events: list[dict] = []
        self.rejected_total = 0

    # -- configuration -----------------------------------------------------

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"guard mode must be one of {MODES}")
        if mode == "ml" and self._ml is None:
            raise ValueError("ml mode needs --guard-model")
        self.mode = mode
        if mode in ("off", "observe"):
            for c in self.conns.values():
                c.verdict, c.confidence = "normal", 0.0

    @property
    def detector(self):
        return self._ml if self.mode == "ml" else self._rules

    # -- request path ------------------------------------------------------

    def register(self, peer) -> ConnStats:
        stats = ConnStats(self._next_id, peer)
        self._next_id += 1
        self.conns[stats.id] = stats
        return stats

    def unregister(self, stats: ConnStats) -> None:
        self.conns.pop(stats.id, None)

    def admit(self, stats: ConnStats) -> str | None:
        """None to run the command; otherwise the error to send instead."""
        if self.mode in ("off", "observe") or stats.verdict == "normal":
            return None
        action, rate = MITIGATION[stats.verdict]
        if action == "close":
            stats.close_requested = True
            return f"ERR guard: connection closed (suspected {stats.verdict})"
        now = time.monotonic()
        stats.tokens = min(rate, stats.tokens + (now - stats.last_refill) * rate)
        stats.last_refill = now
        if stats.tokens >= 1:
            stats.tokens -= 1
            return None
        stats.rejected += 1
        self.rejected_total += 1
        return f"ERR guard: rate limited (suspected {stats.verdict})"

    # -- once per window ---------------------------------------------------

    def tick(self) -> None:
        if self.mode == "off" or not self.conns:
            for c in self.conns.values():
                c.close_window()
            return
        conns = list(self.conns.values())
        rows = [c.close_window() for c in conns]
        if self._log is not None:
            now = time.time()
            for c, row in zip(conns, rows):
                self._log.write(json.dumps({"t": now, "conn": c.id, "name": c.name, "features": row}) + "\n")
            self._log.flush()
        if self.mode == "observe":
            return
        # An empty window (no commands, nothing pending, no AUTH failures) is
        # no evidence either way, and the trainer drops such rows, so the
        # model has never seen one. Count it as normal instead of asking.
        active = [i for i, r in enumerate(rows) if is_active(r)]
        verdicts = dict(zip(active, self.detector.classify([rows[i] for i in active])))
        for i, c in enumerate(conns):
            label, conf = apply_policy(*verdicts.get(i, ("normal", 1.0)), rows[i])
            self._apply(c, label, conf)

    def _apply(self, c: ConnStats, label: str, conf: float) -> None:
        flagged = label != "normal" and conf >= CONFIDENCE
        if flagged:
            c.normal_streak = 0
            if c.verdict != label:
                c.verdict, c.flagged_at = label, time.time()
                c.tokens = 0.0
                self.flag_events.append({"t": c.flagged_at, "conn": c.id, "name": c.name,
                                         "verdict": label, "confidence": round(conf, 3)})
                self.flag_events = self.flag_events[-200:]
                logger.warning("guard: conn %d (%s) flagged %s (%.2f)", c.id, c.name or c.peer, label, conf)
            c.confidence = conf
            # A slowloris connection never completes a command, so admit()
            # never gets a chance to refuse one: close it from here.
            if MITIGATION[label][0] == "close" and c.closer is not None:
                c.close_requested = True
                c.closer()
        else:
            c.normal_streak += 1
            if c.verdict != "normal" and c.normal_streak >= RELEASE_WINDOWS:
                logger.info("guard: conn %d (%s) released from %s", c.id, c.name or c.peer, c.verdict)
                c.verdict, c.confidence = "normal", 0.0

    def status(self) -> list[dict]:
        out = []
        for c in self.conns.values():
            f = dict(zip(FEATURE_NAMES, c.last_features)) if c.last_features else {}
            out.append({"id": c.id, "name": c.name, "verdict": c.verdict,
                        "confidence": round(c.confidence, 3), "rejected": c.rejected,
                        "rate": round(f.get("rate", 0.0), 1), "miss_ratio": round(f.get("miss_ratio", 0.0), 3),
                        "top_key_share": round(f.get("top_key_share", 0.0), 3)})
        return out

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
