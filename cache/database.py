"""
A simulated backing database, so a miss storm can be measured.

A cache exists to protect something slower. Without that something, a
cache miss costs nothing and "the database got flooded" cannot be seen,
let alone measured. This stands in for it: every fetch is counted into
100 ms buckets, and fetches slow down once the load passes the database's
capacity, the way a real database queues under overload.

Overload model: base latency while under `capacity_qps`; above it,
latency grows in proportion to the overload (twice the capacity, twice the
latency). Deliberately simple — the point is that a storm is visible as
load above a line and as slower responses, not to model any one database.

It lives in-process and is thread-safe, so a benchmark's client threads
all hit the same instance.
"""
import threading
import time

BUCKET_SECONDS = 0.1


class SimulatedDatabase:
    def __init__(self, latency_ms: float = 0.5, capacity_qps: float = 2000.0):
        self.latency = latency_ms / 1000.0
        self.capacity_qps = capacity_qps
        self.total = 0
        self._buckets: dict[int, int] = {}
        self._written: dict[bytes, bytes] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _bucket(t: float) -> int:
        return int(t / BUCKET_SECONDS)

    def current_qps(self) -> float:
        """Load over the last full bucket, in queries per second."""
        with self._lock:
            return self._buckets.get(self._bucket(time.time()) - 1, 0) / BUCKET_SECONDS

    def get(self, key) -> bytes:
        if isinstance(key, str):
            key = key.encode()
        now = time.time()
        b = self._bucket(now)
        with self._lock:
            self.total += 1
            self._buckets[b] = self._buckets.get(b, 0) + 1
            recent = self._buckets.get(b - 1, 0) / BUCKET_SECONDS
        overload = max(1.0, recent / self.capacity_qps)
        time.sleep(self.latency * overload)
        with self._lock:
            return self._written.get(key, b"value-" + key)

    def set(self, key, value) -> None:
        """A write. Not counted as load: the experiments measure read storms."""
        key = key.encode() if isinstance(key, str) else key
        value = value.encode() if isinstance(value, str) else value
        with self._lock:
            self._written[key] = value

    def series(self, start: float, end: float) -> list[tuple[float, float]]:
        """(seconds since start, queries/sec) for every bucket in [start, end)."""
        with self._lock:
            first, last = self._bucket(start), self._bucket(end)
            return [((b - first) * BUCKET_SECONDS, self._buckets.get(b, 0) / BUCKET_SECONDS)
                    for b in range(first, last)]
