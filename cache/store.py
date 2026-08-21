"""
Instrumented store with a pluggable eviction policy.

Relationship to cache.lru.LRUCache
----------------------------------
LRUCache is the minimal, fastest thing that can serve traffic: a hashmap
plus a linked list, true LRU, nothing else. It stays the server's default.

PolicyStore is the same idea plus the machinery the eviction experiments
need — O(1) random sampling, and per-key access statistics that the LFU and
learned policies score on. That bookkeeping costs time and memory on every
access, which is exactly why it is a separate class rather than bolted onto
the hot path.

Both expose the same interface, so the server can run either.

Sampling
--------
Python dicts cannot be indexed, so `random.choice` over the keyspace is not
directly possible. KeyVector keeps a dense list of keys plus a key->index
map, giving O(1) insert, O(1) removal (swap with the last element), and
O(1) uniform sampling. This is what makes "sample N candidates" cheap
regardless of how many keys are resident.
"""
import random

from cache.clock import now_ms
from cache.features import GAP_ALPHA
from cache.lru import NO_EXPIRY, NOT_FOUND


class Entry:
    __slots__ = ("key", "value", "expires_at", "last_access", "access_count",
                 "ewma_gap", "prev", "next")

    def __init__(self, key=None, value=None, expires_at=None, op=0):
        self.key = key
        self.value = value
        self.expires_at = expires_at
        # Access statistics, in LOGICAL time (operation ordinals), not
        # wall clock: reuse distance is defined in accesses, and logical
        # time makes replays reproducible regardless of machine speed.
        self.last_access = op
        self.access_count = 1
        # One field instead of a running sum plus the previous gap: every
        # stored field is charged against the memory budget, and an EWMA
        # carries the same signal in a third of the space.
        self.ewma_gap = 0.0
        self.prev: "Entry | None" = None
        self.next: "Entry | None" = None

    @property
    def size(self) -> int:
        """Derived, never stored — len() on bytes is O(1)."""
        return len(self.value) if self.value is not None else 0

    def touch(self, op: int) -> None:
        gap = op - self.last_access
        # Seed on the first observed gap rather than decaying up from zero,
        # which would understate a slow key for its first several accesses.
        self.ewma_gap = float(gap) if self.access_count == 1 else (
            GAP_ALPHA * gap + (1.0 - GAP_ALPHA) * self.ewma_gap
        )
        self.access_count += 1
        self.last_access = op


class KeyVector:
    """Dense key list supporting O(1) add, remove, and uniform sampling."""

    def __init__(self):
        self._keys: list[bytes] = []
        self._pos: dict[bytes, int] = {}

    def add(self, key: bytes) -> None:
        if key in self._pos:
            return
        self._pos[key] = len(self._keys)
        self._keys.append(key)

    def remove(self, key: bytes) -> None:
        idx = self._pos.pop(key, None)
        if idx is None:
            return
        last = self._keys.pop()
        if idx < len(self._keys):  # not the element we just popped
            self._keys[idx] = last
            self._pos[last] = idx

    def sample(self, n: int, rng: random.Random) -> list[bytes]:
        size = len(self._keys)
        if size == 0:
            return []
        if n >= size:
            return list(self._keys)
        # Sampling WITH replacement, like Redis: cheaper than a distinct
        # sample and the duplicate rate is negligible when n << size.
        return [self._keys[rng.randrange(size)] for _ in range(n)]

    def __len__(self) -> int:
        return len(self._keys)


class PolicyStore:
    def __init__(self, capacity: int, policy=None, clock=now_ms, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        from cache.eviction import TrueLRU  # local import avoids a cycle

        self.capacity = capacity
        self.policy = policy or TrueLRU()
        self._clock = clock
        self._rng = random.Random(seed)
        self._map: dict[bytes, Entry] = {}
        self._vector = KeyVector()

        self._head = Entry()
        self._tail = Entry()
        self._head.next = self._tail
        self._tail.prev = self._head

        self._expiring: dict[bytes, int] = {}
        self._sweep_pool: list[bytes] = []
        self._sweep_idx = 0

        self.op = 0            # logical clock: one tick per access
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expired_count = 0

    # -- interface shared with LRUCache -----------------------------------

    def now_ms(self) -> int:
        return self._clock()

    def _unlink(self, entry: Entry) -> None:
        entry.prev.next = entry.next
        entry.next.prev = entry.prev
        entry.prev = entry.next = None

    def _push_front(self, entry: Entry) -> None:
        entry.next = self._head.next
        entry.prev = self._head
        self._head.next.prev = entry
        self._head.next = entry

    def _drop(self, key: bytes, entry: Entry) -> None:
        del self._map[key]
        self._vector.remove(key)
        self._expiring.pop(key, None)
        self._unlink(entry)

    def _expired(self, entry: Entry) -> bool:
        return entry.expires_at is not None and self._clock() >= entry.expires_at

    def get(self, key: bytes) -> bytes | None:
        self.op += 1
        entry = self._map.get(key)
        if entry is None:
            self.misses += 1
            return None
        if self._expired(entry):
            self._drop(key, entry)
            self.expired_count += 1
            self.misses += 1
            return None
        self.hits += 1
        entry.touch(self.op)
        self._unlink(entry)
        self._push_front(entry)
        return entry.value

    def set(self, key: bytes, value: bytes, expires_at: int | None = None) -> None:
        self.op += 1
        if expires_at is not None and self._clock() >= expires_at:
            self.delete(key)
            return

        entry = self._map.get(key)
        if entry is not None:
            entry.value = value
            entry.expires_at = expires_at
            entry.touch(self.op)
            self._track_expiry(key, expires_at)
            self._unlink(entry)
            self._push_front(entry)
            return

        if len(self._map) >= self.capacity:
            self._evict_one()

        entry = Entry(key, value, expires_at, self.op)
        self._map[key] = entry
        self._vector.add(key)
        self._track_expiry(key, expires_at)
        self._push_front(entry)

    def _evict_one(self) -> None:
        victim = self.policy.select_victim(self)
        if victim is None:
            return
        entry = self._map.get(victim)
        if entry is not None:
            self._drop(victim, entry)
            self.evictions += 1

    def _track_expiry(self, key: bytes, expires_at: int | None) -> None:
        if expires_at is None:
            self._expiring.pop(key, None)
        else:
            self._expiring[key] = expires_at

    def delete(self, key: bytes) -> bool:
        entry = self._map.get(key)
        if entry is None:
            return False
        self._drop(key, entry)
        return True

    def ttl_ms(self, key: bytes) -> int:
        entry = self._map.get(key)
        if entry is None:
            return NOT_FOUND
        if entry.expires_at is None:
            return NO_EXPIRY
        remaining = entry.expires_at - self._clock()
        if remaining <= 0:
            self._drop(key, entry)
            self.expired_count += 1
            return NOT_FOUND
        return remaining

    def expire_sample(self, sample_size: int = 20) -> int:
        removed = 0
        checked = 0
        while checked < sample_size and self._expiring:
            if self._sweep_idx >= len(self._sweep_pool):
                self._sweep_pool = list(self._expiring)
                self._sweep_idx = 0
                if not self._sweep_pool:
                    break
            key = self._sweep_pool[self._sweep_idx]
            self._sweep_idx += 1
            checked += 1
            entry = self._map.get(key)
            if entry is None or entry.expires_at is None:
                continue
            if self._expired(entry):
                self._drop(key, entry)
                self.expired_count += 1
                removed += 1
        return removed

    def items(self) -> list[tuple[bytes, bytes, int | None]]:
        out = []
        entry = self._tail.prev
        while entry is not self._head:
            if not self._expired(entry):
                out.append((entry.key, entry.value, entry.expires_at))
            entry = entry.prev
        return out

    def __len__(self) -> int:
        return len(self._map)

    def __contains__(self, key: bytes) -> bool:
        entry = self._map.get(key)
        return entry is not None and not self._expired(entry)

    # -- used by eviction policies ---------------------------------------

    def sample_entries(self, n: int) -> list[Entry]:
        return [self._map[k] for k in self._vector.sample(n, self._rng) if k in self._map]

    def lru_entry(self) -> Entry | None:
        entry = self._tail.prev
        return None if entry is self._head else entry

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def reset_stats(self) -> None:
        """Clear counters after warmup, keeping the cache contents."""
        self.hits = self.misses = self.evictions = 0
