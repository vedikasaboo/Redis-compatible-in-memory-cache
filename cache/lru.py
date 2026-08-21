"""
Fixed-capacity LRU cache with optional per-key expiry.

O(1) get/set/delete. collections.OrderedDict would give that for free, but
this builds the mechanism OrderedDict hides: a hashmap for O(1) lookup plus
a doubly linked list for O(1) reordering by recency, kept in sync by hand.
Sentinel head/tail nodes mean insertion and removal at either end never
needs a None check.

Expiry is LAZY: an expired key stays in memory until something touches it,
at which point it is deleted and reported absent. A key that is never read
again would therefore leak, so expire_sample() provides the active sweep
that reclaims those (see cache.server's background expirer).
"""
from cache.clock import now_ms

NO_EXPIRY = -1  # TTL reply for "exists, but never expires"
NOT_FOUND = -2  # TTL reply for "no such key"


class _Node:
    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: bytes | None = None, value: bytes | None = None,
                 expires_at: int | None = None):
        self.key = key
        self.value = value
        self.expires_at = expires_at  # epoch ms, or None for no expiry
        self.prev: "_Node | None" = None
        self.next: "_Node | None" = None


class LRUCache:
    """
    Recency order: the list's HEAD side is most-recently-used, the TAIL
    side is least-recently-used.

    Invariant every method preserves: after get() or set() touches a live
    key, that key's node is moved to the most-recently-used end. When
    set() inserts a NEW key past capacity, the least-recently-used entry
    (at the tail end) is evicted first.

    `clock` is injectable so tests can drive expiry deterministically
    instead of sleeping.
    """

    def __init__(self, capacity: int, clock=now_ms):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._clock = clock
        self._map: dict[bytes, _Node] = {}
        # Sentinels: the list is never empty, so every splice below is
        # unconditional pointer work with no edge-case branching.
        self._head = _Node()  # MRU side
        self._tail = _Node()  # LRU side
        self._head.next = self._tail
        self._tail.prev = self._head

        # Keys carrying a TTL, so the active sweep can look only at those
        # rather than scanning the whole keyspace.
        self._expiring: dict[bytes, int] = {}
        self._sweep_pool: list[bytes] = []
        self._sweep_idx = 0

        # Observable counters. Two increments on the read path is a price
        # worth paying: without a hit rate you cannot tell a healthy cache
        # from an expensive dictionary.
        self.expired_count = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def now_ms(self) -> int:
        return self._clock()

    # -- internal list surgery -------------------------------------------

    def _unlink(self, node: _Node) -> None:
        """Detach `node` from the list. O(1)."""
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

    def _push_front(self, node: _Node) -> None:
        """Insert `node` at the most-recently-used end. O(1)."""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _drop(self, key: bytes, node: _Node) -> None:
        """Remove a node from every structure that references it."""
        del self._map[key]
        self._expiring.pop(key, None)
        self._unlink(node)

    def _track_expiry(self, key: bytes, expires_at: int | None) -> None:
        if expires_at is None:
            self._expiring.pop(key, None)
        else:
            self._expiring[key] = expires_at

    def _expired(self, node: _Node) -> bool:
        return node.expires_at is not None and self._clock() >= node.expires_at

    # -- public API ------------------------------------------------------

    def get(self, key: bytes) -> bytes | None:
        """Return the value for `key`, or None if absent or expired."""
        node = self._map.get(key)
        if node is None:
            self.misses += 1
            return None
        if self._expired(node):
            # Lazy expiry: the read is what collects the key.
            self._drop(key, node)
            self.expired_count += 1
            self.misses += 1
            return None
        self.hits += 1
        self._unlink(node)
        self._push_front(node)
        return node.value

    def set(self, key: bytes, value: bytes, expires_at: int | None = None) -> None:
        """
        Insert or update `key` -> `value`; `key` becomes most-recently-used.

        `expires_at` is an absolute epoch-ms deadline, or None for no
        expiry. A deadline already in the past deletes the key instead of
        inserting it — that case matters on WAL replay, where a `SET ...
        PXAT` recorded before a crash may well be in the past by the time
        it is replayed, and the key must not come back from the dead.
        """
        if expires_at is not None and self._clock() >= expires_at:
            self.delete(key)
            return

        node = self._map.get(key)
        if node is not None:
            # Update in place — size is unchanged, so no eviction check.
            node.value = value
            node.expires_at = expires_at
            self._track_expiry(key, expires_at)
            self._unlink(node)
            self._push_front(node)
            return

        if len(self._map) >= self.capacity:
            lru = self._tail.prev
            # Guarded by capacity > 0, so this is always a real node, never
            # the head sentinel.
            self._drop(lru.key, lru)
            self.evictions += 1

        node = _Node(key, value, expires_at)
        self._map[key] = node
        self._track_expiry(key, expires_at)
        self._push_front(node)

    def delete(self, key: bytes) -> bool:
        """Remove `key` if present. True if it was removed, False if absent."""
        node = self._map.get(key)
        if node is None:
            return False
        self._drop(key, node)
        return True

    def ttl_ms(self, key: bytes) -> int:
        """Milliseconds left, or NO_EXPIRY (-1) / NOT_FOUND (-2)."""
        node = self._map.get(key)
        if node is None:
            return NOT_FOUND
        if node.expires_at is None:
            return NO_EXPIRY
        remaining = node.expires_at - self._clock()
        if remaining <= 0:
            self._drop(key, node)
            self.expired_count += 1
            return NOT_FOUND
        return remaining

    def expire_sample(self, sample_size: int = 20) -> int:
        """
        Active expiry: check up to `sample_size` keys that carry a TTL and
        delete the expired ones. Returns how many were removed.

        Walks a rotating cursor over a snapshot of the expiring-key set
        rather than sampling randomly, so every key is visited once per
        sweep and none can be starved. The snapshot costs O(n) but is
        rebuilt only once per full sweep, so it amortizes to O(1) per key
        examined. Stale entries (already deleted) are skipped cheaply.
        """
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
            node = self._map.get(key)
            if node is None or node.expires_at is None:
                continue  # stale cursor entry; nothing to do
            if self._expired(node):
                self._drop(key, node)
                self.expired_count += 1
                removed += 1
        return removed

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def reset_stats(self) -> None:
        self.hits = self.misses = self.evictions = 0

    def items(self) -> list[tuple[bytes, bytes, int | None]]:
        """
        All LIVE entries as (key, value, expires_at), in LEAST- to
        MOST-recently-used order.

        Snapshotting uses this: re-inserting the result in the returned
        order rebuilds the same recency ordering, so a recovered cache
        evicts in the same sequence the crashed one would have. Expired
        entries are filtered out rather than persisted.

        Synchronous and atomic by construction — it never awaits, so no
        command handler can mutate the store partway through, and the
        caller gets a consistent point-in-time view.
        """
        out = []
        node = self._tail.prev
        while node is not self._head:
            if not self._expired(node):
                out.append((node.key, node.value, node.expires_at))
            node = node.prev
        return out

    def __len__(self) -> int:
        """
        Number of resident entries, INCLUDING any that have expired but
        not yet been collected. Redis's DBSIZE behaves the same way; use
        get()/ttl_ms() when you need expiry-accurate answers.
        """
        return len(self._map)

    def __contains__(self, key: bytes) -> bool:
        node = self._map.get(key)
        return node is not None and not self._expired(node)
