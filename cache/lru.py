"""
Fixed-capacity LRU cache.

O(1) get/set/delete. collections.OrderedDict would give that for free, but
this builds the mechanism OrderedDict hides: a hashmap for O(1) lookup plus
a doubly linked list for O(1) reordering by recency, kept in sync by hand.
Sentinel head/tail nodes mean insertion and removal at either end never
needs a None check.
"""


class _Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: bytes | None = None, value: bytes | None = None):
        self.key = key
        self.value = value
        self.prev: "_Node | None" = None
        self.next: "_Node | None" = None


class LRUCache:
    """
    Recency order: the list's HEAD side is most-recently-used, the TAIL
    side is least-recently-used.

    Invariant every method preserves: after get() or set() touches an
    existing key, that key's node is moved to the most-recently-used end.
    When set() inserts a NEW key past capacity, the least-recently-used
    entry (at the tail end) is evicted first.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._map: dict[bytes, _Node] = {}
        # Sentinels: the list is never empty, so every splice below is
        # unconditional pointer work with no edge-case branching.
        self._head = _Node()  # MRU side
        self._tail = _Node()  # LRU side
        self._head.next = self._tail
        self._tail.prev = self._head

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

    # -- public API ------------------------------------------------------

    def get(self, key: bytes) -> bytes | None:
        """Return the value for `key`, or None if absent. Refreshes recency."""
        node = self._map.get(key)
        if node is None:
            return None
        self._unlink(node)
        self._push_front(node)
        return node.value

    def set(self, key: bytes, value: bytes) -> None:
        """Insert or update `key` -> `value`; `key` becomes most-recently-used."""
        node = self._map.get(key)
        if node is not None:
            # Update in place — size is unchanged, so no eviction check.
            node.value = value
            self._unlink(node)
            self._push_front(node)
            return

        if len(self._map) >= self.capacity:
            lru = self._tail.prev
            # Guarded by capacity > 0, so this is always a real node, never
            # the head sentinel.
            del self._map[lru.key]
            self._unlink(lru)

        node = _Node(key, value)
        self._map[key] = node
        self._push_front(node)

    def delete(self, key: bytes) -> bool:
        """Remove `key` if present. True if it was removed, False if absent."""
        node = self._map.pop(key, None)
        if node is None:
            return False
        self._unlink(node)
        return True

    def items(self) -> list[tuple[bytes, bytes]]:
        """
        All entries in LEAST- to MOST-recently-used order.

        Snapshotting uses this: re-inserting the result in the returned
        order rebuilds the same recency ordering, so a recovered cache
        evicts in the same sequence the crashed one would have.

        Synchronous and atomic by construction — it never awaits, so no
        command handler can mutate the store partway through, and the
        caller gets a consistent point-in-time view.
        """
        out = []
        node = self._tail.prev
        while node is not self._head:
            out.append((node.key, node.value))
            node = node.prev
        return out

    def __len__(self) -> int:
        return len(self._map)

    def __contains__(self, key: bytes) -> bool:
        return key in self._map
