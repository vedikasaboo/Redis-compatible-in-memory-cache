"""
Fixed-capacity LRU cache. YOU IMPLEMENT THIS.

Must be O(1) for get/set/delete. collections.OrderedDict would give you
O(1) for free, but the point of this exercise is to build the mechanism
OrderedDict hides: a hashmap for O(1) lookup, plus a doubly linked list for
O(1) reordering by recency, kept in sync by hand. Use sentinel head/tail
nodes so insertion and removal at either end never needs a None check.
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

    Invariant every method must preserve: after get() or set() touches an
    existing key, that key's node must be moved to the most-recently-used
    end. When set() would insert a NEW key past capacity, the
    least-recently-used entry (at the tail end) must be evicted first.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._map: dict[bytes, _Node] = {}
        # YOU IMPLEMENT THIS: create head/tail sentinel _Node objects and
        # link them to each other (head.next = tail, tail.prev = head) so
        # the list is never truly empty and every insert/remove is a plain
        # pointer splice with no edge-case branching.
        raise NotImplementedError("YOU IMPLEMENT THIS")

    def get(self, key: bytes) -> bytes | None:
        """
        Return the value for `key`, or None if `key` is absent.
        On a hit, move `key`'s node to the most-recently-used end.
        Must be O(1).
        """
        raise NotImplementedError("YOU IMPLEMENT THIS")

    def set(self, key: bytes, value: bytes) -> None:
        """
        Insert or update `key` -> `value`; `key` becomes most-recently-used.

        If `key` already exists, update its value in place and move it to
        the most-recently-used end (do not change self size accounting
        twice for an update).

        If `key` is new and inserting it would make len(self) exceed
        self.capacity, evict the least-recently-used entry (tail end)
        first — evict exactly one entry per new key over capacity; this
        cache is never allowed to hold more than self.capacity entries.

        Must be O(1) amortized.
        """
        raise NotImplementedError("YOU IMPLEMENT THIS")

    def delete(self, key: bytes) -> bool:
        """
        Remove `key` if present. Return True if it was present and removed,
        False if it was already absent. Must be O(1).
        """
        raise NotImplementedError("YOU IMPLEMENT THIS")

    def __len__(self) -> int:
        return len(self._map)

    def __contains__(self, key: bytes) -> bool:
        return key in self._map
