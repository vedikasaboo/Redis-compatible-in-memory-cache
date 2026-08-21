"""
Consistent hash ring with virtual nodes.

The problem it solves
---------------------
The obvious way to shard N keys over M nodes is `hash(key) % M`. It spreads
keys evenly, but changing M remaps almost EVERYTHING: with 3 nodes going to
4, roughly 3/4 of all keys change owner, so nearly the whole cache is lost
at once and every miss goes to the backing store simultaneously.

Consistent hashing instead maps both keys and nodes onto the same circular
hash space. A key is owned by the first node encountered walking clockwise
from the key's position. Adding a node inserts one new point on the circle
and steals only the keys between it and its predecessor — about 1/N of the
keyspace. Every other key keeps its owner.

Why virtual nodes
-----------------
With one point per node, the arcs between points are wildly uneven — a
handful of random points on a circle do not divide it into equal pieces, so
one node can easily own several times its fair share. Placing V points per
node (150 here) averages V independent arcs per node, and the spread of
that average shrinks like 1/sqrt(V). Measured in tools/plot_distribution.py.

Why not Python's hash()
-----------------------
`hash(b"foo")` is salted per process (PYTHONHASHSEED), so two processes
disagree about where a key lives. Routing decided on the client would not
match routing computed anywhere else, and a restart would silently reshard
everything. This uses xxhash, which is stable across processes and runs.
"""
import bisect

import xxhash

DEFAULT_VNODES = 150


def stable_hash(data: bytes) -> int:
    """Process-stable 64-bit hash. Never use the builtin hash() here."""
    return xxhash.xxh64_intdigest(data)


class HashRing:
    def __init__(self, nodes=None, vnodes: int = DEFAULT_VNODES, hash_fn=stable_hash):
        if vnodes <= 0:
            raise ValueError("vnodes must be positive")
        self.vnodes = vnodes
        self._hash = hash_fn
        # Two parallel sorted arrays rather than a list of tuples: bisect
        # searches _points directly, and the lookup is the hot path.
        self._points: list[int] = []
        self._owners: list[str] = []
        self._nodes: set[str] = set()
        for node in nodes or []:
            self.add_node(node)

    # -- membership ------------------------------------------------------

    def _vnode_points(self, node: str) -> list[int]:
        return [self._hash(f"{node}#{i}".encode()) for i in range(self.vnodes)]

    def add_node(self, node: str) -> None:
        if node in self._nodes:
            return
        self._nodes.add(node)
        for point in self._vnode_points(node):
            idx = bisect.bisect_left(self._points, point)
            # A collision would make ownership depend on insertion order;
            # with 64-bit hashes it is vanishingly rare, but skipping the
            # duplicate keeps the ring deterministic if it ever happens.
            if idx < len(self._points) and self._points[idx] == point:
                continue
            self._points.insert(idx, point)
            self._owners.insert(idx, node)

    def remove_node(self, node: str) -> None:
        if node not in self._nodes:
            return
        self._nodes.discard(node)
        kept = [(p, o) for p, o in zip(self._points, self._owners) if o != node]
        self._points = [p for p, _ in kept]
        self._owners = [o for _, o in kept]

    @property
    def nodes(self) -> list[str]:
        return sorted(self._nodes)

    # -- lookup ----------------------------------------------------------

    def get_node(self, key: bytes) -> str | None:
        """
        Owner of `key`: the first ring point at or clockwise from the key's
        hash. O(log(V*N)) via binary search over the sorted points.
        """
        if not self._points:
            return None
        if isinstance(key, str):
            key = key.encode()
        idx = bisect.bisect(self._points, self._hash(key))
        if idx == len(self._points):
            idx = 0  # past the last point: wrap around the circle
        return self._owners[idx]

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node: str) -> bool:
        return node in self._nodes
