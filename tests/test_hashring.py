"""Tests for the consistent hash ring."""
import subprocess
import sys
from collections import Counter

import pytest

from cache.hashring import HashRing, stable_hash

NODES = [f"node-{i}" for i in range(4)]
KEYS = [f"key:{i}".encode() for i in range(20_000)]


def test_hash_is_stable_across_processes():
    """
    The property everything else depends on: two processes must agree on
    where a key lives. Python's builtin hash() is salted per process and
    would fail this.
    """
    code = "import xxhash; print(xxhash.xxh64_intdigest(b'some-key'))"
    runs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONHASHSEED": seed}).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert len(runs) == 1
    assert runs.pop() == str(stable_hash(b"some-key"))


def test_builtin_hash_would_not_be_stable():
    """Documents WHY stable_hash exists, rather than asserting it by fiat."""
    code = "print(hash('some-key'))"
    runs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONHASHSEED": seed}).stdout.strip()
        for seed in ("1", "2", "3")
    }
    assert len(runs) > 1  # salted: different every process


def test_empty_ring_returns_none():
    assert HashRing([]).get_node(b"k") is None


def test_single_node_owns_everything():
    ring = HashRing(["only"])
    assert all(ring.get_node(k) == "only" for k in KEYS[:500])


def test_lookup_is_deterministic():
    a, b = HashRing(NODES), HashRing(NODES)
    assert [a.get_node(k) for k in KEYS[:500]] == [b.get_node(k) for k in KEYS[:500]]


def test_insertion_order_does_not_change_ownership():
    a = HashRing(NODES)
    b = HashRing(list(reversed(NODES)))
    assert [a.get_node(k) for k in KEYS[:500]] == [b.get_node(k) for k in KEYS[:500]]


def test_str_and_bytes_keys_route_identically():
    ring = HashRing(NODES)
    assert ring.get_node("key:1") == ring.get_node(b"key:1")


def test_all_nodes_receive_keys():
    ring = HashRing(NODES)
    assert set(Counter(ring.get_node(k) for k in KEYS)) == set(NODES)


def test_virtual_nodes_balance_the_load():
    """One point per node divides the circle badly; many points averages out."""
    def imbalance(vnodes):
        counts = Counter(HashRing(NODES, vnodes=vnodes).get_node(k) for k in KEYS)
        loads = [counts.get(n, 0) for n in NODES]
        return max(loads) / min(loads)

    assert imbalance(150) < 1.35
    assert imbalance(150) < imbalance(1)


def test_adding_a_node_remaps_only_about_one_over_n():
    """
    The defining property. With N nodes going to N+1, roughly 1/(N+1) of
    keys should move — versus ~N/(N+1) for `hash(key) % N`.
    """
    ring = HashRing(NODES, vnodes=150)
    before = [ring.get_node(k) for k in KEYS]
    ring.add_node("node-4")
    after = [ring.get_node(k) for k in KEYS]

    moved = sum(1 for b, a in zip(before, after) if b != a) / len(KEYS)
    assert 0.12 < moved < 0.30, f"moved {moved:.1%}, expected around 20%"


def test_keys_that_move_go_only_to_the_new_node():
    """No key should be shuffled between two pre-existing nodes."""
    ring = HashRing(NODES, vnodes=150)
    before = {k: ring.get_node(k) for k in KEYS}
    ring.add_node("node-4")
    for k in KEYS:
        now = ring.get_node(k)
        if now != before[k]:
            assert now == "node-4"


def test_removing_a_node_moves_only_its_keys():
    ring = HashRing(NODES, vnodes=150)
    before = {k: ring.get_node(k) for k in KEYS}
    ring.remove_node("node-0")
    for k in KEYS:
        now = ring.get_node(k)
        assert now != "node-0"
        if before[k] != "node-0":
            assert now == before[k]  # untouched shards keep their keys


def test_add_then_remove_restores_original_mapping():
    ring = HashRing(NODES, vnodes=150)
    before = [ring.get_node(k) for k in KEYS[:2000]]
    ring.add_node("temp")
    ring.remove_node("temp")
    assert [ring.get_node(k) for k in KEYS[:2000]] == before


def test_membership_helpers():
    ring = HashRing(NODES)
    assert len(ring) == 4
    assert "node-0" in ring
    ring.add_node("node-0")  # idempotent
    assert len(ring) == 4
    ring.remove_node("nonexistent")  # no-op
    assert len(ring) == 4
    assert ring.nodes == sorted(NODES)


def test_invalid_vnodes_rejected():
    with pytest.raises(ValueError):
        HashRing(NODES, vnodes=0)
