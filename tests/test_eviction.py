"""Tests for the policy store, the eviction policies, and the workloads."""
import random

import pytest

from cache.eviction import RandomEviction, SampledLFU, SampledLRU, TrueLRU, build
from cache.store import KeyVector, PolicyStore
from tools.workloads import generate


class FakeClock:
    def __init__(self, t=1_000_000):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, ms):
        self.t += ms


# -- KeyVector -----------------------------------------------------------

def test_keyvector_add_remove_len():
    v = KeyVector()
    for k in (b"a", b"b", b"c"):
        v.add(k)
    assert len(v) == 3
    v.add(b"a")  # idempotent
    assert len(v) == 3
    v.remove(b"b")
    assert len(v) == 2
    v.remove(b"missing")  # no-op
    assert len(v) == 2


def test_keyvector_swap_removal_keeps_index_map_consistent():
    """Removal swaps in the last element; the index map must follow."""
    v = KeyVector()
    keys = [f"k{i}".encode() for i in range(50)]
    for k in keys:
        v.add(k)
    rng = random.Random(0)
    for k in rng.sample(keys, 25):
        v.remove(k)
    assert len(v) == 25
    # Every survivor must still be samplable; a broken index map shows up
    # as a KeyError or a phantom key here.
    sampled = set(v.sample(1000, rng))
    assert sampled <= set(keys)
    assert len(sampled) == 25


def test_keyvector_sample_bounds():
    v = KeyVector()
    rng = random.Random(0)
    assert v.sample(5, rng) == []
    for k in (b"a", b"b"):
        v.add(k)
    assert len(v.sample(10, rng)) == 2  # capped at population
    assert len(v.sample(2, rng)) == 2


# -- PolicyStore parity with LRUCache ------------------------------------

def test_store_basic_get_set_delete():
    s = PolicyStore(capacity=10)
    s.set(b"k", b"v")
    assert s.get(b"k") == b"v"
    assert s.delete(b"k") is True
    assert s.get(b"k") is None
    assert s.delete(b"k") is False


def test_store_respects_capacity():
    s = PolicyStore(capacity=5, policy=TrueLRU())
    for i in range(100):
        s.set(f"k{i}".encode(), b"v")
    assert len(s) == 5


def test_store_true_lru_evicts_least_recently_used():
    s = PolicyStore(capacity=3, policy=TrueLRU())
    s.set(b"a", b"v")
    s.set(b"b", b"v")
    s.set(b"c", b"v")
    s.get(b"a")        # a is now most recent; b is the victim
    s.set(b"d", b"v")
    assert s.get(b"b") is None
    assert s.get(b"a") == b"v"
    assert s.get(b"c") == b"v"


def test_store_ttl_and_expiry():
    clock = FakeClock()
    s = PolicyStore(capacity=10, clock=clock)
    s.set(b"k", b"v", expires_at=clock.t + 1000)
    assert s.get(b"k") == b"v"
    clock.advance(1500)
    assert s.get(b"k") is None
    assert len(s) == 0


def test_store_expire_sample():
    clock = FakeClock()
    s = PolicyStore(capacity=1000, clock=clock)
    for i in range(30):
        s.set(f"k{i}".encode(), b"v", expires_at=clock.t + 100)
    clock.advance(200)
    removed = sum(s.expire_sample(20) for _ in range(10))
    assert removed == 30
    assert len(s) == 0


def test_store_items_excludes_expired():
    clock = FakeClock()
    s = PolicyStore(capacity=10, clock=clock)
    s.set(b"live", b"v")
    s.set(b"dead", b"v", expires_at=clock.t + 10)
    clock.advance(100)
    keys = {k for k, _, _ in s.items()}
    assert keys == {b"live"}


def test_store_tracks_hit_rate():
    s = PolicyStore(capacity=10)
    s.set(b"k", b"v")
    s.get(b"k")
    s.get(b"missing")
    assert s.hits == 1
    assert s.misses == 1
    assert s.hit_rate == 0.5


def test_reset_stats_keeps_contents():
    s = PolicyStore(capacity=10)
    s.set(b"k", b"v")
    s.get(b"k")
    s.reset_stats()
    assert s.hits == 0 and s.misses == 0
    assert s.get(b"k") == b"v"  # data survived the counter reset


def test_access_metadata_updates():
    s = PolicyStore(capacity=10)
    s.set(b"k", b"v")
    for _ in range(4):
        s.get(b"k")
    entry = s._map[b"k"]
    assert entry.access_count == 5
    assert entry.prev_gap > 0
    assert entry.last_access == s.op


# -- policies ------------------------------------------------------------

def test_sampled_lru_prefers_older_entries():
    s = PolicyStore(capacity=100, policy=SampledLRU(sample_size=100), seed=1)
    for i in range(10):
        s.set(f"k{i}".encode(), b"v")
    s.get(b"k9")  # freshen the newest
    victim = s.policy.select_victim(s)
    assert victim == b"k0"  # oldest last_access


def test_sampled_lfu_prefers_least_frequent():
    s = PolicyStore(capacity=100, policy=SampledLFU(sample_size=100), seed=1)
    for i in range(5):
        s.set(f"k{i}".encode(), b"v")
    for _ in range(10):  # make everything except k3 popular
        for i in (0, 1, 2, 4):
            s.get(f"k{i}".encode())
    assert s.policy.select_victim(s) == b"k3"


def test_lfu_breaks_ties_by_age():
    s = PolicyStore(capacity=100, policy=SampledLFU(sample_size=100), seed=1)
    for i in range(4):
        s.set(f"k{i}".encode(), b"v")  # all have access_count 1
    assert s.policy.select_victim(s) == b"k0"  # oldest of the tied set


def test_random_policy_returns_a_resident_key():
    s = PolicyStore(capacity=100, policy=RandomEviction(sample_size=1), seed=3)
    for i in range(20):
        s.set(f"k{i}".encode(), b"v")
    assert s.policy.select_victim(s) in s._map


def test_policies_declare_metadata_cost():
    """Fair comparison depends on these being declared, not assumed equal."""
    assert RandomEviction().per_key_metadata_bytes < SampledLRU().per_key_metadata_bytes
    assert SampledLRU().per_key_metadata_bytes < SampledLFU().per_key_metadata_bytes


def test_build_registry():
    assert isinstance(build("true-lru"), TrueLRU)
    assert isinstance(build("sampled-lru", sample_size=7), SampledLRU)
    assert build("sampled-lru", sample_size=7).sample_size == 7
    with pytest.raises(ValueError):
        build("nonexistent")


def test_empty_store_has_no_victim():
    s = PolicyStore(capacity=10, policy=SampledLRU())
    assert s.policy.select_victim(s) is None


# -- workloads -----------------------------------------------------------

@pytest.mark.parametrize("name", ["zipfian", "scan", "cyclic", "mixed"])
def test_workloads_are_reproducible(name):
    a = generate(name, 5000, 1000, 1.1, 500, seed=7)
    b = generate(name, 5000, 1000, 1.1, 500, seed=7)
    assert a == b
    assert len(a) == 5000


def test_different_seeds_give_different_traces():
    a = generate("zipfian", 5000, 1000, 1.1, 500, seed=1)
    b = generate("zipfian", 5000, 1000, 1.1, 500, seed=2)
    assert a != b


def test_zipfian_is_skewed():
    from collections import Counter
    trace = generate("zipfian", 50_000, 5_000, 1.2, 0, seed=3)
    counts = Counter(trace)
    top = sum(c for _, c in counts.most_common(50))
    assert top / len(trace) > 0.25  # a few keys carry a large share


def test_scan_visits_keys_in_order():
    trace = generate("scan", 100, 1000, 1.1, 0, seed=1)
    assert trace[:3] == ["key:0", "key:1", "key:2"]


def test_cyclic_repeats_exactly():
    trace = generate("cyclic", 100, 1000, 1.1, 10, seed=1)
    assert trace[:10] == trace[10:20]
    assert len(set(trace)) == 10


def test_unknown_workload_rejected():
    with pytest.raises(ValueError):
        generate("nope", 10, 10, 1.0, 10, seed=1)
