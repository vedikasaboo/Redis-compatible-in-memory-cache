"""
Tests for the learned eviction policy and its features.

The feature tests matter more than they look: training/serving skew is the
quiet failure mode of an ML system, and the only defence here is that both
sides call the same function. These assert that they do.
"""
from pathlib import Path

import pytest

from cache.features import FEATURE_NAMES, feature_row, features_from_entry
from cache.store import PolicyStore
from tools.train_model import NEVER_SENTINEL, build_position_index, next_use_after

MODEL = Path(__file__).resolve().parent.parent / "models" / "mixed.txt"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="run tools/run_experiments.py first")


# -- features ------------------------------------------------------------

def test_feature_row_shape_matches_names():
    row = feature_row(now_op=100, last_access=90, access_count=5,
                      gap_sum=40, prev_gap=10, size=64)
    assert len(row) == len(FEATURE_NAMES)


def test_feature_values():
    row = feature_row(now_op=100, last_access=90, access_count=5,
                      gap_sum=40, prev_gap=10, size=64)
    age, count, avg_gap, prev_gap, size = row
    assert age == 10
    assert count == 5
    assert avg_gap == 10  # gap_sum 40 over (5-1) gaps
    assert prev_gap == 10
    assert size == 64


def test_avg_gap_safe_on_first_access():
    """A key seen once has zero gaps; the divisor must not be zero."""
    row = feature_row(now_op=5, last_access=5, access_count=1,
                      gap_sum=0, prev_gap=0, size=1)
    assert row[2] == 0.0


def test_live_and_offline_features_agree():
    """
    The anti-skew guarantee: the live path and the trainer must produce
    identical vectors from identical history.
    """
    store = PolicyStore(capacity=10)
    store.set(b"k", b"v" * 64)
    for _ in range(4):
        store.get(b"k")
    entry = store._map[b"k"]

    live = features_from_entry(entry, store.op)
    offline = feature_row(
        now_op=store.op, last_access=entry.last_access,
        access_count=entry.access_count, gap_sum=entry.gap_sum,
        prev_gap=entry.prev_gap, size=entry.size,
    )
    assert live == offline


def test_features_use_only_past_information():
    """Age must grow as time passes without the key being touched."""
    store = PolicyStore(capacity=10)
    store.set(b"k", b"v")
    entry = store._map[b"k"]
    first = features_from_entry(entry, store.op)
    for i in range(20):
        store.set(f"other{i}".encode(), b"v")
    later = features_from_entry(entry, store.op)
    assert later[0] > first[0]      # age increased
    assert later[1] == first[1]     # access_count unchanged


# -- labels --------------------------------------------------------------

def test_position_index():
    index = build_position_index(["a", "b", "a", "c", "a"])
    assert index["a"] == [0, 2, 4]
    assert index["b"] == [1]


def test_next_use_after():
    positions = [0, 2, 4]
    assert next_use_after(positions, 0) == 2   # next use is at 2
    assert next_use_after(positions, 2) == 2   # next use is at 4
    assert next_use_after(positions, 4) == NEVER_SENTINEL


def test_never_reused_gets_sentinel():
    assert next_use_after([5], 10) == NEVER_SENTINEL


# -- policy --------------------------------------------------------------

@needs_model
def test_ml_policy_loads_and_picks_a_resident_key():
    from cache.ml_policy import MLPolicy

    policy = MLPolicy(MODEL, sample_size=16)
    store = PolicyStore(capacity=100, policy=policy, seed=5)
    for i in range(50):
        store.set(f"k{i}".encode(), b"v" * 64)
    victim = policy.select_victim(store)
    assert victim in store._map


@needs_model
def test_ml_policy_evicts_highest_predicted_reuse_distance():
    from cache.ml_policy import MLPolicy

    policy = MLPolicy(MODEL, sample_size=64)
    store = PolicyStore(capacity=100, policy=policy, seed=5)
    for i in range(40):
        store.set(f"k{i}".encode(), b"v" * 64)

    victim = policy.select_victim(store)
    scores = {e.key: policy.score(e, store) for e in store.sample_entries(64)}
    assert scores[victim] == pytest.approx(max(scores.values()))


@needs_model
def test_inference_only_happens_at_eviction():
    """
    The cost argument for the whole design: a model on the GET path would
    be indefensible. This asserts reads and non-evicting writes are free.
    """
    from cache.ml_policy import MLPolicy

    policy = MLPolicy(MODEL, sample_size=16)
    store = PolicyStore(capacity=20, policy=policy, seed=5)
    for i in range(20):  # fills exactly to capacity, no eviction yet
        store.set(f"k{i}".encode(), b"v" * 64)
    assert policy.predictions == 0

    for i in range(20):  # pure reads
        store.get(f"k{i}".encode())
    assert policy.predictions == 0

    store.set(b"overflow", b"v" * 64)  # now it must evict
    assert policy.predictions == 1


@needs_model
def test_missing_model_is_a_clear_error():
    from cache.ml_policy import MLPolicy

    with pytest.raises(FileNotFoundError, match="train one"):
        MLPolicy("/nonexistent/model.txt")


@needs_model
def test_ml_policy_charges_for_its_metadata():
    """Fairness: the learned policy must not get its bookkeeping for free."""
    from cache.eviction import SampledLFU, SampledLRU
    from cache.ml_policy import MLPolicy

    ml = MLPolicy(MODEL)
    assert ml.per_key_metadata_bytes > SampledLFU().per_key_metadata_bytes
    assert ml.per_key_metadata_bytes > SampledLRU().per_key_metadata_bytes
