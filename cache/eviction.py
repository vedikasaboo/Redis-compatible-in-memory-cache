"""
Eviction policies behind one interface: sample N candidates, score each,
evict the worst.

`score()` returns EVICTION PRIORITY — higher means "throw this out first".
A policy only has to answer "how bad is it to keep this key", and the
sampling, the tie-breaking and the memory accounting are shared.

True LRU is the deliberate exception. It does not sample; it takes the
global least-recently-used entry straight off the recency list. It is
included precisely so the sampled policies have an exact baseline to be
measured against — real Redis samples because maintaining exact global
order across a large keyspace is not worth the cost, and the interesting
question is how much accuracy that sacrifices.

Memory fairness
---------------
Policies do not cost the same per key. LFU needs a counter; the learned
policy needs several statistics. Comparing them at equal KEY COUNT would
quietly hand the metadata-hungry policy free memory. Each policy therefore
declares `per_key_metadata_bytes`, and the benchmark converts a fixed BYTE
budget into a per-policy key capacity. See tools/bench_eviction.py.
"""


class EvictionPolicy:
    name = "base"
    sample_size = 0
    # Bytes of per-key bookkeeping this policy needs beyond the key, the
    # value, and the dict/linked-list overhead every policy pays. Counted
    # as 8 bytes per Python-int field on a 64-bit build.
    per_key_metadata_bytes = 0

    def score(self, entry, store) -> float:
        raise NotImplementedError

    def select_victim(self, store) -> bytes | None:
        candidates = store.sample_entries(self.sample_size)
        if not candidates:
            return None
        return max(candidates, key=lambda e: self.score(e, store)).key

    def __repr__(self) -> str:
        return f"<{self.name}>"


class TrueLRU(EvictionPolicy):
    """
    Exact LRU: evict the globally least-recently-used key.

    O(1) because the recency list is already maintained on every access.
    The accuracy ceiling the sampled policies are compared against.
    """
    name = "true-lru"
    per_key_metadata_bytes = 16  # prev/next pointers

    def select_victim(self, store):
        entry = store.lru_entry()
        return entry.key if entry else None

    def score(self, entry, store) -> float:
        return store.op - entry.last_access


class SampledLRU(EvictionPolicy):
    """
    What real Redis does: sample N keys, evict the least recently used of
    those. Approximates true LRU without maintaining exact global order.
    """
    name = "sampled-lru"
    per_key_metadata_bytes = 8  # last_access

    def __init__(self, sample_size: int = 5):
        self.sample_size = sample_size
        self.name = f"sampled-lru-{sample_size}"

    def score(self, entry, store) -> float:
        return store.op - entry.last_access  # older = evict sooner


class SampledLFU(EvictionPolicy):
    """
    Sample N, evict the least frequently used.

    Ties broken by age, otherwise a burst of equally-cold keys would be
    evicted in arbitrary order. Its known weakness is cache pollution: a
    key that was hot long ago keeps a high count and resists eviction even
    after it stops being used.
    """
    name = "sampled-lfu"
    per_key_metadata_bytes = 16  # access_count + last_access

    def __init__(self, sample_size: int = 5):
        self.sample_size = sample_size
        self.name = f"sampled-lfu-{sample_size}"

    def score(self, entry, store):
        return (-entry.access_count, store.op - entry.last_access)


class RandomEviction(EvictionPolicy):
    """
    Evict a uniformly random key. The floor: any policy that cannot beat
    this is not earning the bookkeeping it costs.
    """
    name = "random"
    per_key_metadata_bytes = 0

    def __init__(self, sample_size: int = 1):
        self.sample_size = sample_size

    def score(self, entry, store) -> float:
        return 0.0


REGISTRY = {
    "true-lru": lambda **kw: TrueLRU(),
    "sampled-lru": lambda sample_size=5, **kw: SampledLRU(sample_size),
    "sampled-lfu": lambda sample_size=5, **kw: SampledLFU(sample_size),
    "random": lambda sample_size=1, **kw: RandomEviction(sample_size),
}


def build(name: str, **kwargs) -> EvictionPolicy:
    """
    Construct a policy by name.

    'ml' is registered on import of cache.ml_policy, which is deferred so
    that nothing here depends on lightgbm unless a learned policy is
    actually requested.
    """
    if name == "ml" and "ml" not in REGISTRY:
        import cache.ml_policy  # noqa: F401  (registers itself)
    if name not in REGISTRY:
        raise ValueError(f"unknown eviction policy {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
