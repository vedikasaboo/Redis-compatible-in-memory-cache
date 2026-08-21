"""
Learned eviction policy: predict reuse distance, evict the furthest-away.

Plugs into the same sample-then-score interface as the classical policies,
so nothing else in the cache changes.

Cost model
----------
Inference happens ONLY at eviction, never on the hot path of GET or SET. An
eviction occurs at most once per insertion into a full cache, and each one
costs a single batched prediction over the sampled candidates — not one
prediction per key, and nothing at all on a cache hit. A model on the read
path would be indefensible; a model consulted once per eviction is a real
design.

The model is loaded once at construction (node startup), never per call.
"""
import logging
from pathlib import Path

import numpy as np

from cache.eviction import REGISTRY, EvictionPolicy
from cache.features import features_from_entry

logger = logging.getLogger("cache.ml_policy")


class MLPolicy(EvictionPolicy):
    name = "ml"
    # last_access, access_count, prev_gap, gap_sum, size — five int fields.
    # Charged against the memory budget so the comparison against policies
    # that need less bookkeeping stays honest.
    per_key_metadata_bytes = 40

    def __init__(self, model_path, sample_size: int = 16):
        import lightgbm as lgb

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"no model at {path}; train one with tools/train_model.py"
            )
        self.booster = lgb.Booster(model_file=str(path))
        self.sample_size = sample_size
        self.model_path = str(path)
        self.predictions = 0  # observable, to confirm inference stays off the hot path

    def select_victim(self, store):
        entries = store.sample_entries(self.sample_size)
        if not entries:
            return None
        if len(entries) == 1:
            return entries[0].key
        features = np.array(
            [features_from_entry(e, store.op) for e in entries], dtype=np.float64
        )
        # One batched call for the whole candidate set.
        predicted = self.booster.predict(features)
        self.predictions += 1
        return entries[int(np.argmax(predicted))].key

    def score(self, entry, store) -> float:
        features = np.array([features_from_entry(entry, store.op)], dtype=np.float64)
        return float(self.booster.predict(features)[0])


def _build(model_path="models/reuse.txt", sample_size=16, **_):
    return MLPolicy(model_path, sample_size=sample_size)


REGISTRY["ml"] = _build
