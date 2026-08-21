"""
Feature extraction for the learned eviction policy.

ONE definition, imported by both the offline trainer and the live scorer.
This is the single most important structural decision in the ML part: if
the trainer and the cache computed features separately, any drift between
them — a different gap definition, a different time origin, an off-by-one
in the access count — would silently poison predictions at serving time
while every offline metric still looked healthy. Training/serving skew is
the classic way an ML system fails quietly, and sharing the function is the
cheapest defence.

All features use only PAST information about a key as of some moment `op`.
Nothing here may consult the future; the label does that, and the label is
never available at serving time.

Time is LOGICAL — operation ordinals, not wall clock. Reuse distance is
defined in operations, so measuring in operations keeps features and labels
in the same units and makes replays reproducible across machines.

Memory discipline
-----------------
Every STORED field is charged against the cache's memory budget, so the
feature set is deliberately small: three stored fields per key. Anything
derivable from those (or from the value itself) is computed here for free
rather than stored. `age_over_gap` in particular costs nothing and carries
real signal, so it is exactly the kind of feature worth having.
"""

FEATURE_NAMES = ["age", "access_count", "ewma_gap", "age_over_gap"]

# Weight on the newest gap when updating the running estimate. High enough
# to react when a key's rhythm changes, low enough not to be defined by a
# single outlier.
GAP_ALPHA = 0.3


def feature_row(now_op: int, last_access: int, access_count: int,
                ewma_gap: float) -> list[float]:
    """
    Features for one key as of logical time `now_op`.

    age          — operations since this key was last touched. The signal
                   LRU uses, kept so the model can learn when recency is
                   and is not predictive rather than having it assumed.
    access_count — how often it has been touched. The signal LFU uses.
    ewma_gap     — exponentially weighted mean spacing between this key's
                   accesses. A key touched every ~10 operations behaves
                   very differently from one touched every ~10,000, even at
                   identical age. One field replaces the running sum and
                   the previous gap that an exact mean would need.
    age_over_gap — how overdue the key is, in units of its own rhythm.
                   Derived, so it costs no memory. This is the feature that
                   lets one model serve very different workloads: on
                   Zipfian traffic a hot key sits near 1 and a dead key
                   climbs, while in a cyclic scan the key closest to its
                   next use is the one with the LARGEST ratio — the same
                   number, read correctly in each regime, which a fixed
                   heuristic like LRU cannot do.
    """
    age = now_op - last_access
    return [
        float(age),
        float(access_count),
        float(ewma_gap),
        float(age) / max(1.0, float(ewma_gap)),
    ]


def features_from_entry(entry, now_op: int) -> list[float]:
    """Live path: same computation, sourced from a store Entry."""
    return feature_row(
        now_op=now_op,
        last_access=entry.last_access,
        access_count=entry.access_count,
        ewma_gap=entry.ewma_gap,
    )
