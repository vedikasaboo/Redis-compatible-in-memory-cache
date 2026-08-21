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
defined in accesses, so measuring in accesses keeps features and labels in
the same units and makes replays reproducible across machines.
"""

FEATURE_NAMES = ["age", "access_count", "avg_gap", "prev_gap", "size"]


def feature_row(now_op: int, last_access: int, access_count: int,
                gap_sum: int, prev_gap: int, size: int) -> list[float]:
    """
    Features for one key as of logical time `now_op`.

    age          — accesses since this key was last touched. The signal LRU
                   uses, kept so the model can learn when recency is and is
                   not predictive rather than having it assumed.
    access_count — how often it has been touched. The signal LFU uses.
    avg_gap      — mean spacing between this key's accesses so far. A key
                   accessed every ~10 operations behaves very differently
                   from one accessed every ~10,000, even at equal age.
    prev_gap     — the most recent spacing, which reacts faster than the
                   mean when a key's behaviour changes.
    size         — value size in bytes; larger values cost more to keep.
    """
    return [
        float(now_op - last_access),
        float(access_count),
        float(gap_sum) / max(1, access_count - 1),
        float(prev_gap),
        float(size),
    ]


def features_from_entry(entry, now_op: int) -> list[float]:
    """Live path: same computation, sourced from a store Entry."""
    return feature_row(
        now_op=now_op,
        last_access=entry.last_access,
        access_count=entry.access_count,
        gap_sum=entry.gap_sum,
        prev_gap=entry.prev_gap,
        size=entry.size,
    )
