"""
Train the reuse-distance model used by the learned eviction policy.

What is predicted
-----------------
Reuse distance: how many accesses until this key is next used. Small means
"needed soon, keep it"; large means "not needed for ages, evict it". At
eviction the policy evicts the candidate with the LARGEST predicted reuse
distance, which is a learned approximation of Belady's optimal algorithm
(evict whatever is needed furthest in the future).

Where training examples come from
---------------------------------
Not from moments of ACCESS — from moments of EVICTION. This distinction
matters and is easy to get wrong. If rows were emitted each time a key is
accessed, every row would have age 0, while at serving time the policy only
ever scores keys that have been sitting untouched for a while. The model
would be trained on a distribution it never sees in production and would be
systematically wrong about exactly the keys it is asked about.

So the trainer replays the trace through a real cache of the target
capacity and, at intervals, samples RESIDENT keys and asks the same
question the live policy will ask: "given this key's history as of now, how
far away is its next use?" Those are the training rows.

Labels come from a full index of access positions per key, so the "next
use" lookup is an exact binary search rather than an approximation. A key
never used again gets a large sentinel.

The target is log1p(reuse distance). The raw distribution spans several
orders of magnitude with a long tail; regressing on the log keeps a handful
of enormous distances from dominating the loss. Ranking is unaffected —
log is monotonic, and the policy only compares candidates.

Usage:
    python tools/train_model.py --trace traces/mixed.trace --out models/reuse.txt
"""
import argparse
import bisect
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cache.features import FEATURE_NAMES, feature_row  # noqa: E402
from tools.workloads import read_trace  # noqa: E402

# Stand-in for "never used again". Must exceed any real distance so the
# model learns these are the best possible eviction candidates.
NEVER_SENTINEL = 10_000_000


def build_position_index(trace: list[str]) -> dict[str, list[int]]:
    """key -> ascending list of the positions at which it is accessed."""
    positions: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(trace):
        positions[key].append(i)
    return positions


def next_use_after(positions: list[int], t: int) -> int:
    """Distance from t to the next access, or the sentinel if none."""
    idx = bisect.bisect_right(positions, t)
    return positions[idx] - t if idx < len(positions) else NEVER_SENTINEL


def collect_examples(trace: list[str], capacity: int, sample_every: int,
                     samples_per_point: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Replay the trace through an LRU cache of `capacity`, sampling resident
    keys at intervals. Returns (features, labels, positions).

    `positions` records when each row was produced so the train/test split
    can be made chronologically rather than randomly — a random split would
    leak future information into training and inflate the score.
    """
    index = build_position_index(trace)
    rng = random.Random(seed)

    last_access: dict[str, int] = {}
    access_count: dict[str, int] = defaultdict(int)
    gap_sum: dict[str, int] = defaultdict(int)
    prev_gap: dict[str, int] = defaultdict(int)

    resident: dict[str, None] = {}  # insertion-ordered => LRU order
    rows, labels, at = [], [], []

    for t, key in enumerate(trace):
        if key in resident:
            del resident[key]
        elif len(resident) >= capacity:
            resident.pop(next(iter(resident)))  # evict LRU
        resident[key] = None

        if key in last_access:
            gap = t - last_access[key]
            prev_gap[key] = gap
            gap_sum[key] += gap
        access_count[key] += 1
        last_access[key] = t

        if t % sample_every == 0 and len(resident) > samples_per_point:
            keys = list(resident)
            for candidate in rng.sample(keys, samples_per_point):
                rows.append(feature_row(
                    now_op=t,
                    last_access=last_access[candidate],
                    access_count=access_count[candidate],
                    gap_sum=gap_sum[candidate],
                    prev_gap=prev_gap[candidate],
                    size=64,
                ))
                labels.append(next_use_after(index[candidate], t))
                at.append(t)

    return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.float64), np.array(at)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the reuse-distance model")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", default="models/reuse.txt")
    ap.add_argument("--capacity", type=int, default=5000)
    ap.add_argument("--sample-every", type=int, default=50)
    ap.add_argument("--samples-per-point", type=int, default=8)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--rounds", type=int, default=300)
    args = ap.parse_args()

    trace = read_trace(Path(args.trace))
    print(f"trace: {args.trace} ({len(trace):,} accesses, {len(set(trace)):,} distinct)")

    X, y, at = collect_examples(trace, args.capacity, args.sample_every,
                                args.samples_per_point, args.seed)
    print(f"collected {len(X):,} training rows from eviction-time samples")

    # Chronological split: train on the first 80% of the trace, evaluate on
    # the last 20%. Never a random split — that would let the model see the
    # future of the very sequence it is being tested on.
    cutoff = np.quantile(at, 0.8)
    train_mask = at <= cutoff
    X_train, y_train = X[train_mask], np.log1p(y[train_mask])
    X_test, y_test = X[~train_mask], np.log1p(y[~train_mask])
    print(f"train {len(X_train):,} rows / test {len(X_test):,} rows (chronological 80-20)")

    model = lgb.LGBMRegressor(
        n_estimators=args.rounds, learning_rate=0.06, num_leaves=31,
        min_child_samples=40, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.9, random_state=args.seed, verbose=-1,
        # Pinned to one thread and baked into the saved model. The serving
        # side is a single-threaded event loop, and letting an OpenMP pool
        # spin up inside an eviction would add scheduling jitter to every
        # other connection for no throughput gain on a 16-row batch.
        num_threads=1, n_jobs=1,
    )
    # Fitted on a bare array, deliberately: the live scorer predicts from a
    # numpy array too, and a model carrying feature NAMES would warn on
    # every prediction about the mismatch. Names are attached for reporting
    # below instead.
    model.fit(X_train, y_train,
              eval_X=X_test, eval_y=y_test, eval_metric="l2",
              callbacks=[lgb.early_stopping(30, verbose=False)])

    pred = model.predict(X_test)
    mae = float(np.mean(np.abs(pred - y_test)))
    baseline = float(np.mean(np.abs(np.mean(y_train) - y_test)))

    # What actually matters is RANKING, not absolute error: the policy only
    # ever compares candidates against each other.
    order = np.argsort(pred)
    spearman = float(np.corrcoef(
        np.argsort(np.argsort(pred)), np.argsort(np.argsort(y_test))
    )[0, 1])

    print(f"\nheld-out MAE (log space): {mae:.3f}  (predict-the-mean baseline: {baseline:.3f})")
    print(f"held-out rank correlation: {spearman:.3f}")
    print("\nfeature importances:")
    for name, importance in sorted(zip(FEATURE_NAMES, model.feature_importances_),
                                   key=lambda kv: -kv[1]):
        print(f"  {name:<14} {importance}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out))
    meta = {
        "trace": str(args.trace), "capacity": args.capacity,
        "sample_every": args.sample_every, "samples_per_point": args.samples_per_point,
        "seed": args.seed, "features": FEATURE_NAMES, "target": "log1p(reuse_distance)",
        "never_sentinel": NEVER_SENTINEL, "rows": int(len(X)),
        "held_out_mae_log": mae, "mean_baseline_mae_log": baseline,
        "held_out_rank_correlation": spearman,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nmodel written to {out}")


if __name__ == "__main__":
    main()
