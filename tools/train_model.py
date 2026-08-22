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

from cache.features import FEATURE_NAMES, features_from_entry  # noqa: E402
from cache.store import PolicyStore  # noqa: E402
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
                     samples_per_point: int, seed: int,
                     sim_policy_factory) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Replay the trace through a REAL PolicyStore, sampling resident keys at
    intervals. Returns (features, labels, positions).

    Simulating with the actual store — and reading features through the
    actual feature function — removes the last place training and serving
    could diverge. An earlier version reimplemented an LRU dict here, which
    worked but meant two definitions of "resident" to keep in step.

    `sim_policy_factory` decides WHICH policy shapes the resident set. Round
    one uses LRU; round two uses the model from round one, so the examples
    come from the distribution the deployed policy actually induces. That is
    the standard fix for an off-policy mismatch: a model trained only on
    LRU's resident sets is being asked, at serving time, about keys LRU
    would never have kept.

    Labels are measured in STORE OPERATIONS, not trace positions, because
    that is the clock the live features use. A miss costs two operations
    (the failed GET and the backfilling SET), so the two units differ by the
    miss rate — enough to skew `age` systematically if conflated. `op_at`
    records the operation count at each trace position so the conversion is
    exact.
    """
    index = build_position_index(trace)
    store = PolicyStore(capacity=capacity, policy=sim_policy_factory(), seed=seed)
    value = b"v" * 64

    op_at = [0] * len(trace)
    pending: list[tuple[list[float], str, int]] = []
    groups: list[int] = []   # rows emitted per sampling point, for ranking

    for t, key in enumerate(trace):
        raw = key.encode()
        if store.get(raw) is None:
            store.set(raw, value)      # cache-aside: backfill on miss
        op_at[t] = store.op

        if t % sample_every == 0 and len(store) > samples_per_point:
            sampled = store.sample_entries(samples_per_point)
            for entry in sampled:
                pending.append((features_from_entry(entry, store.op),
                                entry.key.decode(), t))
            groups.append(len(sampled))

    rows, labels, at = [], [], []
    for feats, key, t in pending:
        positions = index.get(key, [])
        i = bisect.bisect_right(positions, t)
        # Distance to the next access, expressed in operations.
        label = op_at[positions[i]] - op_at[t] if i < len(positions) else NEVER_SENTINEL
        rows.append(feats)
        labels.append(label)
        at.append(t)

    return (np.array(rows, dtype=np.float64),
            np.array(labels, dtype=np.float64),
            np.array(at),
            np.array(groups))


def make_sim_factory(policy_name: str, model_path: str | None, sample_size: int):
    from cache.eviction import build as build_policy

    def factory():
        if policy_name == "ml":
            return build_policy("ml", model_path=model_path, sample_size=sample_size)
        return build_policy(policy_name, sample_size=sample_size)

    return factory


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the reuse-distance model")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", default="models/reuse.txt")
    ap.add_argument("--capacity", type=int, default=5000)
    ap.add_argument("--sample-every", type=int, default=50)
    ap.add_argument("--samples-per-point", type=int, default=8)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--sim-policy", default="true-lru",
                    help="policy shaping the resident set while collecting examples")
    ap.add_argument("--sim-model", default=None,
                    help="model for --sim-policy ml (on-policy round two)")
    ap.add_argument("--sample-size", type=int, default=64)
    ap.add_argument("--objective", default="regression", choices=["regression", "rank"],
                    help="rank optimises the ORDERING within each sampled candidate set, "
                         "which is the decision the policy actually makes")
    args = ap.parse_args()

    trace = read_trace(Path(args.trace))
    print(f"trace: {args.trace} ({len(trace):,} accesses, {len(set(trace)):,} distinct)")

    factory = make_sim_factory(args.sim_policy, args.sim_model, args.sample_size)
    print(f"resident set simulated with: {args.sim_policy}"
          + (f" ({args.sim_model})" if args.sim_model else ""))
    X, y, at, groups = collect_examples(trace, args.capacity, args.sample_every,
                                        args.samples_per_point, args.seed, factory)
    print(f"collected {len(X):,} training rows from eviction-time samples")

    # Chronological split: train on the first 80% of the trace, evaluate on
    # the last 20%. Never a random split — that would let the model see the
    # future of the very sequence it is being tested on.
    cutoff = np.quantile(at, 0.8)
    train_mask = at <= cutoff
    X_train, y_train = X[train_mask], np.log1p(y[train_mask])
    X_test, y_test = X[~train_mask], np.log1p(y[~train_mask])
    if args.objective == "regression":
        print(f"train {len(X_train):,} rows / test {len(X_test):,} rows (chronological 80-20)")

    if args.objective == "rank":
        # The policy never uses a predicted distance as a number — it only
        # asks "which of these candidates is worst". LambdaRank optimises
        # exactly that ordering, within each sampled candidate set, instead
        # of spending capacity on absolute accuracy nobody consumes.
        # Relevance grades are the within-group ranking of reuse distance,
        # so the furthest-away key is the one the model must rank top.
        def grade(group_labels):
            order = np.argsort(np.argsort(group_labels))
            return order  # 0 = soonest reuse ... n-1 = furthest away

        graded, offset = [], 0
        for size in groups:
            graded.append(grade(y[offset:offset + size]))
            offset += size
        y_graded = np.concatenate(graded)

        # Split on GROUP boundaries, chronologically — a group split down
        # the middle would leak part of a candidate set across the divide.
        cum = np.cumsum(groups)
        group_time = at[np.concatenate([[0], cum[:-1]])]
        cutoff = np.quantile(group_time, 0.8)
        train_groups = groups[group_time <= cutoff]
        test_groups = groups[group_time > cutoff]
        n_train = int(train_groups.sum())

        X_train, y_train = X[:n_train], y_graded[:n_train]
        X_test, y_test = X[n_train:], y_graded[n_train:]
        print(f"train {len(X_train):,} rows / test {len(X_test):,} rows "
              f"({len(train_groups):,} / {len(test_groups):,} candidate sets)")

        model = lgb.LGBMRanker(
            n_estimators=args.rounds, learning_rate=0.06, num_leaves=31,
            min_child_samples=40, random_state=args.seed, verbose=-1,
            num_threads=1, n_jobs=1, label_gain=list(range(max(groups) + 1)),
        )
        # eval_group is indexed per eval SET, not per group, so it takes a
        # list containing this eval set's group array.
        model.fit(X_train, y_train, group=train_groups,
                  eval_set=[(X_test, y_test)], eval_group=[test_groups],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    else:
        model = lgb.LGBMRegressor(
            n_estimators=args.rounds, learning_rate=0.06, num_leaves=31,
            min_child_samples=40, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.9, random_state=args.seed, verbose=-1,
            # Pinned to one thread and baked into the saved model. The serving
            # side is a single-threaded event loop, and letting an OpenMP pool
            # spin up inside an eviction would add scheduling jitter to every
            # other connection for no throughput gain on a small batch.
            num_threads=1, n_jobs=1,
        )
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
        "seed": args.seed, "features": FEATURE_NAMES, "objective": args.objective,
        "target": "within-group rank of reuse distance" if args.objective == "rank"
                  else "log1p(reuse_distance_ops)",
        "sim_policy": args.sim_policy, "sim_model": args.sim_model,
        "never_sentinel": NEVER_SENTINEL, "rows": int(len(X)),
        "held_out_mae_log": mae, "mean_baseline_mae_log": baseline,
        "held_out_rank_correlation": spearman,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nmodel written to {out}")


if __name__ == "__main__":
    main()
