"""
Replay traces against the cache under each eviction policy and report hit
rate, latency percentiles, and throughput.

Method
------
Replay runs IN-PROCESS, not over a socket. The question here is which
policy retains the more useful keys; routing every access through TCP would
bury that signal under network noise and measure the event loop instead.
Network throughput is measured separately by redis-benchmark.

Each access is a GET; a miss is followed by a SET, which is what a cache in
front of a slow backing store actually does.

Fairness
--------
Policies are compared at a fixed BYTE budget, not a fixed key count.
Per-key bookkeeping differs between them — LFU needs a counter, the learned
policy needs several statistics — so equal key counts would quietly give
the metadata-hungry policy more memory. Capacity is derived per policy:

    capacity = budget_bytes // (key + value + base overhead + policy metadata)

The first N accesses are discarded as warmup so a cold cache does not
penalise every policy equally but unequally by luck of ordering. Each
configuration runs `--repeats` times and the mean is reported.

Usage:
    python tools/bench_eviction.py --traces traces/ --budget-mb 2
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cache.eviction import build  # noqa: E402
from cache.store import PolicyStore  # noqa: E402
from tools.workloads import read_trace  # noqa: E402

RESULTS = REPO_ROOT / "results"

# Per-entry overhead every policy pays regardless of eviction strategy:
# the dict slot, the Entry object header and its common fields.
BASE_ENTRY_BYTES = 120
VALUE_BYTES = 64

DEFAULT_POLICIES = ["true-lru", "sampled-lru", "sampled-lfu", "random"]


def capacity_for(policy, budget_bytes: int, key_bytes: int = 16) -> int:
    per_key = BASE_ENTRY_BYTES + key_bytes + VALUE_BYTES + policy.per_key_metadata_bytes
    return max(1, budget_bytes // per_key)


def replay(trace: list[str], policy, budget_bytes: int, warmup: int, seed: int) -> dict:
    store = PolicyStore(capacity=capacity_for(policy, budget_bytes), policy=policy, seed=seed)
    value = b"v" * VALUE_BYTES
    latencies: list[float] = []

    for i, key in enumerate(trace):
        if i == warmup:
            # Discard warmup: measure steady state, not the cold-start
            # transient that every policy shares.
            store.reset_stats()
            latencies.clear()
        raw = key.encode()
        started = time.perf_counter()
        if store.get(raw) is None:
            store.set(raw, value)
        latencies.append(time.perf_counter() - started)

    latencies.sort()
    n = len(latencies)

    def pct(p: float) -> float:
        return latencies[min(n - 1, int(n * p))] * 1e6 if n else 0.0

    total_time = sum(latencies)
    return {
        "hit_rate": store.hit_rate,
        "hits": store.hits,
        "misses": store.misses,
        "evictions": store.evictions,
        "capacity": store.capacity,
        "p50_us": pct(0.50),
        "p95_us": pct(0.95),
        "p99_us": pct(0.99),
        "throughput_ops": n / total_time if total_time else 0.0,
    }


def resolve_model(model_dir: Path, workload: str, fallback: str) -> str:
    """
    Prefer a model trained on THIS workload. A single model trained on one
    workload carries that workload's prior into the others — a model that
    learned "older means further from reuse" on Zipfian traffic is exactly
    wrong on a cyclic scan, where the oldest key is the next one needed.
    """
    candidate = model_dir / f"{workload}.txt"
    return str(candidate) if candidate.exists() else fallback


def make_policy(name: str, sample_size: int, model_path: str, ml_sample_size: int):
    """
    Every sampling policy gets the SAME candidate count by default.

    An earlier version gave the learned policy 16 candidates while the
    classical ones got 5, on the reasoning that a batched prediction makes a
    wider sample cheap for the model. That reasoning is true but it is not
    exclusive: scoring a candidate for LFU is a couple of attribute reads,
    which is cheaper still. Measured, sampled-LFU gains just as much from a
    wider sample as the model does (mixed: 76.32% at 5 -> 77.72% at 64), so
    the old comparison was crediting the model for extra candidates rather
    than better judgement.
    """
    if name == "ml":
        return build(name, model_path=model_path, sample_size=ml_sample_size or sample_size)
    return build(name, sample_size=sample_size)


def run(trace_dir: Path, policies: list[str], budget_bytes: int,
        warmup_frac: float, repeats: int, seed: int, sample_size: int,
        model_path: str = "models/reuse.txt", ml_sample_size: int = 16,
        model_dir: Path | None = None) -> dict:
    traces = sorted(trace_dir.glob("*.trace"))
    if not traces:
        raise SystemExit(f"no .trace files in {trace_dir} — run tools/workloads.py first")

    results: dict[str, dict[str, dict]] = {}
    for trace_path in traces:
        name = trace_path.stem
        trace = read_trace(trace_path)
        warmup = int(len(trace) * warmup_frac)
        results[name] = {}
        print(f"\n=== {name} ({len(trace):,} accesses, {warmup:,} discarded as warmup) ===")
        print(f"{'policy':<16} {'capacity':>9} {'hit rate':>9} {'p50 us':>8} "
              f"{'p95 us':>8} {'p99 us':>8} {'kops/s':>8}")
        chosen_model = resolve_model(model_dir, name, model_path) if model_dir else model_path
        for policy_name in policies:
            runs = [
                replay(trace, make_policy(policy_name, sample_size, chosen_model, ml_sample_size),
                       budget_bytes, warmup, seed + r)
                for r in range(repeats)
            ]
            mean = {
                key: statistics.mean(r[key] for r in runs)
                for key in ("hit_rate", "p50_us", "p95_us", "p99_us", "throughput_ops", "evictions")
            }
            mean["capacity"] = runs[0]["capacity"]
            mean["hit_rate_stdev"] = statistics.stdev([r["hit_rate"] for r in runs]) if repeats > 1 else 0.0
            results[name][policy_name] = mean
            print(f"{policy_name:<16} {mean['capacity']:>9,} {mean['hit_rate']:>8.2%} "
                  f"{mean['p50_us']:>8.2f} {mean['p95_us']:>8.2f} {mean['p99_us']:>8.2f} "
                  f"{mean['throughput_ops'] / 1000:>8.0f}")
    return results


def plot(results: dict, out_path: Path) -> None:
    workloads = sorted(results)
    policies = sorted({p for w in results.values() for p in w})
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(workloads)), 4.8))
    width = 0.8 / len(policies)
    for i, policy in enumerate(policies):
        xs = [x + i * width for x in range(len(workloads))]
        ys = [results[w].get(policy, {}).get("hit_rate", 0) * 100 for w in workloads]
        bars = ax.bar(xs, ys, width, label=policy)
        for bar, y in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width() / 2, y + 0.8, f"{y:.0f}", ha="center", fontsize=7)
    ax.set_xticks([x + 0.4 - width / 2 for x in range(len(workloads))])
    ax.set_xticklabels(workloads)
    ax.set_ylabel("hit rate (%)")
    ax.set_title("Hit rate by eviction policy and workload")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"\nplot written to {out_path}")


def print_table(results: dict) -> None:
    workloads = sorted(results)
    policies = sorted({p for w in results.values() for p in w})
    width = max(len(p) for p in policies) + 2
    print("\n\n=== hit rate: policy x workload ===\n")
    print("policy".ljust(width) + "".join(w.rjust(12) for w in workloads))
    print("-" * (width + 12 * len(workloads)))
    for policy in policies:
        row = "".join(f"{results[w].get(policy, {}).get('hit_rate', 0):>11.2%} " for w in workloads)
        print(policy.ljust(width) + row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Eviction policy benchmark")
    ap.add_argument("--traces", default="traces")
    ap.add_argument("--budget-mb", type=float, default=2.0, help="memory budget per policy")
    ap.add_argument("--warmup-frac", type=float, default=0.2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sample-size", type=int, default=64,
                    help="candidates sampled per eviction, for ALL sampling policies")
    ap.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    ap.add_argument("--model", default="models/reuse.txt", help="model file for the 'ml' policy")
    ap.add_argument("--ml-sample-size", type=int, default=None,
                    help="override the candidate count for the learned policy only "
                         "(default: same as every other policy, which is the fair comparison)")
    ap.add_argument("--model-dir", default="models", help="per-workload models named <workload>.txt")
    ap.add_argument("--out", default=None, help="results JSON path")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    budget_bytes = int(args.budget_mb * 1024 * 1024)
    results = run(Path(args.traces), args.policies, budget_bytes,
                  args.warmup_frac, args.repeats, args.seed, args.sample_size,
                  model_path=args.model, ml_sample_size=args.ml_sample_size,
                  model_dir=Path(args.model_dir) if args.model_dir else None)
    print_table(results)
    plot(results, RESULTS / "eviction_hit_rate.png")

    out = Path(args.out) if args.out else RESULTS / "eviction_results.json"
    # Config travels with the results so any number here can be reproduced.
    out.write_text(json.dumps({
        "config": {
            "budget_mb": args.budget_mb, "warmup_frac": args.warmup_frac,
            "repeats": args.repeats, "seed": args.seed,
            "sample_size": args.sample_size, "policies": args.policies,
            "ml_sample_size": args.ml_sample_size, "model": args.model,
            "base_entry_bytes": BASE_ENTRY_BYTES, "value_bytes": VALUE_BYTES,
        },
        "results": results,
    }, indent=2) + "\n")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
