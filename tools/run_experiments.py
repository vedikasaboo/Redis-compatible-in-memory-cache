"""
Run the whole eviction study end to end, reproducibly.

  1. Generate an EVALUATION trace and a separate TRAINING trace for each
     workload, from different seeds.
  2. Train one model per workload, on the training trace only.
  3. Benchmark every policy on the evaluation traces.

Why two traces rather than one split in half
--------------------------------------------
The model must never be scored on data it was fitted to. Splitting a single
trace would still let the learned policy warm its cache on the very
sequence it trained on, which flatters it in a way the classical policies
are not flattered. Two independent draws from the same generator is a
cleaner separation: same distribution, different sample, no shared history.

The trainer additionally holds out the last 20% of its own training trace
chronologically, for early stopping.

Usage:
    python tools/run_experiments.py
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def run(cmd: list[str], title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise SystemExit(f"step failed: {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Full eviction experiment pipeline")
    ap.add_argument("--length", type=int, default=200_000)
    ap.add_argument("--universe", type=int, default=20_000)
    ap.add_argument("--cycle", type=int, default=15_000)
    ap.add_argument("--budget-mb", type=float, default=1.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--eval-seed", type=int, default=42)
    ap.add_argument("--train-seed", type=int, default=1234)
    ap.add_argument("--capacity-hint", type=int, default=4500,
                    help="cache size the trainer simulates; should match the benchmark budget")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--rounds", type=int, default=2, choices=[1, 2],
                    help="1 = off-policy only; 2 = add the on-policy refit")
    args = ap.parse_args()

    py = sys.executable
    workloads = ["zipfian", "scan", "cyclic", "mixed"]

    run([py, "tools/workloads.py", "--out", "traces", "--length", str(args.length),
         "--universe", str(args.universe), "--cycle", str(args.cycle),
         "--seed", str(args.eval_seed)], "1a. Evaluation traces")

    run([py, "tools/workloads.py", "--out", "traces_train", "--length", str(args.length),
         "--universe", str(args.universe), "--cycle", str(args.cycle),
         "--seed", str(args.train_seed)], "1b. Training traces (different seed)")

    if not args.skip_train:
        # Round one: examples collected while LRU shapes the resident set.
        for workload in workloads:
            run([py, "tools/train_model.py",
                 "--trace", f"traces_train/{workload}.trace",
                 "--out", f"models/{workload}.round1.txt",
                 "--capacity", str(args.capacity_hint),
                 "--sim-policy", "true-lru"],
                f"2a. Train model (round 1, off-policy): {workload}")

        # Round two: re-collect with the round-one model driving eviction, so
        # the training distribution matches the one the deployed policy
        # actually induces, then refit on that. Without this the model is
        # asked at serving time about keys LRU would never have retained.
        if args.rounds == 2:
            for workload in workloads:
                run([py, "tools/train_model.py",
                     "--trace", f"traces_train/{workload}.trace",
                     "--out", f"models/{workload}.txt",
                     "--capacity", str(args.capacity_hint),
                     "--sim-policy", "ml",
                     "--sim-model", f"models/{workload}.round1.txt"],
                    f"2b. Train model (round 2, on-policy): {workload}")
        else:
            for workload in workloads:
                Path(REPO_ROOT / "models" / f"{workload}.txt").write_text(
                    (REPO_ROOT / "models" / f"{workload}.round1.txt").read_text())

    run([py, "tools/bench_eviction.py",
         "--traces", "traces", "--model-dir", "models",
         "--budget-mb", str(args.budget_mb), "--repeats", str(args.repeats),
         "--policies", "true-lru", "sampled-lru", "sampled-lfu", "random", "ml"],
        "3. Benchmark all policies")

    print("\nDone. Results in results/")


if __name__ == "__main__":
    main()
