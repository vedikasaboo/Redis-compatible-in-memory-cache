"""
Workload generators. Each writes a trace file: one key per line, in access
order.

The four shapes are chosen because they separate the policies. A policy
that wins on all four is genuinely better; one that wins only on Zipfian
has just rediscovered "keep the popular keys".

  zipfian — a few very hot keys and a long cold tail. The friendly case,
            and roughly what real caches see. Recency and frequency both
            work well here, so policies tend to tie.
  scan    — walk a large key range once, in order. The adversarial case for
            LRU: nothing is ever reused before it is evicted, and the scan
            flushes everything useful out on its way through.
  cyclic  — repeat a fixed sequence longer than the cache. Pathological for
            LRU specifically: every key is evicted exactly before its next
            use, so the hit rate collapses toward zero while a policy that
            keeps a stable subset does fine.
  mixed   — Zipfian with a scan injected periodically. The realistic case:
            a healthy workload occasionally disrupted by a batch job.

Everything is seeded, so a trace is reproducible from its config alone.

Usage:
    python tools/workloads.py --out traces/ --length 200000
"""
import argparse
import json
from pathlib import Path

import numpy as np


def zipfian(length: int, universe: int, alpha: float, rng: np.random.Generator) -> list[str]:
    """
    Weighted sampling with Zipf-like weights p(rank) ~ 1/rank^alpha.

    numpy's own `rng.zipf` draws from an unbounded distribution and would
    need rejection sampling to fit a fixed universe; building explicit
    weights over `universe` ranks is clearer and exactly bounded.
    """
    ranks = np.arange(1, universe + 1)
    weights = 1.0 / np.power(ranks, alpha)
    weights /= weights.sum()
    picks = rng.choice(universe, size=length, p=weights)
    return [f"key:{i}" for i in picks]


def scan(length: int, universe: int, **_) -> list[str]:
    """Walk the key range in order, wrapping if the trace is longer."""
    return [f"key:{i % universe}" for i in range(length)]


def cyclic(length: int, cycle: int, **_) -> list[str]:
    """Repeat a fixed sequence. Worst case for LRU when cycle > capacity."""
    return [f"key:{i % cycle}" for i in range(length)]


def mixed(length: int, universe: int, alpha: float, rng: np.random.Generator,
          scan_every: int = 10_000, scan_length: int = 2_000) -> list[str]:
    """Zipfian traffic, interrupted by a scan burst at regular intervals."""
    out: list[str] = []
    scan_cursor = 0
    while len(out) < length:
        chunk = min(scan_every, length - len(out))
        out.extend(zipfian(chunk, universe, alpha, rng))
        if len(out) >= length:
            break
        burst = min(scan_length, length - len(out))
        out.extend(f"key:{(scan_cursor + i) % universe}" for i in range(burst))
        scan_cursor += burst
    return out[:length]


WORKLOADS = {
    "zipfian": lambda length, universe, alpha, cycle, rng: zipfian(length, universe, alpha, rng),
    "scan":    lambda length, universe, alpha, cycle, rng: scan(length, universe),
    "cyclic":  lambda length, universe, alpha, cycle, rng: cyclic(length, cycle),
    "mixed":   lambda length, universe, alpha, cycle, rng: mixed(length, universe, alpha, rng),
}


def generate(name: str, length: int, universe: int, alpha: float, cycle: int, seed: int) -> list[str]:
    if name not in WORKLOADS:
        raise ValueError(f"unknown workload {name!r}; have {sorted(WORKLOADS)}")
    rng = np.random.default_rng(seed)
    return WORKLOADS[name](length, universe, alpha, cycle, rng)


def write_trace(path: Path, keys: list[str], config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(keys) + "\n")
    # Config saved beside the trace so any result can be traced back to
    # exactly the inputs that produced it.
    path.with_suffix(".json").write_text(json.dumps(config, indent=2) + "\n")


def read_trace(path: Path) -> list[str]:
    return path.read_text().split()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate workload traces")
    ap.add_argument("--out", default="traces")
    ap.add_argument("--length", type=int, default=200_000)
    ap.add_argument("--universe", type=int, default=20_000, help="distinct keys")
    ap.add_argument("--alpha", type=float, default=1.1, help="Zipf skew; higher = more concentrated")
    ap.add_argument("--cycle", type=int, default=15_000, help="sequence length for the cyclic workload")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workloads", nargs="+", default=list(WORKLOADS))
    args = ap.parse_args()

    out_dir = Path(args.out)
    for name in args.workloads:
        keys = generate(name, args.length, args.universe, args.alpha, args.cycle, args.seed)
        config = {
            "workload": name, "length": args.length, "universe": args.universe,
            "alpha": args.alpha, "cycle": args.cycle, "seed": args.seed,
        }
        path = out_dir / f"{name}.trace"
        write_trace(path, keys, config)
        distinct = len(set(keys))
        print(f"{name:<8} {len(keys):>8,} accesses  {distinct:>7,} distinct keys  -> {path}")


if __name__ == "__main__":
    main()
