"""
Show that virtual nodes balance the keyspace, and that adding a node
remaps only ~1/N of keys.

Pure ring analysis — no servers needed, so it runs instantly and is
deterministic.

Usage:
    python tools/plot_distribution.py --keys 200000 --nodes 4
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cache.hashring import HashRing  # noqa: E402

RESULTS = REPO_ROOT / "results"


def spread(counts: list[int]) -> float:
    """Max/min load ratio — 1.0 is perfect balance."""
    return max(counts) / min(counts) if min(counts) else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser(description="Key distribution and remap analysis")
    ap.add_argument("--keys", type=int, default=200_000)
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--vnode-settings", type=int, nargs="+", default=[1, 10, 150])
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    keys = [f"key:{i}".encode() for i in range(args.keys)]
    node_ids = [f"node-{i}" for i in range(args.nodes)]

    print(f"{args.keys:,} keys over {args.nodes} nodes\n")
    print(f"{'vnodes':>8} {'max/min load':>14}   per-node share")
    distributions = {}
    for vnodes in args.vnode_settings:
        ring = HashRing(node_ids, vnodes=vnodes)
        counts = Counter(ring.get_node(k) for k in keys)
        ordered = [counts.get(n, 0) for n in node_ids]
        distributions[vnodes] = ordered
        shares = "  ".join(f"{c / args.keys:6.2%}" for c in ordered)
        print(f"{vnodes:>8} {spread(ordered):>13.2f}x   {shares}")

    # --- remap fraction when a node joins --------------------------------
    print()
    ring = HashRing(node_ids, vnodes=150)
    before = {k: ring.get_node(k) for k in keys}
    ring.add_node(f"node-{args.nodes}")
    after = {k: ring.get_node(k) for k in keys}
    moved = sum(1 for k in keys if before[k] != after[k])
    expected = 1 / (args.nodes + 1)
    print(f"adding a {args.nodes + 1}th node moved {moved / args.keys:.2%} of keys "
          f"(1/N = {expected:.2%})")
    print(f"modulo hashing would have moved roughly {args.nodes / (args.nodes + 1):.0%}")

    # --- plot ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    width = 0.8 / len(distributions)
    for i, (vnodes, counts) in enumerate(sorted(distributions.items())):
        xs = [x + i * width for x in range(args.nodes)]
        ax1.bar(xs, [c / args.keys * 100 for c in counts], width, label=f"{vnodes} vnodes")
    ax1.axhline(100 / args.nodes, color="black", ls="--", lw=1, label="perfect balance")
    ax1.set_xticks([x + 0.4 - width / 2 for x in range(args.nodes)])
    ax1.set_xticklabels(node_ids, rotation=20)
    ax1.set_ylabel("share of keyspace (%)")
    ax1.set_title("Virtual nodes even out the load")
    ax1.legend(fontsize=8)

    labels = ["consistent hashing\n(measured)", "modulo hashing\n(analytic)"]
    values = [moved / args.keys * 100, args.nodes / (args.nodes + 1) * 100]
    ax2.bar(labels, values, color=["#2a9d8f", "#e76f51"])
    for i, v in enumerate(values):
        ax2.text(i, v + 1.5, f"{v:.1f}%", ha="center")
    ax2.set_ylabel("keys remapped (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title(f"Cost of adding a node ({args.nodes} -> {args.nodes + 1})")

    fig.tight_layout()
    out = RESULTS / "key_distribution.png"
    fig.savefig(out, dpi=140)
    print(f"\nplot written to {out}")


if __name__ == "__main__":
    main()
