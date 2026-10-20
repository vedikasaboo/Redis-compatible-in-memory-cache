"""
End-to-end guard benchmark: the same attack scenario against a real server
with the guard off, with the rule detector, and with the ML detector.

Scenario (default 24 s):
  t = 0       legitimate traffic starts: 3 apps, a busy service, a cache
              warmer, a viral-key reader, 2 idle connections
  t = 6       all four attacks start together: cache penetration, a hot-key
              flood, a password brute force, and slowloris connections

Measured for each run:
  - database queries/s over time, and how many came from the attacker
  - what each attack achieved: penetration queries that reached the
    database, flood requests served, passwords guessed, slowloris
    connection-seconds held
  - seconds from attack start to the first flag of each attack
  - legitimate requests refused (false positives) and their p99 latency

Usage:
    python tools/bench_guard.py
    python tools/bench_guard.py --modes off,rules,ml --duration 24
"""
import argparse
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cache.client import Client  # noqa: E402
from cache.database import SimulatedDatabase  # noqa: E402
from tools.traffic import ATTACKS, LEGIT, Tally, run_behaviour  # noqa: E402

PASSWORD = "bench-password"
LEGIT_MIX = [("app", 800), ("app", 400), ("app", 1200), ("busy", 5000), ("warmer", 600),
             ("viral", 1500), ("idle", None), ("idle", None), ("buggy", 300), ("uploader", None), ("typo", 400)]
ATTACK_MIX = [("penetration", 800), ("hot_flood", 10000), ("brute_force", 4), ("slowloris", 20)]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port: int, mode: str, model: str) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "cache.server", "--port", str(port), "--capacity", "200000",
           "--requirepass", PASSWORD, "--guard", mode]
    if mode == "ml":
        cmd += ["--guard-model", model]
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 10
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return proc
        time.sleep(0.05)
    raise TimeoutError("server did not start")


def run(mode: str, args) -> dict:
    port = free_port()
    proc = start_server(port, mode, args.model)
    db = SimulatedDatabase(latency_ms=0.5, capacity_qps=args.db_capacity)
    legit_tally, attack_tally = Tally(), Tally()
    stop_legit, stop_attack = threading.Event(), threading.Event()

    def launch(mix, tally, stop, seed0):
        ts = [threading.Thread(target=run_behaviour, daemon=True,
                               args=(kind, port, db, tally, stop, rate, seed0 + i, PASSWORD))
              for i, (kind, rate) in enumerate(mix)]
        for t in ts:
            t.start()
        return ts

    t0 = time.time()
    legit = launch(LEGIT_MIX, legit_tally, stop_legit, 100)
    time.sleep(args.attack_at)
    t_attack = time.time()
    attacks = launch(ATTACK_MIX, attack_tally, stop_attack, 200)

    # Poll the guard through the attack: GUARD STATUS returns only its most
    # recent flags, and slowloris reconnects alone can push early ones out.
    first_seen: dict[str, float] = {}
    flagged_names: set[str] = set()
    with Client(port=port, password=PASSWORD) as c:
        while time.time() < t_attack + args.duration - args.attack_at:
            status = json.loads(c.command("GUARD", "STATUS"))
            for f in status["flags"]:
                flagged_names.add(f["name"])
                if f["t"] >= t_attack:
                    first_seen[f["verdict"]] = min(first_seen.get(f["verdict"], f["t"]), f["t"])
            time.sleep(0.25)
    stop_attack.set()
    stop_legit.set()
    for t in legit + attacks:
        t.join(timeout=5)
    t_end = time.time()
    proc.terminate()
    proc.wait()

    first_flag = {v: round(t - t_attack, 2) for v, t in first_seen.items()}
    false_flags = sorted(n for n in flagged_names if n.startswith("ok:"))

    lat = sorted(d for kind in LEGIT for _, d in legit_tally.latencies.get(kind, []))
    p99 = lat[int(len(lat) * 0.99)] * 1000 if lat else None
    legit_ops = sum(legit_tally.ops.values())
    legit_refused = sum(legit_tally.refused.values())
    slow_events = [e for _, e in attack_tally.events if e.startswith("slowloris connection closed")]
    held = [float(e.split("after ")[1].rstrip("s")) for e in slow_events]
    series = db.series(t0, t_end)
    return {
        "mode": mode,
        "series": series,
        "attack_s": t_attack - t0,
        "db_from_attack": attack_tally.db_calls.get("penetration", 0) + attack_tally.db_calls.get("hot_flood", 0),
        "db_from_legit": sum(legit_tally.db_calls.values()),
        "penetration_db_queries": attack_tally.db_calls.get("penetration", 0),
        "penetration_requests": attack_tally.ops.get("penetration", 0),
        "flood_served": attack_tally.ops.get("hot_flood", 0) - attack_tally.refused.get("hot_flood", 0),
        "flood_refused": attack_tally.refused.get("hot_flood", 0),
        "passwords_tried": attack_tally.ops.get("brute_force", 0),
        "slowloris_closed": len(held),
        "slowloris_median_hold_s": sorted(held)[len(held) // 2] if held else None,
        "first_flag_s": first_flag,
        "false_flags": false_flags,
        "legit_ops": legit_ops,
        "legit_refused": legit_refused,
        "legit_p99_ms": p99,
        "peak_db_qps_attack": max((q for t, q in series if t >= args.attack_at), default=0),
    }


def summarize(results: list[dict]) -> None:
    print(f"\n{'':<26}" + "".join(f"{r['mode']:>14}" for r in results))
    rows = [
        ("penetration -> database", lambda r: f"{r['penetration_db_queries']:,}"),
        ("flood requests served", lambda r: f"{r['flood_served']:,}"),
        ("passwords tried", lambda r: f"{r['passwords_tried']:,}"),
        ("slowloris conns closed", lambda r: f"{r['slowloris_closed']}"),
        ("slowloris median hold s", lambda r: "-" if r["slowloris_median_hold_s"] is None else f"{r['slowloris_median_hold_s']:.1f}"),
        ("detect penetration s", lambda r: str(r["first_flag_s"].get("penetration", "-"))),
        ("detect hot_flood s", lambda r: str(r["first_flag_s"].get("hot_flood", "-"))),
        ("detect brute_force s", lambda r: str(r["first_flag_s"].get("brute_force", "-"))),
        ("detect slowloris s", lambda r: str(r["first_flag_s"].get("slowloris", "-"))),
        ("legit requests refused", lambda r: f"{r['legit_refused']:,} / {r['legit_ops']:,}"),
        ("legit conns flagged", lambda r: str(len(r["false_flags"]))),
        ("legit p99 latency ms", lambda r: "-" if r["legit_p99_ms"] is None else f"{r['legit_p99_ms']:.2f}"),
    ]
    for label, fn in rows:
        print(f"{label:<26}" + "".join(f"{fn(r):>14}" for r in results))


def plot(results: list[dict], args, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"off": "#d64545", "rules": "#c98a0b", "ml": "#2f6fde"}
    labels = {"off": "guard off", "rules": "rule detector", "ml": "ML detector"}
    fig, ax = plt.subplots(figsize=(10, 4))
    for r in results:
        pts = [(t, q) for t, q in r["series"] if t < args.duration]  # drop the shutdown tail
        ax.plot([t for t, _ in pts], [q for _, q in pts], lw=1.6,
                color=colors.get(r["mode"]), label=labels.get(r["mode"], r["mode"]))
    ax.axvline(results[0]["attack_s"], color="#888", ls=":", lw=1)
    ax.text(results[0]["attack_s"] + 0.2, ax.get_ylim()[1] * 0.92, "attacks start", fontsize=9, color="#555")
    ax.axhline(args.db_capacity, color="#444", ls="--", lw=1)
    ax.text(args.duration, args.db_capacity * 0.96, "database capacity", fontsize=9, color="#444", ha="right", va="top")
    ax.set_xlabel("seconds")
    ax.set_ylabel("database queries / second")
    ax.set_title("Database load under attack")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"plot -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modes", default="off,rules,ml")
    ap.add_argument("--duration", type=float, default=24)
    ap.add_argument("--attack-at", type=float, default=6)
    ap.add_argument("--db-capacity", type=float, default=1500)
    ap.add_argument("--model", default="models/guard.txt")
    ap.add_argument("--out", default="results/guard/bench.json")
    ap.add_argument("--plot", default="results/guard/bench.png")
    ap.add_argument("--replot", action="store_true", help="redraw the plot from --out without re-running")
    args = ap.parse_args()

    if args.replot:
        results = json.loads(Path(args.out).read_text())["results"]
        summarize(results)
        plot(results, args, Path(args.plot))
        return 0

    results = []
    for mode in args.modes.split(","):
        print(f"running guard={mode} ...", flush=True)
        results.append(run(mode, args))
    summarize(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"config": vars(args), "results": results}, indent=1))
    print(f"results -> {args.out}")
    plot(results, args, Path(args.plot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
