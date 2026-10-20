"""
Collect labelled traffic windows from a real server, train the guard's
classifier, and compare it against the rule detector on held-out traffic.

Collection
----------
A real cache.server runs with `--guard observe --guard-log`, so it computes
and logs exactly the feature rows it would score in production (through
cache.guard.feature_row, the one shared definition) without acting on them.
Each session launches a random mix of legitimate clients — including the
look-alikes in tools/traffic.py — plus one or two attacks at random
intensities, for `--session-seconds`.

Labels come from each connection's CLIENT SETNAME (`ok:*` -> normal,
`atk:<kind>` -> that kind), applied to every window of that connection, so a
window logged before the name was set is still labelled correctly.
Brute-force connections cannot name themselves (CLIENT needs AUTH,
correctly), and they are the only connections that never get a name, so
those rows are labelled brute_force. Names are never a feature.

Evaluation
----------
Train and test come from SEPARATE collection runs with different seeds, not
a split of one run, so no window of a test session was seen in training.
A third run, `stealth`, uses attack intensities BELOW the training range to
show where detection breaks down rather than only where it works.

Usage:
    python tools/train_guard.py                      # collect, train, evaluate
    python tools/train_guard.py --reuse              # retrain from saved logs
"""
import argparse
import json
import random
import socket
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cache.database import SimulatedDatabase  # noqa: E402
from cache.guard import CLASSES, FEATURE_NAMES, RuleDetector, apply_policy, is_active  # noqa: E402
from tools.traffic import ATTACKS, Tally, label_of, run_behaviour  # noqa: E402

PASSWORD = "guard-demo-password"
DATA_DIR = REPO_ROOT / "results" / "guard"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port: int, log_path: Path) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "cache.server", "--port", str(port), "--capacity", "200000",
         "--requirepass", PASSWORD, "--guard", "observe", "--guard-log", str(log_path)],
        cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 10
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return proc
        time.sleep(0.05)
    raise TimeoutError("server did not start")


ATTACK_RATES = {
    # (low, high) rates or connection counts, per intensity profile
    "train":   {"penetration": (50, 2000), "hot_flood": (4000, 15000), "brute_force": (2, 8), "slowloris": (5, 30)},
    "stealth": {"penetration": (5, 50), "hot_flood": (1500, 4000), "brute_force": (1, 1), "slowloris": (1, 3)},
}


def collect(log_path: Path, sessions: int, seconds: float, seed: int, profile: str) -> None:
    rng = random.Random(seed)
    log_path.unlink(missing_ok=True)
    port = free_port()
    proc = start_server(port, log_path)
    try:
        for s in range(sessions):
            db = SimulatedDatabase(latency_ms=0.2, capacity_qps=1e9)
            tally, stop = Tally(), threading.Event()
            plan = [(k, None) for k in rng.choices(("app", "app", "busy", "warmer", "viral", "idle", "buggy", "uploader", "typo"),
                                                    k=rng.randint(4, 8))]
            # Rotate so every attack appears in every collection, plus a
            # random second one in some sessions.
            attacks = [ATTACKS[s % len(ATTACKS)]]
            if rng.random() < 0.4:
                attacks.append(rng.choice([a for a in ATTACKS if a != attacks[0]]))
            for atk in attacks:
                lo, hi = ATTACK_RATES[profile][atk]
                plan.append((atk, rng.uniform(lo, hi)))
            threads = [threading.Thread(target=run_behaviour, daemon=True,
                                        args=(kind, port, db, tally, stop, rate, rng.randrange(10**9), PASSWORD))
                       for kind, rate in plan]
            for t in threads:
                t.start()
            time.sleep(seconds)
            stop.set()
            for t in threads:
                t.join(timeout=5)
            print(f"  {profile} session {s + 1}/{sessions}: {', '.join(k for k, _ in plan)}", flush=True)
            time.sleep(1.5)  # let the last windows close before the next mix starts
    finally:
        proc.terminate()
        proc.wait()


def load(log_path: Path):
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    # A connection can log a window before it names itself (a typo user's
    # failed AUTH, then CLIENT SETNAME a moment later in the next window).
    # Label every window of a connection by the name it eventually set.
    final_name = {}
    for row in rows:
        if row["name"]:
            final_name[row.get("conn")] = row["name"]
    X, y, names = [], [], []
    for row in rows:
        row["name"] = row["name"] or final_name.get(row.get("conn"), "")
        # Windows with no traffic carry no evidence either way; the server
        # never classifies them either (Guard.tick), so neither do we.
        if not is_active(row["features"]):
            continue
        label = label_of(row["name"]) if row["name"] else "brute_force"
        X.append(row["features"])
        y.append(CLASSES.index(label))
        names.append(row["name"] or "atk:brute_force")
    return np.array(X, dtype=float), np.array(y), names


def train(X, y, out: Path):
    import lightgbm as lgb

    counts = Counter(y.tolist())
    weights = np.array([len(y) / (len(CLASSES) * counts[c]) for c in y])  # balance classes
    params = {"objective": "multiclass", "num_class": len(CLASSES), "learning_rate": 0.1,
              "num_leaves": 15, "min_data_in_leaf": 10, "verbose": -1, "num_threads": 1, "seed": 1}
    booster = lgb.train(params, lgb.Dataset(X, y, weight=weights), num_boost_round=150)
    out.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out))
    return booster


def evaluate(pred: list[int], y, names) -> dict:
    report = {}
    for ci, cls in enumerate(CLASSES):
        tp = sum(1 for p, t in zip(pred, y) if p == ci and t == ci)
        fp = sum(1 for p, t in zip(pred, y) if p == ci and t != ci)
        fn = sum(1 for p, t in zip(pred, y) if p != ci and t == ci)
        report[cls] = {"precision": tp / (tp + fp) if tp + fp else None,
                       "recall": tp / (tp + fn) if tp + fn else None, "support": int(sum(y == ci))}
    # False alarms by legitimate kind: how often each look-alike gets flagged.
    by_kind = defaultdict(lambda: [0, 0])
    for p, t, n in zip(pred, y, names):
        if t == 0:
            by_kind[n][1] += 1
            by_kind[n][0] += p != 0
    report["false_alarm_by_kind"] = {k: v[0] / v[1] for k, v in sorted(by_kind.items())}
    attack = [i for i, t in enumerate(y) if t != 0]
    report["attack_windows_caught"] = sum(1 for i in attack if pred[i] != 0) / max(len(attack), 1)
    normal = [i for i, t in enumerate(y) if t == 0]
    report["normal_windows_flagged"] = sum(1 for i in normal if pred[i] != 0) / max(len(normal), 1)
    return report


def predict_both(booster, X, confidence: float):
    """Both detectors, through the same policy the server applies (Guard.tick)."""
    probs = booster.predict(X)
    ml = []
    for p, row in zip(probs, X.tolist()):
        label, conf = apply_policy(CLASSES[int(p.argmax())], float(p.max()), row)
        ml.append(CLASSES.index(label) if conf >= confidence else 0)
    rules = [CLASSES.index(apply_policy(lbl, c, row)[0])
             for (lbl, c), row in zip(RuleDetector().classify(X.tolist()), X.tolist())]
    return ml, rules


def show(title: str, report: dict) -> None:
    print(f"\n{title}")
    print(f"  attack windows caught: {report['attack_windows_caught']:.1%}   "
          f"normal windows flagged: {report['normal_windows_flagged']:.2%}")
    for cls in CLASSES:
        r = report[cls]
        p = "-" if r["precision"] is None else f"{r['precision']:.3f}"
        rc = "-" if r["recall"] is None else f"{r['recall']:.3f}"
        print(f"  {cls:<12} precision {p:>6}  recall {rc:>6}  ({r['support']} windows)")
    print("  false alarms on legitimate kinds: " +
          ", ".join(f"{k[3:]} {v:.1%}" for k, v in report["false_alarm_by_kind"].items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sessions", type=int, default=16)
    ap.add_argument("--test-sessions", type=int, default=8)
    ap.add_argument("--session-seconds", type=float, default=15)
    ap.add_argument("--reuse", action="store_true", help="skip collection, use saved logs")
    ap.add_argument("--model", default="models/guard.txt")
    ap.add_argument("--confidence", type=float, default=0.8)
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logs = {"train": DATA_DIR / "train.jsonl", "test": DATA_DIR / "test.jsonl", "stealth": DATA_DIR / "stealth.jsonl"}
    if not args.reuse:
        print("collecting training traffic ...")
        collect(logs["train"], args.sessions, args.session_seconds, seed=1, profile="train")
        print("collecting held-out test traffic (different seed) ...")
        collect(logs["test"], args.test_sessions, args.session_seconds, seed=2, profile="train")
        print("collecting stealth traffic (attack intensities below the training range) ...")
        collect(logs["stealth"], args.test_sessions, args.session_seconds, seed=3, profile="stealth")

    Xtr, ytr, _ = load(logs["train"])
    print(f"\ntraining on {len(ytr)} windows: " +
          ", ".join(f"{CLASSES[c]} {n}" for c, n in sorted(Counter(ytr.tolist()).items())))
    booster = train(Xtr, ytr, REPO_ROOT / args.model)
    importance = dict(zip(FEATURE_NAMES, booster.feature_importance("gain").round(1).tolist()))

    results = {"features": FEATURE_NAMES, "importance": importance, "confidence": args.confidence}
    for split in ("test", "stealth"):
        X, y, names = load(logs[split])
        ml, rules = predict_both(booster, X, args.confidence)
        results[split] = {"ml": evaluate(ml, y, names), "rules": evaluate(rules, y, names)}
        show(f"[{split}] ML detector", results[split]["ml"])
        show(f"[{split}] rule detector", results[split]["rules"])

    print("\nfeature importance (gain): " +
          ", ".join(f"{k} {v:.0f}" for k, v in sorted(importance.items(), key=lambda kv: -kv[1])))
    (DATA_DIR / "detection.json").write_text(json.dumps(results, indent=1))
    print(f"\nmodel -> {args.model}\nresults -> {DATA_DIR / 'detection.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
