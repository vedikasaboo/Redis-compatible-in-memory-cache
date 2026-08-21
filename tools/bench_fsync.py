"""
Throughput comparison across durability settings.

Four configurations, all driven at the same concurrency so the only
variable is how often fsync happens:

  always, no group commit  — one fsync per record. The naive design.
  always, group commit     — one fsync per event-loop tick, shared by
                             every writer whose record landed in it.
  everysec                 — one fsync per second, in the background.
  never                    — no explicit fsync; the ceiling set by
                             everything other than durability.

Concurrency is what makes group commit visible: with a single client
there is never more than one record per tick to batch, so it can only
help once writers overlap. The rig therefore also reports the serial
case, where group commit is expected to show no benefit at all.

Usage:
    python tools/bench_fsync.py --duration 3 --clients 32
"""
import argparse
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cache.client import Client  # noqa: E402


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port: int, data_dir: Path, fsync_mode: str, group_commit: bool) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "cache.server",
        "--port", str(port),
        "--capacity", "1000000",
        "--data-dir", str(data_dir),
        "--fsync", fsync_mode,
        "--snapshot-interval", "0",
    ]
    if not group_commit:
        cmd.append("--no-group-commit")
    return subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_ready(port: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.02)
    return False


def _worker(port: int, worker_id: int, stop: threading.Event, counts: list, index: int) -> None:
    n = 0
    try:
        with Client(port=port, timeout=15.0) as c:
            while not stop.is_set():
                c.set(f"w{worker_id}:{n}", "x" * 32)
                n += 1
    except Exception:
        pass
    counts[index] = n


def measure(label: str, fsync_mode: str, group_commit: bool, clients: int, duration: float) -> dict:
    with tempfile.TemporaryDirectory(prefix="evict-bench-") as tmp:
        port = free_port()
        proc = start_server(port, Path(tmp), fsync_mode, group_commit)
        try:
            if not wait_ready(port):
                raise RuntimeError("server did not start")
            stop = threading.Event()
            counts = [0] * clients
            threads = [
                threading.Thread(target=_worker, args=(port, i, stop, counts, i), daemon=True)
                for i in range(clients)
            ]
            started = time.time()
            for t in threads:
                t.start()
            time.sleep(duration)
            stop.set()
            for t in threads:
                t.join(timeout=15.0)
            elapsed = time.time() - started
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    total = sum(counts)
    return {"label": label, "clients": clients, "writes": total, "elapsed": elapsed, "rps": total / elapsed}


def main() -> None:
    ap = argparse.ArgumentParser(description="WAL durability throughput comparison")
    ap.add_argument("--duration", type=float, default=3.0, help="seconds per configuration")
    ap.add_argument("--clients", type=int, default=32, help="concurrent writers")
    args = ap.parse_args()

    configs = [
        ("always, no group commit", "always", False, args.clients),
        ("always, group commit", "always", True, args.clients),
        ("everysec", "everysec", True, args.clients),
        ("never (no fsync)", "never", True, args.clients),
        ("always, group commit (1 client)", "always", True, 1),
        ("always, no group commit (1 client)", "always", False, 1),
    ]

    print(f"SET throughput, {args.duration}s per configuration\n")
    results = []
    for label, mode, group, clients in configs:
        r = measure(label, mode, group, clients, args.duration)
        results.append(r)
        print(f"  {label:<38} {r['clients']:>3} clients  {r['rps']:>10,.0f} writes/sec")

    concurrent = {r["label"]: r["rps"] for r in results if r["clients"] == args.clients}
    serial = {r["label"]: r["rps"] for r in results if r["clients"] == 1}

    print()
    a = concurrent.get("always, no group commit")
    b = concurrent.get("always, group commit")
    if a and b:
        print(f"  Group commit speedup at {args.clients} clients: {b / a:.1f}x")
    sa = serial.get("always, no group commit (1 client)")
    sb = serial.get("always, group commit (1 client)")
    if sa and sb:
        print(f"  Group commit speedup at  1 client:  {sb / sa:.1f}x  (expected ~1x: nothing to batch)")


if __name__ == "__main__":
    main()
