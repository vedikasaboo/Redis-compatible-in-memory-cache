"""
Start a static cluster of N nodes from a config file.

Writes the config if it does not exist, launches one server per node, and
waits. Ctrl-C shuts them all down.

Usage:
    python tools/launch_cluster.py --nodes 3
    python tools/launch_cluster.py --config cluster.json --data-root ./data
"""
import argparse
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cache.cluster import ClusterConfig  # noqa: E402


def wait_ready(host: str, port: int, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def launch(config: ClusterConfig, data_root: Path | None, capacity: int,
           fsync: str, extra: list[str] | None = None) -> list[subprocess.Popen]:
    procs = []
    for node in config.nodes:
        cmd = [
            sys.executable, "-m", "cache.server",
            "--host", node["host"],
            "--port", str(node["port"]),
            "--capacity", str(capacity),
        ]
        if data_root is not None:
            node_dir = data_root / node["id"]
            node_dir.mkdir(parents=True, exist_ok=True)
            cmd += ["--data-dir", str(node_dir), "--fsync", fsync]
        cmd += extra or []
        procs.append(subprocess.Popen(cmd, cwd=str(REPO_ROOT)))
    return procs


def main() -> int:
    ap = argparse.ArgumentParser(description="Launch a static cache cluster")
    ap.add_argument("--nodes", type=int, default=3)
    ap.add_argument("--start-port", type=int, default=6390)
    ap.add_argument("--config", default="cluster.json")
    ap.add_argument("--data-root", default=None, help="enable persistence, one subdir per node")
    ap.add_argument("--capacity", type=int, default=100000)
    ap.add_argument("--fsync", default="everysec")
    args = ap.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        config = ClusterConfig.load(config_path)
        print(f"using existing config {config_path} ({len(config.nodes)} nodes)")
    else:
        config = ClusterConfig.local(args.nodes, start_port=args.start_port)
        config.save(config_path)
        print(f"wrote config {config_path} ({len(config.nodes)} nodes)")

    data_root = Path(args.data_root) if args.data_root else None
    procs = launch(config, data_root, args.capacity, args.fsync)

    ok = True
    for node in config.nodes:
        if wait_ready(node["host"], node["port"]):
            print(f"  {node['id']:<10} {node['host']}:{node['port']}  ready")
        else:
            print(f"  {node['id']:<10} {node['host']}:{node['port']}  FAILED TO START")
            ok = False
    if not ok:
        for p in procs:
            p.terminate()
        return 1

    print("\nCluster up. Ctrl-C to stop.")
    stopping = False

    def _stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        while not stopping:
            time.sleep(0.2)
    finally:
        print("\nstopping cluster...")
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        print("stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
