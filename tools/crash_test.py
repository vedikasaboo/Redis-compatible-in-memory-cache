"""
Crash-test rig: kill -9 the server at a random moment, restart it, and
check the recovered state against what the client was actually told.

The assertion differs by fsync mode, and that difference IS the point:

  always   — every write that received +OK must survive. An acknowledged
             write that vanishes is a broken durability promise.
  everysec — acknowledged writes MAY be lost (that is the advertised
             trade), but what survives must be a PREFIX of the write
             sequence with correct values. Losing a suffix is fine;
             holes, reordering, or wrong values mean the log is corrupt.

Note this is a process crash, not a machine crash: kill -9 destroys the
process but the OS page cache survives, so data that reached write() is
still flushed by the kernel. Testing true power loss needs the machine to
actually lose power. So this rig validates log framing, replay, and
ordering — not the fsync guarantee itself.

Usage:
    python tools/crash_test.py --iterations 100
"""
import argparse
import random
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
from cache.resp import ProtocolError  # noqa: E402

DEAD_CONNECTION = (ConnectionError, OSError, socket.timeout, ProtocolError)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port: int, data_dir: Path, fsync_mode: str) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable, "-m", "cache.server",
            "--port", str(port),
            "--capacity", "1000000",       # no eviction, so expectations stay simple
            "--data-dir", str(data_dir),
            "--fsync", fsync_mode,
            "--snapshot-interval", "0",    # compaction is exercised separately
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_ready(port: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.02)
    return False


def hammer_until_killed(port: int, proc: subprocess.Popen, kill_delay: float) -> list[tuple[bytes, bytes]]:
    """Write sequentially, recording acks, until the server is killed."""
    acked: list[tuple[bytes, bytes]] = []
    killer = threading.Timer(kill_delay, proc.kill)
    killer.start()
    try:
        with Client(port=port, timeout=5.0) as c:
            i = 0
            while True:
                key, value = f"k{i}".encode(), f"v{i}".encode()
                if c.set(key, value) != "OK":
                    break
                acked.append((key, value))
                i += 1
    except DEAD_CONNECTION:
        pass  # expected: this is the crash
    finally:
        killer.cancel()
        proc.kill()
        proc.wait()
    return acked


def read_back(port: int, count: int) -> list[bytes | None]:
    """Pipelined GET of k0..k{count-1}, so verification isn't the bottleneck."""
    out: list[bytes | None] = []
    with Client(port=port, timeout=10.0) as c:
        batch = 1000
        for start in range(0, count, batch):
            end = min(start + batch, count)
            for i in range(start, end):
                c.send("GET", f"k{i}")
            for _ in range(start, end):
                out.append(c.read_reply())
    return out


def check(mode: str, acked: list[tuple[bytes, bytes]], recovered: list[bytes | None]) -> str | None:
    """Return an error description, or None if the recovered state is valid."""
    if mode == "always":
        for i, (key, value) in enumerate(acked):
            if recovered[i] != value:
                return f"acked write {key!r} lost or wrong: expected {value!r}, got {recovered[i]!r}"
        return None

    # everysec: a suffix may be missing, but the survivors must form a
    # clean prefix with correct values.
    seen_missing = False
    for i, (key, value) in enumerate(acked):
        got = recovered[i]
        if got is None:
            seen_missing = True
            continue
        if seen_missing:
            return f"hole in log: {key!r} survived but an earlier write did not"
        if got != value:
            return f"wrong value for {key!r}: expected {value!r}, got {got!r}"
    return None


def run_once(mode: str, workdir: Path, iteration: int) -> tuple[bool, str, int]:
    data_dir = workdir / f"{mode}-{iteration}"
    data_dir.mkdir(parents=True, exist_ok=True)
    port = free_port()

    proc = start_server(port, data_dir, mode)
    if not wait_ready(port):
        proc.kill(); proc.wait()
        return False, "server never became ready", 0

    acked = hammer_until_killed(port, proc, kill_delay=random.uniform(0.05, 0.35))

    port2 = free_port()
    proc2 = start_server(port2, data_dir, mode)
    try:
        if not wait_ready(port2):
            return False, "server failed to restart after crash (recovery error?)", len(acked)
        recovered = read_back(port2, len(acked)) if acked else []
    finally:
        proc2.kill()
        proc2.wait()

    problem = check(mode, acked, recovered)
    return problem is None, problem or "", len(acked)


def main() -> int:
    ap = argparse.ArgumentParser(description="kill -9 crash/recovery test")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--modes", nargs="+", default=["always", "everysec"])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    random.seed(args.seed)

    overall_ok = True
    with tempfile.TemporaryDirectory(prefix="evict-crash-") as tmp:
        workdir = Path(tmp)
        for mode in args.modes:
            print(f"\n=== fsync={mode}: {args.iterations} crash/recovery cycles ===")
            failures, total_writes, started = [], 0, time.time()
            for i in range(args.iterations):
                ok, problem, n = run_once(mode, workdir, i)
                total_writes += n
                if not ok:
                    failures.append((i, problem))
                    print(f"  [{i:3d}] FAIL  {problem}")
                elif (i + 1) % 10 == 0:
                    print(f"  [{i:3d}] ok    ({total_writes} acked writes verified so far)")
            elapsed = time.time() - started
            if failures:
                overall_ok = False
                print(f"  RESULT: {len(failures)}/{args.iterations} FAILED  ({elapsed:.1f}s)")
            else:
                print(f"  RESULT: {args.iterations}/{args.iterations} passed, "
                      f"{total_writes} acked writes verified  ({elapsed:.1f}s)")

    print("\nALL MODES PASSED" if overall_ok else "\nFAILURES DETECTED")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
