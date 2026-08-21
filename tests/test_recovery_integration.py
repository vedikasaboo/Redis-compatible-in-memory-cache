"""
End-to-end: write over a real socket, kill -9 the server, restart it, and
confirm the data is still there. The exhaustive randomized version of this
lives in tools/crash_test.py; this is the fast always-run smoke version.
"""
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cache.client import Client

REPO_ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(port: int, data_dir: Path, fsync: str = "always", snapshot_interval: str = "0"):
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "cache.server",
            "--port", str(port),
            "--capacity", "1000",
            "--data-dir", str(data_dir),
            "--fsync", fsync,
            "--snapshot-interval", snapshot_interval,
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return proc
        except OSError:
            time.sleep(0.02)
    proc.kill()
    raise TimeoutError("server did not start")


def test_data_survives_sigkill(tmp_path):
    port = _free_port()
    proc = _start(port, tmp_path)
    try:
        with Client(port=port) as c:
            for i in range(200):
                assert c.set(f"k{i}", f"v{i}") == "OK"
            assert c.delete("k5") == 1
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait()

    port2 = _free_port()
    proc2 = _start(port2, tmp_path)
    try:
        with Client(port=port2) as c:
            assert c.get("k0") == b"v0"
            assert c.get("k199") == b"v199"
            assert c.get("k5") is None  # the DEL was durable too
    finally:
        proc2.kill()
        proc2.wait()


def test_data_survives_repeated_restarts(tmp_path):
    """Recovery must be idempotent — replaying a replayed log stays correct."""
    port = _free_port()
    for round_num in range(3):
        proc = _start(port, tmp_path)
        try:
            with Client(port=port) as c:
                for i in range(20):
                    assert c.set(f"r{round_num}:{i}", f"v{i}") == "OK"
                # everything written in every previous round is still here
                for prev in range(round_num + 1):
                    assert c.get(f"r{prev}:0") == b"v0"
        finally:
            proc.send_signal(signal.SIGKILL)
            proc.wait()
        port = _free_port()


def test_snapshot_compaction_survives_restart(tmp_path):
    """With snapshots on, the log is compacted but nothing is lost."""
    port = _free_port()
    proc = _start(port, tmp_path, snapshot_interval="1")
    try:
        with Client(port=port) as c:
            for i in range(500):
                c.set(f"k{i}", f"v{i}")
            time.sleep(2.5)  # let at least one snapshot + compaction happen
            c.set("after-snapshot", "yes")
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait()

    assert (tmp_path / "snapshot.rdb").exists()

    port2 = _free_port()
    proc2 = _start(port2, tmp_path, snapshot_interval="0")
    try:
        with Client(port=port2) as c:
            assert c.get("k0") == b"v0"
            assert c.get("k499") == b"v499"
            assert c.get("after-snapshot") == b"yes"
    finally:
        proc2.kill()
        proc2.wait()
