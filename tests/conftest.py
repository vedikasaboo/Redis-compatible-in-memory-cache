"""Shared fixture: launches a real server subprocess for integration tests."""
import socket
import subprocess
import sys
import time

import pytest

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"server on {host}:{port} did not become ready in {timeout}s")


@pytest.fixture
def server(request):
    """Starts cache.server on a free port with a small default capacity.

    Use @pytest.mark.capacity(N) on a test to override the LRU capacity.
    """
    marker = request.node.get_closest_marker("capacity")
    capacity = marker.args[0] if marker else 100
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "cache.server", "--port", str(port), "--capacity", str(capacity)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_port("127.0.0.1", port)
        yield ("127.0.0.1", port)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def pytest_configure(config):
    config.addinivalue_line("markers", "capacity(n): set the LRUCache capacity for the `server` fixture")
