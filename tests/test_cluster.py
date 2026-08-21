"""
Cluster routing tests against a real 3-node cluster.

Covers the two behaviours that matter operationally: a dead shard must
error clearly instead of hanging while the others keep serving, and a
multi-key command spanning shards must be rejected rather than silently
executed piecewise.
"""
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cache.cluster import ClusterClient, ClusterConfig, ClusterError

REPO_ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(port: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.02)
    return False


class Cluster:
    def __init__(self, count: int):
        self.config = ClusterConfig(
            [{"id": f"node-{i}", "host": "127.0.0.1", "port": _free_port()} for i in range(count)]
        )
        self.procs: dict[str, subprocess.Popen] = {}
        for node in self.config.nodes:
            self.procs[node["id"]] = self._spawn(node)
        for node in self.config.nodes:
            assert _wait(node["port"]), f"{node['id']} did not start"

    def _spawn(self, node) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, "-m", "cache.server",
             "--port", str(node["port"]), "--capacity", "10000"],
            cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def kill(self, node_id: str) -> None:
        self.procs[node_id].send_signal(signal.SIGKILL)
        self.procs[node_id].wait()

    def restart(self, node_id: str) -> None:
        node = next(n for n in self.config.nodes if n["id"] == node_id)
        self.procs[node_id] = self._spawn(node)
        assert _wait(node["port"])

    def stop(self) -> None:
        for proc in self.procs.values():
            proc.kill()
            proc.wait()


@pytest.fixture
def cluster():
    c = Cluster(3)
    try:
        yield c
    finally:
        c.stop()


@pytest.fixture
def client(cluster):
    with ClusterClient(cluster.config) as c:
        yield c


def test_all_shards_are_healthy(client):
    assert len(client.healthy_nodes()) == 3


def test_set_and_get_across_shards(client):
    for i in range(300):
        assert client.set(f"key:{i}", f"v{i}") == "OK"
    for i in range(300):
        assert client.get(f"key:{i}") == f"v{i}".encode()


def test_keys_actually_spread_over_shards(client):
    used = {client.node_for(f"key:{i}") for i in range(300)}
    assert len(used) == 3, "keys should not all land on one shard"


def test_key_lands_on_the_shard_the_ring_names(cluster, client):
    """Routing is only correct if the data really is where the ring says."""
    from cache.client import Client

    client.set("probe", "value")
    owner = client.node_for("probe")
    host, port = cluster.config.address(owner)
    with Client(host=host, port=port) as direct:
        assert direct.get("probe") == b"value"

    for node in cluster.config.nodes:
        if node["id"] == owner:
            continue
        with Client(host=node["host"], port=node["port"]) as other:
            assert other.get("probe") is None


def test_dead_shard_errors_clearly_and_does_not_hang(cluster, client):
    for i in range(300):
        client.set(f"key:{i}", f"v{i}")
    victim = client.node_for("key:0")
    cluster.kill(victim)

    started = time.time()
    with pytest.raises(ClusterError) as exc:
        for i in range(300):
            if client.node_for(f"key:{i}") == victim:
                client.get(f"key:{i}")
    assert time.time() - started < 5, "a dead shard must fail fast, not hang"
    assert victim in str(exc.value)


def test_surviving_shards_keep_serving_after_one_dies(cluster, client):
    for i in range(300):
        client.set(f"key:{i}", f"v{i}")
    victim = client.node_for("key:0")
    cluster.kill(victim)

    served = 0
    for i in range(300):
        if client.node_for(f"key:{i}") == victim:
            continue
        assert client.get(f"key:{i}") == f"v{i}".encode()
        served += 1
    assert served > 0
    assert len(client.healthy_nodes()) == 2


def test_shard_is_picked_up_again_after_restart(cluster, client):
    victim = client.node_for("key:0")
    cluster.kill(victim)
    with pytest.raises(ClusterError):
        client.get("key:0")

    cluster.restart(victim)
    # No client restart needed: the connection is retried on next use.
    assert client.set("key:0", "back") == "OK"
    assert client.get("key:0") == b"back"
    assert len(client.healthy_nodes()) == 3


def test_cross_shard_multikey_is_rejected(client):
    """Redis Cluster refuses these too; executing piecewise would break atomicity."""
    keys = [f"key:{i}" for i in range(200)]
    spanning = None
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if client.node_for(a) != client.node_for(b):
                spanning = (a, b)
                break
        if spanning:
            break
    assert spanning is not None

    with pytest.raises(ClusterError, match="spans"):
        client.command("MGET", *spanning)
    with pytest.raises(ClusterError, match="spans"):
        client.delete(*spanning)


def test_same_shard_multikey_is_allowed(client):
    keys = [f"key:{i}" for i in range(400)]
    by_node = {}
    for k in keys:
        by_node.setdefault(client.node_for(k), []).append(k)
    same_shard = max(by_node.values(), key=len)[:3]
    assert len(same_shard) >= 2

    for k in same_shard:
        client.set(k, "v")
    assert client.command("EXISTS", *same_shard) == len(same_shard)
    assert client.delete(*same_shard) == len(same_shard)


def test_ttl_routes_correctly(client):
    client.set("expiring", "v", ex=100)
    assert 95 <= client.ttl("expiring") <= 100


def test_config_roundtrip(tmp_path):
    path = tmp_path / "cluster.json"
    ClusterConfig.local(3, start_port=7000).save(path)
    loaded = ClusterConfig.load(path)
    assert loaded.ids == ["node-0", "node-1", "node-2"]
    assert loaded.address("node-1") == ("127.0.0.1", 7001)


def test_empty_config_rejected():
    with pytest.raises(ValueError):
        ClusterConfig([])
