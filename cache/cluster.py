"""
Client-side cluster routing over a STATIC membership list.

There is no proxy and no gossip: every client reads the same config file,
builds the same ring, and therefore computes the same owner for a key. That
only works because the hash is process-stable (see cache.hashring).

Deliberately out of scope for v1: dynamic join/leave with live key
migration. Adding a node here changes where keys are *looked up* but does
not move any data, so a static config is the honest boundary.

Failure model is per-shard. A dead node makes its OWN keys error
immediately with a clear message rather than hanging; every other shard
keeps serving. The connection is retried on next use, so a node that comes
back is picked up without restarting the client.
"""
import json
from pathlib import Path

from cache.client import CacheError, Client
from cache.hashring import DEFAULT_VNODES, HashRing


class ClusterError(Exception):
    """A shard is unreachable, or a command spans shards."""


class ClusterConfig:
    def __init__(self, nodes: list[dict], vnodes: int = DEFAULT_VNODES):
        if not nodes:
            raise ValueError("cluster config must list at least one node")
        self.nodes = nodes
        self.vnodes = vnodes

    @classmethod
    def load(cls, path) -> "ClusterConfig":
        data = json.loads(Path(path).read_text())
        return cls(nodes=data["nodes"], vnodes=data.get("vnodes", DEFAULT_VNODES))

    @classmethod
    def local(cls, count: int, start_port: int = 6390, vnodes: int = DEFAULT_VNODES) -> "ClusterConfig":
        return cls(
            [{"id": f"node-{i}", "host": "127.0.0.1", "port": start_port + i} for i in range(count)],
            vnodes=vnodes,
        )

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"nodes": self.nodes, "vnodes": self.vnodes}, indent=2) + "\n")

    @property
    def ids(self) -> list[str]:
        return [n["id"] for n in self.nodes]

    def address(self, node_id: str) -> tuple[str, int]:
        for n in self.nodes:
            if n["id"] == node_id:
                return n["host"], n["port"]
        raise KeyError(node_id)


# Commands whose keys are all the arguments after the name. Anything here
# has to be checked for spanning shards before it can be routed.
MULTI_KEY_COMMANDS = {b"MGET", b"DEL", b"EXISTS"}


class ClusterClient:
    """
    Routes each key to its owning node and keeps one connection per node.

    Not thread-safe: give each thread its own instance, as you would with a
    plain socket.
    """

    def __init__(self, config: ClusterConfig, timeout: float = 5.0):
        self.config = config
        self.timeout = timeout
        self.ring = HashRing(config.ids, vnodes=config.vnodes)
        self._conns: dict[str, Client] = {}

    # -- connection management -------------------------------------------

    def _connection(self, node_id: str) -> Client:
        conn = self._conns.get(node_id)
        if conn is not None:
            return conn
        host, port = self.config.address(node_id)
        try:
            conn = Client(host=host, port=port, timeout=self.timeout).connect()
        except OSError as exc:
            raise ClusterError(f"shard {node_id} at {host}:{port} is unreachable: {exc}") from None
        self._conns[node_id] = conn
        return conn

    def _drop(self, node_id: str) -> None:
        """Forget a broken connection so the next call retries the node."""
        conn = self._conns.pop(node_id, None)
        if conn is not None:
            conn.close()

    def close(self) -> None:
        for node_id in list(self._conns):
            self._drop(node_id)

    def __enter__(self) -> "ClusterClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- routing ---------------------------------------------------------

    def node_for(self, key) -> str:
        if isinstance(key, str):
            key = key.encode()
        node = self.ring.get_node(key)
        if node is None:
            raise ClusterError("cluster has no nodes")
        return node

    def _run_on(self, node_id: str, *args):
        conn = self._connection(node_id)
        try:
            return conn.command(*args)
        except CacheError:
            raise  # a server-side error reply is the server's answer, not a failure
        except (OSError, ConnectionError) as exc:
            self._drop(node_id)
            raise ClusterError(f"shard {node_id} failed mid-command: {exc}") from None

    def command(self, *args):
        """
        Route one command by its FIRST key argument.

        Multi-key commands are rejected when their keys do not all live on
        one shard — the same restriction Redis Cluster enforces with hash
        slots. Executing them piecewise would silently break atomicity, so
        erroring is the honest option.
        """
        if not args:
            raise ClusterError("empty command")
        name = args[0].upper() if isinstance(args[0], str) else args[0].upper().decode()
        name_b = name.encode()

        if name_b in (b"PING",):
            # No key: answer from any live shard.
            return self._run_on(self.config.ids[0], *args)
        if len(args) < 2:
            raise ClusterError(f"{name} requires a key when routed through a cluster")

        keys = args[1:] if name_b in MULTI_KEY_COMMANDS else args[1:2]
        owners = {self.node_for(k) for k in keys}
        if len(owners) > 1:
            raise ClusterError(
                f"{name} spans {len(owners)} shards; cross-shard multi-key commands are not supported"
            )
        return self._run_on(owners.pop(), *args)

    # -- conveniences ----------------------------------------------------

    def get(self, key):
        return self.command("GET", key)

    def set(self, key, value, **kwargs):
        for name, val in (("EX", kwargs.get("ex")), ("PX", kwargs.get("px"))):
            if val is not None:
                return self.command("SET", key, value, name, val)
        return self.command("SET", key, value)

    def delete(self, *keys):
        return self.command("DEL", *keys)

    def ttl(self, key):
        return self.command("TTL", key)

    def healthy_nodes(self) -> list[str]:
        """Node ids that currently answer PING."""
        alive = []
        for node_id in self.config.ids:
            try:
                if self._run_on(node_id, "PING") == "PONG":
                    alive.append(node_id)
            except ClusterError:
                pass
        return alive
