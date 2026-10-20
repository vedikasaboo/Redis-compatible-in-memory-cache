#!/usr/bin/env bash
#
# demo.sh, rehearsable end-to-end demo.
#
#   1. Launch a 3-node cluster with persistence enabled
#   2. Show keys spreading across shards via consistent hashing
#   3. Drive load and show a live hit-rate readout
#   4. kill -9 one node: surviving shards keep serving, dead shard errors cleanly
#   5. Restart it: WAL replay restores its data and the hit rate recovers
#
# Every step narrates itself, so it can be talked over.
#
set -uo pipefail

BASE_PORT=6430
DATA_ROOT="$(mktemp -d)"
CONFIG="$DATA_ROOT/cluster.json"
PIDS=()

cleanup() {
    echo ""
    say "Shutting down the cluster..."
    for pid in "${PIDS[@]:-}"; do kill -9 "$pid" 2>/dev/null; done
    rm -rf "$DATA_ROOT"
    echo "Done."
}
trap cleanup EXIT

say()   { echo "  $*"; }
step()  { echo ""; echo "=============================================================="; echo "  $1"; echo "=============================================================="; }
pause() { sleep "${DEMO_PAUSE:-1.2}"; }

if [ ! -d venv ]; then echo "No venv found. Run the setup first."; exit 1; fi
source venv/bin/activate

start_node() {
    local idx=$1
    local port=$((BASE_PORT + idx))
    mkdir -p "$DATA_ROOT/node-$idx"
    python -m cache.server --port "$port" --capacity 1500 \
        --data-dir "$DATA_ROOT/node-$idx" --fsync everysec \
        > "$DATA_ROOT/node-$idx.log" 2>&1 &
    PIDS[$idx]=$!
    disown %% 2>/dev/null || true
    for _ in $(seq 1 100); do
        redis-cli -p "$port" PING >/dev/null 2>&1 && return 0
        sleep 0.1
    done
    return 1
}

# ---------------------------------------------------------------------------
step "STEP 1: Launch a 3-node cluster"
cat > "$CONFIG" <<EOF
{
  "nodes": [
    {"id": "node-0", "host": "127.0.0.1", "port": $((BASE_PORT + 0))},
    {"id": "node-1", "host": "127.0.0.1", "port": $((BASE_PORT + 1))},
    {"id": "node-2", "host": "127.0.0.1", "port": $((BASE_PORT + 2))}
  ],
  "vnodes": 150
}
EOF
say "Each node is an independent asyncio server speaking RESP over TCP,"
say "with its own write-ahead log on disk. There is no proxy, routing"
say "happens in the client, using a consistent hash ring."
echo ""
for i in 0 1 2; do
    start_node "$i" && say "node-$i  127.0.0.1:$((BASE_PORT + i))  ready (pid ${PIDS[$i]})"
done
pause

# ---------------------------------------------------------------------------
step "STEP 2: Consistent hashing spreads the keyspace"
say "Writing 6,000 keys. The client hashes each key onto the ring and"
say "sends it to whichever node owns that arc. Same hash everywhere, so"
say "every client agrees on placement without coordinating."
echo ""
python - "$CONFIG" <<'PY'
import sys
from collections import Counter
from cache.cluster import ClusterClient, ClusterConfig

cfg = ClusterConfig.load(sys.argv[1])
with ClusterClient(cfg) as c:
    for i in range(6000):
        c.set(f"key:{i}", f"value-{i}")
    spread = Counter(c.node_for(f"key:{i}") for i in range(6000))
    for node in sorted(spread):
        n = spread[node]
        print(f"    {node}  {n:>5} keys  {'#' * (n // 120)} {n / 6000:.1%}")
PY
pause

# ---------------------------------------------------------------------------
step "STEP 3: Drive load, watch the hit rate"
say "Replaying a skewed (Zipfian) access pattern, a few hot keys, a long"
say "cold tail, which is what real caches see. Hit rate is read live from"
say "each node's INFO command."
echo ""
python - "$CONFIG" <<'PY'
import random, sys
from cache.client import Client
from cache.cluster import ClusterClient, ClusterConfig

cfg = ClusterConfig.load(sys.argv[1])
rng = random.Random(7)
with ClusterClient(cfg) as c:
    for _ in range(20000):
        # Zipf-ish: heavily favour low-numbered keys.
        k = min(int(rng.paretovariate(1.2)) - 1, 5999)
        if c.get(f"key:{k}") is None:
            c.set(f"key:{k}", f"value-{k}")

for node in cfg.nodes:
    with Client(host=node["host"], port=node["port"]) as direct:
        info = dict(
            line.split(":", 1)
            for line in direct.command("INFO").decode().splitlines()
            if ":" in line
        )
        print(f"    {node['id']}  hit rate {float(info['hit_rate']):>7.1%}   "
              f"keys {info['keys']:>5}   evictions {info['evictions']:>5}")
PY
pause

# ---------------------------------------------------------------------------
step "STEP 4: kill -9 one node"
VICTIM_PORT=$((BASE_PORT + 1))
say "Killing node-1 with SIGKILL, no graceful shutdown, no chance to"
say "flush anything. This is the ungraceful case, not a clean stop."
kill -9 "${PIDS[1]}" 2>/dev/null
sleep 0.5
say "node-1 is gone."
echo ""
say "Now the same workload against the degraded cluster:"
echo ""
python - "$CONFIG" <<'PY'
import sys
from cache.cluster import ClusterClient, ClusterConfig, ClusterError

cfg = ClusterConfig.load(sys.argv[1])
served = failed = 0
with ClusterClient(cfg) as c:
    for i in range(6000):
        try:
            c.get(f"key:{i}")
            served += 1
        except ClusterError:
            failed += 1
    print(f"    served normally : {served:>5} keys  (shards still up)")
    print(f"    failed fast     : {failed:>5} keys  (the dead shard's keys)")
    print()
    try:
        c.get("key:0" if c.node_for("key:0") == "node-1" else next(
            f"key:{i}" for i in range(6000) if c.node_for(f"key:{i}") == "node-1"))
    except ClusterError as exc:
        print(f"    error for a dead shard's key:\n      {exc}")
    print()
    print(f"    healthy shards  : {c.healthy_nodes()}")
PY
say ""
say "Note what did NOT happen: no hang, no cascading failure. Only the"
say "keys owned by node-1 error, and they error immediately."
pause

# ---------------------------------------------------------------------------
step "STEP 5: Restart node-1, recover from the write-ahead log"
say "Restarting. On startup it replays its WAL before it opens the"
say "listening socket, so no client can ever see a half-recovered cache."
echo ""
start_node 1 && say "node-1 back up (pid ${PIDS[1]})"
echo ""
say "What its log says about recovery:"
grep -E "replayed|snapshot|listening" "$DATA_ROOT/node-1.log" | tail -3 | sed 's/^/      /'
echo ""
say "Reading back the keys that were on the dead shard:"
echo ""
python - "$CONFIG" <<'PY'
import sys
from cache.cluster import ClusterClient, ClusterConfig, ClusterError

cfg = ClusterConfig.load(sys.argv[1])
with ClusterClient(cfg) as c:
    owned = [f"key:{i}" for i in range(6000) if c.node_for(f"key:{i}") == "node-1"]
    recovered = sum(1 for k in owned if c.get(k) is not None)
    print(f"    keys owned by node-1 : {len(owned)}")
    print(f"    recovered from WAL   : {recovered}")
    print(f"    healthy shards       : {c.healthy_nodes()}")
PY
say ""
say "The data survived a SIGKILL because every write was appended to the"
say "log before it was acknowledged. Fewer keys come back than were"
say "written: capacity is 1,500 per node, so LRU had already evicted the"
say "rest, and recovery deliberately does not resurrect evicted keys."
echo ""
step "Demo complete"
