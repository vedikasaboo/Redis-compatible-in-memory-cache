#!/usr/bin/env bash
#
# validate.sh — run this AFTER you've implemented cache/resp.py's parse()
# and cache/lru.py's LRUCache. Confirms the server behaves like a real
# Redis server as far as `redis-cli` and `redis-benchmark` are concerned.
#
set -uo pipefail

PORT=6390
HOST=127.0.0.1

cleanup() {
    echo ""
    echo "Stopping server..."
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

source venv/bin/activate

echo "Starting cache server on port $PORT..."
python -m cache.server --port "$PORT" --capacity 1000 &
SERVER_PID=$!

for i in $(seq 1 50); do
    if redis-cli -p "$PORT" PING >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done

if ! redis-cli -p "$PORT" PING >/dev/null 2>&1; then
    echo "FAILED: server never became ready. Check cache/resp.py and cache/lru.py are implemented."
    exit 1
fi

echo ""
echo "=== redis-cli sanity checks ==="
echo "\$ PING"
redis-cli -p "$PORT" PING
echo "\$ SET foo bar"
redis-cli -p "$PORT" SET foo bar
echo "\$ GET foo"
redis-cli -p "$PORT" GET foo
echo "\$ DEL foo"
redis-cli -p "$PORT" DEL foo
echo "\$ GET foo (should be nil after delete)"
redis-cli -p "$PORT" GET foo

echo ""
echo "=== redis-benchmark: SET/GET, 10000 requests ==="
redis-benchmark -p "$PORT" -t set,get -n 10000

echo ""
echo "Validation complete."
