#!/usr/bin/env bash
#
# tour.sh — a guided tour of real Redis behavior via redis-cli.
#
# Starts a throwaway redis-server on port 6380 (so it never collides with a
# default instance on 6379), runs through core commands, shows some edge
# cases (missing key, wrong type, expired key), then benchmarks throughput
# with and without pipelining. Shuts the server down at the end.
#
set -uo pipefail

PORT=6380
DATA_DIR="$(mktemp -d)"
LOG_FILE="$DATA_DIR/redis.log"

section() {
    echo ""
    echo "================================================================"
    echo "== $1"
    echo "================================================================"
}

cleanup() {
    section "SHUTDOWN"
    echo "Stopping redis-server on port $PORT..."
    redis-cli -p "$PORT" SHUTDOWN NOSAVE >/dev/null 2>&1
    wait "$SERVER_PID" 2>/dev/null
    rm -rf "$DATA_DIR"
    echo "Server stopped, temp dir cleaned up."
}
trap cleanup EXIT

section "STARTING redis-server ON PORT $PORT"
# --daemonize no + background job so we keep the PID and can wait() on shutdown.
# --save '' disables RDB snapshotting so this throwaway instance leaves no files behind.
redis-server --port "$PORT" --daemonize no --save '' --dir "$DATA_DIR" --logfile "$LOG_FILE" &
SERVER_PID=$!

# Wait for the server to accept connections rather than sleeping a fixed guess.
for i in $(seq 1 50); do
    if redis-cli -p "$PORT" PING >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done
echo "redis-server started, pid=$SERVER_PID, log=$LOG_FILE"
redis-cli -p "$PORT" PING

section "BASIC COMMANDS: SET / GET / DEL / EXISTS"
echo "\$ SET greeting hello"
redis-cli -p "$PORT" SET greeting hello
echo "\$ GET greeting"
redis-cli -p "$PORT" GET greeting
echo "\$ EXISTS greeting"
redis-cli -p "$PORT" EXISTS greeting
echo "\$ DEL greeting"
redis-cli -p "$PORT" DEL greeting
echo "\$ EXISTS greeting (after delete)"
redis-cli -p "$PORT" EXISTS greeting

section "TTL AND EXPIRY: SET ... EX / TTL"
echo "\$ SET session:1 active EX 5   (expires in 5 seconds)"
redis-cli -p "$PORT" SET session:1 active EX 5
echo "\$ TTL session:1   (seconds remaining)"
redis-cli -p "$PORT" TTL session:1
echo "\$ GET session:1   (still there)"
redis-cli -p "$PORT" GET session:1

section "EDGE CASE: GET on a NONEXISTENT key"
echo "\$ GET this_key_was_never_set"
redis-cli -p "$PORT" GET this_key_was_never_set
echo "(Redis returns a RESP Null Bulk String, redis-cli prints it as (nil))"

section "EDGE CASE: WRONG TYPE operation"
echo "\$ SET mystring hello"
redis-cli -p "$PORT" SET mystring hello
echo "\$ LPUSH mystring item   (mystring holds a string, not a list)"
redis-cli -p "$PORT" LPUSH mystring item
echo "(Redis refuses with a WRONGTYPE error rather than corrupting the value)"

section "EDGE CASE: EXPIRED key"
echo "\$ SET flash:key here PX 300   (expires in 300 milliseconds)"
redis-cli -p "$PORT" SET flash:key here PX 300
echo "\$ GET flash:key   (immediately, should still be here)"
redis-cli -p "$PORT" GET flash:key
echo "Sleeping 0.5s past expiry..."
sleep 0.5
echo "\$ GET flash:key   (after expiry, Redis lazily deletes on access)"
redis-cli -p "$PORT" GET flash:key
echo "\$ EXISTS flash:key   (confirms it's gone)"
redis-cli -p "$PORT" EXISTS flash:key

section "INFO memory"
redis-cli -p "$PORT" INFO memory

section "BENCHMARK: SET/GET, no pipelining (100k requests, default -P 1)"
redis-benchmark -p "$PORT" -q -t set,get -n 100000

section "BENCHMARK: SET/GET, pipelining 16 commands per request (-P 16)"
redis-benchmark -p "$PORT" -q -t set,get -n 100000 -P 16

echo ""
echo "Tour complete."
