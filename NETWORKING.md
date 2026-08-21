# Networking concepts demonstrated

A map from parts of this project to the computer-networking topics they
implement. The first three sections are the networking core; the last two
are extensions built on top of it.

---

## 1. Application-layer protocol design (RESP over TCP)

**Where:** [`cache/resp.py`](cache/resp.py), [`resp_bytes.py`](resp_bytes.py)

TCP gives you a byte stream and nothing else — no notion of "a message".
Any protocol above it must define its own message boundaries. RESP does
this with a one-byte type tag and, for variable-length types, an explicit
length prefix:

```
*3\r\n$3\r\nSET\r\n$3\r\nfoo\r\n$3\r\nbar\r\n
│     │                                      
│     └── bulk string: 3 bytes follow
└──────── array of 3 elements
```

The length prefix is what makes the protocol **binary-safe**: the parser
reads exactly N bytes rather than scanning for a terminator, so a value
containing `\r\n` cannot be mistaken for the end of a field. There is a
test for exactly this (`test_binary_safe_bulk_string`).

Two protocol layers are visible here: a *request* is always an array of
bulk strings, while a *reply* may be any of five types. That asymmetry is
why the encoder and decoder in this project are separate functions rather
than one symmetric codec.

**Talking points:** why length-prefixing beats delimiters; why the protocol
is text-ish but not line-based; how `redis-cli` and `redis-benchmark` — real
third-party clients — drive this server unmodified, which is the practical
proof the protocol was implemented to spec and not to a private convention.

---

## 2. Stream framing and partial reads

**Where:** [`cache/resp.py`](cache/resp.py) `parse()`, [`cache/server.py`](cache/server.py)

The single most important networking fact in the project: **one `read()`
does not equal one message.** The kernel hands you whatever bytes have
arrived. A command can be split across two TCP segments, and several
commands can arrive in one.

`parse(buffer) -> (commands, leftover)` handles both directions:

- **Partial read** — the buffer ends mid-command, so the incomplete tail is
  returned as `leftover` and re-parsed after more bytes arrive. It does not
  raise, and it does not block.
- **Pipelining** — several complete commands in one buffer are all returned
  from a single call.

The server's loop is therefore `buffer = leftover + await reader.read(...)`.

Tests drive this at the extreme: `test_partial_send_byte_by_byte` sends a
command one byte per `send()` call, and `test_incremental_feed_across_multiple_calls`
feeds three bytes at a time.

**Talking points:** why a naive `read()`-then-`parse()` server is silently
broken until the first slow client or pipelined batch; why the parser must
be a resumable state machine over a buffer, not a function over a message.

---

## 3. Event-loop concurrency and multiplexed I/O

**Where:** [`cache/server.py`](cache/server.py), [`DESIGN.md`](DESIGN.md)

One thread serves hundreds of connections. Instead of a thread per client
blocking on `recv()`, `asyncio` multiplexes over `kqueue`/`epoll`: every
coroutine that is waiting on a socket is suspended, and the loop wakes only
the ones whose sockets are ready.

The consequence that matters for correctness: **another client's command can
only interleave at an `await`.** Everything between two awaits runs
atomically. `DESIGN.md` enumerates every await point in the request path,
which is what justifies the store having no locks at all.

Backpressure appears as `await writer.drain()`: if a client stops reading
its replies, the OS send buffer fills, `drain()` suspends that coroutine,
and — because that coroutine is also the one that reads that client's input
— the server stops consuming from it. Flow control propagates from TCP up
into the application without any explicit rate limiting.

Hardening lives here too: a per-connection query-buffer cap and a
multibulk-length cap, so a client cannot pin unbounded memory by
announcing a huge command and then trickling it.

**Talking points:** blocking vs non-blocking I/O; C10K; why the GIL is not
the bottleneck for an I/O-bound server; how `drain()` is TCP flow control
surfaced to the application; measured cost — ~99k SET/s and ~123k GET/s from
Python, against ~215k/s for C Redis on the same machine.

---

## 4. Distributed systems: consistent-hash routing

**Where:** [`cache/hashring.py`](cache/hashring.py), [`cache/cluster.py`](cache/cluster.py)

Routing is **client-side**: no proxy, no gossip, no coordinator. Every
client reads the same static config, builds the same ring, and independently
computes the same owner for a key. That only works because the hash is
**process-stable** — Python's builtin `hash()` is salted per process, so
two clients would disagree. There is a test that demonstrates the builtin
genuinely differs across processes, and another that the chosen hash does
not.

`hash(key) % N` spreads keys evenly but remaps ~N/(N+1) of them when a node
joins. Consistent hashing maps keys and nodes onto one circle and gives each
key to the first node clockwise, so a join steals only the arc between the
new node and its predecessor — about 1/N.

Measured on 200,000 keys over 4 nodes:

| | max/min load | keys moved when a 5th node joins |
|---|---|---|
| 1 virtual node | 4.30× (one node held 55.8%) | — |
| 150 virtual nodes | 1.14× | **22.6%** (1/N ideal = 20%) |
| modulo hashing | 1.00× | ~80% |

Virtual nodes matter because a handful of random points do not divide a
circle evenly; averaging 150 arcs per node shrinks the spread like 1/√V.

**Failure model:** per-shard. A dead node's keys fail fast with an error
naming the shard; every other shard keeps serving; the connection retries on
next use. Cross-shard multi-key commands are refused rather than executed
piecewise — the same restriction Redis Cluster enforces with hash slots.

**Talking points:** why cache invalidation storms make the remap fraction the
whole point; partition tolerance without replication; why static membership
is an honest boundary — the ring changes *lookup*, not data placement, so
live migration is a separate problem.

---

## 5. Durability: write-ahead logging

**Where:** [`cache/wal.py`](cache/wal.py), [`cache/snapshot.py`](cache/snapshot.py)

Storage rather than networking, but it interacts with the event loop in a
way that is squarely a systems-networking concern: **`fsync` never runs on
the event loop.** A disk flush takes milliseconds; doing it inline would
freeze every connected client for that whole time. It goes to a thread pool
via `run_in_executor`.

**Group commit** is the throughput lever: all writes staged in one
event-loop tick share a single `fsync`. Measured, 8 concurrent clients:

| mode | writes/sec |
|---|---|
| `always`, no group commit | 9,195 |
| `always`, group commit | 22,535 (**2.5×**) |
| `everysec` | 59,771 |

With one client the speedup is 1.0× — there is nothing to batch, which is
the check that the gain is real batching and not measurement noise.

**Ordering:** a record is staged and applied to memory with no `await`
between the two, so log order and memory order are identical. If the
durability wait happened *between* them, two clients writing the same key
could be logged in one order and applied in the other, and replay would
rebuild a state no client ever observed.

**Talking points:** the durability/latency trade (`always` vs `everysec`);
why compaction uses a snapshot rather than a log rewrite (a rewrite would
resurrect keys the cache had already *evicted*); why a torn tail from
`kill -9` is handled for free by the same parser that handles partial TCP
reads — both are "the bytes stop mid-record".

---

## Quick reference

| Concept | File | Evidence |
|---|---|---|
| Application-layer protocol | `cache/resp.py` | `redis-cli`/`redis-benchmark` drive it unmodified |
| Message framing | `cache/resp.py` `parse()` | byte-by-byte and pipelined send tests |
| Non-blocking multiplexed I/O | `cache/server.py` | ~99k SET/s single-threaded |
| Flow control / backpressure | `await writer.drain()` | slow-client and non-reading-client tests |
| Resource exhaustion defence | query-buffer + multibulk caps | oversized-command tests |
| Consistent hashing | `cache/hashring.py` | 22.6% remap vs ~80% for modulo |
| Partial failure | `cache/cluster.py` | `kill -9` one shard, others keep serving |
| Write-ahead durability | `cache/wal.py` | 100× `kill -9` crash test |
