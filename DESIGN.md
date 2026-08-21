# Design notes

Running document. Sections are added as phases land.

---

## Concurrency: where the request path yields, and why there are no locks

The server handles every connection in its own coroutine on a **single
thread**. There is exactly one OS thread touching the store, so the
question is never "can two CPUs race" — it is "where can the event loop
switch from one client's coroutine to another's". That can only happen at
an `await`. Between two `await` points, a coroutine runs to completion with
no interleaving whatsoever.

That is the whole concurrency argument: **the store is only ever mutated by
synchronous code between await points, so it needs no locks.**

### The await points in `handle_client` (cache/server.py)

In order, per iteration of the read loop:

| # | Await | What it yields on |
|---|---|---|
| 1 | `data = await reader.read(READ_CHUNK)` | Bytes arriving on this socket, or the peer closing. This is where a connection spends nearly all its life. |
| 2 | `await writer.drain()` *(query-buffer-limit branch)* | The error reply reaching the OS send buffer, just before disconnecting a buffer hog. |
| 3 | `await writer.drain()` *(protocol-error branch)* | Same, for a malformed-RESP disconnect. |
| 4 | `await persistence.sync()` | The WAL reporting this batch durable. Returns immediately unless `fsync=always`. |
| 5 | `await writer.drain()` | The OS send buffer having room — i.e. backpressure from a client that is not reading fast enough. |
| 6 | `await writer.wait_closed()` *(in `finally`)* | The socket finishing teardown. |

### What runs synchronously — and therefore atomically

Everything between those points, notably:

- `parse(buffer)` — RESP decoding
- `normalize_write(command, store.now_ms())` — expiry rewriting
- `persistence.stage_write(record)` — appending to the WAL's pending list
- `dispatch(command, store)` — **all store reads, writes, deletes, evictions and lazy expiry**
- `encode(reply)` — RESP encoding

Because none of these await, no other client's command can be observed
partway through any of them. A `SET` that evicts an LRU victim, updates the
hashmap, and splices the linked list is indivisible from every other
coroutine's point of view.

### The one ordering constraint this creates

Staging to the WAL and applying to the store **must not be separated by an
await**, and in `handle_client` they are not:

```python
record = normalize_write(command, store.now_ms())
if record is not None:
    if persistence is not None:
        persistence.stage_write(record)      # establishes log order
    replies.append(encode(dispatch(record, store)))   # applies, same order
```

If an `await` sat between those two lines, two clients writing the same key
could be logged in one order and applied in memory in the other. Replay
would then rebuild a state that no client ever observed. The durability
wait (`await persistence.sync()`, point 4) happens *after* both, because it
governs only when the client is *told* the write succeeded — not when it
takes effect.

### Background coroutines

Three other coroutines touch shared state. Each is safe for the same
reason — its mutations are synchronous:

- **WAL commit loop** (`cache/wal.py::_commit_loop`) — swaps the pending
  list out *before* awaiting the executor, so records staged during the
  flush land in a fresh list. Holds an `asyncio.Lock` shared with
  compaction, so the two can never rewrite the file concurrently.
- **Active expiry** (`cache/server.py::_expiry_loop`) — `await sleep(...)`,
  then a synchronous bounded `store.expire_sample(n)`. Bounded so the sweep
  can never itself become a latency spike.
- **Snapshot loop** (`cache/persistence.py::snapshot_now`) — the delicate
  one. It flushes, then reads the log offset and `store.items()` with **no
  await between them**, so the pair is a consistent point-in-time view.
  `items()` returns a materialized list, so the subsequent chunked write
  (which *does* await, to yield the loop back to clients) cannot observe a
  torn store.

### Where fsync goes

Never on the event loop. Every `os.fsync` runs via
`loop.run_in_executor(...)`, because a single blocking disk flush on the
loop thread would freeze *every* connected client, not just the one
waiting on durability. Measured effect: with `everysec`, throughput is
within 0.3% of not syncing at all (68,088 vs 68,292 writes/sec).

---

## Hardening: what a misbehaving client cannot do

Each limit below is covered by a test in `tests/test_robustness.py` that
was verified to **fail** when the limit is removed.

| Limit | Where | Without it |
|---|---|---|
| `MAX_BULK_LEN` (8MB) | `cache/resp.py` | A client could declare a huge bulk length and make the server buffer it. Rejected on the *declared length*, before any payload is read. |
| `MAX_MULTIBULK_LEN` (1M elements) | `cache/resp.py` | `*999999999\r\n` followed by a slow trickle pins a partial command of unbounded size. **Verified**: without the cap the server accepts the header and waits forever. |
| `--max-query-buffer` (32MB) | `cache/server.py` | An incomplete command can pin unbounded memory per connection. **Verified**: without the cap the server buffered 3MB of a never-completing command against a 1MB limit without complaint. |

Three client pathologies, and why each is contained:

- **One byte per second.** Costs one idle coroutine parked at await #1.
  The partial command sits in that connection's own buffer (now capped).
  Other clients see no latency change.
- **Malformed RESP.** `parse()` raises `ProtocolError`; the server replies
  with an error and closes *that* connection only.
- **Never reads its replies.** The server's writes fill the socket buffers,
  then await #5 (`drain()`) suspends that handler — and only that handler.
  Since the handler does not return to await #1 until it drains, it also
  stops consuming that client's input, so TCP backpressure propagates all
  the way back to the sender. Memory stays bounded without any explicit
  output-buffer accounting.

**Not implemented:** an idle-connection timeout. A client that connects and
sends nothing holds a socket and a coroutine indefinitely. Real Redis has a
`timeout` config for this; this server does not.

---

## TTL

Expiry deadlines are **absolute epoch milliseconds**, never relative
durations, everywhere below the command layer.

`SET k v EX 10` is rewritten to `SET k v PXAT <deadline>` by
`normalize_write()` *before* it is logged or applied, so:

- The WAL records an absolute instant. Replaying a logged `EX 10` an hour
  after a crash would otherwise grant a fresh 10 seconds — the key would
  come back from the dead. Redis rewrites its AOF the same way.
- Snapshots store the deadline alongside the value, so a key whose TTL
  elapsed while the server was *down* is dropped on load rather than
  re-leased.
- The value logged and the value applied are byte-identical, because the
  server dispatches the *normalized* command rather than the original.

Expiry is **lazy plus sampled**:

- **Lazy** — `get()`, `ttl_ms()` and `in` collect an expired key at the
  moment they touch it. Cost is O(1) and paid by the reader.
- **Sampled** — `expire_sample(n)` walks a rotating cursor over the set of
  TTL-carrying keys, deleting expired ones. Lazy expiry alone leaks: a key
  nobody ever reads again would hold its memory forever. The cursor is used
  instead of random sampling so no key can be starved, and the snapshot it
  walks is rebuilt only once per full sweep, amortizing to O(1) per key
  examined.

`len(store)` counts entries that have expired but not yet been collected,
matching Redis's `DBSIZE`. Use `get()` / `ttl_ms()` / `in` for
expiry-accurate answers.
