# Phase 0 Notes — Exploration

These notes are generated from the actual output of `tour.sh` and `resp_bytes.py`
run on this machine (macOS, Redis 8.8.0, Python 3.11.9). No content here is
recalled from memory — every byte example and benchmark number below was
captured from a real run.

## Environment

- OS: macOS (Darwin 23.4.0, arm64)
- Redis: `8.8.0` (installed via Homebrew, `/opt/homebrew/bin/redis-server`)
- Python: `3.11.9` (via pyenv, venv at `./venv`)
- Python deps: numpy 2.4.6, lightgbm 4.7.0, xxhash 3.8.1, pytest 9.1.1

## RESP (REdis Serialization Protocol) — real byte captures

Every Redis command a client sends is a RESP **array of bulk strings**,
regardless of what the reply type is. The reply type varies. Below are real
captures from `resp_bytes.py` talking to Redis over a raw socket (no redis-py).

### Simple String (`+`)

```
SENT:     *1\r\n$4\r\nPING\r\n
RECEIVED: +PONG\r\n
```

### Error (`-`)

```
SENT:     *2\r\n$11\r\nNOTACOMMAND\r\n$4\r\narg1\r\n
RECEIVED: -ERR unknown command 'NOTACOMMAND', with args beginning with: 'arg1' \r\n
```

### Integer (`:`)

```
SENT:     *2\r\n$4\r\nINCR\r\n$7\r\ncounter\r\n
RECEIVED: :1\r\n
```

### Bulk String (`$`)

Length-prefixed — the number after `$` is the exact byte length of the payload
that follows, so a parser knows exactly how many bytes to read regardless of
what characters (including `\r\n`) appear inside the value itself.

```
SENT:     *2\r\n$3\r\nGET\r\n$8\r\ngreeting\r\n
RECEIVED: $11\r\nhello world\r\n
```

### Null Bulk String (`$-1`)

Redis's way of saying "no value" (e.g. key doesn't exist) — distinct from an
empty string (`$0\r\n\r\n`), and distinct from an error.

```
SENT:     *2\r\n$3\r\nGET\r\n$23\r\nthis_key_does_not_exist\r\n
RECEIVED: $-1\r\n
```

### Array (`*`)

Arrays can nest other RESP types, including a mix of bulk strings and nulls in
the same reply (seen here in `MGET` on two existing keys + one missing key):

```
SENT:     *4\r\n$4\r\nMGET\r\n$2\r\nk1\r\n$2\r\nk2\r\n$11\r\nmissing_key\r\n
RECEIVED: *3\r\n$2\r\nv1\r\n$2\r\nv2\r\n$-1\r\n
```

### Pipelining — two commands, one write, one read

```
SENT (1 write, 2 commands):
  *3\r\n$3\r\nSET\r\n$8\r\npipe_key\r\n$8\r\npipe_val\r\n*2\r\n$3\r\nGET\r\n$8\r\npipe_key\r\n

RECEIVED (1 read, 2 replies):
  +OK\r\n$8\r\npipe_val\r\n
```

**Implication for the Phase 1 parser:** the server does not wait for a client
to "ask nicely" for one reply at a time — replies are written back-to-back
with no extra framing beyond each reply's own leading type byte and, for bulk
strings/arrays, its length prefix. A single `recv()`/`read()` on the client
side can return zero, one, or several complete replies, or even a partial
reply that got split across a TCP segment. The parser must:
1. Never assume one read = one reply (or one reply = one read).
2. Buffer incoming bytes and only "consume" a reply once it has enough bytes
   to fully decode it (a byte count for bulk strings/arrays, or a `\r\n`
   terminator for simple strings/errors/integers).
3. Loop: decode as many complete replies as are available in the buffer,
   then go back for more bytes if the buffer runs out mid-reply.

## Command behavior — edge cases (from `tour.sh`)

- **Nonexistent key** (`GET this_key_was_never_set`): redis-cli prints `(nil)`,
  which on the wire is the RESP null bulk string `$-1\r\n` above.
- **Wrong type** (`LPUSH` on a key holding a string): Redis refuses outright —
  `WRONGTYPE Operation against a key holding the wrong kind of value` — rather
  than silently coercing or corrupting the value.
- **Expired key** (`SET flash:key here PX 300`, read after 500ms): the key
  reads back as `(nil)` and `EXISTS` returns `0`. Redis uses **lazy expiration**
  (deleted on access after its TTL passes) combined with an active background
  sweep — a key isn't necessarily purged from memory the instant it expires,
  but any read after expiry treats it as gone.

## Benchmark numbers (this machine, `redis-benchmark`, 100,000 requests)

| | No pipelining (`-P 1`) | Pipelining (`-P 16`) | Speedup |
|---|---|---|---|
| SET | 215,517 req/s (p50 0.119ms) | 1,562,500 req/s (p50 0.447ms) | ~7.25x |
| GET | 221,239 req/s (p50 0.119ms) | 2,127,660 req/s (p50 0.319ms) | ~9.62x |

**Why pipelining wins:** without pipelining, each request pays a full
round-trip (client write → network → server processes → network → client
read) before the next request can be sent, so throughput is capped by
round-trip latency, not by how fast the server can execute commands. With
`-P 16`, 16 commands are written in one shot and their 16 replies are read
back together, amortizing the round-trip cost across many commands. Note p50
latency *per batch* goes up (more work per round trip) even as aggregate
throughput goes way up — this is a throughput/latency trade-off, not a free
lunch.

This is the same reason the RESP parser has to handle multiple commands per
read (see above): pipelining is what makes it happen, and it's a normal,
expected client behavior — not an edge case.

## asyncio warm-up (`echo_server.py` / `echo_client.py`)

Confirmed working end-to-end: client sent `hello`, `asyncio`, `world` and got
each echoed back exactly. Every `await` point in both scripts is commented
with what it yields on:

- `reader.read()` — yields until bytes are available on the socket
- `writer.drain()` — yields until the OS write buffer has room (backpressure)
- `writer.wait_closed()` — yields until the socket is fully torn down
- `asyncio.start_server(...)` / `asyncio.open_connection(...)` — yield during
  the bind/connect handshake
- `server.serve_forever()` — yields indefinitely, waking only to dispatch work

This matters for Phase 1: every one of these `await` points is a place where
the event loop can run a *different* client's coroutine. A server handling many
connections concurrently will have their `handle_client` coroutines interleaved
at exactly these points — never in the middle of a single non-async statement.

## References

- RESP protocol spec: https://redis.io/docs/latest/develop/reference/protocol-spec/
- Redis persistence (RDB/AOF): https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/

## Explicitly out of scope for Phase 0

No parser, server, cache, or eviction logic was written here — that's Phase 1.
This phase only confirms: real Redis is installed and behaves as documented,
the wire format is understood at the byte level, and the asyncio primitives
needed to write a concurrent server are warmed up.
