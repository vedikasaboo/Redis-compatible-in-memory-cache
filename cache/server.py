"""
Single-node asyncio RESP server.

One handler coroutine per connection: read bytes -> resp.parse() into
commands -> commands.dispatch() against the shared LRUCache -> resp.encode()
replies -> write back.

With persistence enabled the request path gains two steps: a write is
staged to the WAL before it is applied, and replies for a batch are held
until the WAL reports them durable (see cache.persistence).

See DESIGN.md for exactly where this path awaits, and why that makes the
store safe without locks.
"""
import argparse
import asyncio
import logging

from cache.commands import dispatch, normalize_write
from cache.lru import LRUCache
from cache.persistence import DEFAULT_SNAPSHOT_INTERVAL, Persistence
from cache.resp import MAX_BULK_LEN, ErrorReply, ProtocolError, encode, parse
from cache.wal import FSYNC_MODES

logger = logging.getLogger("cache.server")

READ_CHUNK = 65536

# A connection's unparsed input is capped so an incomplete command cannot
# grow without bound. It must exceed MAX_BULK_LEN, or a legitimate maximum
# -size bulk string would be rejected while still arriving.
DEFAULT_MAX_QUERY_BUFFER = 4 * MAX_BULK_LEN  # 32MB

DEFAULT_EXPIRY_INTERVAL = 0.1
DEFAULT_EXPIRY_SAMPLE = 20


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    store: LRUCache,
    persistence: "Persistence | None" = None,
    max_query_buffer: int = DEFAULT_MAX_QUERY_BUFFER,
    idle_timeout: float = 0.0,
) -> None:
    peer = writer.get_extra_info("peername")
    logger.debug("connection from %s", peer)
    buffer = b""
    try:
        while True:
            if idle_timeout > 0:
                # An idle client otherwise holds a socket and a coroutine
                # forever. Redis calls this `timeout` and defaults it to 0
                # (disabled); so does this server.
                try:
                    data = await asyncio.wait_for(reader.read(READ_CHUNK), idle_timeout)
                except asyncio.TimeoutError:
                    logger.debug("closing %s: idle for %.1fs", peer, idle_timeout)
                    break
            else:
                data = await reader.read(READ_CHUNK)  # yields until bytes arrive or the peer closes
            if not data:
                break
            buffer += data

            if len(buffer) > max_query_buffer:
                # Either a genuinely oversized command or a client dribbling
                # an endless partial one. Both are handled the same way:
                # refuse and disconnect, rather than buffer indefinitely.
                writer.write(encode(ErrorReply("ERR Protocol error: query buffer limit exceeded")))
                await writer.drain()
                logger.warning("closing %s: query buffer limit exceeded", peer)
                break

            try:
                commands, buffer = parse(buffer)
            except ProtocolError as exc:
                writer.write(encode(ErrorReply(f"ERR Protocol error: {exc}")))
                await writer.drain()
                break  # a protocol error is unrecoverable for this connection

            if not commands:
                continue

            replies = []
            for command in commands:
                if not command:
                    continue
                # Stage-then-apply, with no await in between: the log order
                # and the in-memory apply order are the same order. If this
                # awaited between the two, two clients writing the same key
                # could be logged in one order and applied in the other, and
                # replay would rebuild a state no client ever saw.
                #
                # Dispatch the NORMALIZED command so the expiry deadline
                # applied in memory is identical to the one logged.
                record = normalize_write(command, store.now_ms())
                if record is not None:
                    if persistence is not None:
                        persistence.stage_write(record)
                    replies.append(encode(dispatch(record, store)))
                else:
                    replies.append(encode(dispatch(command, store)))

            # Durability before acknowledgement. Every writer whose records
            # landed in this tick waits on the same commit, so one fsync
            # serves the whole batch.
            if persistence is not None:
                await persistence.sync()

            for reply in replies:
                writer.write(reply)
            await writer.drain()  # backpressure: yield until the OS write buffer has room
    except (ConnectionResetError, BrokenPipeError):
        pass  # peer went away mid-request; nothing to report
    finally:
        logger.debug("closing %s", peer)
        # A peer that vanishes abruptly (redis-benchmark does this on exit)
        # can make close()/wait_closed() raise too, so the teardown path
        # needs the same guard as the read loop — otherwise the connection
        # dies with an unhandled traceback in the server log.
        try:
            writer.close()
            await writer.wait_closed()
        except (ConnectionResetError, BrokenPipeError):
            pass


async def _expiry_loop(store: LRUCache, interval: float, sample_size: int) -> None:
    """
    Active expiry.

    Lazy expiry alone leaks: a key nobody ever reads again keeps its memory
    forever. This sweeps a bounded number of TTL-carrying keys per tick —
    bounded so the sweep itself can never become a latency spike, and
    awaiting between ticks so it yields the loop back to clients.
    """
    while True:
        await asyncio.sleep(interval)
        removed = store.expire_sample(sample_size)
        if removed:
            logger.debug("active expiry removed %d keys", removed)


async def run_server(
    host: str,
    port: int,
    capacity: int,
    data_dir=None,
    fsync_mode: str = "everysec",
    group_commit: bool = True,
    snapshot_interval: float = DEFAULT_SNAPSHOT_INTERVAL,
    max_query_buffer: int = DEFAULT_MAX_QUERY_BUFFER,
    expiry_interval: float = DEFAULT_EXPIRY_INTERVAL,
    expiry_sample: int = DEFAULT_EXPIRY_SAMPLE,
    idle_timeout: float = 0.0,
) -> None:
    store = LRUCache(capacity)

    persistence = None
    if data_dir is not None:
        persistence = Persistence(
            store,
            data_dir,
            fsync_mode=fsync_mode,
            group_commit=group_commit,
            snapshot_interval=snapshot_interval,
        )
        # Recovery completes BEFORE the listening socket exists, so no
        # client can ever observe a partially-rebuilt cache.
        persistence.recover()
        await persistence.start()
        logger.info(
            "persistence on: dir=%s fsync=%s group_commit=%s snapshot_interval=%ss",
            data_dir, fsync_mode, group_commit, snapshot_interval,
        )

    expirer = None
    if expiry_interval > 0:
        expirer = asyncio.create_task(_expiry_loop(store, expiry_interval, expiry_sample))

    async def _handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await handle_client(reader, writer, store, persistence, max_query_buffer, idle_timeout)

    server = await asyncio.start_server(_handler, host, port)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    logger.info("listening on %s (capacity=%d keys)", addrs, capacity)
    try:
        async with server:
            await server.serve_forever()
    finally:
        if expirer is not None:
            expirer.cancel()
            try:
                await expirer
            except asyncio.CancelledError:
                pass
        if persistence is not None:
            await persistence.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Single-node RESP-compatible cache server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6390)
    ap.add_argument("--capacity", type=int, default=10_000, help="max number of keys before LRU eviction")
    ap.add_argument("--verbose", action="store_true", help="log every connection open/close")
    ap.add_argument("--data-dir", default=None, help="enable persistence, storing wal.log + snapshot.rdb here")
    ap.add_argument("--fsync", choices=FSYNC_MODES, default="everysec", help="WAL durability mode")
    ap.add_argument("--no-group-commit", action="store_true",
                    help="fsync once per record instead of once per batch (benchmark comparison only)")
    ap.add_argument("--snapshot-interval", type=float, default=DEFAULT_SNAPSHOT_INTERVAL,
                    help="seconds between snapshots; 0 disables")
    ap.add_argument("--max-query-buffer", type=int, default=DEFAULT_MAX_QUERY_BUFFER,
                    help="max unparsed bytes buffered per connection before disconnecting it")
    ap.add_argument("--expiry-interval", type=float, default=DEFAULT_EXPIRY_INTERVAL,
                    help="seconds between active-expiry sweeps; 0 disables (lazy expiry still applies)")
    ap.add_argument("--expiry-sample", type=int, default=DEFAULT_EXPIRY_SAMPLE,
                    help="TTL-carrying keys examined per sweep")
    ap.add_argument("--idle-timeout", type=float, default=0.0,
                    help="disconnect a connection idle this many seconds; 0 disables (Redis default)")
    args = ap.parse_args()
    if args.verbose:
        logging.getLogger("cache").setLevel(logging.DEBUG)
    try:
        asyncio.run(
            run_server(
                args.host,
                args.port,
                args.capacity,
                data_dir=args.data_dir,
                fsync_mode=args.fsync,
                group_commit=not args.no_group_commit,
                snapshot_interval=args.snapshot_interval,
                max_query_buffer=args.max_query_buffer,
                expiry_interval=args.expiry_interval,
                expiry_sample=args.expiry_sample,
                idle_timeout=args.idle_timeout,
            )
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
