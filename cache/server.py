"""
Single-node asyncio RESP server.

One handler coroutine per connection: read bytes -> resp.parse() into
commands -> commands.dispatch() against the shared LRUCache -> resp.encode()
replies -> write back.

With persistence enabled the request path gains two steps: a write is
staged to the WAL before it is applied, and replies for a batch are held
until the WAL reports them durable (see cache.persistence).
"""
import argparse
import asyncio
import logging

from cache.commands import dispatch, is_durable_write
from cache.lru import LRUCache
from cache.persistence import DEFAULT_SNAPSHOT_INTERVAL, Persistence
from cache.resp import ErrorReply, ProtocolError, encode, parse
from cache.wal import FSYNC_MODES

logger = logging.getLogger("cache.server")

READ_CHUNK = 65536


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    store: LRUCache,
    persistence: "Persistence | None" = None,
) -> None:
    peer = writer.get_extra_info("peername")
    logger.debug("connection from %s", peer)
    buffer = b""
    try:
        while True:
            data = await reader.read(READ_CHUNK)  # yields until bytes arrive or the peer closes
            if not data:
                break
            buffer += data

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
                if persistence is not None and is_durable_write(command):
                    persistence.stage_write(command)
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


async def run_server(
    host: str,
    port: int,
    capacity: int,
    data_dir=None,
    fsync_mode: str = "everysec",
    group_commit: bool = True,
    snapshot_interval: float = DEFAULT_SNAPSHOT_INTERVAL,
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

    async def _handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await handle_client(reader, writer, store, persistence)

    server = await asyncio.start_server(_handler, host, port)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    logger.info("listening on %s (capacity=%d keys)", addrs, capacity)
    try:
        async with server:
            await server.serve_forever()
    finally:
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
            )
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
