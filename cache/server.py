"""
Single-node asyncio RESP server.

One handler coroutine per connection: read bytes -> resp.parse() into
commands -> commands.dispatch() against the shared LRUCache -> resp.encode()
replies -> write back. This module is plumbing around the two pieces you
implement (cache.resp.parse, cache.lru.LRUCache) — it won't do anything
useful until those are filled in.
"""
import argparse
import asyncio
import logging

from cache.commands import dispatch
from cache.lru import LRUCache
from cache.resp import ErrorReply, ProtocolError, encode, parse

logger = logging.getLogger("cache.server")

READ_CHUNK = 65536


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, store: LRUCache) -> None:
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

            for command in commands:
                if not command:
                    continue
                reply = dispatch(command, store)
                writer.write(encode(reply))
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


async def run_server(host: str, port: int, capacity: int) -> None:
    store = LRUCache(capacity)

    async def _handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await handle_client(reader, writer, store)

    server = await asyncio.start_server(_handler, host, port)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    logger.info("listening on %s (capacity=%d keys)", addrs, capacity)
    async with server:
        await server.serve_forever()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Single-node RESP-compatible cache server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6390)
    ap.add_argument("--capacity", type=int, default=10_000, help="max number of keys before LRU eviction")
    ap.add_argument("--verbose", action="store_true", help="log every connection open/close")
    args = ap.parse_args()
    if args.verbose:
        logging.getLogger("cache").setLevel(logging.DEBUG)
    asyncio.run(run_server(args.host, args.port, args.capacity))


if __name__ == "__main__":
    main()
