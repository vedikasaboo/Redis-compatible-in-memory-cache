"""
Point-in-time snapshots of the store, used for log compaction.

Why snapshot rather than rewrite the log
----------------------------------------
The obvious way to shrink a WAL is to rewrite it as the minimal set of
commands reproducing current state. That is subtly wrong for a CACHE:
records for keys the LRU has since EVICTED are still in the log, and a
naive rewrite that only drops superseded writes would resurrect them on
replay — the recovered cache would hold keys the live one had already
thrown away, over capacity and diverged.

Serializing the CURRENT dict sidesteps that entirely. Whatever is not in
the dict is not in the snapshot, evicted keys included. The snapshot
records the log offset it corresponds to; recovery loads it and replays
only the log tail after that offset.

File format: a fixed-size header carrying that offset, then one RESP array
per entry — `[key, value]`, or `[key, value, expires_at]` for a key with a
TTL — so loading reuses resp.parse() just like the WAL does. The deadline
is stored ABSOLUTE (epoch ms), so a key whose TTL elapsed while the server
was down comes back already expired instead of getting a fresh lease.
"""
import asyncio
import logging
import os
from pathlib import Path

from cache.resp import ProtocolError, encode, parse

logger = logging.getLogger("cache.snapshot")

SNAP_MAGIC = b"EVICTSNAP1 "
SNAP_HEADER_LEN = len(SNAP_MAGIC) + 20 + 2

DEFAULT_CHUNK = 512


def _encode_header(log_offset: int) -> bytes:
    return SNAP_MAGIC + f"{log_offset:020d}".encode() + b"\r\n"


def _decode_header(raw: bytes) -> int:
    if len(raw) != SNAP_HEADER_LEN or not raw.startswith(SNAP_MAGIC):
        raise ValueError("not a valid snapshot header")
    try:
        return int(raw[len(SNAP_MAGIC) : -2])
    except ValueError:
        raise ValueError("corrupt snapshot header offset") from None


def _encode_entry(entry: tuple) -> bytes:
    """(key, value) or (key, value, expires_at) -> one RESP array."""
    key, value = entry[0], entry[1]
    expires_at = entry[2] if len(entry) > 2 else None
    if expires_at is None:
        return encode([key, value])
    return encode([key, value, str(expires_at).encode()])


async def write_snapshot(
    path: Path,
    items: list[tuple],
    log_offset: int,
    chunk_size: int = DEFAULT_CHUNK,
) -> None:
    """
    Write `items` as the snapshot at `log_offset`.

    Each item is (key, value) or (key, value, expires_at).

    `items` must already be a materialized point-in-time list (see
    LRUCache.items()), NOT a live view — this coroutine awaits between
    chunks, so anything lazily iterating the real store would observe
    mutations partway through and produce a torn snapshot.

    Written to a temp file and renamed, so a crash partway through leaves
    the previous snapshot untouched; the rename is the commit point.
    """
    loop = asyncio.get_running_loop()
    tmp = path.with_suffix(path.suffix + ".tmp")

    f = open(tmp, "wb")
    try:
        f.write(_encode_header(log_offset))
        buf: list[bytes] = []
        for i, entry in enumerate(items, 1):
            buf.append(_encode_entry(entry))
            if i % chunk_size == 0:
                await loop.run_in_executor(None, f.write, b"".join(buf))
                buf.clear()
                # Yield: snapshotting a large store must not monopolize the
                # event loop, or every client stalls for the whole dump.
                await asyncio.sleep(0)
        if buf:
            await loop.run_in_executor(None, f.write, b"".join(buf))
        await loop.run_in_executor(None, f.flush)
        await loop.run_in_executor(None, os.fsync, f.fileno())
    finally:
        f.close()

    def _commit() -> None:
        os.replace(tmp, path)
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)  # make the rename durable, not just the data
        finally:
            os.close(dir_fd)

    await loop.run_in_executor(None, _commit)
    logger.info("wrote snapshot: %d entries at log offset %d", len(items), log_offset)


def load_snapshot(path: Path) -> tuple[list[tuple[bytes, bytes, int | None]], int]:
    """
    Load (items, log_offset), each item (key, value, expires_at | None).
    Returns ([], 0) if there is no snapshot.

    A snapshot that fails to parse is treated as absent rather than fatal:
    the WAL alone is always sufficient to rebuild state, just more slowly,
    so a damaged snapshot should cost time and not data.
    """
    if not path.exists():
        return [], 0
    try:
        with open(path, "rb") as f:
            log_offset = _decode_header(f.read(SNAP_HEADER_LEN))
            body = f.read()
        entries, _ = parse(body)
    except (ValueError, ProtocolError) as exc:
        logger.warning("ignoring unreadable snapshot (%s); replaying full log instead", exc)
        return [], 0

    items = []
    for entry in entries:
        if len(entry) == 2:
            items.append((entry[0], entry[1], None))
        elif len(entry) == 3:
            try:
                items.append((entry[0], entry[1], int(entry[2])))
            except ValueError:
                logger.warning("ignoring snapshot entry with unreadable expiry")
        else:
            logger.warning("ignoring malformed snapshot entry of length %d", len(entry))
    return items, log_offset
