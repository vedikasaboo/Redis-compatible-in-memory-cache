"""
Write-ahead log.

Record format is just RESP: a logged write is the exact same byte sequence
the client sent, `*3\\r\\n$3\\r\\nSET\\r\\n...`. That means replay can reuse
resp.parse() rather than inventing a second format and a second parser —
and it means a log truncated by a crash mid-write is handled for free,
because parse() already reports a partial trailing command as `leftover`
instead of raising.

Logical offsets
---------------
Snapshotting needs to say "the snapshot reflects every record up to byte
offset N". But compaction physically deletes the front of the log, so a
byte's position in the file changes over time. So the log carries a
LOGICAL offset space that only ever moves forward: a fixed-size header
records `base_offset`, the logical offset of the first byte after the
header. Logical offset of a byte at physical position p is therefore
`base_offset + (p - LOG_HEADER_LEN)`. Compaction rewrites the file with a
larger base_offset; logical offsets recorded in a snapshot stay valid
across that.

Durability
----------
fsync never runs on the event loop — it goes through run_in_executor, or a
single slow disk flush would freeze every connected client. Three modes:

  always   — a write is acknowledged only after fsync. Group commit means
             all writes staged in the same event-loop tick share ONE fsync.
  everysec — a write is acknowledged once buffered; a background task
             fsyncs about once a second. Up to ~1s of acknowledged writes
             can be lost, which is the standard trade.
  never    — never fsync explicitly; rely on the OS. For benchmarking.
"""
import asyncio
import logging
import os
import time
from pathlib import Path

from cache.resp import ProtocolError, encode, parse

logger = logging.getLogger("cache.wal")

LOG_MAGIC = b"EVICTWAL1 "
LOG_HEADER_LEN = len(LOG_MAGIC) + 20 + 2  # magic + zero-padded offset + CRLF

FSYNC_MODES = ("always", "everysec", "never")
EVERYSEC_INTERVAL = 1.0


def encode_record(command: list[bytes]) -> bytes:
    """A log record is the command re-encoded as a RESP array of bulk strings."""
    return encode(command)


def _encode_header(base_offset: int) -> bytes:
    return LOG_MAGIC + f"{base_offset:020d}".encode() + b"\r\n"


def _decode_header(raw: bytes) -> int:
    if len(raw) != LOG_HEADER_LEN or not raw.startswith(LOG_MAGIC):
        raise ValueError("not a valid WAL header")
    try:
        return int(raw[len(LOG_MAGIC) : -2])
    except ValueError:
        raise ValueError("corrupt WAL header offset") from None


def _parse_recoverable(data: bytes) -> tuple[list[list[bytes]], int]:
    """
    Parse as many records as possible, tolerating a damaged tail.

    A clean crash mid-write leaves a TRUNCATED record, which parse()
    already reports as leftover without raising. Genuine corruption
    (garbled bytes rather than a short write) makes parse() raise instead,
    and since it decodes the whole buffer in one call, one bad record
    would otherwise cost us every good record too. So on ProtocolError we
    binary-search for the longest prefix that still parses, and treat
    everything after it as lost. Returns (commands, bytes_consumed).
    """
    try:
        commands, leftover = parse(data)
        return commands, len(data) - len(leftover)
    except ProtocolError:
        pass

    logger.warning("WAL contains corrupt data; recovering longest intact prefix")
    lo, hi = 0, len(data)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            parse(data[:mid])
            lo = mid
        except ProtocolError:
            hi = mid - 1
    commands, leftover = parse(data[:lo])
    return commands, lo - len(leftover)


def read_log(path: Path, from_offset: int = 0) -> tuple[list[list[bytes]], int]:
    """
    Read records at or after logical `from_offset`.

    Returns (commands, good_physical_len) where good_physical_len is the
    file length containing only intact records — the caller truncates to
    it, discarding a torn tail so the next append starts on a clean
    record boundary.
    """
    with open(path, "rb") as f:
        base = _decode_header(f.read(LOG_HEADER_LEN))
        body = f.read()

    skip = min(max(0, from_offset - base), len(body))
    commands, consumed = _parse_recoverable(body[skip:])
    return commands, LOG_HEADER_LEN + skip + consumed


class WAL:
    def __init__(self, path, fsync_mode: str = "everysec", group_commit: bool = True):
        if fsync_mode not in FSYNC_MODES:
            raise ValueError(f"fsync_mode must be one of {FSYNC_MODES}")
        self.path = Path(path)
        self.fsync_mode = fsync_mode
        self.group_commit = group_commit

        self._file = None
        self._base_offset = 0
        self._written_bytes = 0  # physical bytes after the header

        self._pending: list[bytes] = []
        self._staged_seq = 0   # records handed to stage()
        self._written_seq = 0  # records written(2) to the file
        self._synced_seq = 0   # records known durable
        self._waiters: list[tuple[int, asyncio.Future]] = []

        self._wake = asyncio.Event()
        self._commit_task = None
        self._ticker_task = None
        self._lock = asyncio.Lock()  # serializes flush against compaction
        self._closed = False
        self._last_fsync = 0.0
        self._loop = None

        self.fsync_count = 0  # observable, for the fsync-mode benchmark

    # -- lifecycle -------------------------------------------------------

    def open(self) -> None:
        """Open (creating if needed) for appending. Synchronous, pre-serving."""
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "wb") as f:
                f.write(_encode_header(0))
                f.flush()
                os.fsync(f.fileno())
            self._base_offset = 0
            self._written_bytes = 0
        else:
            with open(self.path, "rb") as f:
                self._base_offset = _decode_header(f.read(LOG_HEADER_LEN))
            self._written_bytes = max(0, self.path.stat().st_size - LOG_HEADER_LEN)
        self._file = open(self.path, "r+b")
        self._file.seek(0, os.SEEK_END)

    def truncate_to(self, physical_len: int) -> None:
        """Drop a torn tail left by a crash. Synchronous, pre-serving."""
        with open(self.path, "r+b") as f:
            f.truncate(physical_len)
            f.flush()
            os.fsync(f.fileno())
        self._written_bytes = max(0, physical_len - LOG_HEADER_LEN)

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._last_fsync = time.monotonic()
        self._commit_task = asyncio.create_task(self._commit_loop())
        if self.fsync_mode == "everysec":
            self._ticker_task = asyncio.create_task(self._ticker())

    async def close(self) -> None:
        self._closed = True
        for task in (self._ticker_task, self._commit_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        # Final flush is unconditional: on a clean shutdown even everysec
        # mode should not drop the last second of writes.
        await self._flush(force_fsync=True)
        if self._file is not None:
            self._file.close()
            self._file = None

    # -- append path -----------------------------------------------------

    @property
    def logical_end(self) -> int:
        """Logical offset just past the last WRITTEN record."""
        return self._base_offset + self._written_bytes

    def stage(self, record: bytes) -> None:
        """
        Queue a record. Synchronous on purpose: staging establishes log
        ORDER, and it must happen in the same uninterrupted step as
        applying the command to the store. If this awaited, two concurrent
        writers to the same key could land in the log in one order and in
        memory in the other, and replay would then disagree with the state
        the clients actually observed.
        """
        self._pending.append(record)
        self._staged_seq += 1
        self._wake.set()

    async def sync(self) -> None:
        """
        Return once everything staged so far is durable.

        In `always` mode this is what the server awaits before replying, so
        an acknowledged write is a persisted write. Concurrent callers
        queue behind the same commit and share one fsync — that sharing is
        the whole point of group commit.
        """
        if self._closed or self.fsync_mode != "always":
            self._wake.set()  # everysec/never: don't wait, but do get it written
            return
        target = self._staged_seq
        if self._synced_seq >= target:
            return
        fut = self._loop.create_future()
        self._waiters.append((target, fut))
        self._wake.set()
        await fut

    # -- background commit ----------------------------------------------

    async def _ticker(self) -> None:
        while not self._closed:
            await asyncio.sleep(EVERYSEC_INTERVAL)
            self._wake.set()

    async def _commit_loop(self) -> None:
        while not self._closed:
            await self._wake.wait()
            self._wake.clear()
            # Yield once so every coroutine already runnable in this tick
            # gets to stage its record before we cut the batch. This is
            # what makes one fsync cover many concurrent writers.
            await asyncio.sleep(0)
            try:
                await self._flush()
            except Exception:
                logger.exception("WAL commit failed")

    def _write_bytes(self, data: bytes) -> None:
        self._file.write(data)
        self._file.flush()

    def _do_fsync(self) -> None:
        os.fsync(self._file.fileno())

    async def _flush(self, force_fsync: bool = False) -> None:
        async with self._lock:
            records, self._pending = self._pending, []

            if records and not self.group_commit and self.fsync_mode == "always":
                # Degenerate mode kept for the benchmark: one fsync per
                # record, so group commit has something to be measured
                # against.
                for i, rec in enumerate(records, 1):
                    await self._loop.run_in_executor(None, self._write_bytes, rec)
                    self._written_bytes += len(rec)
                    self._written_seq += 1
                    await self._loop.run_in_executor(None, self._do_fsync)
                    self.fsync_count += 1
                self._last_fsync = time.monotonic()
                self._synced_seq = self._written_seq
                self._release()
                return

            if records:
                data = b"".join(records)
                # Even the plain write goes to the executor: a large write
                # to a slow disk blocks, and blocking here would stall
                # every other connection.
                await self._loop.run_in_executor(None, self._write_bytes, data)
                self._written_bytes += len(data)
                self._written_seq += len(records)

            due = force_fsync
            if self.fsync_mode == "always" and self._waiters:
                due = True
            elif self.fsync_mode == "everysec" and time.monotonic() - self._last_fsync >= EVERYSEC_INTERVAL:
                due = True

            if due and self._file is not None and self._written_seq > self._synced_seq:
                await self._loop.run_in_executor(None, self._do_fsync)
                self.fsync_count += 1
                self._last_fsync = time.monotonic()
                self._synced_seq = self._written_seq
            elif self.fsync_mode == "never":
                # Nothing is fsynced, but callers must not block forever.
                self._synced_seq = self._written_seq

            self._release()

    def _release(self) -> None:
        still_waiting = []
        for target, fut in self._waiters:
            if target <= self._synced_seq and not fut.done():
                fut.set_result(None)
            elif not fut.done():
                still_waiting.append((target, fut))
        self._waiters = still_waiting

    # -- compaction ------------------------------------------------------

    async def compact(self, discard_before: int) -> None:
        """
        Physically drop every record before logical offset `discard_before`
        (the offset a just-written snapshot corresponds to).

        Rewrites the file as `header(new_base) + tail` and renames it into
        place, so a crash mid-compaction leaves the original log intact —
        the rename is the atomic commit point. Held under the same lock the
        flush path uses, so no append can slip in between reading the tail
        and swapping the file.
        """
        async with self._lock:
            if self._pending:
                records, self._pending = self._pending, []
                data = b"".join(records)
                await self._loop.run_in_executor(None, self._write_bytes, data)
                self._written_bytes += len(data)
                self._written_seq += len(records)

            new_base = max(self._base_offset, discard_before)
            keep_from = LOG_HEADER_LEN + (new_base - self._base_offset)
            if keep_from <= LOG_HEADER_LEN:
                return  # nothing to drop

            def _rewrite() -> int:
                with open(self.path, "rb") as src:
                    src.seek(keep_from)
                    tail = src.read()
                tmp = self.path.with_suffix(self.path.suffix + ".compact")
                with open(tmp, "wb") as dst:
                    dst.write(_encode_header(new_base))
                    dst.write(tail)
                    dst.flush()
                    os.fsync(dst.fileno())
                os.replace(tmp, self.path)
                dir_fd = os.open(str(self.path.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)  # make the rename itself durable
                finally:
                    os.close(dir_fd)
                return len(tail)

            self._file.close()
            tail_len = await self._loop.run_in_executor(None, _rewrite)
            self._file = open(self.path, "r+b")
            self._file.seek(0, os.SEEK_END)
            self._base_offset = new_base
            self._written_bytes = tail_len
            logger.info("compacted WAL: new base offset %d, %d bytes retained", new_base, tail_len)
