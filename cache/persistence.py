"""
Persistence coordinator: owns the WAL, the snapshot file, and the
background snapshot task, and defines the exact recovery order.

Module boundary: the server knows only stage_write()/sync()/recover().
Everything about offsets, compaction, and fsync scheduling stays here.

Recovery order (this order is the whole correctness argument)
-------------------------------------------------------------
1. Load the snapshot. It reflects all writes up to logical offset S.
2. Replay WAL records at or after S. Records before S are already
   represented in the snapshot, so replaying them would be redundant —
   and worse, could resurrect keys the LRU evicted before S.
3. Truncate the WAL to its last intact record, discarding a torn tail
   from the crash, so the next append starts on a clean boundary.
4. Only then start accepting connections.

The offset is what makes step 2 safe across a crash at any point. If we
die after writing the snapshot but before compacting the log, the log
still holds the pre-S prefix — and step 2 skips it by offset rather than
assuming compaction ran.
"""
import asyncio
import logging
from pathlib import Path

from cache.commands import dispatch
from cache.lru import LRUCache
from cache.snapshot import DEFAULT_CHUNK, load_snapshot, write_snapshot
from cache.wal import WAL, encode_record, read_log

logger = logging.getLogger("cache.persistence")

DEFAULT_SNAPSHOT_INTERVAL = 60.0


class Persistence:
    def __init__(
        self,
        store: LRUCache,
        data_dir,
        fsync_mode: str = "everysec",
        group_commit: bool = True,
        snapshot_interval: float = DEFAULT_SNAPSHOT_INTERVAL,
        snapshot_chunk: int = DEFAULT_CHUNK,
    ):
        self.store = store
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "wal.log"
        self.snapshot_path = self.dir / "snapshot.rdb"
        self.snapshot_interval = snapshot_interval
        self.snapshot_chunk = snapshot_chunk
        self.wal = WAL(self.log_path, fsync_mode=fsync_mode, group_commit=group_commit)
        self._snapshot_task = None
        self._closed = False

    # -- startup ---------------------------------------------------------

    def recover(self) -> int:
        """
        Rebuild the store from snapshot + WAL. Synchronous and called
        before the listening socket opens, so no client can observe a
        half-recovered cache. Returns the number of records replayed.
        """
        items, snap_offset = load_snapshot(self.snapshot_path)
        for key, value, expires_at in items:
            # set() drops anything whose deadline already passed, so keys
            # that expired while the server was down stay dead.
            self.store.set(key, value, expires_at)
        if items:
            logger.info("loaded snapshot: %d entries at offset %d", len(items), snap_offset)

        replayed = 0
        if self.log_path.exists():
            try:
                commands, good_len = read_log(self.log_path, from_offset=snap_offset)
            except ValueError as exc:
                logger.error("unreadable WAL header (%s); starting from snapshot only", exc)
                commands, good_len = [], None
            for command in commands:
                if command:
                    dispatch(command, self.store)
                    replayed += 1
            if good_len is not None and good_len < self.log_path.stat().st_size:
                logger.warning(
                    "discarding %d bytes of torn WAL tail",
                    self.log_path.stat().st_size - good_len,
                )
                self.wal.truncate_to(good_len)

        self.wal.open()
        if replayed:
            logger.info("replayed %d WAL records", replayed)
        return replayed

    async def start(self) -> None:
        await self.wal.start()
        if self.snapshot_interval > 0:
            self._snapshot_task = asyncio.create_task(self._snapshot_loop())

    async def close(self) -> None:
        self._closed = True
        if self._snapshot_task is not None:
            self._snapshot_task.cancel()
            try:
                await self._snapshot_task
            except asyncio.CancelledError:
                pass
        await self.wal.close()

    # -- write path ------------------------------------------------------

    def stage_write(self, command: list[bytes]) -> None:
        """Log the command before the server applies it. Never awaits."""
        self.wal.stage(encode_record(command))

    async def sync(self) -> None:
        """Await durability per the configured fsync mode."""
        await self.wal.sync()

    # -- compaction ------------------------------------------------------

    async def _snapshot_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(self.snapshot_interval)
            try:
                await self.snapshot_now()
            except Exception:
                logger.exception("snapshot failed")

    async def snapshot_now(self) -> int:
        """
        Take a snapshot and compact the log behind it.

        The pairing of `offset` with `items` is the delicate part: flush
        first so every staged record is really in the file, then read the
        offset and the store's contents with NO await between them. That
        gap is the only place another coroutine could mutate the store,
        and skipping it is what guarantees the snapshot means exactly
        "state as of this offset".
        """
        await self.wal._flush(force_fsync=True)
        offset = self.wal.logical_end
        items = self.store.items()  # atomic: synchronous, no await inside

        await write_snapshot(self.snapshot_path, items, offset, chunk_size=self.snapshot_chunk)
        await self.wal.compact(offset)
        return len(items)
