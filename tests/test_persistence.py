"""
Tests for the WAL, snapshots, and recovery.

Async pieces are driven with asyncio.run() inside sync test functions so
the suite needs no pytest-asyncio dependency.
"""
import asyncio
import time

from cache.commands import dispatch
from cache.lru import LRUCache
from cache.persistence import Persistence
from cache.resp import encode
from cache.snapshot import load_snapshot, write_snapshot
from cache.wal import LOG_HEADER_LEN, WAL, read_log


def _run(coro):
    return asyncio.run(coro)


# -- WAL basics ----------------------------------------------------------

def test_wal_roundtrip(tmp_path):
    path = tmp_path / "wal.log"

    async def go():
        wal = WAL(path, fsync_mode="always")
        wal.open()
        await wal.start()
        wal.stage(encode([b"SET", b"a", b"1"]))
        wal.stage(encode([b"DEL", b"a"]))
        await wal.sync()
        await wal.close()

    _run(go())
    commands, good_len = read_log(path)
    assert commands == [[b"SET", b"a", b"1"], [b"DEL", b"a"]]
    assert good_len == path.stat().st_size


def test_wal_recovers_from_torn_tail(tmp_path):
    """A crash mid-write leaves a truncated record; it must be dropped."""
    path = tmp_path / "wal.log"

    async def go():
        wal = WAL(path, fsync_mode="always")
        wal.open()
        await wal.start()
        wal.stage(encode([b"SET", b"a", b"1"]))
        await wal.sync()
        await wal.close()

    _run(go())
    intact_len = path.stat().st_size
    with open(path, "ab") as f:  # simulate a half-written second record
        f.write(b"*3\r\n$3\r\nSET\r\n$1\r\nb")

    commands, good_len = read_log(path)
    assert commands == [[b"SET", b"a", b"1"]]
    assert good_len == intact_len  # torn bytes excluded


def test_wal_recovers_longest_prefix_from_corruption(tmp_path):
    """Genuine garbage (not just truncation) must not cost the good records."""
    path = tmp_path / "wal.log"

    async def go():
        wal = WAL(path, fsync_mode="always")
        wal.open()
        await wal.start()
        for i in range(5):
            wal.stage(encode([b"SET", f"k{i}".encode(), b"v"]))
        await wal.sync()
        await wal.close()

    _run(go())
    with open(path, "ab") as f:
        f.write(b"\x00\x01garbage-not-resp\xff")

    commands, _ = read_log(path)
    assert commands == [[b"SET", f"k{i}".encode(), b"v"] for i in range(5)]


def test_wal_read_from_offset_skips_prefix(tmp_path):
    path = tmp_path / "wal.log"

    async def go():
        wal = WAL(path, fsync_mode="always")
        wal.open()
        await wal.start()
        wal.stage(encode([b"SET", b"a", b"1"]))
        await wal.sync()
        mid = wal.logical_end
        wal.stage(encode([b"SET", b"b", b"2"]))
        await wal.sync()
        await wal.close()
        return mid

    mid = _run(go())
    commands, _ = read_log(path, from_offset=mid)
    assert commands == [[b"SET", b"b", b"2"]]


def test_wal_compaction_preserves_tail_and_offsets(tmp_path):
    path = tmp_path / "wal.log"

    async def go():
        wal = WAL(path, fsync_mode="always")
        wal.open()
        await wal.start()
        wal.stage(encode([b"SET", b"old", b"1"]))
        await wal.sync()
        cut = wal.logical_end
        wal.stage(encode([b"SET", b"new", b"2"]))
        await wal.sync()
        await wal.compact(cut)
        end_after = wal.logical_end
        await wal.close()
        return cut, end_after

    cut, end_after = _run(go())
    # The retained tail is only the post-cut record...
    commands, _ = read_log(path)
    assert commands == [[b"SET", b"new", b"2"]]
    # ...and logical offsets did not rewind despite bytes being deleted.
    assert end_after > cut
    assert path.stat().st_size < LOG_HEADER_LEN + 60


def test_wal_group_commit_shares_one_fsync(tmp_path):
    """Records staged in the same tick must cost a single fsync."""
    path = tmp_path / "wal.log"

    async def go():
        wal = WAL(path, fsync_mode="always", group_commit=True)
        wal.open()
        await wal.start()
        for i in range(20):
            wal.stage(encode([b"SET", f"k{i}".encode(), b"v"]))
        await asyncio.gather(*(wal.sync() for _ in range(20)))
        count = wal.fsync_count
        await wal.close()
        return count

    assert _run(go()) == 1


def test_wal_without_group_commit_fsyncs_per_record(tmp_path):
    path = tmp_path / "wal.log"

    async def go():
        wal = WAL(path, fsync_mode="always", group_commit=False)
        wal.open()
        await wal.start()
        for i in range(10):
            wal.stage(encode([b"SET", f"k{i}".encode(), b"v"]))
        await wal.sync()
        count = wal.fsync_count
        await wal.close()
        return count

    assert _run(go()) == 10


# -- snapshots -----------------------------------------------------------

def test_snapshot_roundtrip(tmp_path):
    path = tmp_path / "snapshot.rdb"
    items = [(f"k{i}".encode(), f"v{i}".encode()) for i in range(50)]
    _run(write_snapshot(path, items, log_offset=4321, chunk_size=7))
    loaded, offset = load_snapshot(path)
    assert loaded == [(k, v, None) for k, v in items]
    assert offset == 4321


def test_snapshot_roundtrip_with_expiry(tmp_path):
    path = tmp_path / "snapshot.rdb"
    items = [(b"forever", b"v", None), (b"doomed", b"v", 1_700_000_000_000)]
    _run(write_snapshot(path, items, log_offset=0))
    loaded, _ = load_snapshot(path)
    assert loaded == items


def test_snapshot_preserves_binary_values(tmp_path):
    path = tmp_path / "snapshot.rdb"
    items = [(b"key", b"has\r\nCRLF\x00inside")]
    _run(write_snapshot(path, items, log_offset=0))
    loaded, _ = load_snapshot(path)
    assert loaded == [(b"key", b"has\r\nCRLF\x00inside", None)]


def test_missing_snapshot_is_not_an_error(tmp_path):
    assert load_snapshot(tmp_path / "nope.rdb") == ([], 0)


def test_corrupt_snapshot_is_ignored_not_fatal(tmp_path):
    path = tmp_path / "snapshot.rdb"
    path.write_bytes(b"totally not a snapshot")
    assert load_snapshot(path) == ([], 0)


# -- recovery ------------------------------------------------------------

def test_recover_from_wal_only(tmp_path):
    async def write():
        store = LRUCache(100)
        p = Persistence(store, tmp_path, fsync_mode="always", snapshot_interval=0)
        p.recover()
        await p.start()
        for cmd in ([b"SET", b"a", b"1"], [b"SET", b"b", b"2"], [b"DEL", b"a"]):
            p.stage_write(cmd)
            await p.sync()
        await p.close()

    _run(write())

    store2 = LRUCache(100)
    p2 = Persistence(store2, tmp_path, fsync_mode="always", snapshot_interval=0)
    replayed = p2.recover()
    _run(p2.close())
    assert replayed == 3
    assert store2.get(b"a") is None  # the DEL was replayed too
    assert store2.get(b"b") == b"2"


def test_recover_from_snapshot_plus_log_tail(tmp_path):
    async def write():
        store = LRUCache(100)
        p = Persistence(store, tmp_path, fsync_mode="always", snapshot_interval=0)
        p.recover()
        await p.start()
        for i in range(10):
            cmd = [b"SET", f"k{i}".encode(), f"v{i}".encode()]
            p.stage_write(cmd)
            store.set(cmd[1], cmd[2])
            await p.sync()
        await p.snapshot_now()
        # writes after the snapshot live only in the compacted log tail
        for i in range(10, 15):
            cmd = [b"SET", f"k{i}".encode(), f"v{i}".encode()]
            p.stage_write(cmd)
            store.set(cmd[1], cmd[2])
            await p.sync()
        await p.close()

    _run(write())
    assert (tmp_path / "snapshot.rdb").exists()

    store2 = LRUCache(100)
    p2 = Persistence(store2, tmp_path, fsync_mode="always", snapshot_interval=0)
    replayed = p2.recover()
    _run(p2.close())
    assert replayed == 5  # only the tail, not the 10 already in the snapshot
    for i in range(15):
        assert store2.get(f"k{i}".encode()) == f"v{i}".encode()


def test_snapshot_does_not_resurrect_evicted_keys(tmp_path):
    """
    The reason compaction is snapshot-based rather than a log rewrite: the
    log still contains SETs for keys the LRU has since evicted, and they
    must not come back.
    """
    async def write():
        store = LRUCache(5)  # tiny, so most keys get evicted
        p = Persistence(store, tmp_path, fsync_mode="always", snapshot_interval=0)
        p.recover()
        await p.start()
        for i in range(100):
            cmd = [b"SET", f"k{i}".encode(), f"v{i}".encode()]
            p.stage_write(cmd)
            store.set(cmd[1], cmd[2])
            await p.sync()
        await p.snapshot_now()
        await p.close()

    _run(write())

    store2 = LRUCache(5)
    p2 = Persistence(store2, tmp_path, fsync_mode="always", snapshot_interval=0)
    p2.recover()
    _run(p2.close())
    assert len(store2) == 5
    assert store2.get(b"k0") is None
    assert store2.get(b"k99") == b"v99"


def test_recovery_preserves_lru_ordering(tmp_path):
    """A recovered cache should evict in the same order the crashed one would."""
    async def write():
        store = LRUCache(3)
        p = Persistence(store, tmp_path, fsync_mode="always", snapshot_interval=0)
        p.recover()
        await p.start()
        for key in (b"a", b"b", b"c"):
            p.stage_write([b"SET", key, b"v"])
            store.set(key, b"v")
            await p.sync()
        store.get(b"a")  # a becomes most-recently-used, b is now the LRU victim
        await p.snapshot_now()
        await p.close()

    _run(write())
    store2 = LRUCache(3)
    p2 = Persistence(store2, tmp_path, fsync_mode="always", snapshot_interval=0)
    p2.recover()
    _run(p2.close())

    store2.set(b"d", b"v")  # over capacity -> should evict b, matching pre-crash order
    assert store2.get(b"b") is None
    assert store2.get(b"a") == b"v"


def test_malformed_writes_are_not_logged(tmp_path):
    """A command that errors changes nothing, so it must not reach the log."""
    from cache.commands import normalize_write

    now = 1_000_000
    assert normalize_write([b"SET", b"k", b"v"], now) == [b"SET", b"k", b"v"]
    assert normalize_write([b"SET", b"k"], now) is None  # wrong arity -> error reply
    assert normalize_write([b"DEL", b"k"], now) == [b"DEL", b"k"]
    assert normalize_write([b"GET", b"k"], now) is None
    assert normalize_write([b"PING"], now) is None
    assert normalize_write([b"SET", b"k", b"v", b"EX", b"0"], now) is None  # rejected -> not logged
    assert normalize_write([b"SET", b"k", b"v", b"EX", b"nope"], now) is None


def test_relative_expiry_is_logged_as_absolute(tmp_path):
    """
    The replay-safety property: a logged `SET ... EX 10` replayed an hour
    later would grant a fresh 10 seconds. Logging an absolute deadline
    instead keeps replay faithful no matter when it happens.
    """
    from cache.commands import normalize_write

    now = 1_000_000
    assert normalize_write([b"SET", b"k", b"v", b"EX", b"10"], now) == \
        [b"SET", b"k", b"v", b"PXAT", b"1010000"]
    assert normalize_write([b"SET", b"k", b"v", b"PX", b"500"], now) == \
        [b"SET", b"k", b"v", b"PXAT", b"1000500"]
    # Absolute forms pass through unchanged in meaning.
    assert normalize_write([b"SET", b"k", b"v", b"PXAT", b"1234"], now) == \
        [b"SET", b"k", b"v", b"PXAT", b"1234"]


def test_ttl_survives_restart_and_elapses_while_down(tmp_path):
    """A key whose deadline passes during downtime must not come back."""
    async def write():
        store = LRUCache(100)
        p = Persistence(store, tmp_path, fsync_mode="always", snapshot_interval=0)
        p.recover()
        await p.start()
        now = store.now_ms()
        for cmd in (
            [b"SET", b"lives", b"v", b"PXAT", str(now + 3_600_000).encode()],
            [b"SET", b"dies", b"v", b"PXAT", str(now + 50).encode()],
        ):
            p.stage_write(cmd)
            dispatch(cmd, store)
            await p.sync()
        await p.close()

    _run(write())
    time.sleep(0.1)  # let "dies" elapse while the server is down

    store2 = LRUCache(100)
    p2 = Persistence(store2, tmp_path, fsync_mode="always", snapshot_interval=0)
    p2.recover()
    _run(p2.close())
    assert store2.get(b"lives") == b"v"
    assert store2.get(b"dies") is None
