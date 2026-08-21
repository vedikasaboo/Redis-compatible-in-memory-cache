"""
Contract tests for cache.lru.LRUCache — YOU implement this.

These will fail with NotImplementedError until you fill it in. That's
expected. Once implemented, every test here must pass.
"""
from cache.lru import LRUCache


def test_set_and_get():
    c = LRUCache(capacity=2)
    c.set(b"a", b"1")
    assert c.get(b"a") == b"1"


def test_get_missing_returns_none():
    c = LRUCache(capacity=2)
    assert c.get(b"nope") is None


def test_len_tracks_size():
    c = LRUCache(capacity=3)
    assert len(c) == 0
    c.set(b"a", b"1")
    c.set(b"b", b"2")
    assert len(c) == 2


def test_update_existing_key_does_not_grow_size():
    c = LRUCache(capacity=2)
    c.set(b"a", b"1")
    c.set(b"a", b"2")
    assert len(c) == 1
    assert c.get(b"a") == b"2"


def test_delete_present_key():
    c = LRUCache(capacity=2)
    c.set(b"a", b"1")
    assert c.delete(b"a") is True
    assert c.get(b"a") is None
    assert len(c) == 0


def test_delete_absent_key():
    c = LRUCache(capacity=2)
    assert c.delete(b"nope") is False


def test_contains():
    c = LRUCache(capacity=2)
    c.set(b"a", b"1")
    assert b"a" in c
    assert b"z" not in c


def test_eviction_when_over_capacity():
    c = LRUCache(capacity=2)
    c.set(b"a", b"1")
    c.set(b"b", b"2")
    c.set(b"c", b"3")  # over capacity: a is least-recently-used, must be evicted
    assert len(c) == 2
    assert c.get(b"a") is None
    assert c.get(b"b") == b"2"
    assert c.get(b"c") == b"3"


def test_get_refreshes_recency():
    c = LRUCache(capacity=2)
    c.set(b"a", b"1")
    c.set(b"b", b"2")
    c.get(b"a")        # touch a, so b becomes least-recently-used
    c.set(b"c", b"3")  # must evict b, not a
    assert c.get(b"a") == b"1"
    assert c.get(b"b") is None
    assert c.get(b"c") == b"3"


def test_set_on_existing_key_refreshes_recency():
    c = LRUCache(capacity=2)
    c.set(b"a", b"1")
    c.set(b"b", b"2")
    c.set(b"a", b"1-updated")  # touch a via set, b becomes least-recently-used
    c.set(b"c", b"3")
    assert c.get(b"a") == b"1-updated"
    assert c.get(b"b") is None


def test_repeated_eviction_under_sustained_load():
    c = LRUCache(capacity=3)
    for i in range(100):
        c.set(str(i).encode(), str(i).encode())
    assert len(c) == 3
    # only the last 3 keys inserted should survive
    for i in range(97, 100):
        assert c.get(str(i).encode()) == str(i).encode()
    assert c.get(b"0") is None
