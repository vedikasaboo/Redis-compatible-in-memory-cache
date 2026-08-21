"""
TTL tests.

The store takes an injectable clock, so these drive expiry by advancing a
fake clock rather than sleeping — deterministic and instant.
"""
from cache.commands import dispatch
from cache.lru import NO_EXPIRY, NOT_FOUND, LRUCache
from cache.resp import ErrorReply, SimpleString


class FakeClock:
    def __init__(self, t=1_000_000):
        self.t = t

    def __call__(self) -> int:
        return self.t

    def advance(self, ms: int) -> None:
        self.t += ms


def make_store(capacity=100):
    clock = FakeClock()
    return LRUCache(capacity, clock=clock), clock


# -- store-level ---------------------------------------------------------

def test_key_without_expiry_never_expires():
    store, clock = make_store()
    store.set(b"k", b"v")
    clock.advance(10_000_000)
    assert store.get(b"k") == b"v"
    assert store.ttl_ms(b"k") == NO_EXPIRY


def test_key_expires_after_deadline():
    store, clock = make_store()
    store.set(b"k", b"v", expires_at=clock.t + 1000)
    clock.advance(999)
    assert store.get(b"k") == b"v"
    clock.advance(1)  # now exactly at the deadline
    assert store.get(b"k") is None


def test_expired_key_is_actually_removed_not_just_hidden():
    store, clock = make_store()
    store.set(b"k", b"v", expires_at=clock.t + 10)
    clock.advance(20)
    assert store.get(b"k") is None
    assert len(store) == 0  # the read collected it


def test_ttl_ms_reports_remaining_and_sentinels():
    store, clock = make_store()
    assert store.ttl_ms(b"missing") == NOT_FOUND
    store.set(b"plain", b"v")
    assert store.ttl_ms(b"plain") == NO_EXPIRY
    store.set(b"timed", b"v", expires_at=clock.t + 5000)
    assert store.ttl_ms(b"timed") == 5000
    clock.advance(2000)
    assert store.ttl_ms(b"timed") == 3000
    clock.advance(3000)
    assert store.ttl_ms(b"timed") == NOT_FOUND


def test_setting_past_deadline_deletes_instead_of_inserting():
    store, clock = make_store()
    store.set(b"k", b"old")
    store.set(b"k", b"new", expires_at=clock.t - 1)
    assert store.get(b"k") is None
    assert len(store) == 0


def test_overwriting_clears_previous_expiry():
    store, clock = make_store()
    store.set(b"k", b"v", expires_at=clock.t + 100)
    store.set(b"k", b"v2")  # no expiry this time
    clock.advance(1000)
    assert store.get(b"k") == b"v2"
    assert store.ttl_ms(b"k") == NO_EXPIRY


def test_contains_respects_expiry():
    store, clock = make_store()
    store.set(b"k", b"v", expires_at=clock.t + 10)
    assert b"k" in store
    clock.advance(20)
    assert b"k" not in store


def test_expired_key_does_not_consume_capacity_after_collection():
    store, clock = make_store(capacity=2)
    store.set(b"a", b"v", expires_at=clock.t + 10)
    store.set(b"b", b"v")
    clock.advance(20)
    store.get(b"a")  # collects a
    store.set(b"c", b"v")
    assert store.get(b"b") == b"v"  # b was not evicted to make room
    assert store.get(b"c") == b"v"


# -- active expiry -------------------------------------------------------

def test_expire_sample_reclaims_untouched_keys():
    """Lazy expiry alone leaks keys nobody reads again; the sweep fixes that."""
    store, clock = make_store(capacity=1000)
    for i in range(50):
        store.set(f"k{i}".encode(), b"v", expires_at=clock.t + 100)
    clock.advance(200)
    assert len(store) == 50  # expired but not yet collected

    removed = 0
    for _ in range(10):
        removed += store.expire_sample(20)
    assert removed == 50
    assert len(store) == 0


def test_expire_sample_leaves_live_keys_alone():
    store, clock = make_store()
    store.set(b"live", b"v", expires_at=clock.t + 10_000)
    store.set(b"dead", b"v", expires_at=clock.t + 10)
    store.set(b"plain", b"v")
    clock.advance(100)
    for _ in range(5):
        store.expire_sample(20)
    assert store.get(b"live") == b"v"
    assert store.get(b"plain") == b"v"
    assert store.get(b"dead") is None


def test_expire_sample_is_bounded():
    """The sweep must never scan the whole keyspace in one tick."""
    store, clock = make_store(capacity=10_000)
    for i in range(1000):
        store.set(f"k{i}".encode(), b"v", expires_at=clock.t + 10)
    clock.advance(100)
    assert store.expire_sample(20) == 20  # not 1000


def test_expire_sample_on_empty_store_is_noop():
    store, _ = make_store()
    assert store.expire_sample(20) == 0
    store.set(b"k", b"v")  # no TTL -> nothing for the sweep to consider
    assert store.expire_sample(20) == 0


# -- command-level -------------------------------------------------------

def test_set_with_ex_and_ttl_command():
    store, clock = make_store()
    assert dispatch([b"SET", b"k", b"v", b"EX", b"10"], store) == SimpleString("OK")
    assert dispatch([b"TTL", b"k"], store) == 10
    clock.advance(4000)
    assert dispatch([b"TTL", b"k"], store) == 6
    clock.advance(6000)
    assert dispatch([b"TTL", b"k"], store) == NOT_FOUND
    assert dispatch([b"GET", b"k"], store) is None


def test_set_with_px():
    store, clock = make_store()
    dispatch([b"SET", b"k", b"v", b"PX", b"1500"], store)
    assert dispatch([b"PTTL", b"k"], store) == 1500
    assert dispatch([b"TTL", b"k"], store) == 2  # rounded up, like Redis


def test_set_with_pxat_absolute():
    store, clock = make_store()
    dispatch([b"SET", b"k", b"v", b"PXAT", str(clock.t + 2000).encode()], store)
    assert dispatch([b"PTTL", b"k"], store) == 2000


def test_set_with_exat_absolute():
    store, clock = make_store()
    deadline_s = (clock.t + 5000) // 1000
    dispatch([b"SET", b"k", b"v", b"EXAT", str(deadline_s).encode()], store)
    assert dispatch([b"GET", b"k"], store) == b"v"
    clock.advance(6000)
    assert dispatch([b"GET", b"k"], store) is None


def test_ttl_sentinels_via_commands():
    store, _ = make_store()
    assert dispatch([b"TTL", b"missing"], store) == NOT_FOUND
    dispatch([b"SET", b"k", b"v"], store)
    assert dispatch([b"TTL", b"k"], store) == NO_EXPIRY


def test_invalid_expiry_options_are_errors():
    store, _ = make_store()
    assert isinstance(dispatch([b"SET", b"k", b"v", b"EX", b"0"], store), ErrorReply)
    assert isinstance(dispatch([b"SET", b"k", b"v", b"EX", b"-5"], store), ErrorReply)
    assert isinstance(dispatch([b"SET", b"k", b"v", b"EX", b"abc"], store), ErrorReply)
    assert isinstance(dispatch([b"SET", b"k", b"v", b"BOGUS", b"10"], store), ErrorReply)
    assert isinstance(dispatch([b"SET", b"k", b"v", b"EX"], store), ErrorReply)
    # a rejected SET must not have stored anything
    assert dispatch([b"GET", b"k"], store) is None


def test_expiry_option_is_case_insensitive():
    store, _ = make_store()
    dispatch([b"SET", b"k", b"v", b"ex", b"10"], store)
    assert dispatch([b"TTL", b"k"], store) == 10


def test_exists_respects_expiry():
    store, clock = make_store()
    dispatch([b"SET", b"k", b"v", b"PX", b"100"], store)
    assert dispatch([b"EXISTS", b"k"], store) == 1
    clock.advance(200)
    assert dispatch([b"EXISTS", b"k"], store) == 0
