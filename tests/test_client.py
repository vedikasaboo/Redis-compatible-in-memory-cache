"""Tests for the client library (used for routing in Phase 4)."""
import pytest

from cache.client import CacheError, Client


@pytest.fixture
def client(server):
    host, port = server
    with Client(host=host, port=port) as c:
        yield c


def test_ping(client):
    assert client.ping() == "PONG"


def test_set_get_delete(client):
    assert client.set("k", "v") == "OK"
    assert client.get("k") == b"v"
    assert client.delete("k") == 1
    assert client.get("k") is None


def test_exists(client):
    client.set("a", "1")
    assert client.exists("a") == 1
    assert client.exists("nope") == 0


def test_ttl_helpers(client):
    client.set("k", "v", ex=100)
    assert 95 <= client.ttl("k") <= 100
    assert 90_000 <= client.pttl("k") <= 100_000
    client.set("plain", "v")
    assert client.ttl("plain") == -1
    assert client.ttl("missing") == -2


def test_error_reply_raises(client):
    with pytest.raises(CacheError):
        client.command("NOSUCHCOMMAND")


def test_binary_safe_values(client):
    payload = b"has\r\nCRLF\x00and\xffbytes"
    client.set("bin", payload)
    assert client.get("bin") == payload


def test_pipeline_returns_all_replies_in_order(client):
    replies = client.pipeline([
        ("SET", "a", "1"),
        ("SET", "b", "2"),
        ("GET", "a"),
        ("GET", "b"),
        ("GET", "missing"),
    ])
    assert replies == ["OK", "OK", b"1", b"2", None]


def test_pipeline_keeps_going_past_an_error(client):
    replies = client.pipeline([("SET", "a", "1"), ("BOGUS",), ("GET", "a")])
    assert replies[0] == "OK"
    assert isinstance(replies[1], CacheError)
    assert replies[2] == b"1"


def test_client_survives_reconnect(server):
    host, port = server
    c = Client(host=host, port=port).connect()
    c.set("k", "v")
    c.close()
    c2 = Client(host=host, port=port).connect()
    assert c2.get("k") == b"v"
    c2.close()
