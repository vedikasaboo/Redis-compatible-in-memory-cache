"""Tests for the implemented encoder (cache.resp.encode)."""
import pytest

from cache.resp import ErrorReply, SimpleString, encode


def test_simple_string():
    assert encode(SimpleString("PONG")) == b"+PONG\r\n"


def test_error():
    assert encode(ErrorReply("ERR bad")) == b"-ERR bad\r\n"


def test_integer():
    assert encode(1) == b":1\r\n"
    assert encode(0) == b":0\r\n"
    assert encode(-5) == b":-5\r\n"


def test_bool_rejected():
    with pytest.raises(TypeError):
        encode(True)


def test_bulk_string():
    assert encode(b"hello world") == b"$11\r\nhello world\r\n"


def test_empty_bulk_string():
    assert encode(b"") == b"$0\r\n\r\n"


def test_null_bulk_string():
    assert encode(None) == b"$-1\r\n"


def test_array():
    assert encode([b"a", b"b", b"c"]) == b"*3\r\n$1\r\na\r\n$1\r\nb\r\n$1\r\nc\r\n"


def test_array_with_mixed_null():
    assert encode([b"v1", None]) == b"*2\r\n$2\r\nv1\r\n$-1\r\n"


def test_empty_array():
    assert encode([]) == b"*0\r\n"


def test_unencodable_type_raises():
    with pytest.raises(TypeError):
        encode(object())
