"""
Contract tests for cache.resp.parse — the decoder YOU implement.

These will fail with NotImplementedError until you fill in parse(). That's
expected. Once implemented, every test here must pass.
"""
import pytest

from cache.resp import MAX_BULK_LEN, ProtocolError, parse


def test_single_command_no_args():
    assert parse(b"*1\r\n$4\r\nPING\r\n") == ([[b"PING"]], b"")


def test_single_command_with_args():
    commands, leftover = parse(b"*3\r\n$3\r\nSET\r\n$3\r\nfoo\r\n$3\r\nbar\r\n")
    assert commands == [[b"SET", b"foo", b"bar"]]
    assert leftover == b""


def test_partial_buffer_returns_no_commands_and_preserves_bytes():
    partial = b"*1\r\n$4\r\nPI"
    assert parse(partial) == ([], partial)


def test_partial_buffer_missing_trailing_crlf():
    partial = b"*1\r\n$4\r\nPING"  # payload complete, trailing \r\n not yet arrived
    commands, leftover = parse(partial)
    assert commands == []
    assert leftover == partial


def test_empty_buffer():
    assert parse(b"") == ([], b"")


def test_pipelined_commands_in_one_call():
    buf = b"*1\r\n$4\r\nPING\r\n" * 2
    commands, leftover = parse(buf)
    assert commands == [[b"PING"], [b"PING"]]
    assert leftover == b""


def test_pipelined_commands_with_trailing_partial():
    complete = b"*1\r\n$4\r\nPING\r\n"
    partial = b"*1\r\n$3\r\nGE"
    commands, leftover = parse(complete + partial)
    assert commands == [[b"PING"]]
    assert leftover == partial


def test_incremental_feed_across_multiple_calls():
    """Simulates bytes trickling in over several reader.read() calls."""
    full = b"*2\r\n$3\r\nGET\r\n$3\r\nfoo\r\n"
    buffer = b""
    collected = []
    for i in range(0, len(full), 3):  # feed 3 bytes at a time
        buffer += full[i : i + 3]
        commands, buffer = parse(buffer)
        collected.extend(commands)
    assert collected == [[b"GET", b"foo"]]
    assert buffer == b""


def test_binary_safe_bulk_string():
    """Bulk string payloads can contain arbitrary bytes, including \\r\\n."""
    payload = b"a\r\nb"
    buf = b"*2\r\n$3\r\nSET\r\n$4\r\n" + payload + b"\r\n"
    commands, leftover = parse(buf)
    assert commands == [[b"SET", payload]]
    assert leftover == b""


def test_oversized_bulk_string_raises_before_buffering():
    huge_len = MAX_BULK_LEN + 1
    # Deliberately do NOT include the huge_len bytes of payload — a correct
    # implementation must reject based on the length header alone.
    buf = f"*1\r\n${huge_len}\r\n".encode()
    with pytest.raises(ProtocolError):
        parse(buf)


def test_malformed_leading_byte_raises():
    with pytest.raises(ProtocolError):
        parse(b"+PONG\r\n")  # a reply type, not a valid client request


def test_malformed_length_header_raises():
    with pytest.raises(ProtocolError):
        parse(b"*1\r\n$notanumber\r\nfoo\r\n")
