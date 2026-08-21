"""
RESP (REdis Serialization Protocol) encoding and decoding.

Two halves live here on purpose:
  - encode(): turns a Python value into RESP reply bytes. IMPLEMENTED.
  - parse():  turns raw bytes off the wire into commands. YOU IMPLEMENT THIS.

Splitting them this way mirrors the real asymmetry in the protocol: a
client's *request* is always one shape (an array of bulk strings), while a
server's *reply* can be any of the five RESP types. Writing the encoder
first is what lets you see that shape clearly before tackling the harder,
stateful problem of decoding a byte stream that arrives in arbitrary chunks.
"""
from dataclasses import dataclass

MAX_BULK_LEN = 8 * 1024 * 1024  # 8MB — reject a declared bulk length above this
                                 # as soon as the header is parsed, before
                                 # buffering that many bytes.


class ProtocolError(Exception):
    """Raised when a byte stream does not conform to RESP framing rules."""


@dataclass(frozen=True)
class SimpleString:
    """Wraps a str that must be encoded as a RESP Simple String (+), e.g. PONG/OK."""
    value: str


@dataclass(frozen=True)
class ErrorReply:
    """Wraps a str that must be encoded as a RESP Error (-)."""
    value: str


# --------------------------------------------------------------------------
# Encoder — IMPLEMENTED
# --------------------------------------------------------------------------

def encode(value) -> bytes:
    """
    Encode a Python value as RESP reply bytes.

        SimpleString(s) -> +s\\r\\n
        ErrorReply(s)    -> -s\\r\\n
        int              -> :n\\r\\n
        bytes            -> $<len>\\r\\n<bytes>\\r\\n
        None             -> $-1\\r\\n                     (null bulk string)
        list             -> *<len>\\r\\n<encode(item)>...  (each item encoded recursively)

    A plain Python str is intentionally NOT accepted — callers must be
    explicit about whether a string is a SimpleString or should be sent as
    a bulk string (bytes), since real Redis replies use both and they are
    not interchangeable on the wire.
    """
    if isinstance(value, SimpleString):
        return f"+{value.value}\r\n".encode()
    if isinstance(value, ErrorReply):
        return f"-{value.value}\r\n".encode()
    if isinstance(value, bool):
        # bool is an int subclass in Python; guard so True/False never
        # silently become RESP integers :1 / :0 by accident.
        raise TypeError("encode() does not accept bool; encode an int explicitly")
    if isinstance(value, int):
        return f":{value}\r\n".encode()
    if value is None:
        return b"$-1\r\n"
    if isinstance(value, bytes):
        return f"${len(value)}\r\n".encode() + value + b"\r\n"
    if isinstance(value, list):
        parts = [f"*{len(value)}\r\n".encode()]
        parts.extend(encode(item) for item in value)
        return b"".join(parts)
    raise TypeError(f"cannot encode {type(value)!r} as a RESP value")


# --------------------------------------------------------------------------
# Decoder — YOU IMPLEMENT THIS
# --------------------------------------------------------------------------

def parse(buffer: bytes) -> tuple[list[list[bytes]], bytes]:
    """
    Decode as many COMPLETE client commands as `buffer` contains.

    Every command a real Redis client sends is a RESP Array of Bulk Strings.
    For example `SET foo bar` on the wire is exactly:

        *3\\r\\n$3\\r\\nSET\\r\\n$3\\r\\nfoo\\r\\n$3\\r\\nbar\\r\\n

    This function only needs to decode THAT shape (array of bulk strings)
    as a top-level request. Decoding simple strings / errors / integers /
    nested arrays as a top-level value is the ENCODER's job (see encode()
    above), not this function's — those only ever appear in replies, never
    in what a client sends.

    Contract
    --------
    Input:
        `buffer` — the raw bytes accumulated so far for one client
        connection. It may contain zero, one, or many complete commands,
        and may end mid-command (the rest hasn't arrived over the socket
        yet).

    Output: `(commands, leftover)`
        `commands` — a list of fully-parsed commands, each itself a list
            of bytes (one entry per RESP bulk string argument, in order;
            the command name is commands[i][0], e.g. b"SET"). Preserve the
            order commands appear in the buffer.
        `leftover` — the bytes NOT yet consumed:
            * b"" if `buffer` ended exactly on a command boundary.
            * the tail bytes of a not-yet-complete command if `buffer` was
              cut off mid-command (missing part of a length header, or the
              payload/its trailing \\r\\n hasn't fully arrived).

    Must handle PARTIAL reads:
        If `buffer` ends mid-command, do NOT raise and do NOT block. Return
        whatever complete commands appeared before the cut-off point
        (possibly an empty list) plus the incomplete tail as `leftover`, so
        the caller can do `buffer = leftover + await reader.read(...)` and
        call parse() again.

    Must handle PIPELINING:
        If `buffer` contains two or more complete commands back-to-back
        (as real clients send when pipelining), return ALL of them in one
        call — never just the first one, and never require the caller to
        invoke parse() once per command.

    Must reject oversized bulk strings:
        As soon as a bulk string's length header ($<len>) is parsed, if
        `len > MAX_BULK_LEN`, raise ProtocolError immediately — before
        waiting for that many payload bytes to actually arrive. This is
        what stops a client from making the server buffer unbounded memory
        just by claiming a huge length.

    Must raise ProtocolError on any other malformed input:
        e.g. a top-level byte that isn't `*` (only arrays are valid
        requests), a non-integer length header, an array/bulk-string
        length that's negative in an invalid position, etc. Use a short,
        human-readable message — the caller turns this into a RESP error
        reply and then closes the connection.

    Examples this must satisfy (see tests/test_resp_decode_contract.py):

        parse(b"*1\\r\\n$4\\r\\nPING\\r\\n") == ([[b"PING"]], b"")

        parse(b"*1\\r\\n$4\\r\\nPI") == ([], b"*1\\r\\n$4\\r\\nPI")
        # incomplete — wait for more bytes, don't raise

        parse(b"*1\\r\\n$4\\r\\nPING\\r\\n*1\\r\\n$4\\r\\nPING\\r\\n") \\
            == ([[b"PING"], [b"PING"]], b"")
        # two pipelined commands, both returned from one call
    """
    raise NotImplementedError("YOU IMPLEMENT THIS")
