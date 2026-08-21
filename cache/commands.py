"""
Command dispatch: routes a parsed command (list[bytes]) to a handler
function that operates on the store and returns a value ready for
resp.encode(). This is plumbing, not the interrogatable core — the
interesting parts (resp.parse, cache.lru.LRUCache) live elsewhere.
"""
from cache.lru import LRUCache
from cache.resp import ErrorReply, SimpleString

SYNTAX_ERROR = ErrorReply("ERR syntax error")
NOT_AN_INT = ErrorReply("ERR value is not an integer or out of range")


def _wrong_args(name: str) -> ErrorReply:
    return ErrorReply(f"ERR wrong number of arguments for '{name}' command")


def _expiry_from_options(options: list[bytes], now: int) -> tuple[int | None, ErrorReply | None]:
    """
    Turn SET's expiry options into an ABSOLUTE epoch-ms deadline.

    Collapsing EX/PX (relative) into an absolute deadline here, once, is
    what lets the WAL record something replay-safe: a logged `SET k v EX
    10` replayed an hour later would wrongly grant a fresh 10 seconds,
    while a logged absolute deadline stays correct no matter when it is
    replayed. Redis solves this the same way when rewriting its AOF.

    Returns (expires_at, error). Exactly one is non-None.
    """
    if not options:
        return None, None
    if len(options) != 2:
        return None, SYNTAX_ERROR
    name = options[0].upper()
    try:
        n = int(options[1])
    except ValueError:
        return None, NOT_AN_INT

    if name in (b"EX", b"PX"):
        # Redis rejects non-positive relative TTLs rather than treating
        # them as "delete immediately".
        if n <= 0:
            return None, ErrorReply("ERR invalid expire time in 'set' command")
        return now + (n * 1000 if name == b"EX" else n), None
    if name in (b"EXAT", b"PXAT"):
        # Absolute deadlines in the past ARE legal; the key simply never
        # becomes visible.
        return (n * 1000 if name == b"EXAT" else n), None
    return None, SYNTAX_ERROR


def normalize_write(command: list[bytes], now: int) -> list[bytes] | None:
    """
    Return the form of `command` that should be written to the WAL, or
    None if it is not a durable write (a read, or malformed).

    The arity/option validation matters: a command that will error changes
    nothing, so logging it would make replay diverge from the state
    clients actually observed. And relative expiries are rewritten to
    absolute PXAT so replay stays faithful — see _expiry_from_options.

    The server dispatches the RETURNED command, not the original, so the
    deadline written to the log and the deadline applied in memory are
    byte-for-byte the same value.
    """
    if not command:
        return None
    name = command[0].upper()

    if name == b"DEL":
        return command if len(command) >= 2 else None

    if name != b"SET":
        return None
    if len(command) == 3:
        return command
    if len(command) == 5:
        expires_at, error = _expiry_from_options(command[3:], now)
        if error is not None:
            return None
        return [b"SET", command[1], command[2], b"PXAT", str(expires_at).encode()]
    return None


def cmd_ping(args: list[bytes], store: LRUCache):
    if len(args) > 1:
        return _wrong_args("ping")
    if args:
        return args[0]  # PING <msg> echoes msg back as a bulk string
    return SimpleString("PONG")


def cmd_get(args: list[bytes], store: LRUCache):
    if len(args) != 1:
        return _wrong_args("get")
    return store.get(args[0])  # bytes, or None -> encoded as null bulk string


def cmd_set(args: list[bytes], store: LRUCache):
    if len(args) < 2:
        return _wrong_args("set")
    key, value = args[0], args[1]
    expires_at, error = _expiry_from_options(args[2:], store.now_ms())
    if error is not None:
        return error
    store.set(key, value, expires_at)
    return SimpleString("OK")


def cmd_del(args: list[bytes], store: LRUCache):
    if not args:
        return _wrong_args("del")
    return sum(1 for key in args if store.delete(key))


def cmd_ttl(args: list[bytes], store: LRUCache):
    if len(args) != 1:
        return _wrong_args("ttl")
    ms = store.ttl_ms(args[0])
    if ms < 0:
        return ms  # -1 (no expiry) / -2 (no such key), same as Redis
    return (ms + 999) // 1000  # Redis rounds TTL up to whole seconds


def cmd_pttl(args: list[bytes], store: LRUCache):
    if len(args) != 1:
        return _wrong_args("pttl")
    return store.ttl_ms(args[0])


def cmd_exists(args: list[bytes], store: LRUCache):
    if not args:
        return _wrong_args("exists")
    return sum(1 for key in args if key in store)


DISPATCH = {
    b"PING": cmd_ping,
    b"GET": cmd_get,
    b"SET": cmd_set,
    b"DEL": cmd_del,
    b"TTL": cmd_ttl,
    b"PTTL": cmd_pttl,
    b"EXISTS": cmd_exists,
}


def dispatch(command: list[bytes], store: LRUCache):
    """Route a parsed command to its handler; unknown commands -> RESP error."""
    name = command[0].upper()
    handler = DISPATCH.get(name)
    if handler is None:
        shown = command[0].decode("utf-8", errors="replace")
        return ErrorReply(f"ERR unknown command '{shown}'")
    return handler(command[1:], store)
