"""
Command dispatch: routes a parsed command (list[bytes]) to a handler
function that operates on the store and returns a value ready for
resp.encode(). This is plumbing, not the interrogatable core — the
interesting parts (resp.parse, cache.lru.LRUCache) live elsewhere.
"""
from cache.lru import LRUCache
from cache.resp import ErrorReply, SimpleString


def _wrong_args(name: str) -> ErrorReply:
    return ErrorReply(f"ERR wrong number of arguments for '{name}' command")


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
    if len(args) != 2:
        return _wrong_args("set")
    key, value = args
    store.set(key, value)
    return SimpleString("OK")


def cmd_del(args: list[bytes], store: LRUCache):
    if not args:
        return _wrong_args("del")
    return sum(1 for key in args if store.delete(key))


DISPATCH = {
    b"PING": cmd_ping,
    b"GET": cmd_get,
    b"SET": cmd_set,
    b"DEL": cmd_del,
}


def is_durable_write(command: list[bytes]) -> bool:
    """
    True if this command mutates state AND is well-formed enough to apply.

    The arity check matters: a malformed `SET` gets an error reply and
    changes nothing, so logging it would make replay diverge from the
    state clients actually observed.
    """
    if not command:
        return False
    name = command[0].upper()
    if name == b"SET":
        return len(command) == 3
    if name == b"DEL":
        return len(command) >= 2
    return False


def dispatch(command: list[bytes], store: LRUCache):
    """Route a parsed command to its handler; unknown commands -> RESP error."""
    name = command[0].upper()
    handler = DISPATCH.get(name)
    if handler is None:
        shown = command[0].decode("utf-8", errors="replace")
        return ErrorReply(f"ERR unknown command '{shown}'")
    return handler(command[1:], store)
