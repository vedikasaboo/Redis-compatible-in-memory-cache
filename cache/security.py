"""
Connection-level security: authentication, connection caps, and the
session commands that need per-connection state.

The rest of the server is stateless per command (commands.dispatch takes a
command and the store, nothing else). AUTH, CLIENT and GUARD cannot be: they
read or change state belonging to ONE connection, so they are answered here,
before a command ever reaches dispatch.

Authentication follows Redis's own protocol, so redis-cli -a works:

  - with --requirepass set, every command except AUTH gets
        -NOAUTH Authentication required.
  - a wrong password gets
        -WRONGPASS invalid username-password pair or user is disabled.
  - both `AUTH <password>` and `AUTH <user> <password>` (user "default") work.

Brute-force resistance, without locking out legitimate clients that share an
IP (everything on localhost does):

  - the password comparison is constant-time (hmac.compare_digest), so the
    reply time does not leak how many leading characters were right;
  - each failure doubles the reply delay for failed AUTHs from that IP
    (0.1 s, 0.2 s, 0.4 s ... capped at 2 s), decaying by half every 30 s.
    Only FAILED attempts are delayed, so a legitimate client sharing the IP
    and authenticating correctly is never slowed. Keyed by IP rather than
    by connection because disconnecting a guesser would otherwise hand it
    a fresh, short delay on reconnect — the first benchmark run measured
    exactly that, with the guard ON letting through 3x more guesses;
  - after MAX_AUTH_FAILURES on one connection, that connection is closed;
  - every failure is counted into the guard's features, so the detector sees
    a guessing campaign spread over many connections too.

handle() is deliberately synchronous: it returns a delay for the caller to
sleep rather than awaiting itself, so the common path (an authenticated
GET) costs no extra coroutine.
"""
import hmac
import json
import time

from cache.guard import Guard, ConnStats
from cache.resp import ErrorReply, SimpleString

NOAUTH = ErrorReply("NOAUTH Authentication required.")
WRONGPASS = ErrorReply("WRONGPASS invalid username-password pair or user is disabled.")
NO_PASSWORD_SET = ErrorReply(
    "ERR AUTH <password> called without any password configured for the default user. "
    "Are you sure your configuration is correct?"
)
AUTH_BASE_DELAY = 0.1
AUTH_MAX_DELAY = 2.0
MAX_AUTH_FAILURES = 10
AUTH_DECAY_SECONDS = 30.0  # an IP's failure count halves over this long

# Commands answered here rather than by dispatch, because they need the session.
SESSION_COMMANDS = {b"AUTH", b"CLIENT", b"GUARD"}

# Commands whose first argument is a key, for the guard's per-key features.
KEYED = {b"GET", b"SET", b"DEL", b"EXISTS", b"TTL", b"PTTL"}


class Session:
    __slots__ = ("peer", "ip", "authenticated", "auth_failures", "stats", "close_after")

    def __init__(self, peer, stats: ConnStats, authenticated: bool):
        self.peer = peer
        self.ip = peer[0] if isinstance(peer, tuple) else str(peer)
        self.authenticated = authenticated
        self.auth_failures = 0
        self.stats = stats
        self.close_after = False


class Security:
    def __init__(self, guard: Guard, password: str | None = None, max_conns_per_ip: int = 0):
        self.guard = guard
        self.password = password.encode() if password else None
        self.max_conns_per_ip = max_conns_per_ip
        self._per_ip: dict[str, int] = {}
        self._ip_failures: dict[str, tuple[float, float]] = {}  # ip -> (count, when)

    # -- connection lifecycle ----------------------------------------------

    def open(self, peer) -> Session | None:
        """A new session, or None if this IP already holds its connection quota."""
        ip = peer[0] if isinstance(peer, tuple) else str(peer)
        if self.max_conns_per_ip and self._per_ip.get(ip, 0) >= self.max_conns_per_ip:
            return None
        self._per_ip[ip] = self._per_ip.get(ip, 0) + 1
        return Session(peer, self.guard.register(peer), authenticated=self.password is None)

    def close(self, session: Session) -> None:
        self._per_ip[session.ip] -= 1
        if not self._per_ip[session.ip]:
            del self._per_ip[session.ip]
        self.guard.unregister(session.stats)

    # -- per command -------------------------------------------------------

    def handle(self, session: Session, command: list[bytes]):
        """
        (reply, delay). reply None = not a session command and allowed
        through: dispatch it. delay = seconds to wait before replying.
        """
        name = command[0].upper()
        if name == b"AUTH":
            return self._auth(session, command[1:])
        if self.password is not None and not session.authenticated:
            return NOAUTH, 0.0
        if name == b"CLIENT":
            return self._client(session, command[1:]), 0.0
        if name == b"GUARD":
            return self._guard(command[1:]), 0.0
        return None, 0.0

    def _auth(self, session: Session, args: list[bytes]):
        if len(args) not in (1, 2):
            return ErrorReply("ERR wrong number of arguments for 'auth' command"), 0.0
        if self.password is None:
            return NO_PASSWORD_SET, 0.0
        user, given = (b"default", args[0]) if len(args) == 1 else (args[0], args[1])
        if user == b"default" and hmac.compare_digest(given, self.password):
            session.authenticated = True
            return SimpleString("OK"), 0.0
        session.auth_failures += 1
        session.stats.auth_failures += 1
        if session.auth_failures >= MAX_AUTH_FAILURES:
            session.close_after = True
        now = time.monotonic()
        count, when = self._ip_failures.get(session.ip, (0.0, now))
        count = count * 0.5 ** ((now - when) / AUTH_DECAY_SECONDS) + 1
        self._ip_failures[session.ip] = (count, now)
        delay = min(AUTH_MAX_DELAY, AUTH_BASE_DELAY * 2 ** (count - 1))
        return WRONGPASS, delay

    def _client(self, session: Session, args: list[bytes]):
        sub = args[0].upper() if args else b""
        if sub == b"SETNAME" and len(args) == 2:
            session.stats.name = args[1].decode("utf-8", errors="replace")
            return SimpleString("OK")
        if sub == b"GETNAME":
            return session.stats.name.encode() or None
        if sub == b"ID":
            return session.stats.id
        return ErrorReply("ERR unknown CLIENT subcommand (supported: SETNAME, GETNAME, ID)")

    def _guard(self, args: list[bytes]):
        sub = args[0].upper() if args else b"STATUS"
        if sub == b"STATUS":
            body = {"mode": self.guard.mode, "rejected": self.guard.rejected_total,
                    "connections": self.guard.status(), "flags": self.guard.flag_events[-50:]}
            return json.dumps(body).encode()
        if sub == b"MODE" and len(args) == 2:
            try:
                self.guard.set_mode(args[1].decode().lower())
            except ValueError as exc:
                return ErrorReply(f"ERR {exc}")
            return SimpleString("OK")
        return ErrorReply("ERR usage: GUARD STATUS | GUARD MODE off|observe|rules|ml")


def keyed_arg(name: bytes, command: list[bytes]) -> bytes | None:
    return command[1] if name in KEYED and len(command) > 1 else None
