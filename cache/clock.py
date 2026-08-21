"""
Single source of time.

Expiry uses WALL-CLOCK epoch milliseconds, not a monotonic counter,
because a TTL has to survive a restart: the WAL records an absolute
deadline, and after recovery that deadline must still mean the same
instant. A monotonic clock resets with the process and would silently
resurrect keys that should have expired while the server was down.

Milliseconds (not float seconds) to match the units Redis itself uses on
the wire for PX/PXAT, so nothing is lost converting back and forth.
"""
import time


def now_ms() -> int:
    return int(time.time() * 1000)
