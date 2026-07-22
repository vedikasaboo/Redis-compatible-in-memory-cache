"""
resp_bytes.py — a raw look at the RESP (REdis Serialization Protocol) wire format.

Connects to Redis with nothing but the stdlib `socket` module (no redis-py),
hand-encodes commands as RESP arrays of bulk strings (the format real clients
use to send commands), and prints the *exact* bytes sent and received with
\r\n made visible. The goal is to see, with your own eyes, the five RESP
reply types before writing a parser for them in Phase 1:

    +   simple string   e.g. +PONG\r\n
    -   error           e.g. -ERR unknown command '...'\r\n
    :   integer         e.g. :42\r\n
    $   bulk string     e.g. $5\r\nhello\r\n   (length-prefixed)
    $-1 null bulk string e.g. $-1\r\n          (used for "key not found")
    *   array           e.g. *2\r\n$3\r\nfoo\r\n$3\r\nbar\r\n

RESP spec: https://redis.io/docs/latest/develop/reference/protocol-spec/
"""
import socket

HOST = "localhost"
PORT = 6380


def visible(data: bytes) -> str:
    """Render bytes with \r\n as literal escapes so wire framing is visible."""
    return data.decode("utf-8", errors="backslashreplace").replace("\r", "\\r").replace("\n", "\\n\n")


def encode_command(*args: str) -> bytes:
    """Hand-encode a command as a RESP array of bulk strings — the wire
    format every real Redis client sends, regardless of reply type."""
    out = [f"*{len(args)}\r\n".encode()]
    for arg in args:
        arg_bytes = arg.encode()
        out.append(f"${len(arg_bytes)}\r\n".encode())
        out.append(arg_bytes)
        out.append(b"\r\n")
    return b"".join(out)


def send_and_recv(sock: socket.socket, *args: str, label: str = "", bufsize: int = 65536) -> bytes:
    request = encode_command(*args)
    if label:
        print(f"--- {label} ---")
    print(f"SENT     ({len(request)} bytes): {visible(request)}")
    sock.sendall(request)
    reply = sock.recv(bufsize)
    print(f"RECEIVED ({len(reply)} bytes): {visible(reply)}")
    print()
    return reply


def main():
    sock = socket.create_connection((HOST, PORT), timeout=5)
    print(f"Connected raw TCP socket to {HOST}:{PORT}\n")

    # --- Simple string (+) ---------------------------------------------
    print("### RESP type: Simple String (+) ###")
    send_and_recv(sock, "PING", label="PING -> +PONG")

    # --- Error (-) --------------------------------------------------------
    print("### RESP type: Error (-) ###")
    send_and_recv(sock, "NOTACOMMAND", "arg1", label="deliberately bad command -> -ERR ...")

    # --- Integer (:) --------------------------------------------------
    print("### RESP type: Integer (:) ###")
    send_and_recv(sock, "DEL", "counter", label="cleanup: DEL counter")
    send_and_recv(sock, "INCR", "counter", label="INCR counter -> :1")
    send_and_recv(sock, "INCR", "counter", label="INCR counter again -> :2")

    # --- Bulk string ($) on an existing key -----------------------------
    print("### RESP type: Bulk String ($) ###")
    send_and_recv(sock, "SET", "greeting", "hello world", label="SET greeting 'hello world'")
    send_and_recv(sock, "GET", "greeting", label="GET greeting -> $11\\r\\nhello world\\r\\n")

    # --- Null bulk string ($-1) on a missing key ------------------------
    print("### RESP type: Null Bulk String ($-1) ###")
    send_and_recv(sock, "GET", "this_key_does_not_exist", label="GET on a missing key -> $-1\\r\\n")

    # --- Array (*) -------------------------------------------------------
    print("### RESP type: Array (*) ###")
    send_and_recv(sock, "RPUSH", "mylist", "a", "b", "c", label="RPUSH mylist a b c")
    send_and_recv(sock, "LRANGE", "mylist", "0", "-1", label="LRANGE mylist 0 -1 -> array of bulk strings")

    send_and_recv(sock, "SET", "k1", "v1", label="SET k1 v1")
    send_and_recv(sock, "SET", "k2", "v2", label="SET k2 v2")
    send_and_recv(sock, "MGET", "k1", "k2", "missing_key",
                  label="MGET k1 k2 missing_key -> array mixing bulk strings and a null")

    # --- Pipelining: two commands in ONE write, read back BOTH replies --
    print("### PIPELINING: two commands sent in a single write() ###")
    cmd1 = encode_command("SET", "pipe_key", "pipe_val")
    cmd2 = encode_command("GET", "pipe_key")
    combined = cmd1 + cmd2
    print(f"SENT     ({len(combined)} bytes, 2 commands in 1 write): {visible(combined)}")
    sock.sendall(combined)
    reply = sock.recv(65536)
    print(f"RECEIVED ({len(reply)} bytes, both replies in 1 read): {visible(reply)}")
    print()
    print("Note: the server pipelines replies back-to-back with no framing")
    print("between them beyond each reply's own RESP type marker — a parser")
    print("must be able to read multiple complete replies out of one buffer,")
    print("and must not assume one read() == one reply.")

    sock.close()


if __name__ == "__main__":
    main()
