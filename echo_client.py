"""echo_client.py — matching client for echo_server.py."""
import asyncio

MESSAGES = [b"hello", b"asyncio", b"world"]


async def main():
    reader, writer = await asyncio.open_connection("127.0.0.1", 8888)  # await #1: yields while the TCP handshake completes
    for msg in MESSAGES:
        print(f"[client] sending {msg!r}")
        writer.write(msg)
        await writer.drain()                # await #2: yields until the message is handed to the OS send buffer
        data = await reader.read(1024)       # await #3: yields until the echoed reply arrives back
        print(f"[client] received {data!r}")
    writer.close()
    await writer.wait_closed()               # await #4: yields until the connection is fully closed


if __name__ == "__main__":
    asyncio.run(main())
