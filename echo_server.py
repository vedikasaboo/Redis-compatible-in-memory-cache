"""echo_server.py — minimal asyncio TCP echo server (asyncio warm-up)."""
import asyncio

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info("peername")
    print(f"[server] connection from {addr}")
    while True:
        data = await reader.read(1024)  # await #1: yields until bytes arrive on this socket
        if not data:                    # empty bytes = client closed
            break
        print(f"[server] echoing {data!r} back to {addr}")
        writer.write(data)
        await writer.drain()            # await #2: yields until OS write buffer has room
    writer.close()
    await writer.wait_closed()          # await #3: yields until socket is fully torn down

async def main():
    server = await asyncio.start_server(handle_client, "127.0.0.1", 8888)  # await #4: yields while binding the socket
    print("[server] listening on 127.0.0.1:8888")
    async with server:
        await server.serve_forever()    # await #5: yields forever, waking to dispatch connections/data

if __name__ == "__main__":
    asyncio.run(main())
