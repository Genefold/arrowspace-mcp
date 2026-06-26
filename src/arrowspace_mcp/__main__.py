from __future__ import annotations

import argparse

import anyio
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount, Route

from arrowspace_mcp.config import ServerConfig
from arrowspace_mcp.server import build_server


def run_stdio(config: ServerConfig) -> None:
    from mcp.server.stdio import stdio_server

    server = build_server(config)

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server._mcp_server.run(
                read_stream,
                write_stream,
                server._mcp_server.create_initialization_options(),
            )

    anyio.run(_run)


def run_sse(host: str, port: int, config: ServerConfig) -> None:
    from mcp.server.sse import SseServerTransport

    server = build_server(config)
    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server._mcp_server.run(
                streams[0],
                streams[1],
                server._mcp_server.create_initialization_options(),
            )

    routes = [
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ]
    app = Starlette(routes=routes)
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="ArrowSpace MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (SSE only, default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind (SSE only, default: 8765)",
    )
    parser.add_argument(
        "--allowed-paths",
        default=".",
        help="Colon-separated allowed directories for file_path (default: current dir)",
    )
    parser.add_argument(
        "--ttl",
        type=float,
        default=3600.0,
        help="Index TTL in seconds (default: 3600 = 1 hour)",
    )
    parser.add_argument(
        "--max-indexes",
        type=int,
        default=100,
        help="Maximum number of concurrent indexes (default: 100)",
    )
    args = parser.parse_args()

    config = ServerConfig(
        allowed_paths=[p.strip() for p in args.allowed_paths.split(":")],
        ttl=args.ttl,
        max_indexes=args.max_indexes,
    )

    if args.transport == "stdio":
        run_stdio(config)
    else:
        run_sse(args.host, args.port, config)


if __name__ == "__main__":
    main()
