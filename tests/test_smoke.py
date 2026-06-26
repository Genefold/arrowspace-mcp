from __future__ import annotations

import json

import anyio
import numpy as np
import pytest
from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client


@pytest.fixture(scope="module")
def server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command="uv",
        args=["run", "arrowspace-mcp", "--transport", "stdio", "--ttl", "30"],
    )


@pytest.fixture
def sample_vectors() -> list[list[float]]:
    rng = np.random.default_rng(3407)
    return rng.normal(size=(20, 4)).astype(np.float64).tolist()


def _ok(result) -> dict:
    assert not result.isError, f"Tool call failed: {result.content[0].text}"
    return json.loads(result.content[0].text)


@pytest.mark.asyncio
async def test_initialize(server_params: StdioServerParameters) -> None:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            result = await session.initialize()
            assert result.serverInfo.name == "arrowspace-mcp"


@pytest.mark.asyncio
async def test_list_tools(server_params: StdioServerParameters) -> None:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            for name in (
                "build_index", "search", "lambdas",
                "lambdas_sorted", "spectral_analysis", "delete_index",
            ):
                assert name in tool_names, f"Missing tool: {name}"


@pytest.mark.asyncio
async def test_build_index(
    server_params: StdioServerParameters,
    sample_vectors: list[list[float]],
) -> None:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("build_index", {
                "vectors": sample_vectors,
                "k": 4,
            })
            data = _ok(result)
            assert "index_id" in data
            assert data["n_items"] == 20
            assert data["n_components"] == 4


@pytest.mark.asyncio
async def test_search(server_params: StdioServerParameters) -> None:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            build = _ok(await session.call_tool("build_index", {
                "vectors": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                "k": 2,
            }))
            idx = build["index_id"]

            result = _ok(await session.call_tool("search", {
                "index_id": idx,
                "query": [0.9, 0.1],
                "tau": 1.0,
            }))
            assert len(result["hits"]) > 0
            assert "index" in result["hits"][0]
            assert "score" in result["hits"][0]


@pytest.mark.asyncio
async def test_lambdas(server_params: StdioServerParameters) -> None:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            build = _ok(await session.call_tool("build_index", {
                "vectors": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                "k": 2,
            }))
            idx = build["index_id"]

            result = _ok(await session.call_tool("lambdas", {"index_id": idx}))
            assert result["n_items"] == 3
            assert len(result["lambdas"]) == 3

            result = _ok(await session.call_tool("lambdas", {
                "index_id": idx,
                "include_stats": True,
            }))
            assert "stats" in result
            assert "min" in result["stats"]


@pytest.mark.asyncio
async def test_lambdas_sorted(server_params: StdioServerParameters) -> None:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            build = _ok(await session.call_tool("build_index", {
                "vectors": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                "k": 2,
            }))
            idx = build["index_id"]

            result = _ok(await session.call_tool("lambdas_sorted", {"index_id": idx}))
            assert result["n_items"] == 3
            assert len(result["ranked"]) == 3
            assert result["ranked"][0]["score"] <= result["ranked"][1]["score"]

            result = _ok(await session.call_tool("lambdas_sorted", {
                "index_id": idx,
                "top_n": 1,
            }))
            assert len(result["ranked"]) == 1
            assert result.get("truncated") is True


@pytest.mark.asyncio
async def test_spectral_analysis(server_params: StdioServerParameters) -> None:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            build = _ok(await session.call_tool("build_index", {
                "vectors": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                "k": 2,
            }))
            idx = build["index_id"]

            result = _ok(await session.call_tool("spectral_analysis", {
                "index_id": idx,
                "mode": "spectrum",
            }))
            assert "eigenvalues" in result
            assert "spectral_gap" in result

            result = _ok(await session.call_tool("spectral_analysis", {
                "index_id": idx,
                "mode": "components",
            }))
            assert "n_components" in result

            result = _ok(await session.call_tool("spectral_analysis", {
                "index_id": idx,
                "mode": "clusters",
            }))
            assert "assignments" in result


@pytest.mark.asyncio
async def test_delete_index(server_params: StdioServerParameters) -> None:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            build = _ok(await session.call_tool("build_index", {
                "vectors": [[1.0, 0.0], [0.0, 1.0]],
                "k": 1,
            }))
            idx = build["index_id"]

            result = _ok(await session.call_tool("delete_index", {"index_id": idx}))
            assert result["status"] == "deleted"

            result = await session.call_tool("lambdas", {"index_id": idx})
            assert result.isError
            assert "not found" in result.content[0].text.lower()


@pytest.mark.asyncio
async def test_file_path_input(
    server_params: StdioServerParameters,
    tmp_path,
) -> None:
    npy_path = tmp_path / "test.npy"
    np.save(str(npy_path), np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64))

    params = StdioServerParameters(
        command="uv",
        args=[
            "run", "arrowspace-mcp",
            "--transport", "stdio",
            "--ttl", "30",
            "--allowed-paths", str(tmp_path),
        ],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = _ok(await session.call_tool("build_index", {
                "file_path": str(npy_path),
                "file_format": "npy",
                "k": 1,
            }))
            assert "index_id" in result
            assert result["n_items"] == 2


@pytest.mark.asyncio
async def test_list_resources(server_params: StdioServerParameters) -> None:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_resources()
            uris = [str(r.uri) for r in result.resources]
            for uri in (
                "arrowspace://skills/core",
                "arrowspace://skills/search",
                "arrowspace://skills/spectral",
                "arrowspace://skills/hyperparameters",
                "arrowspace://paper/abstract",
            ):
                assert uri in uris, f"Missing resource: {uri}"


@pytest.mark.asyncio
async def test_read_skill_resource(server_params: StdioServerParameters) -> None:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.read_resource("arrowspace://skills/core")
            assert len(result.contents) == 1
            text = result.contents[0].text
            assert len(text) > 100
            assert "ArrowSpace" in text


@pytest.mark.asyncio
async def test_read_paper_abstract(server_params: StdioServerParameters) -> None:
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.read_resource("arrowspace://paper/abstract")
            text = result.contents[0].text
            assert "λτ" in text
            assert "Rayleigh quotient" in text
