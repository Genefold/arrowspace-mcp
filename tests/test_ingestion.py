from __future__ import annotations

import json

import httpx
import numpy as np
import pytest


def test_npy_build_survives_reload(tmp_path):
    from arrowspace import load_arrowspace

    from arrowspace_mcp.ingestion import ingest_npy

    items = np.random.default_rng(3407).normal(size=(20, 4))
    source = tmp_path / "vectors.npy"
    np.save(source, items)
    output = tmp_path / "index"
    result = ingest_npy(source, output, graph_params={"eps": 2.0, "k": 4})

    assert result["n_items"] == 20
    assert list(output.glob("*.parquet"))
    manifest = json.loads((output / "manifest.json").read_text())
    aspace, graph = load_arrowspace(
        str(output), manifest["dataset_name"], manifest["graph_params"], False
    )
    np.testing.assert_allclose(aspace.get_all_items(), items)
    assert len(aspace.lambdas()) == 20
    hits = aspace.search(items[0], graph, tau=1.0)
    assert hits[0][0] == 0
    assert np.isfinite(hits[0][1])


@pytest.mark.parametrize(
    "items",
    [
        np.array([1.0, 2.0]),
        np.empty((0, 3)),
        np.empty((3, 0)),
        np.array([[np.nan, 1.0]]),
        np.array([[np.inf, 1.0]]),
        np.array([[1 + 2j]]),
        np.array([["1", "2"]]),
        np.zeros((2, 3)),
        np.array([[object()]], dtype=object),
    ],
)
def test_invalid_vectors_do_not_create_an_index(tmp_path, items):
    from arrowspace_mcp.ingestion import ingest_npy

    source = tmp_path / "vectors.npy"
    np.save(source, items)
    with pytest.raises(ValueError):
        ingest_npy(source, tmp_path / "index")
    assert not (tmp_path / "index").exists()


@pytest.mark.parametrize(
    "params",
    [
        {"k": 0},
        {"k": 1.5},
        {"eps": float("nan")},
        {"topk": True},
        {"sigma": -1},
        {"unknown": 2},
    ],
)
def test_invalid_graph_parameters_are_rejected(tmp_path, params):
    from arrowspace_mcp.ingestion import ingest_npy

    source = tmp_path / "vectors.npy"
    np.save(source, np.ones((3, 2)))
    with pytest.raises((TypeError, ValueError)):
        ingest_npy(source, tmp_path / "index", graph_params=params)
    assert not (tmp_path / "index").exists()


def test_existing_output_is_preserved(tmp_path):
    from arrowspace_mcp.ingestion import ingest_npy

    source = tmp_path / "vectors.npy"
    np.save(source, np.ones((3, 2)))
    output = tmp_path / "index"
    output.mkdir()
    marker = output / "manifest.json"
    marker.write_text("user data")
    with pytest.raises(FileExistsError):
        ingest_npy(source, output)
    assert marker.read_text() == "user data"


def test_zarr_registration_and_build_use_rest():
    from arrowspace_mcp.ingestion import ingest_zarr

    requests = []

    def server(request):
        requests.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/api/upload/commit":
            return httpx.Response(200, json={"registered": True})
        return httpx.Response(200, json={"built": True, "id": "main--vectors"})

    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        result = ingest_zarr(
            "http://arro.test",
            "main--vectors",
            "/data/vectors.zarr",
            graph_params={"eps": 2.0},
            client=client,
        )
    assert result["built"] is True
    assert requests == [
        (
            "/api/upload/commit",
            {"dataset_id": "main--vectors", "fs_path": "/data/vectors.zarr"},
        ),
        ("/api/datasets/main--vectors/index", {"graph_params": {"eps": 2.0}}),
    ]


def test_zarr_keeps_server_tuned_defaults():
    from arrowspace_mcp.ingestion import ingest_zarr

    def server(request):
        if request.url.path.endswith("/index"):
            assert json.loads(request.content) == {}
        return httpx.Response(200, json={"built": True})

    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        ingest_zarr(
            "http://arro.test", "main--vectors", "/data/vectors.zarr", client=client
        )


def test_zarr_registration_error_stops_build():
    from arrowspace_mcp.ingestion import ingest_zarr

    def server(request):
        assert request.url.path == "/api/upload/commit"
        return httpx.Response(422, json={"detail": "not a Zarr array"})

    with (
        httpx.Client(transport=httpx.MockTransport(server)) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        ingest_zarr("http://arro.test", "main--bad", "/data/bad", client=client)
