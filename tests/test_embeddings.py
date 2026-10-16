from __future__ import annotations

import json

import numpy as np
import pytest


class FakeEncoder:
    """Offline boundary double; encode text content into distinct vectors."""

    def encode(self, texts, **kwargs):
        prefix = kwargs.get("prompt_name") == "query"
        return np.array([[len(text) + prefix, 1.0] for text in texts], dtype=np.float32)


def test_jsonl_embeddings_preserve_row_identity(tmp_path):
    from arrowspace_mcp.embeddings import embed_file

    source = tmp_path / "documents.jsonl"
    source.write_text('{"id":"a","body":"first"}\n\n{"id":"b","body":"second"}\n')
    output = tmp_path / "embedded"
    result = embed_file(
        source, output, text_field="body", batch_size=1, encoder=FakeEncoder()
    )
    vectors = np.load(output / "embeddings.npy", allow_pickle=False)
    assert vectors.dtype == np.float64
    assert vectors.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)
    rows = [
        json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()
    ]
    assert [row["id"] for row in rows] == ["a", "b"]
    assert [row["row"] for row in rows] == [0, 1]
    assert [row["text"] for row in rows] == ["first", "second"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert manifest["seed"] == 3407
    assert manifest["input_sha256"]
    assert result["n_items"] == 2


@pytest.mark.parametrize("model", ["minilm", "granite", "snowflake"])
def test_query_encoding_uses_model_specific_prompt(tmp_path, model):
    from arrowspace_mcp.embeddings import embed_file

    source = tmp_path / "query.txt"
    source.write_text("find this\n")
    output = tmp_path / "queries"
    embed_file(
        source,
        output,
        model=model,
        role="query",
        normalise=False,
        scale=2.0,
        encoder=FakeEncoder(),
    )
    expected_length = 10 if model == "snowflake" else 9
    np.testing.assert_array_equal(
        np.load(output / "embeddings.npy"), [[expected_length * 2, 2]]
    )


@pytest.mark.parametrize(
    "content",
    [
        "",
        " \n\n",
        '{"text":null}\n',
        '{"other":"missing"}\n',
        '{"text":"  "}\n',
        "not json\n",
    ],
)
def test_invalid_text_does_not_leave_output(tmp_path, content):
    from arrowspace_mcp.embeddings import embed_file

    source = tmp_path / "bad.jsonl"
    source.write_text(content)
    with pytest.raises(ValueError):
        embed_file(source, tmp_path / "out", encoder=FakeEncoder())
    assert not (tmp_path / "out").exists()


def test_encoder_failure_removes_partial_bundle(tmp_path):
    from arrowspace_mcp.embeddings import embed_file

    class FailingEncoder:
        def encode(self, texts, **kwargs):
            raise RuntimeError("model failed")

    source = tmp_path / "text.txt"
    source.write_text("hello\n")
    with pytest.raises(RuntimeError, match="model failed"):
        embed_file(source, tmp_path / "out", encoder=FailingEncoder())
    assert not (tmp_path / "out").exists()


def test_embedding_does_not_overwrite_user_data(tmp_path):
    from arrowspace_mcp.embeddings import embed_file

    source = tmp_path / "text.txt"
    source.write_text("hello\n")
    output = tmp_path / "out"
    output.mkdir()
    (output / "embeddings.npy").write_bytes(b"existing data")
    with pytest.raises(FileExistsError):
        embed_file(source, output, encoder=FakeEncoder())
    assert (output / "embeddings.npy").read_bytes() == b"existing data"
