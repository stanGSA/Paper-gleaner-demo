from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import openai
import pytest
from pydantic import SecretStr

from gleaner.config import Config
from gleaner.ranking import EmbeddingError, build_embedder
from gleaner.ranking.embedder import ApiEmbedder, FastEmbedEmbedder, normalize


class FakeTextEmbedding:
    instances: list["FakeTextEmbedding"] = []

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_sizes: list[int] = []
        FakeTextEmbedding.instances.append(self)

    def embed(self, texts: list[str], batch_size: int):
        self.batch_sizes.append(batch_size)
        for text in texts:
            yield np.array([len(text), 0.0, 0.0])


@pytest.fixture
def fake_fastembed(monkeypatch: pytest.MonkeyPatch) -> type[FakeTextEmbedding]:
    import fastembed

    FakeTextEmbedding.instances = []
    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)
    return FakeTextEmbedding


def test_normalize() -> None:
    result = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(result, [[0.6, 0.8], [0.0, 0.0]])


def test_fastembed_loads_lazily_and_normalizes(
    fake_fastembed: type[FakeTextEmbedding], tmp_path: Path
) -> None:
    embedder = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", cache_dir=tmp_path, batch_size=8)
    assert fake_fastembed.instances == []

    vectors = embedder.embed(["ab", "abcd"])
    embedder.embed(["x"])

    np.testing.assert_allclose(vectors, [[1, 0, 0], [1, 0, 0]])
    assert len(fake_fastembed.instances) == 1  # 模型只加载一次
    model = fake_fastembed.instances[0]
    assert model.model_name == "BAAI/bge-small-en-v1.5"
    assert model.cache_dir == str(tmp_path)
    assert model.batch_sizes == [8, 8]


def test_fastembed_cache_dir_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GLEANER_MODEL_CACHE", str(tmp_path))
    assert FastEmbedEmbedder("m").cache_dir == tmp_path


def test_fastembed_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import fastembed

    def broken(**kwargs: Any) -> None:
        raise ValueError("download failed")

    monkeypatch.setattr(fastembed, "TextEmbedding", broken)
    with pytest.raises(EmbeddingError, match="download failed"):
        FastEmbedEmbedder("m").embed(["x"])


def test_empty_input() -> None:
    assert FastEmbedEmbedder("m").embed([]).shape == (0, 0)


class FakeEmbeddings:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        data = [
            SimpleNamespace(index=i, embedding=[float(len(text)), 1.0])
            for i, text in enumerate(kwargs["input"])
        ]
        return SimpleNamespace(data=list(reversed(data)))  # 故意打乱顺序


def test_api_embedder_batches_and_keeps_order() -> None:
    embeddings = FakeEmbeddings()
    client = SimpleNamespace(embeddings=embeddings)
    embedder = ApiEmbedder("text-embedding-v3", client=client, batch_size=2)

    vectors = embedder.embed(["a", "bbb", "cc"])

    assert [len(r["input"]) for r in embeddings.requests] == [2, 1]
    assert embeddings.requests[0]["model"] == "text-embedding-v3"
    np.testing.assert_allclose(vectors, normalize(np.array([[1, 1], [3, 1], [2, 1]], dtype=float)))


def test_api_embedder_wraps_errors() -> None:
    request = httpx.Request("POST", "https://api.example/embeddings")

    def fail(**kwargs: Any) -> None:
        raise openai.APIConnectionError(request=request)

    client = SimpleNamespace(embeddings=SimpleNamespace(create=fail))
    with pytest.raises(EmbeddingError, match="m"):
        ApiEmbedder("m", client=client).embed(["x"])


def test_build_local_embedder() -> None:
    embedder = build_embedder(Config())
    assert isinstance(embedder, FastEmbedEmbedder)
    assert embedder.model == "BAAI/bge-small-en-v1.5"


def test_build_api_embedder_falls_back_to_llm_credentials() -> None:
    config = Config.model_validate({"ranking": {"embedding": {"provider": "api", "model": "e"}}})
    config.llm.api_key = SecretStr("sk-llm")
    config.llm.base_url = "https://llm.example/v1"

    embedder = build_embedder(config)

    assert isinstance(embedder, ApiEmbedder)
    assert embedder._client.api_key == "sk-llm"
    assert str(embedder._client.base_url).startswith("https://llm.example/v1")


def test_build_api_embedder_prefers_embedding_credentials() -> None:
    config = Config.model_validate(
        {"ranking": {"embedding": {"provider": "api", "base_url": "https://e.example/v1"}}}
    )
    config.ranking.embedding.api_key = SecretStr("sk-embed")
    config.llm.api_key = SecretStr("sk-llm")

    embedder = build_embedder(config)

    assert embedder._client.api_key == "sk-embed"
    assert str(embedder._client.base_url).startswith("https://e.example/v1")


def test_build_api_embedder_without_key() -> None:
    config = Config.model_validate({"ranking": {"embedding": {"provider": "api"}}})
    with pytest.raises(EmbeddingError, match="EMBEDDING_API_KEY"):
        build_embedder(config)
