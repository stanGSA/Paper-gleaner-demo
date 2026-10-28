import json
from pathlib import Path
from typing import Any

import pytest

from gleaner.llm import LLMOutputError
from gleaner.profile import ProfileError, get_anchors, load_interests
from gleaner.profile.anchors import ANCHORS_PATH
from gleaner.store import DataStore
from tests.conftest import fake_llm, user_prompt

ANCHORS_REPLY = json.dumps(
    {
        "anchors": [
            {
                "name": "长上下文推理",
                "query": "Efficient long-context inference via KV cache compression.",
            },
            {"name": "检索增强生成", "query": "Retrieval-augmented generation for science QA."},
            {"name": "医学影像", "query": "Medical image segmentation.", "negative": True},
        ]
    },
    ensure_ascii=False,
)


def reply_with(text: str):
    def handler(request: dict[str, Any]) -> str:
        return text

    return handler


def test_load_interests_strips_comments(tmp_path: Path) -> None:
    path = tmp_path / "interests.md"
    path.write_text("<!--\n填写说明\n-->\n\n我研究长上下文推理。\n", encoding="utf-8")
    assert load_interests(path) == "我研究长上下文推理。"


def test_load_interests_missing(tmp_path: Path) -> None:
    with pytest.raises(ProfileError, match="interests.example.md"):
        load_interests(tmp_path / "interests.md")


def test_load_interests_only_comments(tmp_path: Path) -> None:
    path = tmp_path / "interests.md"
    path.write_text("<!-- 只有说明 -->\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="是空的"):
        load_interests(path)


def test_example_interests_file_is_valid() -> None:
    text = load_interests(Path(__file__).parent.parent / "profile/interests.example.md")
    assert "长上下文推理" in text
    assert "<!--" not in text


def test_generates_and_caches(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    llm, fake = fake_llm(reply_with(ANCHORS_REPLY))

    anchors = get_anchors(store, "我研究长上下文推理和 RAG，不关心医学影像。", llm)

    assert [a.name for a in anchors.positive] == ["长上下文推理", "检索增强生成"]
    assert [a.name for a in anchors.negative] == ["医学影像"]
    assert anchors.model == "deepseek-chat"
    assert "我研究长上下文推理" in user_prompt(fake.requests[0])
    assert (tmp_path / ANCHORS_PATH).is_file()

    again = get_anchors(store, "我研究长上下文推理和 RAG，不关心医学影像。", llm)
    assert again == anchors
    assert len(fake.requests) == 1  # 第二次命中缓存


def test_regenerates_when_interests_change(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    llm, fake = fake_llm(reply_with(ANCHORS_REPLY))

    get_anchors(store, "版本一", llm)
    get_anchors(store, "版本二", llm)

    assert len(fake.requests) == 2


def test_regenerates_when_cache_is_corrupted(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    store.write_text(ANCHORS_PATH, '{"unexpected": true}')
    llm, fake = fake_llm(reply_with(ANCHORS_REPLY))

    anchors = get_anchors(store, "兴趣", llm)

    assert len(anchors.anchors) == 3
    assert len(fake.requests) == 1


def test_rejects_only_negative_anchors(tmp_path: Path) -> None:
    only_negative = json.dumps({"anchors": [{"name": "x", "query": "y", "negative": True}]})
    llm, fake = fake_llm(reply_with(only_negative))

    with pytest.raises(LLMOutputError):
        get_anchors(DataStore(tmp_path), "兴趣", llm)
    assert len(fake.requests) == 2  # 校验失败后重试过一次
