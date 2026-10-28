from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from pydantic import BaseModel, SecretStr

from gleaner.config import LLMConfig
from gleaner.llm import BudgetExceededError, LLMClient, LLMError, LLMOutputError, UsageTracker


class FakeCompletions:
    """按顺序返回预设回复，并记录每次请求的参数。"""

    def __init__(self, replies: list[str | Exception], usage: tuple[int, int] = (10, 5)) -> None:
        self.replies = list(replies)
        self.usage = usage
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        prompt_tokens, completion_tokens = self.usage
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        )


def make_client(replies: list[str | Exception], **config: Any) -> tuple[LLMClient, FakeCompletions]:
    completions = FakeCompletions(replies)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return LLMClient(LLMConfig(**config), client=fake), completions


class Verdict(BaseModel):
    score: int
    reason: str


def test_chat_uses_model_for_tier() -> None:
    client, completions = make_client(["a", "b"])

    assert client.chat([{"role": "user", "content": "hi"}]) == "a"
    client.chat([{"role": "user", "content": "hi"}], tier="deep", max_tokens=100)

    assert completions.requests[0]["model"] == "deepseek-chat"
    assert "max_tokens" not in completions.requests[0]
    assert completions.requests[1]["model"] == "deepseek-reasoner"
    assert completions.requests[1]["max_tokens"] == 100


def test_chat_records_usage() -> None:
    client, _ = make_client(["a", "b", "c"])
    client.chat([{"role": "user", "content": "1"}])
    client.chat([{"role": "user", "content": "2"}])
    client.chat([{"role": "user", "content": "3"}], tier="deep")

    assert client.tracker.by_model["deepseek-chat"].calls == 2
    assert client.tracker.by_model["deepseek-chat"].total_tokens == 30
    assert client.tracker.total.total_tokens == 45
    assert client.tracker.total.calls == 3


def test_budget_blocks_further_calls() -> None:
    client, completions = make_client(["a", "b"], daily_token_budget=15)
    client.chat([{"role": "user", "content": "1"}])

    with pytest.raises(BudgetExceededError):
        client.chat([{"role": "user", "content": "2"}])
    assert len(completions.requests) == 1


def test_shared_tracker_across_clients() -> None:
    tracker = UsageTracker(budget=None)
    for _ in range(2):
        completions = FakeCompletions(["ok"])
        fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        LLMClient(LLMConfig(), client=fake, tracker=tracker).chat(
            [{"role": "user", "content": "x"}]
        )
    assert tracker.total.calls == 2


def test_empty_reply_raises() -> None:
    client, _ = make_client([""])
    with pytest.raises(LLMOutputError):
        client.chat([{"role": "user", "content": "hi"}])


def test_openai_error_is_wrapped() -> None:
    request = httpx.Request("POST", "https://api.example/chat/completions")
    client, _ = make_client([openai.APIConnectionError(request=request)])
    with pytest.raises(LLMError, match="deepseek-chat"):
        client.chat([{"role": "user", "content": "hi"}])


def test_chat_json_parses_fenced_output() -> None:
    client, completions = make_client(['```json\n{"score": 8, "reason": "相关"}\n```'])

    verdict = client.chat_json([{"role": "user", "content": "打分"}], Verdict)

    assert verdict == Verdict(score=8, reason="相关")
    request = completions.requests[0]
    assert request["response_format"] == {"type": "json_object"}
    assert request["messages"][0]["role"] == "system"
    assert '"score"' in request["messages"][0]["content"]


def test_chat_json_retries_once_with_feedback() -> None:
    client, completions = make_client(['{"score": "high"}', '{"score": 7, "reason": "ok"}'])

    verdict = client.chat_json([{"role": "user", "content": "打分"}], Verdict)

    assert verdict.score == 7
    retry_messages = completions.requests[1]["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": '{"score": "high"}'}
    assert "不符合要求" in retry_messages[-1]["content"]


def test_chat_json_gives_up_after_second_failure() -> None:
    client, _ = make_client(["not json", "still not json"])
    with pytest.raises(LLMOutputError):
        client.chat_json([{"role": "user", "content": "打分"}], Verdict)


def test_chat_json_without_json_mode() -> None:
    client, completions = make_client(['{"score": 3, "reason": "弱相关"}'])
    client.chat_json([{"role": "user", "content": "打分"}], Verdict, tier="deep", json_mode=False)
    assert "response_format" not in completions.requests[0]


def test_missing_api_key() -> None:
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        LLMClient(LLMConfig())


def test_builds_openai_client_from_config() -> None:
    config = LLMConfig(
        api_key=SecretStr("sk-test"), base_url="https://api.example/v1", max_retries=5
    )
    client = LLMClient(config)
    assert isinstance(client._client, openai.OpenAI)
    assert str(client._client.base_url).startswith("https://api.example/v1")
    assert client._client.max_retries == 5


def test_chat_json_merges_system_prompt() -> None:
    client, completions = make_client(['{"score": 5, "reason": "ok"}'])
    messages = [
        {"role": "system", "content": "你是科研助理。"},
        {"role": "user", "content": "打分"},
    ]

    client.chat_json(messages, Verdict, temperature=0.2)

    sent = completions.requests[0]["messages"]
    assert [m["role"] for m in sent] == ["system", "user"]
    assert sent[0]["content"].startswith("你是科研助理。")
    assert "JSON Schema" in sent[0]["content"]
    assert completions.requests[0]["temperature"] == 0.2


def test_chat_passes_request_timeout() -> None:
    client, completions = make_client(["a", "b"])
    client.chat([{"role": "user", "content": "hi"}])
    client.chat([{"role": "user", "content": "hi"}], tier="deep", timeout=180)

    assert "timeout" not in completions.requests[0]  # 默认沿用 llm.timeout_seconds
    assert completions.requests[1]["timeout"] == 180
