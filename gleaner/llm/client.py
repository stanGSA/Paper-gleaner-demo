"""OpenAI 兼容的 LLM 客户端。

只依赖 OpenAI 兼容接口，改 base_url 和模型名即可切换 DeepSeek、Qwen、Kimi、GLM 等服务。
网络错误、限流和 5xx 的重试交给 openai SDK（max_retries）；本模块负责模型分级、
token 用量统计与预算，以及结构化输出的校验和重试。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from gleaner.config import LLMConfig

Tier = Literal["fast", "deep"]
Message = dict[str, str]
T = TypeVar("T", bound=BaseModel)

_THINK_RE = re.compile(r"\A\s*<think>.*?</think>", re.DOTALL)
_JSON_INSTRUCTION = (
    "请只输出一个 JSON 对象，不要输出任何解释或 Markdown 代码块。"
    "JSON 必须符合以下 JSON Schema：\n{schema}"
)


class LLMError(Exception):
    """调用 LLM 失败。"""


class BudgetExceededError(LLMError):
    """token 用量已达到预算上限。"""


class LLMOutputError(LLMError):
    """模型输出为空或不符合要求的格式。"""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class UsageTracker:
    """按模型累计 token 用量，并在超出预算时阻止新的调用。"""

    budget: int | None = None
    by_model: dict[str, Usage] = field(default_factory=dict)

    @property
    def total(self) -> Usage:
        total = Usage()
        for usage in self.by_model.values():
            total.prompt_tokens += usage.prompt_tokens
            total.completion_tokens += usage.completion_tokens
            total.calls += usage.calls
        return total

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        usage = self.by_model.setdefault(model, Usage())
        usage.prompt_tokens += prompt_tokens
        usage.completion_tokens += completion_tokens
        usage.calls += 1

    def ensure_within_budget(self) -> None:
        if self.budget is not None and self.total.total_tokens >= self.budget:
            raise BudgetExceededError(
                f"token 用量 {self.total.total_tokens} 已达到预算 {self.budget}"
            )


class LLMClient:
    def __init__(
        self,
        config: LLMConfig,
        *,
        tracker: UsageTracker | None = None,
        client: Any = None,
    ) -> None:
        self.config = config
        self.tracker = tracker or UsageTracker(budget=config.daily_token_budget)
        if client is None:
            if config.api_key is None:
                raise LLMError("未设置 LLM_API_KEY")
            client = openai.OpenAI(
                api_key=config.api_key.get_secret_value(),
                base_url=config.base_url,
                timeout=config.timeout_seconds,
                max_retries=config.max_retries,
            )
        self._client = client

    def model_for(self, tier: Tier) -> str:
        return getattr(self.config.models, tier)

    def chat(
        self,
        messages: list[Message],
        *,
        tier: Tier = "fast",
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> str:
        """发送一次对话请求，返回模型输出的文本。

        timeout 覆盖配置中的 llm.timeout_seconds，用于深读这类输入长、输出慢的请求。
        """
        self.tracker.ensure_within_budget()
        model = self.model_for(tier)

        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if timeout is not None:
            kwargs["timeout"] = timeout

        try:
            response = self._client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
        except openai.OpenAIError as e:
            raise LLMError(f"调用 {model} 失败：{e}") from e

        if response.usage is not None:
            self.tracker.record(
                model, response.usage.prompt_tokens, response.usage.completion_tokens
            )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise LLMOutputError(f"{model} 返回了空内容")
        return content

    def chat_json(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        tier: Tier = "fast",
        json_mode: bool = True,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> T:
        """请求结构化输出并用 pydantic 校验；不合格时把错误反馈给模型重试一次。

        部分推理模型不支持 JSON mode，此时传 ``json_mode=False``，仍会按 Schema 校验。
        """
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        instruction = _JSON_INSTRUCTION.format(schema=schema_json)
        # 合并进调用方的 system 消息：部分服务商不接受多条 system 消息
        if messages and messages[0]["role"] == "system":
            system = {"role": "system", "content": f"{messages[0]['content']}\n\n{instruction}"}
            conversation: list[Message] = [system, *messages[1:]]
        else:
            conversation = [{"role": "system", "content": instruction}, *messages]

        for attempt in range(2):
            content = self.chat(
                conversation,
                tier=tier,
                json_mode=json_mode,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            try:
                return schema.model_validate_json(_strip_code_fence(content))
            except ValidationError as e:
                if attempt == 1:
                    raise LLMOutputError(f"模型输出不符合 Schema：{e}") from e
                conversation = [
                    *conversation,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": f"上面的输出不符合要求：{e}\n请只输出符合 Schema 的 JSON。",
                    },
                ]
        raise AssertionError("unreachable")


def _strip_code_fence(text: str) -> str:
    """去掉推理模型输出的 <think> 思考过程，以及有时会包在 JSON 外面的 ``` 代码块标记。"""
    text = _THINK_RE.sub("", text).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.removesuffix("```").strip()
    return text
