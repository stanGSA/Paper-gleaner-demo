"""推送渠道的统一接口（docs/architecture.md 第 9 节）。

交互统一在 GitHub Issue 中进行：邮件、飞书、企业微信只发日报摘要和 Issue 链接。
各渠道相互独立，某个渠道失败不影响其它渠道，也不影响日报（见 notify/dispatch.py）。

标题、相关理由、TL;DR 和概览都是外部内容，每个渠道自己做转义：邮件里不能注入 HTML，
飞书里不能 @ 人（<at>），企业微信里不能写 <@userid>。

渠道的密钥（Webhook 地址、SMTP 密码等）只从环境变量读取，失败时抛出的 NotifyError
已经脱敏，可以直接写进日志、运行摘要和 gleaner-data。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import UTC, date, datetime
from typing import Any, Literal, Protocol

import httpx

from gleaner.config import NotifyConfig
from gleaner.models import Digest, ScoredPaper
from gleaner.redaction import redact
from gleaner.render import issue_title

ChannelName = Literal["email", "feishu", "wecom"]
CHANNELS: tuple[ChannelName, ...] = ("email", "feishu", "wecom")
CHANNEL_LABELS: dict[str, str] = {"email": "邮件", "feishu": "飞书", "wecom": "企业微信"}
TEST_PREFIX = "【测试】"
DEFAULT_SMTP_PORT = 465  # 没有设置 SMTP_PORT 时用 SSL 端口
NO_ISSUE_NOTE = "没有启用 GitHub Issue，这份日报只通过推送渠道发送，不能评论交互。"
INTERACT_NOTE = "在 Issue 中评论 /deep、/compare 追问，给论文评论点 👍/👎 反馈"
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")


class NotifyError(Exception):
    """推送失败。信息已经脱敏，不含 Webhook 地址、密码和收件人。"""


class NotifyConfigError(NotifyError):
    """渠道已启用，但缺少需要的环境变量。"""


class Notifier(Protocol):
    name: ChannelName

    def send(self, digest: Digest, *, test: bool = False) -> None:
        """发送一份日报摘要，失败时抛出 NotifyError。test=True 时在标题中注明是测试消息。"""
        ...


def missing_settings(config: NotifyConfig, channel: ChannelName) -> list[str]:
    """渠道缺少的环境变量（GitHub Secrets）。"""
    if channel == "email":
        email = config.email
        required = {
            "SMTP_HOST": email.smtp_host,
            "SMTP_USER": email.smtp_user,
            "SMTP_PASSWORD": email.smtp_password,
            "EMAIL_TO": email.to,
        }
    elif channel == "feishu":
        required = {"FEISHU_WEBHOOK": config.feishu.webhook}
    else:
        required = {"WECOM_WEBHOOK": config.wecom.webhook}
    return [name for name, value in required.items() if not value]


def heading(digest: Digest, *, test: bool = False) -> str:
    """消息标题，与日报 Issue 的标题相同，如「📚 Paper Gleaner · 2026-10-25 · 6 篇」。"""
    return (TEST_PREFIX if test else "") + issue_title(digest)


def score_text(paper: ScoredPaper) -> str:
    return "" if paper.llm_score is None else f"{paper.llm_score}/10"


def shorten(text: str, limit: int) -> str:
    """合并空白（含换行）后截断到 limit 个字符。"""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def issue_link(digest: Digest) -> str | None:
    """日报 Issue 的链接；只接受不含空白和引号的 http(s) 地址。"""
    url = digest.issue_url or ""
    if url.startswith(("https://", "http://")) and not any(c in url for c in " \t\n\"'<>"):
        return url
    return None


def arxiv_url(paper: ScoredPaper) -> str | None:
    """论文的 arXiv 链接；ID 不是 arXiv 格式（如手改过的 digest）时返回 None，不拼进链接。"""
    return paper.url if _ARXIV_ID_RE.match(paper.id) else None


def dump_json(payload: dict[str, Any]) -> bytes:
    """紧凑的 UTF-8 JSON：中文不转成 \\uXXXX，按这个结果计算请求体的大小。"""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def post_webhook(
    url: str,
    payload: dict[str, Any],
    *,
    label: str,
    secrets: Iterable[str],
    client: httpx.Client | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """向群机器人 Webhook 发送 JSON，返回解析后的返回体（调用方再检查其中的错误码）。

    httpx 的异常信息会带上请求地址，而地址本身就是密钥：只保留脱敏后的说明，不保留原始异常。
    """
    request = {
        "content": dump_json(payload),
        "headers": {"Content-Type": "application/json; charset=utf-8"},
    }
    try:
        if client is not None:
            response = client.post(url, **request)
        else:
            with httpx.Client(timeout=timeout) as owned:
                response = owned.post(url, **request)
    except httpx.HTTPError as e:
        reason = redact(f"{type(e).__name__}: {e}", secrets)
        raise NotifyError(f"请求{label}机器人失败（{reason}）") from None
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise NotifyError(f"{label}机器人返回 HTTP {response.status_code}，返回体不是 JSON")
    return body


def service_message(value: object, secrets: Iterable[str] = ()) -> str:
    """服务端返回的错误说明：截短并脱敏后再写进错误信息。"""
    return redact(shorten(str(value or ""), 120), secrets)


def sample_digest(day: date, issue_url: str | None = None) -> Digest:
    """gleaner notify-test 发送的示例日报。"""
    paper = ScoredPaper(
        id="1706.03762",
        title="Attention Is All You Need",
        abstract="The dominant sequence transduction models are based on complex recurrent "
        "or convolutional neural networks.",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        categories=["cs.CL", "cs.LG"],
        llm_score=8,
        reason="这是一条测试消息：收到它说明这个渠道配置成功了。",
        tldr="提出完全基于注意力机制的 Transformer。",
    )
    return Digest(
        day=day,
        generated_at=datetime.now(UTC),
        overview="Paper Gleaner 推送渠道测试：正式的日报会在这里用三句话概括当天的论文。",
        papers=[paper],
        issue_url=issue_url,
    )
