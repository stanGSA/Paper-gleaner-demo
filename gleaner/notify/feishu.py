"""飞书推送：自定义机器人，消息卡片列出论文，附「去 Issue 讨论」按钮。

- 签名校验：以 timestamp + "\\n" + secret 为密钥，对空消息做 HMAC-SHA256，再 base64（飞书文档中的
  算法）；timestamp 是秒级时间戳，与飞书服务器相差超过 1 小时会被拒绝（code 19021）。没有设置
  FEISHU_SECRET 时不签名，适用于没有开启签名校验的机器人；
- 出错时也可能返回 HTTP 200，要看返回体里的 code（旧版接口为 StatusCode），0 才是成功；
- 外部文本（标题、理由、TL;DR、概览）只放进 plain_text 元素，不经 Markdown 解析，不能 @ 人（<at>）
  或插入链接；保险起见再在 "<" 后插入零宽空格。lark_md 元素里只有程序拼出的 arXiv 链接；
- 请求体不能超过 20 KB：放不下时减少列出的篇数，注明其余的去 Issue 查看。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from collections.abc import Callable
from typing import Any

import httpx

from gleaner.config import FeishuNotifyConfig, NotifyConfig
from gleaner.models import Digest
from gleaner.notify.base import (
    INTERACT_NOTE,
    NO_ISSUE_NOTE,
    ChannelName,
    NotifyConfigError,
    NotifyError,
    arxiv_url,
    dump_json,
    heading,
    issue_link,
    missing_settings,
    post_webhook,
    score_text,
    service_message,
    shorten,
)
from gleaner.redaction import protect

# 飞书限制请求体不超过 20 KB；签名字段和余量另留出约 1 KB
MAX_PAYLOAD_BYTES = 19 * 1024
MAX_OVERVIEW_CHARS = 500
MAX_REASON_CHARS = 300
_ZERO_WIDTH_SPACE = "​"
# 常见错误码的处理建议
_HINTS = {
    19021: "签名校验失败：检查 FEISHU_SECRET 是否与机器人安全设置中的密钥一致，以及运行环境的时钟",
    19022: "IP 白名单校验失败：GitHub Actions 的出口 IP 不固定，请关闭 IP 白名单，改用签名校验",
    19024: "关键词校验失败：请改用签名校验，或把「Paper Gleaner」设为自定义关键词",
    11232: "发送太频繁，被飞书限流",
    9499: "请求格式不对或 Webhook 地址有误",
}


def sign(timestamp: str, secret: str) -> str:
    """飞书自定义机器人的签名：HMAC-SHA256(key=timestamp + "\\n" + secret, msg="")，再 base64。"""
    key = f"{timestamp}\n{secret}".encode()
    return base64.b64encode(hmac.new(key, digestmod=hashlib.sha256).digest()).decode()


class FeishuNotifier:
    name: ChannelName = "feishu"

    def __init__(
        self,
        config: FeishuNotifyConfig,
        *,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        assert config.webhook is not None
        self._webhook = config.webhook.get_secret_value()
        self._secret = config.secret.get_secret_value() if config.secret else None
        self.max_papers = config.max_papers
        self._client = client
        self._clock = clock
        protect(self._webhook, self._secret)

    @classmethod
    def from_config(cls, config: NotifyConfig) -> FeishuNotifier:
        if missing := missing_settings(config, "feishu"):
            raise NotifyConfigError(f"缺少环境变量 {'、'.join(missing)}")
        return cls(config.feishu)

    @property
    def secrets(self) -> list[str]:
        return [value for value in (self._webhook, self._secret) if value]

    def send(self, digest: Digest, *, test: bool = False) -> None:
        payload: dict[str, Any] = build_card(digest, max_papers=self.max_papers, test=test)
        if self._secret is not None:
            timestamp = str(int(self._clock()))
            payload = {"timestamp": timestamp, "sign": sign(timestamp, self._secret), **payload}
        body = post_webhook(
            self._webhook, payload, label="飞书", secrets=self.secrets, client=self._client
        )
        code = body.get("code", body.get("StatusCode"))
        if code != 0:
            message = service_message(body.get("msg") or body.get("StatusMessage"), self.secrets)
            hint = _HINTS.get(code) if isinstance(code, int) else None
            detail = f"{hint}（{message}）" if hint else message
            raise NotifyError(f"飞书返回错误 {code}：{detail}")


def build_card(digest: Digest, *, max_papers: int, test: bool = False) -> dict[str, Any]:
    """消息卡片；超过请求体上限时减少列出的篇数。"""
    shown = min(len(digest.papers), max_papers)
    while True:
        payload = {"msg_type": "interactive", "card": _card(digest, shown, test)}
        if shown == 0 or len(dump_json(payload)) <= MAX_PAYLOAD_BYTES:
            return payload
        shown -= 1


def _card(digest: Digest, shown: int, test: bool) -> dict[str, Any]:
    elements: list[dict[str, Any]] = []
    if digest.overview:
        elements.append(_plain_div(shorten(digest.overview, MAX_OVERVIEW_CHARS)))
    for index, paper in enumerate(digest.papers[:shown], start=1):
        elements.append({"tag": "hr"})
        score = score_text(paper)
        lines = [f"[{index}] {shorten(paper.title, 300)}" + (f"（{score}）" if score else "")]
        if paper.reason:
            lines.append(f"🎯 {shorten(paper.reason, MAX_REASON_CHARS)}")
        if paper.tldr:
            lines.append(f"📝 {shorten(paper.tldr, MAX_REASON_CHARS)}")
        elements.append(_plain_div("\n".join(lines)))
        if url := arxiv_url(paper):
            links = f"[arXiv]({url}) · [PDF]({paper.pdf_url})"
            elements.append({"tag": "div", "text": {"tag": "lark_md", "content": links}})
    elements.append({"tag": "hr"})
    if (rest := len(digest.papers) - shown) > 0:
        elements.append(_plain_div(f"另有 {rest} 篇，请到 Issue 中查看。"))
    if url := issue_link(digest):
        button = {
            "tag": "button",
            "text": {"tag": "plain_text", "content": "去 Issue 讨论"},
            "type": "primary",
            "url": url,
        }
        elements.append({"tag": "action", "actions": [button]})
        elements.append(_note(INTERACT_NOTE))
    else:
        elements.append(_note(NO_ISSUE_NOTE))
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "blue",
            "title": {"tag": "plain_text", "content": _plain(heading(digest, test=test))},
        },
        "elements": elements,
    }


def _plain(text: str) -> str:
    """外部文本放进 plain_text 前，在 "<" 后插入零宽空格：即使被当成标记解析，也构不成 <at>。"""
    return text.replace("<", "<" + _ZERO_WIDTH_SPACE)


def _plain_div(text: str) -> dict[str, Any]:
    return {"tag": "div", "text": {"tag": "plain_text", "content": _plain(text)}}


def _note(text: str) -> dict[str, Any]:
    return {"tag": "note", "elements": [{"tag": "plain_text", "content": _plain(text)}]}
