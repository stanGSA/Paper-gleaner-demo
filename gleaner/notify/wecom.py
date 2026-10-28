"""企业微信推送：群机器人的 markdown 消息。

- content 最多 4096 字节（UTF-8 编码后），只发前 max_papers 篇（默认 5）和日报 Issue 的链接；
  放不下时先截短理由，再减少篇数；
- 出错时也返回 HTTP 200，要看返回体里的 errcode，0 才是成功；
- 外部文本（标题、理由、概览）先合并成一行，不能借换行写出标题或引用；再在 "<" 后插入零宽空格、
  把 [ ] * ` 换成全角或形近字符：不能写 <@userid> 提醒群成员，也不能插入链接、<font> 标签或加粗。
"""

from __future__ import annotations

import httpx

from gleaner.config import NotifyConfig, WecomNotifyConfig
from gleaner.models import Digest
from gleaner.notify.base import (
    NO_ISSUE_NOTE,
    ChannelName,
    NotifyConfigError,
    NotifyError,
    arxiv_url,
    heading,
    issue_link,
    missing_settings,
    post_webhook,
    score_text,
    service_message,
    shorten,
)
from gleaner.redaction import protect

MAX_CONTENT_BYTES = 4096
# 先按这个长度截短理由，放不下时再改用更短的
REASON_LIMITS = (120, 60)
MAX_OVERVIEW_CHARS = 150
_ZERO_WIDTH_SPACE = "​"
_MARKDOWN_CHARS = str.maketrans({"[": "［", "]": "］", "*": "＊", "`": "'"})
_HINTS = {
    93000: "Webhook 地址无效：检查 WECOM_WEBHOOK，或机器人是否已被移出群聊",
    45009: "发送太频繁，被企业微信限流（每个机器人每分钟最多 20 条）",
}


class WecomNotifier:
    name: ChannelName = "wecom"

    def __init__(self, config: WecomNotifyConfig, *, client: httpx.Client | None = None) -> None:
        assert config.webhook is not None
        self._webhook = config.webhook.get_secret_value()
        self.max_papers = config.max_papers
        self._client = client
        protect(self._webhook)

    @classmethod
    def from_config(cls, config: NotifyConfig) -> WecomNotifier:
        if missing := missing_settings(config, "wecom"):
            raise NotifyConfigError(f"缺少环境变量 {'、'.join(missing)}")
        return cls(config.wecom)

    @property
    def secrets(self) -> list[str]:
        return [self._webhook]

    def send(self, digest: Digest, *, test: bool = False) -> None:
        content = build_markdown(digest, max_papers=self.max_papers, test=test)
        payload = {"msgtype": "markdown", "markdown": {"content": content}}
        body = post_webhook(
            self._webhook, payload, label="企业微信", secrets=self.secrets, client=self._client
        )
        code = body.get("errcode")
        if code != 0:
            message = service_message(body.get("errmsg"), self.secrets)
            hint = _HINTS.get(code) if isinstance(code, int) else None
            detail = f"{hint}（{message}）" if hint else message
            raise NotifyError(f"企业微信返回错误 {code}：{detail}")


def build_markdown(digest: Digest, *, max_papers: int, test: bool = False) -> str:
    """不超过 4096 字节的 markdown：先截短理由，再减少篇数，最后硬截断兜底。"""
    count = min(len(digest.papers), max_papers)
    while True:
        for limit in REASON_LIMITS:
            content = _render(digest, count, limit, test)
            if len(content.encode("utf-8")) <= MAX_CONTENT_BYTES:
                return content
        if count == 0:
            return content.encode("utf-8")[:MAX_CONTENT_BYTES].decode("utf-8", errors="ignore")
        count -= 1


def _render(digest: Digest, count: int, reason_limit: int, test: bool) -> str:
    lines = [f"### {_text(heading(digest, test=test), 120)}"]
    if digest.overview:
        lines.append(f"> {_text(digest.overview, MAX_OVERVIEW_CHARS)}")
    for index, paper in enumerate(digest.papers[:count], start=1):
        score = score_text(paper)
        score_tag = f' <font color="comment">{score}</font>' if score else ""
        lines += ["", f"**[{index}] {_text(paper.title, 200)}**{score_tag}"]
        if text := paper.reason or paper.tldr:
            lines.append(_text(text, reason_limit))
        if url := arxiv_url(paper):
            lines.append(f"[arXiv]({url})")
    lines.append("")
    if (rest := len(digest.papers) - count) > 0:
        lines.append(f"另有 {rest} 篇，请到 Issue 中查看。")
    if url := issue_link(digest):
        lines.append(f"[去 Issue 讨论]({url})：评论 /deep、/compare 追问，点 👍/👎 反馈")
    else:
        lines.append(NO_ISSUE_NOTE)
    return "\n".join(lines)


def _text(text: str, limit: int) -> str:
    """外部文本：合并成一行、截短，再去掉 markdown 和标签的作用。"""
    text = shorten(text, limit).translate(_MARKDOWN_CHARS)
    return text.replace("<", "<" + _ZERO_WIDTH_SPACE)
