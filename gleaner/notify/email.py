"""邮件推送：HTML 简版加纯文本备用版，内容是标题、相关理由、TL;DR 和日报 Issue 的链接。

用 smtplib 发送：465 端口走 SSL，其它端口（通常是 587）先连接再 STARTTLS。
QQ、163 邮箱的 SMTP_PASSWORD 填授权码，Gmail 填应用专用密码，都不是登录密码。
标题和正文都用 UTF-8；外部文本（论文标题、理由、概览）在 HTML 中一律转义。
"""

from __future__ import annotations

import html
import logging
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid

from gleaner.config import EmailNotifyConfig, NotifyConfig
from gleaner.models import Digest
from gleaner.notify.base import (
    DEFAULT_SMTP_PORT,
    NO_ISSUE_NOTE,
    TEST_PREFIX,
    ChannelName,
    NotifyConfigError,
    NotifyError,
    heading,
    issue_link,
    missing_settings,
    score_text,
    shorten,
)
from gleaner.redaction import redact

SSL_PORT = 465
SENDER_NAME = "Paper Gleaner"
MAX_OVERVIEW_CHARS = 1_000
MAX_REASON_CHARS = 500

logger = logging.getLogger(__name__)


class EmailNotifier:
    name: ChannelName = "email"

    def __init__(
        self,
        config: EmailNotifyConfig,
        *,
        timeout: float = 30.0,
    ) -> None:
        assert config.smtp_host and config.smtp_user and config.smtp_password
        self.config = config
        self.host = config.smtp_host
        self.port = config.smtp_port or DEFAULT_SMTP_PORT
        self.user = config.smtp_user.get_secret_value()
        self._password = config.smtp_password.get_secret_value()
        self.recipients = list(config.to)
        self.timeout = timeout

    @classmethod
    def from_config(cls, config: NotifyConfig) -> EmailNotifier:
        if missing := missing_settings(config, "email"):
            raise NotifyConfigError(f"缺少环境变量 {'、'.join(missing)}")
        return cls(config.email)

    @property
    def secrets(self) -> list[str]:
        """不能出现在错误信息中的值：密码、账号和收件人地址。"""
        return [self._password, self.user, *self.recipients]

    def send(self, digest: Digest, *, test: bool = False) -> None:
        message = build_message(
            digest,
            sender=self.user,
            recipients=self.recipients,
            subject_prefix=self.config.subject_prefix,
            test=test,
        )
        context = ssl.create_default_context()
        try:
            if self.port == SSL_PORT:
                server = smtplib.SMTP_SSL(
                    self.host, self.port, timeout=self.timeout, context=context
                )
            else:
                server = smtplib.SMTP(self.host, self.port, timeout=self.timeout)
            with server:
                if self.port != SSL_PORT:
                    server.starttls(context=context)
                server.login(self.user, self._password)
                server.send_message(message)
        except smtplib.SMTPAuthenticationError:
            raise NotifyError(
                "SMTP 登录失败：请检查 SMTP_USER，以及 SMTP_PASSWORD 是否为邮箱的授权码"
                "（QQ、163 邮箱）或应用专用密码（Gmail），而不是登录密码"
            ) from None
        except (smtplib.SMTPException, OSError) as e:
            # 不保留原始异常：其中可能带着收件人地址
            reason = redact(f"{type(e).__name__}: {e}", self.secrets)
            raise NotifyError(f"通过 {self.host}:{self.port} 发送邮件失败（{reason}）") from None


def build_message(
    digest: Digest,
    *,
    sender: str,
    recipients: list[str],
    subject_prefix: str,
    test: bool = False,
) -> EmailMessage:
    message = EmailMessage()
    subject = f"{subject_prefix} · {digest.day.isoformat()} · {len(digest.papers)} 篇"
    message["Subject"] = (TEST_PREFIX if test else "") + subject
    message["From"] = formataddr((SENDER_NAME, sender))
    message["To"] = ", ".join(recipients)
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid(domain=sender.rpartition("@")[2] or None)
    # 正文用 base64 编码：不依赖服务器支持 8BITMIME，中文也不会乱码
    message.set_content(render_text(digest, test=test), charset="utf-8", cte="base64")
    message.add_alternative(
        render_html(digest, test=test), subtype="html", charset="utf-8", cte="base64"
    )
    return message


def render_text(digest: Digest, *, test: bool = False) -> str:
    lines = [heading(digest, test=test), ""]
    if digest.overview:
        lines += [shorten(digest.overview, MAX_OVERVIEW_CHARS), ""]
    for index, paper in enumerate(digest.papers, start=1):
        score = score_text(paper)
        lines.append(f"[{index}] {paper.title}" + (f"（{score}）" if score else ""))
        if paper.reason:
            lines.append(f"为什么和你相关：{shorten(paper.reason, MAX_REASON_CHARS)}")
        if paper.tldr:
            lines.append(f"TL;DR：{shorten(paper.tldr, MAX_REASON_CHARS)}")
        lines += [paper.url, ""]
    if url := issue_link(digest):
        lines.append(f"去 Issue 讨论（评论 /deep、/compare 追问，点 👍/👎 反馈）：{url}")
    else:
        lines.append(NO_ISSUE_NOTE)
    return "\n".join(lines) + "\n"


def render_html(digest: Digest, *, test: bool = False) -> str:
    """简单的 HTML 版本：外部文本全部经过 html.escape，链接只用 arXiv 和日报 Issue 的地址。"""
    parts = [
        '<div style="font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;'
        'max-width:680px;line-height:1.6;color:#1f2328">',
        f'<h2 style="margin:0 0 12px">{_escape(heading(digest, test=test))}</h2>',
    ]
    if digest.overview:
        overview = _escape(shorten(digest.overview, MAX_OVERVIEW_CHARS))
        parts.append(
            '<p style="margin:0 0 16px;padding:8px 12px;background:#f6f8fa;'
            f'border-left:4px solid #d0d7de">{overview}</p>'
        )
    for index, paper in enumerate(digest.papers, start=1):
        score = score_text(paper)
        score_html = f' <span style="color:#57606a">（{score}）</span>' if score else ""
        parts.append(
            f'<h3 style="margin:16px 0 4px;font-size:16px">[{index}] '
            f'<a href="{_escape(paper.url)}">{_escape(paper.title)}</a>{score_html}</h3>'
        )
        if paper.reason:
            reason = _escape(shorten(paper.reason, MAX_REASON_CHARS))
            parts.append(f'<p style="margin:4px 0">🎯 <b>为什么和你相关</b>：{reason}</p>')
        if paper.tldr:
            tldr = _escape(shorten(paper.tldr, MAX_REASON_CHARS))
            parts.append(f'<p style="margin:4px 0">📝 <b>TL;DR</b>：{tldr}</p>')
    if url := issue_link(digest):
        parts.append(
            f'<p style="margin:24px 0 0"><a href="{_escape(url)}" style="display:inline-block;'
            "padding:8px 16px;background:#1f6feb;color:#fff;border-radius:6px;"
            'text-decoration:none">去 Issue 讨论</a></p>'
            '<p style="color:#57606a;font-size:13px">在 Issue 中评论 /deep、/compare 追问，'
            "给论文评论点 👍/👎 反馈。</p>"
        )
    else:
        parts.append(f'<p style="color:#57606a;font-size:13px">{_escape(NO_ISSUE_NOTE)}</p>')
    parts.append("</div>")
    return "\n".join(parts) + "\n"


def _escape(text: str) -> str:
    return html.escape(text, quote=True)
