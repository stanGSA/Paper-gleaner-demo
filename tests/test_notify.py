import json
import smtplib
from datetime import UTC, date, datetime
from email import message_from_bytes, policy
from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr

from gleaner.config import NotifyConfig
from gleaner.models import Digest, ScoredPaper
from gleaner.notify.base import NotifyError, sample_digest
from gleaner.notify.dispatch import build_notifiers, failed_channels, send_all
from gleaner.notify.email import EmailNotifier, build_message
from gleaner.notify.feishu import MAX_PAYLOAD_BYTES, FeishuNotifier, build_card, sign
from gleaner.notify.wecom import MAX_CONTENT_BYTES, WecomNotifier, build_markdown
from tests.conftest import FakeNotifier

FEISHU = "https://open.feishu.cn/open-apis/bot/v2/hook/feishu-token-123"
WECOM = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=wecom-key-456"
NOW = datetime(2026, 10, 26, 1, 40, tzinfo=UTC)
EVIL = "<at id=all></at> <@all> <script>alert(1)</script> [点我](https://evil.example) **粗**"


def paper(n: int, **fields: Any) -> ScoredPaper:
    values = {
        "id": f"2610.0000{n}",
        "title": f"Paper {n}: 长上下文推理",
        "abstract": "We study things.",
        "llm_score": 8,
        "reason": "压缩 KV cache，与你的长上下文推理方向直接相关。",
        "tldr": "显存降低 4 倍。",
    } | fields
    return ScoredPaper(**values)


def digest(papers: list[ScoredPaper] | None = None, **fields: Any) -> Digest:
    values = {
        "day": date(2026, 10, 26),
        "generated_at": NOW,
        "overview": "今天以长上下文为主。",
        "papers": papers if papers is not None else [paper(1), paper(2, llm_score=None)],
        "issue_number": 5,
        "issue_url": "https://github.com/o/r/issues/5",
    } | fields
    return Digest(**values)


def notify_config(**env: str) -> NotifyConfig:
    config = NotifyConfig.model_validate(
        {"email": {"enabled": True}, "feishu": {"enabled": True}, "wecom": {"enabled": True}}
    )
    if "webhook" in env:
        config.feishu.webhook = SecretStr(env["webhook"])
    if "secret" in env:
        config.feishu.secret = SecretStr(env["secret"])
    if "wecom" in env:
        config.wecom.webhook = SecretStr(env["wecom"])
    if "smtp" in env:
        config.email.smtp_host = "smtp.qq.com"
        config.email.smtp_port = int(env["smtp"])
        config.email.smtp_user = SecretStr("me@qq.com")
        config.email.smtp_password = SecretStr("auth-code-789")
        config.email.to = ["alice@example.com", "bob@example.com"]
    return config


# --- 邮件 -----------------------------------------------------------------------


class FakeSMTP:
    """代替 smtplib.SMTP / SMTP_SSL，记录调用；error 不为空时在 send_message 时抛出。"""

    instances: list["FakeSMTP"] = []
    error: Exception | None = None

    def __init__(self, host: str, port: int, timeout: float | None = None, context=None) -> None:
        self.host, self.port, self.context = host, port, context
        self.calls: list[Any] = []
        self.messages: list[Any] = []
        FakeSMTP.instances.append(self)

    def __enter__(self) -> "FakeSMTP":
        return self

    def __exit__(self, *exc: object) -> None:
        self.calls.append("quit")

    def starttls(self, context=None) -> None:
        self.calls.append("starttls")

    def login(self, user: str, password: str) -> None:
        self.calls.append(("login", user, password))

    def send_message(self, message) -> None:
        if FakeSMTP.error is not None:
            raise FakeSMTP.error
        self.messages.append(message)


@pytest.fixture
def smtp(monkeypatch: pytest.MonkeyPatch) -> type[FakeSMTP]:
    FakeSMTP.instances, FakeSMTP.error = [], None
    ssl_class = type("FakeSMTPSSL", (FakeSMTP,), {"ssl": True})
    monkeypatch.setattr(smtplib, "SMTP_SSL", ssl_class)
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    return FakeSMTP


def parse(message) -> Any:
    return message_from_bytes(message.as_bytes(), policy=policy.default)


def test_email_message_is_utf8_with_html_and_text() -> None:
    message = build_message(
        digest(),
        sender="me@qq.com",
        recipients=["a@example.com", "b@example.com"],
        subject_prefix="📚 论文推送",
    )
    raw = message.as_bytes()
    raw.decode("ascii")  # 头部经过 RFC 2047 编码、正文 base64：整封邮件都是 ASCII
    parsed = parse(message)

    assert parsed["Subject"] == "📚 论文推送 · 2026-10-26 · 2 篇"
    assert parsed["To"] == "a@example.com, b@example.com"
    assert "me@qq.com" in parsed["From"]
    text = parsed.get_body(("plain",)).get_content()
    html = parsed.get_body(("html",)).get_content()
    assert "[1] Paper 1: 长上下文推理（8/10）" in text
    assert "为什么和你相关：压缩 KV cache" in text
    assert "TL;DR：显存降低 4 倍。" in text
    assert "https://github.com/o/r/issues/5" in text
    assert '<a href="https://arxiv.org/abs/2610.00001">Paper 1: 长上下文推理</a>' in html
    assert 'href="https://github.com/o/r/issues/5"' in html
    assert "去 Issue 讨论" in html


def test_email_escapes_external_html() -> None:
    message = build_message(
        digest([paper(1, title=EVIL, reason=EVIL)], overview=EVIL),
        sender="me@qq.com",
        recipients=["a@example.com"],
        subject_prefix="P",
        test=True,
    )
    parsed = parse(message)
    html = parsed.get_body(("html",)).get_content()

    assert parsed["Subject"].startswith("【测试】P · ")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert 'href="https://evil.example"' not in html


def test_email_without_issue_link() -> None:
    message = build_message(
        digest(issue_url=None), sender="me@qq.com", recipients=["a@example.com"], subject_prefix="P"
    )
    text = parse(message).get_body(("plain",)).get_content()
    assert "没有启用 GitHub Issue" in text


def test_email_sends_over_ssl(smtp: type[FakeSMTP]) -> None:
    notifier = EmailNotifier.from_config(notify_config(smtp="465"))

    notifier.send(digest())

    [server] = smtp.instances
    assert (server.host, server.port, getattr(server, "ssl", False)) == ("smtp.qq.com", 465, True)
    assert server.context is not None
    assert server.calls == [("login", "me@qq.com", "auth-code-789"), "quit"]
    assert len(server.messages) == 1


def test_email_uses_starttls_on_587(smtp: type[FakeSMTP]) -> None:
    EmailNotifier.from_config(notify_config(smtp="587")).send(digest())

    [server] = smtp.instances
    assert not getattr(server, "ssl", False)
    assert server.calls[:2] == ["starttls", ("login", "me@qq.com", "auth-code-789")]


def test_email_authentication_error(smtp: type[FakeSMTP]) -> None:
    smtp.error = smtplib.SMTPAuthenticationError(535, b"Login fail. auth-code-789")

    with pytest.raises(NotifyError, match="授权码") as info:
        EmailNotifier.from_config(notify_config(smtp="465")).send(digest())
    assert "auth-code-789" not in str(info.value)


def test_email_errors_do_not_leak_addresses_or_password(smtp: type[FakeSMTP]) -> None:
    smtp.error = smtplib.SMTPRecipientsRefused(
        {"alice@example.com": (550, b"no such user"), "bob@example.com": (550, b"auth-code-789")}
    )

    with pytest.raises(NotifyError) as info:
        EmailNotifier.from_config(notify_config(smtp="465")).send(digest())

    message = str(info.value)
    assert "smtp.qq.com:465" in message
    for secret in ("alice@example.com", "bob@example.com", "auth-code-789", "me@qq.com"):
        assert secret not in message
    assert info.value.__cause__ is None and info.value.__suppress_context__


# --- 飞书 -----------------------------------------------------------------------


def test_feishu_signature() -> None:
    # 以 timestamp + "\n" + secret 为密钥、对空消息做 HMAC-SHA256，再 base64
    assert sign("1599360473", "demo-secret") == "3/MaVZ8JLIy4TUG+7KSFJqvUkTKd+HWY8g+56DZWq8s="


@respx.mock
def test_feishu_sends_signed_card() -> None:
    route = respx.post(FEISHU).mock(
        return_value=httpx.Response(200, json={"code": 0, "data": {}, "msg": "success"})
    )
    notifier = FeishuNotifier(
        notify_config(webhook=FEISHU, secret="demo-secret").feishu, clock=lambda: 1599360473.9
    )

    notifier.send(digest())

    body = json.loads(route.calls.last.request.content)
    assert body["timestamp"] == "1599360473"
    assert body["sign"] == "3/MaVZ8JLIy4TUG+7KSFJqvUkTKd+HWY8g+56DZWq8s="
    assert body["msg_type"] == "interactive"
    card = body["card"]
    assert card["header"]["title"]["content"] == "📚 Paper Gleaner · 2026-10-26 · 2 篇"
    texts = [e["text"]["content"] for e in card["elements"] if e["tag"] == "div"]
    assert (
        "[1] Paper 1: 长上下文推理（8/10）\n🎯 压缩 KV cache，与你的长上下文推理方向直接相关。"
        in (texts[1])
    )
    assert "[arXiv](https://arxiv.org/abs/2610.00001)" in texts[2]
    [action] = [e for e in card["elements"] if e["tag"] == "action"]
    assert action["actions"][0]["url"] == "https://github.com/o/r/issues/5"
    assert action["actions"][0]["text"]["content"] == "去 Issue 讨论"
    assert "charset=utf-8" in route.calls.last.request.headers["content-type"]


@respx.mock
def test_feishu_without_secret_is_unsigned_and_accepts_legacy_response() -> None:
    route = respx.post(FEISHU).mock(
        return_value=httpx.Response(200, json={"Extra": None, "StatusCode": 0})
    )
    FeishuNotifier(notify_config(webhook=FEISHU).feishu).send(digest(issue_url=None))

    body = json.loads(route.calls.last.request.content)
    assert "sign" not in body and "timestamp" not in body
    notes = [e for e in body["card"]["elements"] if e["tag"] == "note"]
    assert "没有启用 GitHub Issue" in notes[-1]["elements"][0]["content"]
    assert not [e for e in body["card"]["elements"] if e["tag"] == "action"]


def test_feishu_external_text_cannot_mention() -> None:
    card = build_card(digest([paper(1, title=EVIL, reason=EVIL)], overview=EVIL), max_papers=10)
    text = json.dumps(card, ensure_ascii=False)

    assert "<at" not in text
    assert "<\u200bat id=all>" in text  # 插入了零宽空格
    lark_md = [
        e["text"]["content"]
        for e in card["card"]["elements"]
        if e["tag"] == "div" and e["text"]["tag"] == "lark_md"
    ]
    assert lark_md == [
        "[arXiv](https://arxiv.org/abs/2610.00001) · [PDF](https://arxiv.org/pdf/2610.00001)"
    ]


def test_feishu_card_fits_size_limit() -> None:
    long_reason = "很长的理由，" * 60
    papers = [paper(i, reason=long_reason, tldr=long_reason) for i in range(1, 10)] * 3
    card = build_card(digest(papers), max_papers=30)

    size = len(json.dumps(card, ensure_ascii=False, separators=(",", ":")).encode())
    assert size <= MAX_PAYLOAD_BYTES
    texts = [e["text"]["content"] for e in card["card"]["elements"] if e["tag"] == "div"]
    assert any(text.startswith("另有 ") and "请到 Issue 中查看" in text for text in texts)


def test_feishu_card_respects_max_papers() -> None:
    card = build_card(digest([paper(i) for i in range(1, 6)]), max_papers=2)
    titles = [
        e["text"]["content"]
        for e in card["card"]["elements"]
        if e["tag"] == "div" and e["text"]["content"].startswith("[")
    ]
    assert len([t for t in titles if not t.startswith("[arXiv]")]) == 2
    assert "另有 3 篇，请到 Issue 中查看。" in json.dumps(card, ensure_ascii=False)


@respx.mock
def test_feishu_error_code_with_http_200() -> None:
    respx.post(FEISHU).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 19021,
                "data": {},
                "msg": "sign match fail or timestamp is not within one hour from current time",
            },
        )
    )
    notifier = FeishuNotifier(notify_config(webhook=FEISHU, secret="s3cret").feishu)

    with pytest.raises(NotifyError, match="飞书返回错误 19021：签名校验失败") as info:
        notifier.send(digest())
    assert "sign match fail" in str(info.value)
    assert "feishu-token-123" not in str(info.value)


@respx.mock
def test_feishu_network_error_hides_webhook() -> None:
    respx.post(FEISHU).mock(side_effect=httpx.ConnectError(f"cannot connect to {FEISHU}"))

    with pytest.raises(NotifyError, match="请求飞书机器人失败") as info:
        FeishuNotifier(notify_config(webhook=FEISHU).feishu).send(digest())
    assert "feishu-token-123" not in str(info.value)
    assert info.value.__cause__ is None


@respx.mock
def test_feishu_non_json_response() -> None:
    respx.post(FEISHU).mock(return_value=httpx.Response(502, text="<html>bad gateway</html>"))

    with pytest.raises(NotifyError, match="HTTP 502"):
        FeishuNotifier(notify_config(webhook=FEISHU).feishu).send(digest())


# --- 企业微信 -------------------------------------------------------------------


@respx.mock
def test_wecom_sends_markdown() -> None:
    route = respx.post(WECOM).mock(
        return_value=httpx.Response(200, json={"errcode": 0, "errmsg": "ok"})
    )

    WecomNotifier(notify_config(wecom=WECOM).wecom).send(digest())

    body = json.loads(route.calls.last.request.content)
    assert body["msgtype"] == "markdown"
    content = body["markdown"]["content"]
    assert content.startswith("### 📚 Paper Gleaner · 2026-10-26 · 2 篇\n> 今天以长上下文为主。")
    assert '**[1] Paper 1: 长上下文推理** <font color="comment">8/10</font>' in content
    assert "[arXiv](https://arxiv.org/abs/2610.00001)" in content
    assert "[去 Issue 讨论](https://github.com/o/r/issues/5)" in content


def test_wecom_top_n_and_byte_limit() -> None:
    long_title = "很长的中文标题" * 30
    papers = [paper(i, title=long_title, reason="理由" * 200) for i in range(1, 9)]

    content = build_markdown(digest(papers), max_papers=5)

    assert len(content.encode("utf-8")) <= MAX_CONTENT_BYTES
    assert "**[6]" not in content
    assert "另有 " in content and "请到 Issue 中查看" in content
    assert content.endswith("点 👍/👎 反馈")


def test_wecom_external_text_cannot_mention_or_link() -> None:
    content = build_markdown(
        digest([paper(1, title=EVIL, reason=EVIL)], overview=EVIL + "\n### 伪造标题"), max_papers=5
    )

    assert "<@" not in content and "<at" not in content and "<script" not in content
    assert "](https://evil.example)" not in content
    assert "\n### 伪造标题" not in content
    assert "＊＊粗＊＊" in content


@respx.mock
def test_wecom_error_code_with_http_200() -> None:
    respx.post(WECOM).mock(
        return_value=httpx.Response(
            200,
            json={
                "errcode": 93000,
                "errmsg": f"invalid webhook url, hint: [1729], from ip: 1.2.3.4, url: {WECOM}",
            },
        )
    )

    with pytest.raises(NotifyError, match="企业微信返回错误 93000：Webhook 地址无效") as info:
        WecomNotifier(notify_config(wecom=WECOM).wecom).send(digest())
    assert "wecom-key-456" not in str(info.value)


# --- 分发 -----------------------------------------------------------------------


def test_build_notifiers_skips_disabled_and_reports_missing_settings() -> None:
    config = notify_config(webhook=FEISHU)
    config.wecom.enabled = False

    notifiers, problems = build_notifiers(config)

    assert [n.name for n in notifiers] == ["feishu"]
    [problem] = problems
    assert problem.startswith("邮件已启用但没有配置好，这次不发送：缺少环境变量 SMTP_HOST")

    only, problems = build_notifiers(config, ["feishu"])
    assert [n.name for n in only] == ["feishu"] and problems == []


def test_send_all_isolates_failures() -> None:
    email = FakeNotifier("email", NotifyError("SMTP 登录失败"))
    feishu = FakeNotifier("feishu", RuntimeError(f"boom {FEISHU}"))
    wecom = FakeNotifier("wecom")

    results = send_all(digest(), [email, feishu, wecom], now=NOW)

    assert results["email"].status == "failed" and results["email"].error == "SMTP 登录失败"
    assert results["feishu"].error == "意外错误（RuntimeError）"  # 不保留可能带着密钥的异常信息
    assert results["wecom"].status == "sent" and results["wecom"].at == NOW
    assert len(wecom.sent) == 1


def test_failed_channels_only_retries_failures() -> None:
    saved = digest(
        notifications={
            "email": {"status": "sent", "at": NOW},
            "feishu": {"status": "failed", "error": "x", "at": NOW},
        }
    )
    notifiers = [FakeNotifier("email"), FakeNotifier("feishu"), FakeNotifier("wecom")]

    assert [n.name for n in failed_channels(saved, notifiers)] == ["feishu"]


def test_sample_digest_for_notify_test() -> None:
    sample = sample_digest(date(2026, 10, 26), "https://github.com/o/r/issues?q=label%3Adaily")
    assert sample.papers[0].id == "1706.03762"
    assert "测试消息" in sample.papers[0].reason
