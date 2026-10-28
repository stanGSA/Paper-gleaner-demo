import logging

import pytest

from gleaner import redaction
from gleaner.redaction import MASK, protect, redact

FEISHU = "https://open.feishu.cn/open-apis/bot/v2/hook/0f1e2d3c-token"
WECOM = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=wecom-secret-key"


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(redaction, "_secrets", set())


def test_redacts_given_secrets() -> None:
    text = f"login failed for me@qq.com with auth-code-123 at {FEISHU}"
    cleaned = redact(text, ["me@qq.com", "auth-code-123", FEISHU])
    assert cleaned == f"login failed for {MASK} with {MASK} at {MASK}"


def test_redacts_webhook_tokens_without_registration() -> None:
    assert "0f1e2d3c-token" not in redact(f"Client error for url '{FEISHU}'")
    assert "wecom-secret-key" not in redact(f"POST {WECOM} failed")
    assert redact(WECOM).endswith(f"key={MASK}")


def test_short_values_are_not_treated_as_secrets() -> None:
    assert redact("port 465 refused", ["465"]) == "port 465 refused"


def test_httpx_log_records_are_redacted(caplog: pytest.LogCaptureFixture) -> None:
    protect("zotero-key-xyz", None, "")
    caplog.set_level(logging.INFO, logger="httpx")

    logging.getLogger("httpx").info('HTTP Request: GET %s "%s"', "/users/zotero-key-xyz", "200")

    assert "zotero-key-xyz" not in caplog.text
    assert f"/users/{MASK}" in caplog.text
    protect("another-secret")  # 只装一个过滤器
    assert len([f for f in logging.getLogger("httpx").filters]) == 1
