"""密钥脱敏。

Webhook 地址本身就是密钥（拿到地址就能往群里发消息），SMTP 密码、收件人地址、Zotero 的 ID 和
Key 也不能公开。公开仓库的 Actions 日志任何人都能看，gleaner-data 分支也是公开的，所以这些值
不能出现在日志、异常信息、运行摘要和回复里：

- 推送渠道和 Zotero 客户端把自己的异常转成不含原始异常的错误信息，并用 redact() 去掉密钥；
- httpx 在 INFO 级别会记录请求的完整 URL（加 -v 时会打印出来），protect() 给 httpx 的日志
  加一个过滤器，把登记过的密钥换成 ***。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

MASK = "***"
# 太短的值（如端口号）不当作密钥，避免把日志里无关的数字也替换掉
_MIN_SECRET_CHARS = 4
# 即使没有登记，也把常见 Webhook 地址中的令牌部分去掉
_TOKEN_RE = re.compile(r"(hook/|webhook/send\?key=|[?&]key=)[^\s&'\"<>]+", re.IGNORECASE)

_secrets: set[str] = set()


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """把 text 中的密钥（secrets 加上 protect() 登记过的）和 Webhook 令牌换成 ***。"""
    values = {value for value in (*secrets, *_secrets) if len(value) >= _MIN_SECRET_CHARS}
    for value in sorted(values, key=len, reverse=True):
        text = text.replace(value, MASK)
    return _TOKEN_RE.sub(rf"\1{MASK}", text)


def protect(*secrets: str | None) -> None:
    """登记密钥：之后 httpx 记录的日志中出现这些值时会被替换掉。"""
    _secrets.update(value for value in secrets if value and len(value) >= _MIN_SECRET_CHARS)
    logger = logging.getLogger("httpx")
    if not any(isinstance(f, _RedactingFilter) for f in logger.filters):
        logger.addFilter(_RedactingFilter())


class _RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        cleaned = redact(message)
        if cleaned != message:
            record.msg, record.args = cleaned, None
        return True
