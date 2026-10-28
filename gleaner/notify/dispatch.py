"""创建已启用的推送渠道，并逐个发送。

各渠道相互独立：一个渠道失败（包括意料之外的异常）只记录失败原因，其它渠道照常发送，
日报也照常完成。结果记在 digest.notifications 里，兜底运行据此只补发失败的渠道。
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from datetime import datetime

from gleaner.config import NotifyConfig
from gleaner.models import ChannelResult, Digest
from gleaner.notify.base import (
    CHANNEL_LABELS,
    CHANNELS,
    ChannelName,
    Notifier,
    NotifyConfigError,
    NotifyError,
)
from gleaner.notify.email import EmailNotifier
from gleaner.notify.feishu import FeishuNotifier
from gleaner.notify.wecom import WecomNotifier

_FACTORIES = {"email": EmailNotifier, "feishu": FeishuNotifier, "wecom": WecomNotifier}

logger = logging.getLogger(__name__)


def build_notifiers(
    config: NotifyConfig, channels: Collection[ChannelName] | None = None
) -> tuple[list[Notifier], list[str]]:
    """已启用且配置齐全的渠道，以及缺少配置、这次不发送的说明。

    channels 不为 None 时只取其中的渠道（gleaner notify-test --channel）。
    """
    notifiers: list[Notifier] = []
    problems: list[str] = []
    for channel in CHANNELS:
        if channels is not None and channel not in channels:
            continue
        if not getattr(config, channel).enabled:
            continue
        try:
            notifiers.append(_FACTORIES[channel].from_config(config))
        except NotifyConfigError as e:
            problems.append(f"{CHANNEL_LABELS[channel]}已启用但没有配置好，这次不发送：{e}")
    return notifiers, problems


def send_all(
    digest: Digest, notifiers: Sequence[Notifier], *, now: datetime, test: bool = False
) -> dict[str, ChannelResult]:
    """逐个渠道发送，返回各渠道的结果；不抛出异常。"""
    results: dict[str, ChannelResult] = {}
    for notifier in notifiers:
        label = CHANNEL_LABELS[notifier.name]
        try:
            notifier.send(digest, test=test)
        except NotifyError as e:
            logger.warning("%s推送失败：%s", label, e)
            results[notifier.name] = ChannelResult(status="failed", error=str(e), at=now)
        except Exception as e:  # 渠道的 bug 也不能影响日报；异常信息可能带着密钥，只记类型
            logger.warning("%s推送时出现意外错误（%s）", label, type(e).__name__)
            error = f"意外错误（{type(e).__name__}）"
            results[notifier.name] = ChannelResult(status="failed", error=error, at=now)
        else:
            results[notifier.name] = ChannelResult(status="sent", at=now)
    return results


def failed_channels(digest: Digest, notifiers: Sequence[Notifier]) -> list[Notifier]:
    """上次发送失败、这次需要补发的渠道；上次没有启用的渠道不补发。"""
    return [
        notifier
        for notifier in notifiers
        if (result := digest.notifications.get(notifier.name)) and result.status == "failed"
    ]


def failure_warnings(results: dict[str, ChannelResult]) -> list[str]:
    return [
        f"{CHANNEL_LABELS.get(name, name)}推送失败：{result.error}"
        for name, result in results.items()
        if result.status == "failed"
    ]


def results_line(results: dict[str, ChannelResult]) -> str:
    """如「推送：邮件 ✓ · 飞书 ✗」。"""
    marks = [
        f"{CHANNEL_LABELS.get(name, name)} {'✓' if result.status == 'sent' else '✗'}"
        for name, result in results.items()
    ]
    return "推送：" + " · ".join(marks)
