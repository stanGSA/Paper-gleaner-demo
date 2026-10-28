"""Issue 指令交互流水线（docs/architecture.md 3.2、8.2）。

读取 issue_comment 事件 → 给指令评论加 👀 → 解析并执行指令 → 回复结果（超长时拆成多条）
→ 有数据写入时提交 gleaner-data → 加 🚀。出错时回复简短原因和 /help 提示，并加 😕。

interact.yml 的 if 条件已经过滤了作者身份、标签和指令前缀，这里再校验一遍：
本地用 --event 调试时同样安全，workflow 被改错时也不会响应陌生人的评论。
LLM 只负责生成文本，发评论、写数据等副作用都在这里由代码完成。
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel

from gleaner.agent import (
    Command,
    CommandError,
    Context,
    Services,
    UnknownCommandError,
    execute,
    parse_command,
)
from gleaner.config import Config
from gleaner.notify import GitHubError
from gleaner.render.markdown import split_comment
from gleaner.render.replies import error_reply, help_reply, short_error, unknown_reply
from gleaner.store import DataBranchError, DataStore

ALLOWED_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
REQUIRED_LABEL = "gleaner"
# 日报 Issue 正文末尾的标记，见 render.issue_body
_DIGEST_MARKER = re.compile(r"<!-- gleaner:digest day=(\d{4}-\d{2}-\d{2}) -->")
_TITLE_DAY = re.compile(r"Paper Gleaner · (\d{4}-\d{2}-\d{2})")

Status = Literal[
    "ignored",  # 不是需要处理的评论（PR 评论、不是指令、作者没有权限等）
    "done",  # 已回复
    "rejected",  # 指令写错了或无法执行，已回复原因
    "failed",  # 内部错误（已尽量回复）
]

logger = logging.getLogger(__name__)


class EventError(Exception):
    """事件文件缺失或不是 issue_comment 事件。"""


class Replier(Protocol):
    """发评论和加表情；GitHubClient 实现了这两个方法，本地试运行时换成打印。"""

    def create_comment(self, issue_number: int, body: str) -> int: ...

    def add_reaction(self, comment_id: int, content: str) -> None: ...


class Branch(Protocol):
    def checkout(self) -> DataStore: ...

    def commit_and_push(self, message: str) -> bool: ...


class CommentEvent(BaseModel):
    action: str
    repository: str
    issue_number: int
    issue_title: str
    issue_body: str
    labels: list[str]
    is_pull_request: bool
    comment_id: int
    comment_body: str
    author: str
    author_association: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> CommentEvent:
        try:
            issue, comment = payload["issue"], payload["comment"]
            return cls(
                action=payload.get("action") or "",
                repository=(payload.get("repository") or {}).get("full_name") or "",
                issue_number=issue["number"],
                issue_title=issue.get("title") or "",
                issue_body=issue.get("body") or "",
                labels=[label["name"] for label in issue.get("labels") or []],
                is_pull_request=bool(issue.get("pull_request")),
                comment_id=comment["id"],
                comment_body=comment.get("body") or "",
                author=(comment.get("user") or {}).get("login") or "",
                author_association=comment.get("author_association") or "",
            )
        except (KeyError, TypeError, ValueError) as e:
            raise EventError(f"不是 issue_comment 事件：缺少 {e}") from e


@dataclass
class InteractResult:
    status: Status
    command: Command | None = None
    reply_ids: list[int] = field(default_factory=list)
    message: str = ""


def load_event(path: Path) -> CommentEvent:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise EventError(f"找不到事件文件 {path}") from e
    except OSError as e:
        raise EventError(f"读取事件文件 {path} 失败：{e}") from e
    except json.JSONDecodeError as e:
        raise EventError(f"事件文件 {path} 不是合法的 JSON：{e}") from e
    if not isinstance(payload, dict):
        raise EventError(f"事件文件 {path} 的顶层必须是对象")
    return CommentEvent.from_payload(payload)


def ignore_reason(event: CommentEvent) -> str | None:
    """不该处理的事件返回原因；与 interact.yml 的 if 条件保持一致。"""
    if event.action != "created":
        return f"不处理 {event.action or '未知'} 事件，只处理新评论"
    if event.is_pull_request:
        return "不处理 Pull Request 下的评论"
    if not event.comment_body.startswith("/"):
        return "评论不是以 / 开头的指令"
    if event.author_association not in ALLOWED_ASSOCIATIONS:
        return f"{event.author} 不是仓库所有者或协作者（{event.author_association}）"
    if REQUIRED_LABEL not in event.labels:
        return f"Issue 没有 {REQUIRED_LABEL} 标签"
    return None


def issue_day(event: CommentEvent) -> date | None:
    """Issue 对应的日报日期：优先取正文中的标记，其次取标题中的日期。"""
    for pattern, text in ((_DIGEST_MARKER, event.issue_body), (_TITLE_DAY, event.issue_title)):
        if match := pattern.search(text):
            try:
                return date.fromisoformat(match.group(1))
            except ValueError:
                continue
    return None


def handle_comment(
    event: CommentEvent,
    *,
    config: Config,
    replier: Replier,
    branch: Branch,
    services: Services,
    persist: bool = True,
    now: datetime | None = None,
    pause_seconds: float = 1.0,
) -> InteractResult:
    """处理一条指令评论。persist=False 时不写缓存和反馈、不提交数据分支（本地试运行）。"""
    if reason := ignore_reason(event):
        return InteractResult("ignored", message=reason)
    _react(replier, event.comment_id, "eyes")

    command: Command | None = None
    try:
        command = parse_command(event.comment_body)
        if command.name == "help":
            body, store = help_reply(), None
        else:
            store = branch.checkout()
            ctx = Context(
                config,
                services,
                store,
                day=issue_day(event),
                issue_number=event.issue_number,
                persist=persist,
                now=now or datetime.now(UTC),
                comments=_comment_reader(replier, event.issue_number),
                comment_id=event.comment_id,
            )
            body = execute(command, ctx)
    except UnknownCommandError as e:
        return _reject(event, replier, unknown_reply(e.name), command, pause_seconds)
    except CommandError as e:
        return _reject(event, replier, error_reply(str(e)), command, pause_seconds)
    except Exception as e:  # 意料之外的错误也要回复，不能让用户一直等着
        expected = isinstance(e, (DataBranchError, GitHubError, httpx.HTTPError))
        if not expected:
            logger.exception("处理指令时出错")
        reason = f"处理指令时出错（{type(e).__name__}）：{short_error(e)}"
        result = _reject(event, replier, error_reply(reason), command, pause_seconds)
        result.status = "failed"
        return result

    assert command is not None
    try:
        reply_ids = post_reply(replier, event.issue_number, body, pause_seconds=pause_seconds)
    except (GitHubError, httpx.HTTPError) as e:
        _react(replier, event.comment_id, "confused")
        return InteractResult("failed", command, message=f"回复发表失败：{e}")

    message = f"已回复 {len(reply_ids)} 条评论"
    if store is not None and persist:
        try:
            # 提交说明里不写用户的原始参数，避免其中的 #123 之类关联到别的 Issue
            if branch.commit_and_push(f"data: /{command.name} from issue {event.issue_number}"):
                message += "，数据已提交到 gleaner-data"
        except DataBranchError as e:
            # 回复已经发出，缓存没写进去只影响下次是否重新生成
            logger.warning("提交 gleaner-data 失败：%s", e)
            message += f"；提交 gleaner-data 失败：{e}"
    _react(replier, event.comment_id, "rocket")
    return InteractResult("done", command, reply_ids, message)


def report_failure(
    event: CommentEvent, replier: Replier, reason: str, *, pause_seconds: float = 1.0
) -> InteractResult:
    """还没开始执行就失败了（如配置文件有误）：符合条件的指令仍然回复原因。"""
    if ignore_reason(event):
        return InteractResult("ignored", message=reason)
    result = _reject(event, replier, error_reply(reason), None, pause_seconds)
    result.status = "failed"
    return result


def post_reply(
    replier: Replier, issue_number: int, body: str, *, pause_seconds: float = 1.0
) -> list[int]:
    """发表回复，超过评论长度上限时拆成多条，按顺序发出。"""
    ids = []
    for index, part in enumerate(split_comment(body)):
        if index:
            time.sleep(pause_seconds)  # GitHub 建议连续创建内容时间隔约 1 秒
        ids.append(replier.create_comment(issue_number, part))
    return ids


def _reject(
    event: CommentEvent,
    replier: Replier,
    body: str,
    command: Command | None,
    pause_seconds: float,
) -> InteractResult:
    try:
        reply_ids = post_reply(replier, event.issue_number, body, pause_seconds=pause_seconds)
    except (GitHubError, httpx.HTTPError) as e:
        logger.warning("回复错误信息失败：%s", e)
        reply_ids = []
    _react(replier, event.comment_id, "confused")
    first_line = body.splitlines()[0] if body else ""
    return InteractResult("rejected", command, reply_ids, first_line)


def _comment_reader(
    replier: Replier, issue_number: int
) -> Callable[[], list[dict[str, Any]]] | None:
    """/ask 的上下文：用到时才读取 Issue 的评论（GitHubClient.list_comments 返回的评论
    带有 author_association）；试运行时的 replier 不能读评论，返回 None。"""
    list_comments = getattr(replier, "list_comments", None)
    if not callable(list_comments):
        return None
    return lambda: list_comments(issue_number)


def _react(replier: Replier, comment_id: int, content: str) -> None:
    """加表情只是提示进度，失败不影响指令本身。"""
    try:
        replier.add_reaction(comment_id, content)
    except (GitHubError, httpx.HTTPError) as e:
        logger.warning("给评论 %d 加 %s 失败：%s", comment_id, content, e)
