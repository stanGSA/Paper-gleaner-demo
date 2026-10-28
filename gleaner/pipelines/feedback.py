"""日报第 0 步：同步论文评论上的表情（docs/architecture.md 3.1、3.4）。

GitHub 的表情不会触发 Actions 事件，所以每次运行日报时扫描最近 7 天的日报，逐条读取论文评论的
表情，把和上次同步不同的状态（新增的、撤销的）追加为反馈记录。

- 只统计仓库所有者（GITHUB_REPOSITORY_OWNER，不区分大小写）的 👍 / 👎：一个 fork 对应一个用户，
  协作者、机器人和其他人的表情都忽略；
- 一篇论文的状态：有 👍 为 +1，有 👎 为 -1，两个都点了为 0；同一篇论文出现在多份日报里时合并计算；
- 记录的 weight 就是新的状态，撤销时写一条 weight=0 的记录（signal 为被撤销的那个）；
  「上次同步的状态」取自已有的反馈记录，不另存状态文件；
- 失败（限流、网络问题）不影响日报：已经读完的论文照常写入，其余的留到下次，并在运行结果中给出警告。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Protocol

import httpx

from gleaner.models import Digest, Feedback, ScoredPaper
from gleaner.notify import GitHubError
from gleaner.profile import load_feedback, record_feedback
from gleaner.profile.memory import latest_reactions
from gleaner.store import DataStore

REACTION_SYNC_DAYS = 7
_STATES = {"+1": 1, "-1": -1}  # 只统计 👍 和 👎

logger = logging.getLogger(__name__)


class ReactionReader(Protocol):
    def list_reactions(self, comment_id: int) -> list[dict[str, Any]]: ...


@dataclass
class SyncResult:
    comments: int = 0  # 读取了表情的评论数
    records: int = 0  # 新写入的反馈记录数
    error: str | None = None  # 同步中断的原因；None 表示全部读完


def sync_reactions(
    store: DataStore,
    digests: Sequence[Digest],
    reader: ReactionReader,
    *,
    owner: str,
    today: date,
    now: datetime,
    timezone: str,
) -> SyncResult:
    """读取 digests 中论文评论的表情，追加状态有变化的反馈记录。"""
    owner_key = owner.casefold()
    papers: dict[str, ScoredPaper] = {}
    comments: dict[str, list[int]] = {}  # 论文 ID → 它在各份日报中的评论 ID
    for digest in digests:
        by_id = {paper.id: paper for paper in digest.papers}
        for paper_id, comment_id in digest.comment_ids.items():
            if paper_id in by_id:
                papers.setdefault(paper_id, by_id[paper_id])
                comments.setdefault(paper_id, []).append(comment_id)

    result = SyncResult()
    states: dict[str, set[int]] = {}  # 读完了全部评论的论文 → 所有者点过的表情
    for paper_id, comment_ids in comments.items():
        found: set[int] = set()
        read = 0
        for comment_id in comment_ids:
            try:
                reactions = reader.list_reactions(comment_id)
            except GitHubError as e:
                if e.status == 404:  # 评论被删掉了：没有新信息，保留已记录的状态
                    logger.info("论文 %s 的评论 %d 已不存在，跳过", paper_id, comment_id)
                    continue
                result.error = str(e)
                break
            except httpx.HTTPError as e:
                result.error = f"{type(e).__name__}：{e}"
                break
            read += 1
            for reaction in reactions:
                login = str((reaction.get("user") or {}).get("login") or "")
                state = _STATES.get(str(reaction.get("content")))
                if state is not None and login.casefold() == owner_key:
                    found.add(state)
        if result.error is not None:
            break
        result.comments += read
        if read:
            states[paper_id] = found

    previous = latest_reactions(load_feedback(store, today))
    records = []
    for paper_id, found in states.items():
        state = float(sum(found))  # {+1}、{-1}、{+1, -1} → +1、-1、0
        before = previous.get(paper_id, 0.0)
        if state == before:
            continue
        # 撤销时 signal 记被撤销的那个表情
        signal = "up" if (state or before) > 0 else "down"
        paper = papers[paper_id]
        records.append(
            Feedback(
                paper_id=paper_id,
                title=paper.title,
                abstract=paper.abstract,
                signal=signal,
                weight=state,
                source="reaction",
                created_at=now,
            )
        )
    record_feedback(store, records, timezone=timezone)
    result.records = len(records)
    if result.error is not None:
        logger.warning("表情同步中断：%s", result.error)
    return result
