"""每日推送流水线（docs/architecture.md 3.1）。

同步表情 → 抓取 → 去重与关键词过滤 → 向量粗筛（可选加入 Zotero 锚点）→ LLM 精排 → 生成概览
→ 自动深读前 M 篇 → 发布 Issue → 推送到邮件、飞书、企业微信 → 写入 gleaner-data。

Zotero 文库只在本机缓存，读取失败时用旧缓存或跳过，不影响日报；digest 和统计里只有数量。

自动深读（deep_read.auto_top_m，默认 1 篇）用 deep 档模型，报告以折叠块附在论文评论里，同时写入
reports/ 缓存，之后 /deep 直接返回缓存。获取全文或 LLM 失败、剩余 token 预算不够时跳过，
在运行结果中说明，不影响日报；兜底运行补发评论时从缓存读取报告，不重新生成。

推送渠道只在日报发布成功时发送（没有启用 GitHub Issue 时照样发送，只是不带链接）；
没有达到 min_score 的论文、没有新公告和 dry-run 都不发送。各渠道的结果记在 digest 里，
兜底运行只补发失败的渠道。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Literal, Protocol

from pydantic import ValidationError

from gleaner.config import Config
from gleaner.filters import apply_filters, load_seen, record_seen
from gleaner.llm import BudgetExceededError, LLMClient, LLMError
from gleaner.models import ChannelResult, DeepReport, Digest, RunStats, ScoredPaper
from gleaner.notify import IssueRef
from gleaner.notify.base import Notifier
from gleaner.notify.dispatch import failed_channels, failure_warnings, send_all
from gleaner.pipelines.feedback import (
    REACTION_SYNC_DAYS,
    ReactionReader,
    SyncResult,
    sync_reactions,
)
from gleaner.profile import ensure_queries, get_anchors, load_interests, load_memory
from gleaner.ranking import Embedder, coarse_rank, rerank, summarize
from gleaner.reader import FulltextFetcher, deep_read, load_report, save_report
from gleaner.render.markdown import kilo
from gleaner.sources import ArxivSource, Source
from gleaner.sources.zotero import ZoteroResult
from gleaner.store import DataStore

Status = Literal[
    "published",  # 已发布 Issue
    "preview",  # 没有发布渠道（dry-run），只生成了日报
    "already_published",  # 当天的日报已发布，跳过
    "repaired",  # 当天的日报已发布，补发了上次没发出去的评论或推送
    "no_announcements",  # arXiv 今天没有公告
    "nothing_relevant",  # 没有达到 min_score 的论文，不发布
]

# 判断剩余预算够不够自动深读：输入按 deep_read.max_input_tokens 计，另加这么多输出
AUTO_DEEP_READ_OUTPUT_TOKENS = 4_000

logger = logging.getLogger(__name__)


class Publisher(Protocol):
    def find_existing(self, day: date) -> IssueRef | None: ...

    def publish(
        self, digest: Digest, reports: Mapping[str, DeepReport] | None = None
    ) -> Digest: ...

    def repair(self, digest: Digest, reports: Mapping[str, DeepReport] | None = None) -> Digest: ...

    def close_stale(self, now: datetime | None = None) -> list[int]: ...


class ZoteroSource(Protocol):
    def load(self) -> ZoteroResult:
        """读取 Zotero 锚点；失败时降级，不抛出异常。"""
        ...


@dataclass
class DailyResult:
    status: Status
    digest: Digest | None = None
    existing: IssueRef | None = None
    sampled: bool = False  # 没有新公告，改用了最近提交的论文
    sync: SyncResult = field(default_factory=SyncResult)  # 第 0 步：表情同步
    warnings: list[str] = field(default_factory=list)  # 不影响日报的问题，如表情同步失败
    notified: dict[str, ChannelResult] = field(default_factory=dict)  # 这次发送的渠道及结果
    repaired_comments: int = 0  # 兜底运行补发的论文评论数


DIGEST_DIR = "history/digests"


def digest_path(day: date) -> str:
    return f"{DIGEST_DIR}/{day.isoformat()}.json"


def load_recent_digests(store: DataStore, today: date, days: int) -> list[Digest]:
    """最近 days 天（含今天）保存的日报，从早到晚；格式不对的跳过。"""
    digests = []
    for offset in reversed(range(days)):
        day = today - timedelta(days=offset)
        try:
            digest = store.read_model(digest_path(day), Digest)
        except ValidationError as e:
            logger.warning("%s 格式不正确，跳过：%s", digest_path(day), e)
            continue
        if digest is not None:
            digests.append(digest)
    return digests


def run_daily(
    config: Config,
    *,
    store: DataStore,
    llm: LLMClient,
    embedder: Embedder,
    today: date,
    source: Source | None = None,
    publisher: Publisher | None = None,
    reactions: ReactionReader | None = None,
    owner: str | None = None,
    notifiers: Sequence[Notifier] = (),
    fetcher: FulltextFetcher | None = None,
    zotero: ZoteroSource | None = None,
    force: bool = False,
    sample_size: int | None = None,
    now: datetime | None = None,
) -> DailyResult:
    """运行一次日报流水线。

    reactions、owner：从哪里读取论文评论的表情、只统计谁的；没有时跳过表情同步。
    notifiers：发布后推送摘要的渠道（邮件、飞书、企业微信），dry-run 时为空。
    fetcher：自动深读时获取全文，默认直接请求 arXiv。
    zotero：启用 Zotero 时读取文库锚点（A4）。
    sample_size：没有新公告时，改用最近提交的这么多篇论文（不去重、不记录为已处理）。
    只在没有发布渠道时生效，用于周末等时候试运行、检查配置。
    """
    started = time.monotonic()
    now = now or datetime.now(UTC)
    # 先读兴趣描述：没配置好就立刻失败，不白跑抓取
    interests = load_interests(config.profile.interests_file)

    # 第 0 步放在最前面：当天日报已经发布（兜底运行）或没有新公告时也同步
    sync = SyncResult()
    if reactions is not None and owner:
        digests = load_recent_digests(store, today, REACTION_SYNC_DAYS)
        sync = sync_reactions(
            store, digests, reactions, owner=owner, today=today, now=now, timezone=config.timezone
        )
    warnings = _sync_warnings(sync)

    existing = None if publisher is None or force else publisher.find_existing(today)
    # 没有启用 GitHub Issue 时，按保存的 digest 判断今天是否已经推送过
    pushed = publisher is None and bool(notifiers) and not force and _was_notified(store, today)
    if existing is not None or pushed:
        result = _repair(store, publisher, existing, today, notifiers, now)
        result.sync, result.warnings = sync, warnings + result.warnings
        return result

    source = source or ArxivSource(config.sources.arxiv)
    papers = source.fetch_latest()
    sampled = not papers and bool(sample_size) and publisher is None and not notifiers
    if sampled and sample_size:
        papers = source.fetch_recent(sample_size)
    if not papers:
        return DailyResult("no_announcements", sync=sync, warnings=warnings)

    seen = set() if sampled else load_seen(store, today)
    filtered = apply_filters(papers, seen=seen, config=config.filters)
    if not sampled:
        record_seen(store, [p for p in papers if p.id not in seen], today)

    anchors = get_anchors(store, interests, llm)
    # 反馈锚点、屏蔽主题和学到的偏好；手改过的屏蔽主题缺少英文描述时在这里补上
    memory = load_memory(store, today)
    warnings += ensure_queries(store, memory.muted, llm)
    library = zotero.load() if zotero is not None else None
    if library is not None:
        warnings += library.warnings
    ranking = config.ranking
    candidates = coarse_rank(
        filtered.papers,
        anchors,
        embedder,
        top_k=ranking.coarse_top_k,
        weights=ranking.weights,
        forced_ids=filtered.forced_ids,
        memory=memory,
        half_life_days=ranking.feedback_half_life_days,
        now=now,
        zotero=library.papers if library is not None else (),
    )
    reranked = rerank(
        candidates,
        interests=interests,
        anchors=anchors,
        llm=llm,
        top_n=ranking.final_top_n,
        min_score=ranking.min_score,
        language=config.language,
        preferences=memory.preferences,
        muted_topics=[topic.topic for topic in memory.muted],
    )
    overview = summarize(reranked.selected, llm, config.language)
    reports, deep_read_notes = _auto_deep_read(
        reranked.selected[: config.deep_read.auto_top_m],
        config=config,
        store=store,
        llm=llm,
        fetcher=fetcher,
        interests=interests,
        now=now,
    )
    warnings += deep_read_notes

    usage = llm.tracker.total
    selected_ids = {p.id for p in reranked.selected}
    digest = Digest(
        day=today,
        generated_at=now,
        overview=overview,
        papers=reranked.selected,
        others=[p for p in reranked.ranked if p.id not in selected_ids],
        deep_reads=list(reports),
        stats=RunStats(
            fetched=len(papers),
            duplicates=filtered.duplicates,
            excluded=filtered.excluded,
            candidates=len(candidates),
            selected=len(reranked.selected),
            failed_batches=reranked.failed_batches,
            degraded=reranked.degraded,
            llm_calls=usage.calls,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            duration_seconds=round(time.monotonic() - started, 1),
            feedback=memory.stats(now, ranking.feedback_half_life_days).model_copy(
                update={"synced": sync.records, "sync_error": sync.error}
            ),
            zotero=library.stats if library is not None else None,
            deep_read=len(reports),
            deep_read_notes=deep_read_notes,
        ),
    )

    status: Status = "preview"
    notified: dict[str, ChannelResult] = {}
    if not digest.papers:
        status = "nothing_relevant"
    else:
        if publisher is not None:
            digest = publisher.publish(digest, reports)
            status = "published"
        if notifiers:
            # 发布 Issue 之后再推送，消息里才有 Issue 链接；某个渠道失败不影响其它渠道
            notified = send_all(digest, notifiers, now=now)
            digest = digest.model_copy(update={"notifications": notified})
            warnings += failure_warnings(notified)
            status = "published"
    if publisher is not None:
        publisher.close_stale(now)

    store.write_model(digest_path(today), digest)
    store.append_jsonl(
        f"stats/{today:%Y-%m}.jsonl",
        [
            {
                "day": today.isoformat(),
                "status": status,
                "issue": digest.issue_number,
                **digest.stats.model_dump(),
            }
        ],
    )
    return DailyResult(
        status, digest, sampled=sampled, sync=sync, warnings=warnings, notified=notified
    )


def _auto_deep_read(
    papers: Sequence[ScoredPaper],
    *,
    config: Config,
    store: DataStore,
    llm: LLMClient,
    fetcher: FulltextFetcher | None,
    interests: str,
    now: datetime,
) -> tuple[dict[str, DeepReport], list[str]]:
    """给排在最前的几篇论文生成深读报告并写入缓存；返回报告和没有深读的原因。

    已有缓存的直接用缓存。任何一步失败都只跳过这一篇，不影响日报。
    """
    reports: dict[str, DeepReport] = {}
    notes: list[str] = []
    budget = llm.tracker.budget
    need = config.deep_read.max_input_tokens + AUTO_DEEP_READ_OUTPUT_TOKENS
    for index, paper in enumerate(papers, start=1):
        if (cached := load_report(store, paper.id)) is not None:
            reports[paper.id] = cached
            continue
        used = llm.tracker.total.total_tokens
        if budget is not None and used + need > budget:
            notes.append(
                f"第 {index} 篇没有自动深读：今天已用 {kilo(used)} token，深读预计还要约 "
                f"{kilo(need)}，会超出 llm.daily_token_budget（{kilo(budget)}）"
            )
            continue
        try:
            fulltext = (fetcher or FulltextFetcher()).fetch(paper)
        except Exception as e:  # 全文解析出错也不能影响日报
            logger.warning("自动深读获取 %s 的全文失败：%s", paper.id, e)
            notes.append(f"第 {index} 篇没有自动深读：获取全文失败（{type(e).__name__}）")
            continue
        if not fulltext.sections:
            notes.append(f"第 {index} 篇没有自动深读：没能获取全文和摘要")
            continue
        try:
            report = deep_read(
                paper,
                fulltext,
                llm=llm,
                interests=interests,
                config=config.deep_read,
                language=config.language,
                now=now,
            )
        except BudgetExceededError as e:
            notes.append(f"第 {index} 篇没有自动深读：{e}")
            continue
        except LLMError as e:
            logger.warning("自动深读 %s 失败：%s", paper.id, e)
            notes.append(f"第 {index} 篇自动深读失败：{e}")
            continue
        save_report(store, report)
        reports[paper.id] = report
    return reports, notes


def _sync_warnings(sync: SyncResult) -> list[str]:
    if sync.error is None:
        return []
    written = f"已写入 {sync.records} 条，其余下次再同步"
    return [f"表情同步没有完成（{written}）：{sync.error}"]


def _saved_digest(store: DataStore, today: date) -> Digest | None:
    try:
        return store.read_model(digest_path(today), Digest)
    except ValidationError:
        return None


def _was_notified(store: DataStore, today: date) -> bool:
    saved = _saved_digest(store, today)
    return saved is not None and bool(saved.notifications)


def _repair(
    store: DataStore,
    publisher: Publisher | None,
    existing: IssueRef | None,
    today: date,
    notifiers: Sequence[Notifier],
    now: datetime,
) -> DailyResult:
    """当天日报已发布：按保存的 digest 补发上次没发出去的论文评论，以及推送失败的渠道。

    已经发送成功的渠道不再发送；上次没有启用的渠道也不补发。
    """
    saved = _saved_digest(store, today)
    if saved is None or (existing is not None and saved.issue_number != existing.number):
        return DailyResult("already_published", existing=existing)

    repaired = saved
    if publisher is not None:
        # 补发的评论里也附上自动深读的报告：从缓存读取，不重新生成
        reports = {
            paper_id: report
            for paper_id in saved.deep_reads
            if (report := load_report(store, paper_id)) is not None
        }
        repaired = publisher.repair(saved, reports)
    retry = failed_channels(repaired, notifiers)
    notified = send_all(repaired, retry, now=now) if retry else {}
    if notified:
        notifications = {**repaired.notifications, **notified}
        repaired = repaired.model_copy(update={"notifications": notifications})
    if repaired == saved:
        return DailyResult("already_published", saved, existing)
    store.write_model(digest_path(today), repaired)
    return DailyResult(
        "repaired",
        repaired,
        existing,
        warnings=failure_warnings(notified),
        notified=notified,
        repaired_comments=len(repaired.comment_ids) - len(saved.comment_ids),
    )
