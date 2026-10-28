"""Issue 评论指令的解析与分发（docs/architecture.md 3.2）。

指令写在评论第一行，例如 "/deep 3 --force"、"/compare 3 5"、"/like 3 有开源代码"、"/mute 医学影像"；
只有 /ask 的问题可以写多行（"/ask" 之后的全部内容都是问题）。
<ref> 支持三种写法：当前 Issue 中的序号、arXiv ID（可带版本号）、arXiv 链接。
序号按这个 Issue 对应的那一天解析，所以在旧日报下评论同样有效；
日报推荐的论文编号 1…N，/more 列出的候选接着编号 N+1…。

这里只负责「做什么」：返回回复的 Markdown，并把报告缓存、反馈写进 DataStore；
发评论、加表情、提交数据分支由 pipelines/interact.py 完成。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import cached_property
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from gleaner.config import Config
from gleaner.llm import LLMClient, LLMError
from gleaner.models import Digest, Paper
from gleaner.pipelines.daily import DIGEST_DIR, digest_path
from gleaner.profile import (
    ProfileError,
    clean_note,
    command_feedback,
    load_feedback,
    load_interests,
    record_feedback,
)
from gleaner.profile.memory import CommandSignal
from gleaner.profile.muted import (
    MAX_TOPIC_CHARS,
    MuteEvent,
    append_event,
    describe_topic,
    find_topic,
    load_muted,
)
from gleaner.reader import (
    MAX_COMPARE_PAPERS,
    FullText,
    FulltextFetcher,
    compare_papers,
    compare_related,
    deep_read,
    load_report,
    save_report,
)
from gleaner.render.replies import (
    ask_fallback_reply,
    ask_reply,
    code,
    compare_fallback_reply,
    compare_reply,
    deep_fallback_reply,
    deep_reply,
    feedback_reply,
    help_reply,
    more_reply,
    mute_reply,
    muted_list_reply,
    muted_topics_text,
    related_fallback_reply,
    related_reply,
    unmute_reply,
)
from gleaner.sources import ArxivSource, SemanticScholar, SourceError
from gleaner.store import DataStore

COMMANDS = ("deep", "compare", "more", "like", "dislike", "mute", "unmute", "ask", "help")
DEFAULT_MORE = 5
MAX_MORE = 20
MAX_QUESTION_CHARS = 1_000

_FORCE_FLAGS = {"--force", "-f"}
# 参数之间可以用空格、逗号或顿号分隔："/compare 3 5"、"/compare 3,5"、"/compare 3、5"
_ARG_SEPARATOR = re.compile(r"[\s,，、]+")
# 指令名和其后的原文
_NAME_AND_REST = re.compile(r"[\s,，、]*([^\s,，、]*)[\s,，、]*(.*)", re.DOTALL)
# /like、/dislike 的 <ref> 和原因："/like 3 有开源代码"、"/like 3，有开源代码"、"/like 3：…"
_REF_AND_NOTE = re.compile(r"([^\s,，、：]+)[\s,，、：]*(.*)", re.DOTALL)
_ARXIV_ID_RE = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?$", re.IGNORECASE)
_ARXIV_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.|export\.)?arxiv\.org/(?:abs|pdf|html)/"
    r"(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?/?(?:[?#].*)?$",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)

Ref = int | str  # 序号，或不带版本号的 arXiv ID


class CommandError(Exception):
    """指令写错了或无法执行（参数不对、序号越界、找不到论文等）；原因会回复给用户。"""


class UnknownCommandError(CommandError):
    def __init__(self, name: str) -> None:
        super().__init__(f"不认识的指令 /{name}")
        self.name = name


@dataclass(frozen=True)
class Command:
    name: str
    args: tuple[str, ...] = ()
    force: bool = False
    # 指令名之后的原文（/ask 是之后的全部内容，其它指令只取第一行）：
    # /like 的原因、/mute 的主题、/ask 的问题取自这里，保留其中的标点和空格
    rest: str = field(default="", compare=False)

    @property
    def text(self) -> str:
        parts = [f"/{self.name}", *self.args, *(["--force"] if self.force else [])]
        return " ".join(parts)


@dataclass
class Services:
    """指令用到的外部服务；测试和本地调试时可以替换。"""

    arxiv: ArxivSource
    fetcher: FulltextFetcher
    s2: SemanticScholar
    # 用到时才创建：没有配置 LLM_API_KEY 时，/more、/help 照常可用
    llm_factory: Callable[[], LLMClient]


@dataclass
class Context:
    config: Config
    services: Services
    store: DataStore
    day: date | None = None  # 当前 Issue 对应的日报日期；None 表示不是日报 Issue
    issue_number: int | None = None
    persist: bool = True  # 写入报告缓存、反馈和屏蔽主题；本地调试时为 False
    now: datetime = field(default_factory=lambda: datetime.now(UTC))
    # 读取当前 Issue 的评论，作为 /ask 的上下文；None 表示没有（本地运行）
    comments: Callable[[], list[dict[str, Any]]] | None = None
    comment_id: int | None = None  # 触发指令的那条评论，不放进上下文

    @cached_property
    def llm(self) -> LLMClient:
        return self.services.llm_factory()

    @cached_property
    def interests(self) -> str | None:
        try:
            return load_interests(self.config.profile.interests_file)
        except ProfileError as e:
            logger.warning("读取兴趣描述失败，报告中不结合研究兴趣：%s", e)
            return None

    @cached_property
    def digest(self) -> Digest | None:
        if self.day is None:
            return None
        try:
            return self.store.read_model(digest_path(self.day), Digest)
        except ValidationError as e:
            logger.warning("%s 格式不正确：%s", digest_path(self.day), e)
            return None


def parse_command(body: str) -> Command:
    """解析评论第一行的指令。"""
    lines = body.strip().splitlines()
    first = lines[0].strip() if lines else ""
    if not first.startswith("/"):
        raise CommandError("指令需要以 / 开头")
    match = _NAME_AND_REST.match(first[1:])
    assert match is not None  # 两个分组都可以为空，总能匹配
    name, rest = match.group(1).lower(), match.group(2).strip()
    if name not in COMMANDS:
        raise UnknownCommandError(name)
    if name == "ask":
        # 问题里可能有 --force 之类的字样，不当作参数
        return Command(name, (), False, "\n".join([rest, *lines[1:]]).strip())
    tokens = [token for token in _ARG_SEPARATOR.split(rest) if token]
    force = any(token.lower() in _FORCE_FLAGS for token in tokens)
    args = tuple(token for token in tokens if token.lower() not in _FORCE_FLAGS)
    return Command(name, args, force, rest)


def parse_ref(token: str) -> Ref:
    """把 <ref> 解析成序号（int）或不带版本号的 arXiv ID（str）。"""
    token = token.strip().strip("<>()[]")  # 链接可能被 Markdown 包了一层
    if token.isdigit():
        index = int(token)
        if index < 1:
            raise CommandError("序号从 1 开始")
        return index
    if match := _ARXIV_ID_RE.match(token) or _ARXIV_URL_RE.match(token):
        return match.group(1)
    raise CommandError(f"无法识别 {code(token)}：请写序号、arXiv ID 或 arXiv 链接")


def latest_digest_day(store: DataStore) -> date | None:
    """data 分支上最近一份日报的日期，本地调试按序号引用论文时使用。"""
    directory = store.path(DIGEST_DIR)
    stems = sorted(path.stem for path in directory.glob("*.json")) if directory.is_dir() else []
    for stem in reversed(stems):
        try:
            return date.fromisoformat(stem)
        except ValueError:
            continue
    return None


def execute(command: Command, ctx: Context) -> str:
    """执行指令，返回回复的 Markdown。用户能修正的问题抛出 CommandError。"""
    handlers = {
        "deep": _deep,
        "compare": _compare,
        "more": _more,
        "like": _like,
        "dislike": _dislike,
        "mute": _mute,
        "unmute": _unmute,
        "ask": _ask,
        "help": _help,
    }
    return handlers[command.name](command, ctx)


def _help(command: Command, ctx: Context) -> str:
    return help_reply()


def _deep(command: Command, ctx: Context) -> str:
    if len(command.args) != 1:
        raise CommandError("用法：`/deep <ref> [--force]`，一次深读一篇；多篇请用 `/compare`")
    [paper], notes = resolve_papers([parse_ref(command.args[0])], ctx)

    cached = None if command.force else load_report(ctx.store, paper.id)
    if cached is not None:
        if paper.title == paper.id:  # arXiv API 不可用时只有 ID，用报告里记下的标题
            paper = paper.model_copy(update={"title": cached.title})
        _record_feedback(ctx, [paper], "deep")
        return deep_reply(paper, cached, cached=True)

    paper, fulltext = _fetch(paper, ctx)
    try:
        llm = ctx.llm
        report = deep_read(
            paper,
            fulltext,
            llm=llm,
            interests=ctx.interests,
            config=ctx.config.deep_read,
            language=ctx.config.language,
            now=ctx.now,
        )
    except LLMError as e:
        logger.warning("深读 %s 失败：%s", paper.id, e)
        return deep_fallback_reply(paper, fulltext, error=e)

    if ctx.persist:
        save_report(ctx.store, report)
    _record_feedback(ctx, [paper], "deep")
    tokens = llm.tracker.total.total_tokens
    return deep_reply(
        paper, report, cached=False, notes=notes, fulltext_notes=fulltext.notes, tokens=tokens
    )


def _compare(command: Command, ctx: Context) -> str:
    if not command.args:
        raise CommandError(
            "用法：`/compare <ref> <ref> ...` 横向对比多篇，或 `/compare <ref>` 与相关工作对比"
        )
    if len(command.args) > MAX_COMPARE_PAPERS:
        raise CommandError(f"一次最多对比 {MAX_COMPARE_PAPERS} 篇论文")
    papers, notes = resolve_papers([parse_ref(arg) for arg in command.args], ctx)
    if len(papers) == 1:
        return _compare_related(papers[0], notes, ctx)

    fetched = [_fetch(paper, ctx) for paper in papers]
    papers = [paper for paper, _ in fetched]
    fulltexts = [fulltext for _, fulltext in fetched]
    try:
        llm = ctx.llm
        body = compare_papers(
            papers,
            fulltexts,
            llm=llm,
            interests=ctx.interests,
            config=ctx.config.deep_read,
            language=ctx.config.language,
        )
    except LLMError as e:
        logger.warning("对比失败：%s", e)
        return compare_fallback_reply(papers, fulltexts, error=e)

    _record_feedback(ctx, papers, "compare")
    tokens = llm.tracker.total.total_tokens
    return compare_reply(
        papers, fulltexts, body, model=llm.model_for("deep"), notes=notes, tokens=tokens
    )


def _compare_related(paper: Paper, notes: list[str], ctx: Context) -> str:
    paper, fulltext = _fetch(paper, ctx)
    related = ctx.services.s2.related_works(paper.id)
    try:
        llm = ctx.llm
        body = compare_related(
            paper,
            fulltext,
            related.papers,
            llm=llm,
            interests=ctx.interests,
            config=ctx.config.deep_read,
            language=ctx.config.language,
        )
    except LLMError as e:
        logger.warning("%s 与相关工作对比失败：%s", paper.id, e)
        return related_fallback_reply(paper, fulltext, related, error=e)

    _record_feedback(ctx, [paper], "compare")
    tokens = llm.tracker.total.total_tokens
    return related_reply(
        paper, fulltext, related, body, model=llm.model_for("deep"), notes=notes, tokens=tokens
    )


def _more(command: Command, ctx: Context) -> str:
    count = DEFAULT_MORE
    if command.args:
        if len(command.args) > 1 or not command.args[0].isdigit():
            raise CommandError("用法：`/more [n]`，n 是要展开的篇数，默认 5")
        count = min(max(int(command.args[0]), 1), MAX_MORE)
    digest = _require_digest(ctx)
    start = len(digest.papers) + 1
    items = list(enumerate(digest.others, start=start))[:count]
    return more_reply(digest, items)


def _like(command: Command, ctx: Context) -> str:
    return _explicit_feedback(command, ctx, "like")


def _dislike(command: Command, ctx: Context) -> str:
    return _explicit_feedback(command, ctx, "dislike")


def _explicit_feedback(command: Command, ctx: Context, signal: CommandSignal) -> str:
    """/like、/dislike：立即写入 ±2 的反馈，原因存进 note。"""
    match = _REF_AND_NOTE.match(command.rest)
    if match is None:
        example = "有开源代码" if signal == "like" else "纯 benchmark"
        raise CommandError(f"用法：`/{signal} <ref> [原因]`，例如 `/{signal} 3 {example}`")
    [paper], notes = resolve_papers([parse_ref(match.group(1))], ctx)
    note = clean_note(match.group(2))
    if paper.title == paper.id:
        # arXiv API 不可用：不为一条反馈去下载全文，只记下 ID
        notes = [
            "arXiv API 暂时不可用，只记下了论文 ID：这条反馈不参与粗筛，原因仍会进入每周的偏好总结"
        ]

    today = ctx.now.astimezone(ZoneInfo(ctx.config.timezone)).date()
    earlier = [
        record.signal
        for record in load_feedback(ctx.store, today)
        if record.paper_id == paper.id and record.signal in ("like", "dislike")
    ]
    _record_feedback(ctx, [paper], signal, note=note)
    replaced = bool(earlier) and earlier[-1] != signal
    return feedback_reply(paper, signal, note=note, replaced=replaced, notes=notes)


def _mute(command: Command, ctx: Context) -> str:
    """/mute <主题>：追加一条 mute 事件；不带参数时列出当前屏蔽的主题。"""
    muted = load_muted(ctx.store)
    topic = " ".join(command.rest.split())
    if not topic:
        return muted_list_reply([item.topic for item in muted])
    if len(topic) > MAX_TOPIC_CHARS:
        raise CommandError(
            f"主题太长（{len(topic)} 个字）：请写成 {MAX_TOPIC_CHARS} 个字以内的简短主题，"
            "例如 `/mute 医学影像`"
        )
    if (existing := find_topic(muted, topic)) is not None:
        return mute_reply(existing.topic, [item.topic for item in muted], already=True)

    query, model = None, None
    try:
        query = describe_topic(topic, ctx.llm)
        model = ctx.llm.model_for("fast")
    except LLMError as e:
        # 先记下屏蔽，英文描述留到下次运行日报时补上
        logger.warning("生成屏蔽主题的英文描述失败：%s", e)
    if ctx.persist:
        event = MuteEvent(action="mute", topic=topic, query=query, model=model, created_at=ctx.now)
        append_event(ctx.store, event)
    return mute_reply(topic, [*(item.topic for item in muted), topic], query=query)


def _unmute(command: Command, ctx: Context) -> str:
    """/unmute <主题>：追加一条 unmute 事件。"""
    muted = load_muted(ctx.store)
    topic = " ".join(command.rest.split())
    current = muted_topics_text([item.topic for item in muted])
    if not topic:
        raise CommandError(f"用法：`/unmute <主题>`。{current}")
    existing = find_topic(muted, topic)
    if existing is None:
        raise CommandError(f"{code(topic)} 不在屏蔽列表中。{current}")
    if ctx.persist:
        append_event(
            ctx.store, MuteEvent(action="unmute", topic=existing.topic, created_at=ctx.now)
        )
    remaining = [item.topic for item in muted if item is not existing]
    return unmute_reply(existing.topic, remaining)


def _ask(command: Command, ctx: Context) -> str:
    """/ask <问题>：deep 档模型用只读工具查资料后作答（agent/loop.py）。"""
    # loop 依赖 tools，tools 又依赖本模块：用到时再导入，避免循环导入
    from gleaner.agent.loop import ask

    question = command.rest.strip()
    if not question:
        raise CommandError(
            "用法：`/ask <问题>`，例如 `/ask 今天哪篇的方法最适合用在单源域泛化的检测器上？`"
        )
    if len(question) > MAX_QUESTION_CHARS:
        raise CommandError(
            f"问题太长（{len(question)} 个字）：请精简到 {MAX_QUESTION_CHARS} 个字以内"
        )
    try:
        result = ask(question, ctx)
    except LLMError as e:
        logger.warning("/ask 失败：%s", e)
        return ask_fallback_reply(question, error=e)
    return ask_reply(
        question,
        result.answer,
        model=result.model,
        steps=result.steps,
        max_steps=result.max_steps,
        tokens=result.tokens,
        tool_calls=result.tool_calls,
        cited=result.cited,
        note=result.limit_note,
    )


def resolve_papers(refs: list[Ref], ctx: Context) -> tuple[list[Paper], list[str]]:
    """把 <ref> 解析成论文（去掉重复的），同时返回降级说明。

    序号和当天日报中的 arXiv ID 直接用 digest 中的元数据；其余的通过 arXiv API 查询，
    API 不可用时只保留 ID，标题和摘要稍后从全文中补全。
    """
    needs_digest = any(isinstance(ref, int) for ref in refs)
    digest = _require_digest(ctx) if needs_digest else ctx.digest
    candidates: list[Paper] = [*digest.papers, *digest.others] if digest else []
    by_id = {paper.id: paper for paper in candidates}

    resolved: list[Paper | str] = []
    for ref in refs:
        if isinstance(ref, str):
            resolved.append(by_id.get(ref, ref))
        elif ref <= len(candidates):
            resolved.append(candidates[ref - 1])
        else:
            assert digest is not None
            raise CommandError(_out_of_range(ref, digest))

    missing = list(dict.fromkeys(item for item in resolved if isinstance(item, str)))
    found: dict[str, Paper] = {}
    notes: list[str] = []
    if missing:
        try:
            found = {paper.id: paper for paper in ctx.services.arxiv.fetch_by_ids(missing)}
        except SourceError as e:
            logger.warning("查询 arXiv 元数据失败：%s", e)
            notes.append("arXiv API 暂时不可用，论文信息从全文中获取")
            found = {
                paper_id: Paper(id=paper_id, title=paper_id, abstract="") for paper_id in missing
            }
        unknown = [paper_id for paper_id in missing if paper_id not in found]
        if unknown:
            raise CommandError(
                f"arXiv 上找不到 {'、'.join(code(paper_id) for paper_id in unknown)}"
            )

    papers = [found[item] if isinstance(item, str) else item for item in resolved]
    unique = list({paper.id: paper for paper in papers}.values())
    return unique, notes


def _require_digest(ctx: Context) -> Digest:
    if ctx.day is None:
        raise CommandError("这个 Issue 不是日报，没法按序号找论文，请改用 arXiv ID 或链接")
    digest = ctx.digest
    if digest is None:
        raise CommandError(
            f"找不到 {ctx.day.isoformat()} 的日报数据（gleaner-data 分支上的 "
            f"{digest_path(ctx.day)}），请改用 arXiv ID 或链接"
        )
    if (
        ctx.issue_number is not None
        and digest.issue_number is not None
        and digest.issue_number != ctx.issue_number
    ):
        # 同一天的日报用 --force 重新发布过：序号以新的 Issue 为准
        raise CommandError(
            f"{ctx.day.isoformat()} 的日报已重新发布在 #{digest.issue_number}，"
            "这里的序号可能对不上：请到新的日报下评论，或改用 arXiv ID"
        )
    return digest


def _out_of_range(index: int, digest: Digest) -> str:
    recommended = len(digest.papers)
    message = f"序号 {index} 超出范围：{digest.day.isoformat()} 的日报推荐了 {recommended} 篇"
    if digest.others:
        last = recommended + len(digest.others)
        message += f"，另有 {len(digest.others)} 篇候选（序号 {recommended + 1}–{last}，"
        message += "评论 `/more` 查看）"
    return message


def _fetch(paper: Paper, ctx: Context) -> tuple[Paper, FullText]:
    """获取全文；论文信息不全时（arXiv API 不可用）用全文中的标题和摘要补上。"""
    fulltext = ctx.services.fetcher.fetch(paper)
    updates = {}
    if paper.title == paper.id and fulltext.title:
        updates["title"] = fulltext.title
    if not paper.abstract and fulltext.abstract:
        updates["abstract"] = fulltext.abstract
    if updates:
        paper = paper.model_copy(update=updates)
    if not fulltext.sections:
        reasons = "；".join(fulltext.notes) or "没有摘要"
        raise CommandError(f"没能获取 {paper.id} 的全文和摘要（{reasons}），请稍后重试")
    return paper, fulltext


def _record_feedback(
    ctx: Context, papers: Iterable[Paper], signal: CommandSignal, note: str | None = None
) -> None:
    """写入指令产生的反馈（/deep、/compare 为隐式正反馈），本地调试时不写。"""
    if not ctx.persist:
        return
    records = [command_feedback(paper, signal, now=ctx.now, note=note) for paper in papers]
    record_feedback(ctx.store, records, timezone=ctx.config.timezone)
