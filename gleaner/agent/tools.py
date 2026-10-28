"""Agent 工具（docs/architecture.md 3.2「Agent 工具（全部只读）」），/ask 和 MCP Server 共用。

每个工具有名称、给模型看的说明、用 pydantic 定义并校验的参数，以及一个返回纯文本的函数：

- 全部只读：不写 gleaner-data，不发评论；写反馈这类副作用只由编排代码完成；
- 参数都要校验：日期只接受 YYYY-MM-DD，arXiv ID 只接受新式 ID（或 arXiv 链接），
  查询不超过 MAX_QUERY_CHARS 个字符，不认识的参数直接报错；
- 返回的文本含有论文标题、摘要、全文、反馈原因等外部内容，调用方要按不可信数据包裹后再放进
  prompt；每个结果都截断到 MAX_RESULT_TOKENS 以内；
- 参数不合法或外部服务失败时抛出 ToolError，/ask 把错误信息作为工具结果交给模型，指令不会崩溃。

found 参数收集结果中出现过的论文（arXiv ID → 标题），/ask 回复中据此给引用的论文加链接。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from gleaner.agent.commands import (
    CommandError,
    Context,
    latest_digest_day,
    parse_ref,
    resolve_papers,
)
from gleaner.llm.prompt import truncate_to_tokens
from gleaner.models import Digest, Paper, ScoredPaper
from gleaner.pipelines.daily import DIGEST_DIR, digest_path
from gleaner.profile import aggregate_feedback, load_feedback, load_muted, load_preferences
from gleaner.reader import load_report
from gleaner.reader.prompt import SOURCE_LABELS
from gleaner.sources import SourceError

MAX_QUERY_CHARS = 200
MAX_RESULT_TOKENS = 4_000
FULLTEXT_TOKENS = 3_500  # read_fulltext 返回的正文上限，留出余量给章节列表
REPORT_EXCERPT_TOKENS = 1_200  # get_paper 附上的深读报告摘录
PROFILE_TOKENS = 1_500
SEARCH_RESULTS = 8
HISTORY_RESULTS = 10
MAX_SECTIONS = 8
_ABSTRACT_CHARS = 300
_TERM_RE = re.compile(r"[\w+\-.]+")

Found = dict[str, str]


class ToolError(Exception):
    """工具参数不合法或执行失败；信息会作为工具结果交给模型，或返回给 MCP 的调用方。"""


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DigestArgs(_Args):
    date: str | None = Field(
        None, description="日期，YYYY-MM-DD；不写时为当前 Issue 对应的那天（没有时为最近一份）"
    )

    @field_validator("date")
    @classmethod
    def _valid_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
            raise ValueError("日期必须写成 YYYY-MM-DD")
        date.fromisoformat(value.strip())  # 月、日越界时抛出 ValueError
        return value.strip()


class PaperArgs(_Args):
    ref: str = Field(
        min_length=1, max_length=100, description="当天日报中的序号、arXiv ID 或 arXiv 链接"
    )


class ArxivIdArgs(_Args):
    arxiv_id: str = Field(min_length=1, max_length=100, description="arXiv ID，如 2410.12345")

    @field_validator("arxiv_id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        return arxiv_id_arg(value)


class FulltextArgs(ArxivIdArgs):
    sections: list[str] | None = Field(
        None,
        max_length=MAX_SECTIONS,
        description="只读这些章节（标题中的关键词，如 method、experiments）；不写时按优先级读全文",
    )

    @field_validator("sections")
    @classmethod
    def _short_sections(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(len(item) > 80 for item in value):
            raise ValueError("章节名太长")
        return value


class QueryArgs(_Args):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS, description="英文关键词")

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("查询不能为空")
        return " ".join(value.split())


class NoArgs(_Args):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    params: type[_Args]
    run: Callable[[Context, Any, Found], str]

    def call(self, ctx: Context, arguments: dict[str, Any], found: Found | None = None) -> str:
        """校验参数后执行，结果截断到 MAX_RESULT_TOKENS；失败时抛出 ToolError。"""
        try:
            args = self.params.model_validate(arguments)
        except ValidationError as e:
            raise ToolError(f"参数不合法：{_validation_message(e)}") from None
        try:
            text = self.run(ctx, args, found if found is not None else {})
        except CommandError as e:
            raise ToolError(str(e)) from None
        except SourceError as e:
            raise ToolError(f"外部服务暂时不可用：{e}") from None
        return truncate_to_tokens(text, MAX_RESULT_TOKENS)

    def schema(self) -> dict[str, Any]:
        """参数的 JSON Schema（写进 /ask 的 system prompt）。"""
        schema = self.params.model_json_schema()
        properties = {
            name: {key: value for key, value in spec.items() if key != "title"}
            for name, spec in schema.get("properties", {}).items()
        }
        return {"properties": properties, "required": schema.get("required", [])}


# --- 各个工具 -------------------------------------------------------------------


def get_digest(ctx: Context, args: DigestArgs, found: Found) -> str:
    """某一天的日报：推荐的论文（序号 1…N）和其余候选（接着编号）。"""
    day = date.fromisoformat(args.date) if args.date else ctx.day or latest_digest_day(ctx.store)
    if day is None:
        raise ToolError("gleaner-data 上还没有任何日报")
    digest = load_digest(ctx, day)
    if digest is None:
        latest = latest_digest_day(ctx.store)
        hint = f"，最近一份是 {latest.isoformat()}" if latest else ""
        raise ToolError(f"没有 {day.isoformat()} 的日报{hint}")
    lines = [
        f"{day.isoformat()} 的日报：推荐 {len(digest.papers)} 篇，"
        f"另有 {len(digest.others)} 篇打过分但没有推送的候选（序号接着往下编）。"
    ]
    if digest.issue_number is not None:
        lines.append(f"日报 Issue：#{digest.issue_number}")
    if digest.overview:
        lines.append(f"概览：{digest.overview}")
    for index, paper in enumerate(digest.papers, start=1):
        lines.append(_paper_line(index, paper, found, detail=True))
    if digest.others:
        lines.append("其余候选：")
        start = len(digest.papers) + 1
        for index, paper in enumerate(digest.others, start=start):
            lines.append(_paper_line(index, paper, found, detail=False))
    return "\n".join(lines)


def get_paper(ctx: Context, args: PaperArgs, found: Found) -> str:
    """一篇论文的元数据和摘要；在当天日报中时附上分数和理由，有深读报告缓存时附上摘录。"""
    [paper], notes = resolve_papers([parse_ref(args.ref)], ctx)
    found[paper.id] = paper.title
    lines = [
        f"arXiv ID：{paper.id}",
        f"标题：{paper.title}",
        f"作者：{', '.join(paper.authors[:10]) or '未知'}",
        f"分类：{', '.join(paper.categories) or '未知'}",
        f"摘要：{paper.abstract or '（无）'}",
    ]
    if isinstance(paper, ScoredPaper):
        if paper.llm_score is not None:
            lines.append(f"日报打分：{paper.llm_score}/10")
        if paper.reason:
            lines.append(f"推荐理由：{paper.reason}")
        if paper.tldr:
            lines.append(f"TL;DR：{paper.tldr}")
    lines += [f"说明：{note}" for note in notes]
    if (report := load_report(ctx.store, paper.id)) is not None:
        excerpt = truncate_to_tokens(report.body, REPORT_EXCERPT_TOKENS)
        lines.append(f"已有深读报告（{report.generated_at:%Y-%m-%d}，{report.model}）：\n{excerpt}")
    return "\n".join(lines)


def read_fulltext(ctx: Context, args: FulltextArgs, found: Found) -> str:
    """论文全文（arXiv HTML，没有时读 PDF，都失败时只有摘要），可以只读指定章节。"""
    paper = _known_paper(ctx, args.arxiv_id) or Paper(
        id=args.arxiv_id, title=args.arxiv_id, abstract=""
    )
    fulltext = ctx.services.fetcher.fetch(paper)
    if fulltext.title and paper.title == paper.id:
        paper = paper.model_copy(update={"title": fulltext.title})
    found[paper.id] = paper.title
    titles = [section.title or section.kind for section in fulltext.sections]
    lines = [
        f"arXiv ID：{paper.id}",
        f"标题：{paper.title}",
        f"全文来源：{SOURCE_LABELS[fulltext.source]}",
        f"章节：{'；'.join(titles) or '（无）'}",
    ]
    if fulltext.notes:
        lines.append(f"说明：{'；'.join(fulltext.notes)}")
    if fulltext.links:
        lines.append(f"正文中的代码链接：{'、'.join(fulltext.links)}")
    if not fulltext.sections:
        raise ToolError(f"没能获取 {paper.id} 的全文和摘要")
    if args.sections:
        wanted = [name.casefold() for name in args.sections]
        chosen = [
            section
            for section in fulltext.sections
            if any(name in section.title.casefold() or name == section.kind for name in wanted)
        ]
        if not chosen:
            lines.append(
                f"没有找到标题含有 {'、'.join(args.sections)} 的章节，下面按优先级给出全文。"
            )
        else:
            fulltext.sections = chosen
    lines += ["", fulltext.render(FULLTEXT_TOKENS)]
    return "\n".join(lines)


def search_arxiv(ctx: Context, args: QueryArgs, found: Found) -> str:
    """在 arXiv 上按关键词检索（各个词都要出现），按相关度排序。"""
    papers = ctx.services.arxiv.search(args.query, limit=SEARCH_RESULTS)
    if not papers:
        return f"arXiv 上没有找到与 {args.query} 匹配的论文（只支持英文关键词）。"
    lines = [f"arXiv 检索 {args.query} 的前 {len(papers)} 条结果："]
    for paper in papers:
        found[paper.id] = paper.title
        when = f"（{paper.announced:%Y-%m-%d}）" if paper.announced else ""
        lines.append(f"- {paper.id} · {paper.title}{when}\n  摘要：{_short(paper.abstract)}")
    return "\n".join(lines)


def related_works(ctx: Context, args: ArxivIdArgs, found: Found) -> str:
    """Semantic Scholar 上的参考文献和推荐论文。"""
    related = ctx.services.s2.related_works(args.arxiv_id)
    lines = [f"{args.arxiv_id} 的相关工作（Semantic Scholar）："]
    for work in related.papers:
        if work.arxiv_id:
            found[work.arxiv_id] = work.title
        relation = {"reference": "参考文献", "recommended": "推荐论文"}[work.relation]
        meta = [relation, str(work.year or "年份未知")]
        if work.citation_count is not None:
            meta.append(f"被引 {work.citation_count}")
        if work.arxiv_id:
            meta.append(f"arXiv {work.arxiv_id}")
        lines.append(f"- {work.title}（{'，'.join(meta)}）")
        if work.abstract:
            lines.append(f"  摘要：{_short(work.abstract)}")
    if related.errors:
        lines.append(f"部分数据没有获取到：{'；'.join(related.errors)}")
    if not related.papers and not related.errors:
        lines.append("（没有找到）")
    return "\n".join(lines)


def get_profile(ctx: Context, args: NoArgs, found: Found) -> str:
    """研究者的兴趣描述、学到的偏好和屏蔽的主题。"""
    interests = ctx.interests
    lines = ["兴趣描述（profile/interests.md）：", truncate_to_tokens(interests, PROFILE_TOKENS)]
    if not interests:
        lines[-1] = "（没有兴趣描述）"
    preferences = load_preferences(ctx.store)
    lines += ["", "学到的偏好（learned_preferences.md）：", preferences or "（还没有）"]
    muted = [topic.topic for topic in load_muted(ctx.store)]
    lines += ["", f"屏蔽的主题：{'、'.join(muted) if muted else '（没有）'}"]
    return "\n".join(lines)


def search_history(ctx: Context, args: QueryArgs, found: Found) -> str:
    """在推送过的论文（history/digests）和有反馈的论文中按关键词检索。

    按词出现的位置打分：标题中出现记 3 分，摘要、推荐理由或反馈原因中出现记 1 分；
    同分时最近推送或反馈的在前。
    """
    terms = list(dict.fromkeys(term.casefold() for term in _TERM_RE.findall(args.query)))
    entries = _history_entries(ctx)
    scored = []
    for entry in entries.values():
        title = entry.title.casefold()
        body = " ".join([entry.abstract, entry.reason, *entry.notes]).casefold()
        score = sum(3 if term in title else 1 if term in body else 0 for term in terms)
        if score:
            scored.append((score, entry.latest, entry))
    if not scored:
        return (
            f"历史中没有与 {args.query} 匹配的论文"
            f"（共检索 {len(entries)} 篇；用英文关键词效果更好）。"
        )
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    lines = [f"历史中与 {args.query} 最匹配的 {min(len(scored), HISTORY_RESULTS)} 篇："]
    for _, _, entry in scored[:HISTORY_RESULTS]:
        found[entry.paper_id] = entry.title
        lines.append(f"- {entry.paper_id} · {entry.title}")
        if entry.pushed:
            lines.append(f"  推送：{'；'.join(entry.pushed)}")
        if entry.signals:
            feedback = "、".join(entry.signals)
            if entry.notes:
                feedback += f"（原因：{'；'.join(entry.notes)}）"
            lines.append(f"  反馈：{feedback}")
        if entry.reason:
            lines.append(f"  推荐理由：{_short(entry.reason)}")
    return "\n".join(lines)


TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in (
        Tool(
            "get_digest",
            "获取某一天的日报：推荐的论文、分数、理由和其余候选",
            DigestArgs,
            get_digest,
        ),
        Tool(
            "get_paper",
            "获取一篇论文的元数据和摘要；在日报中时附上推荐理由，有深读报告时附上摘录",
            PaperArgs,
            get_paper,
        ),
        Tool(
            "read_fulltext",
            "读取论文全文或指定章节（截断到约 3500 token）",
            FulltextArgs,
            read_fulltext,
        ),
        Tool("search_arxiv", "用英文关键词检索 arXiv", QueryArgs, search_arxiv),
        Tool(
            "related_works",
            "通过 Semantic Scholar 获取一篇论文的参考文献和推荐论文",
            ArxivIdArgs,
            related_works,
        ),
        Tool("get_profile", "读取研究者的兴趣描述、学到的偏好和屏蔽的主题", NoArgs, get_profile),
        Tool(
            "search_history",
            "用关键词在推送过的论文和有反馈的论文中检索（英文关键词效果更好）",
            QueryArgs,
            search_history,
        ),
    )
}


# --- 公用 -----------------------------------------------------------------------


def load_digest(ctx: Context, day: date) -> Digest | None:
    if day == ctx.day:
        return ctx.digest
    try:
        return ctx.store.read_model(digest_path(day), Digest)
    except ValidationError as e:
        raise ToolError(f"{digest_path(day)} 格式不正确") from e


@dataclass
class _HistoryEntry:
    paper_id: str
    title: str = ""
    abstract: str = ""
    reason: str = ""
    latest: str = ""  # 最近一次推送或反馈的日期，用于排序
    pushed: list[str] = field(default_factory=list)  # 出现在哪些日报里
    signals: list[str] = field(default_factory=list)  # 生效的反馈，如 👍、/like
    notes: list[str] = field(default_factory=list)  # /like、/dislike 的原因


def _history_entries(ctx: Context) -> dict[str, _HistoryEntry]:
    entries: dict[str, _HistoryEntry] = {}
    directory = ctx.store.path(DIGEST_DIR)
    days = sorted(path.stem for path in directory.glob("*.json")) if directory.is_dir() else []
    for stem in days:
        try:
            digest = ctx.store.read_model(f"{DIGEST_DIR}/{stem}.json", Digest)
        except (ValidationError, ValueError):
            continue
        if digest is None:
            continue
        for index, paper in enumerate([*digest.papers, *digest.others], start=1):
            entry = entries.setdefault(paper.id, _HistoryEntry(paper.id))
            entry.title, entry.abstract = paper.title, paper.abstract
            entry.reason = paper.reason or entry.reason
            entry.latest = max(entry.latest, stem)
            score = "" if paper.llm_score is None else f"，{paper.llm_score}/10"
            kind = "推荐" if index <= len(digest.papers) else "候选"
            entry.pushed.append(f"{stem} 第 {index} 篇（{kind}{score}）")
    today = ctx.now.astimezone(ZoneInfo(ctx.config.timezone)).date()
    for item in aggregate_feedback(load_feedback(ctx.store, today)):
        entry = entries.setdefault(item.paper_id, _HistoryEntry(item.paper_id))
        if item.has_text:
            entry.title = entry.title or item.title
            entry.abstract = entry.abstract or item.abstract
        entry.title = entry.title or item.paper_id
        entry.signals = item.signals
        entry.notes = item.notes
        if item.updated_at:
            entry.latest = max(entry.latest, item.updated_at.date().isoformat())
    return entries


def _paper_line(index: int, paper: ScoredPaper, found: Found, *, detail: bool) -> str:
    found[paper.id] = paper.title
    score = "未打分" if paper.llm_score is None else f"{paper.llm_score}/10"
    line = f"[{index}] {paper.id} · {score} · {paper.title}"
    if paper.reason:
        line += f"\n    理由：{paper.reason if detail else _short(paper.reason)}"
    if detail and paper.tldr:
        line += f"\n    TL;DR：{paper.tldr}"
    return line


def find_in_history(ctx: Context, paper_id: str) -> Paper | None:
    """在推送过的论文和有反馈的论文中找这篇论文的元数据（标题、摘要），找不到时返回 None。"""
    entry = _history_entries(ctx).get(paper_id)
    if entry is None or not entry.title or entry.title == paper_id:
        return None
    return Paper(id=paper_id, title=entry.title, abstract=entry.abstract)


def _known_paper(ctx: Context, paper_id: str) -> Paper | None:
    digest = ctx.digest
    candidates = [*digest.papers, *digest.others] if digest else []
    return next((paper for paper in candidates if paper.id == paper_id), None)


def arxiv_id_arg(value: str) -> str:
    """校验并规范化 arXiv ID 参数（可以带版本号或写成链接）；不合法时抛出 ValueError。"""
    try:
        ref = parse_ref(value)
    except CommandError as e:
        raise ValueError(str(e).replace("`", "")) from None
    if not isinstance(ref, str):
        raise ValueError("这里要写 arXiv ID，不是序号")
    return ref


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _ABSTRACT_CHARS else text[:_ABSTRACT_CHARS].rstrip() + "…"


def _validation_message(error: ValidationError) -> str:
    parts = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "参数"
        parts.append(f"{location}：{item['msg']}")
    return "；".join(parts)
