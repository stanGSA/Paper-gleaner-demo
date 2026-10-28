"""Issue 指令回复的 Markdown 渲染。

论文标题、作者、模型输出和用户写的参数都是外部内容：标题和作者经过 sanitize，
模型输出经过 sanitize_markdown，用户参数（包括反馈原因、屏蔽主题）放进行内代码（code()）。
回复中不写 gleaner:paper 标记，避免被当成日报的论文评论。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from gleaner.models import DeepReport, Digest, Paper, ScoredPaper
from gleaner.reader.fulltext import FullText, FulltextSource
from gleaner.render.markdown import (
    REPORT_SOURCE_NAMES,
    kilo,
    paper_link,
    sanitize,
    sanitize_markdown,
)
from gleaner.sources import RelatedPaper, RelatedWorks

SOURCE_NAMES: dict[FulltextSource, str] = REPORT_SOURCE_NAMES
RELATION_NAMES = {"reference": "参考文献", "recommended": "推荐论文"}
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")
_S2_URL_PREFIX = "https://www.semanticscholar.org/"
_MAX_ERROR_CHARS = 200
_MAX_NOTE_CHARS = 200  # 与 profile.memory.MAX_NOTE_CHARS 一致

HELP = """\
### 🤖 Paper Gleaner 指令

| 指令 | 说明 |
| --- | --- |
| `/deep <ref>` | 深读全文，生成结构化报告；已有缓存时直接返回，加 `--force` 重新生成 |
| `/compare <ref> <ref> ...` | 多篇论文横向对比（最多 5 篇） |
| `/compare <ref>` | 单篇论文与相关工作对比（相关工作来自 Semantic Scholar） |
| `/more [n]` | 展开这份日报中打过分但没有推送的候选，默认 5 篇 |
| `/like <ref> [原因]` | 喜欢这篇论文，之后多推荐类似的；原因每周汇总进学到的偏好 |
| `/dislike <ref> [原因]` | 不喜欢这篇论文，之后少推荐类似的 |
| `/mute <主题>` | 屏蔽一个主题，如 `/mute 医学影像`；不带参数时列出已屏蔽的主题 |
| `/unmute <主题>` | 取消屏蔽 |
| `/ask <问题>` | 自由提问：Agent 会查日报、读全文、检索 arXiv 和你的历史后作答，最多 8 步 |
| `/help` | 显示本帮助 |

`<ref>` 可以是本 Issue 中的序号（如 `3`）、arXiv ID（如 `2410.12345`，版本号会被忽略）\
或 arXiv 链接。`/more` 列出的候选接着日报往下编号，同样可以用序号引用。\
给论文评论点 👍 / 👎 也算反馈：下一次运行日报时同步，只统计仓库所有者的表情。"""


def help_reply() -> str:
    return HELP


def error_reply(reason: str) -> str:
    return f"⚠️ {reason}\n\n评论 `/help` 查看可用指令。"


def unknown_reply(name: str) -> str:
    return f"⚠️ 不认识的指令 {code('/' + name)}。\n\n{HELP}"


def code(text: str, limit: int = 60) -> str:
    """把用户写的参数放进行内代码：其中的 @、#123 和 HTML 都不会生效。"""
    text = " ".join(text.replace("`", "'").split())
    if len(text) > limit:
        text = text[:limit] + "…"
    return f"`{text}`"


def short_error(error: object) -> str:
    """异常信息可能包含外部服务返回的内容，转义并截短后再放进回复。"""
    text = " ".join(str(error).split())
    if len(text) > _MAX_ERROR_CHARS:
        text = text[:_MAX_ERROR_CHARS] + "…"
    return sanitize(text)


# --- /deep ----------------------------------------------------------------------


def deep_reply(
    paper: Paper,
    report: DeepReport,
    *,
    cached: bool,
    notes: Sequence[str] = (),
    fulltext_notes: Sequence[str] = (),
    tokens: int | None = None,
) -> str:
    """notes 是一般说明（如 arXiv API 不可用）；fulltext_notes 是全文获取失败的原因。"""
    lines = [f"## 🔍 深读 · {paper_link(paper)}", "", *_paper_meta(paper), ""]
    if report.source == "abstract":
        lines.append(f"> ⚠️ 没能获取全文，以下分析仅基于摘要{_details(fulltext_notes)}。")
    if notes:
        lines.append(f"> ℹ️ {_join(notes)}")
    if report.source == "abstract" or notes:
        lines.append("")
    lines += [sanitize_markdown(report.body), "", "---"]
    footer = [
        f"全文来源：{SOURCE_NAMES[report.source]}",
        f"模型 {sanitize(report.model)}",
        f"{report.generated_at:%Y-%m-%d} 生成",
    ]
    if cached:
        footer.append(f"来自缓存，评论 `/deep {paper.id} --force` 重新生成")
    elif tokens:
        footer.append(f"token {kilo(tokens)}")
    lines.append(f"<sub>{' · '.join(footer)}</sub>")
    return "\n".join(lines)


def deep_fallback_reply(paper: Paper, fulltext: FullText, *, error: object) -> str:
    """深读报告没有生成（LLM 调用失败）时，先给出摘要和代码链接。"""
    lines = [
        f"## 🔍 深读 · {paper_link(paper)}",
        "",
        *_paper_meta(paper),
        "",
        f"> ⚠️ 深读报告没有生成：{short_error(error)}",
        *_fulltext_warning(fulltext),
        f"> 下面先给出摘要和找到的链接，稍后可以再评论 `/deep {paper.id}` 重试。",
        "",
        "### 摘要",
        "",
        sanitize(paper.abstract or fulltext.abstract or "（无摘要）"),
        "",
        "### 代码与资源",
        "",
        _links(fulltext.links),
    ]
    return "\n".join(lines)


# --- /compare -------------------------------------------------------------------


def compare_reply(
    papers: Sequence[Paper],
    fulltexts: Sequence[FullText],
    body: str,
    *,
    model: str,
    notes: Sequence[str] = (),
    tokens: int | None = None,
) -> str:
    lines = [
        f"## ⚖️ 对比 · {len(papers)} 篇论文",
        "",
        *_paper_table(papers, fulltexts),
        "",
        *_source_warnings(papers, fulltexts, notes),
        sanitize_markdown(body),
        "",
        "---",
        f"<sub>{_usage(model, tokens)}</sub>",
    ]
    return "\n".join(lines)


def compare_fallback_reply(
    papers: Sequence[Paper], fulltexts: Sequence[FullText], *, error: object
) -> str:
    lines = [
        f"## ⚖️ 对比 · {len(papers)} 篇论文",
        "",
        *_paper_table(papers, fulltexts),
        "",
        f"> ⚠️ 对比没有生成：{short_error(error)}",
        "> 下面先列出各篇正文中找到的链接，稍后可以再评论 `/compare` 重试。",
    ]
    for index, (paper, fulltext) in enumerate(zip(papers, fulltexts, strict=True), start=1):
        lines += ["", f"**[{index}] {sanitize(paper.title)}**", "", _links(fulltext.links)]
    return "\n".join(lines)


def related_reply(
    paper: Paper,
    fulltext: FullText,
    related: RelatedWorks,
    body: str,
    *,
    model: str,
    notes: Sequence[str] = (),
    tokens: int | None = None,
) -> str:
    lines = [f"## ⚖️ 与相关工作对比 · {paper_link(paper)}", "", *_paper_meta(paper), ""]
    lines += _related_warnings(related)
    lines += _source_warnings([paper], [fulltext], notes)
    lines += [sanitize_markdown(body), ""]
    lines += _related_table(related.papers)
    lines += [
        "---",
        f"<sub>全文来源：{SOURCE_NAMES[fulltext.source]} · {_usage(model, tokens)}</sub>",
    ]
    return "\n".join(lines)


def related_fallback_reply(
    paper: Paper, fulltext: FullText, related: RelatedWorks, *, error: object
) -> str:
    lines = [f"## ⚖️ 与相关工作对比 · {paper_link(paper)}", "", *_paper_meta(paper), ""]
    lines += [
        f"> ⚠️ 对比没有生成：{short_error(error)}",
        f"> 下面先列出找到的相关工作，稍后可以再评论 `/compare {paper.id}` 重试。",
        "",
    ]
    lines += _related_warnings(related)
    lines += _related_table(related.papers, folded=False)
    return "\n".join(lines).rstrip()


# --- /more ----------------------------------------------------------------------


def more_reply(digest: Digest, items: Sequence[tuple[int, ScoredPaper]]) -> str:
    day = digest.day.isoformat()
    if not items:
        return f"📭 {day} 的日报没有其它候选。"
    first, last = items[0][0], items[-1][0]
    total = len(digest.papers) + len(digest.others)
    lines = [
        f"## 📋 更多候选 · {day}",
        "",
        f"这份日报推荐了 {len(digest.papers)} 篇，另有 {len(digest.others)} 篇打过分但没有推送，"
        f"下面是第 {first}–{last} 篇：",
    ]
    for index, paper in items:
        score = "" if paper.llm_score is None else f" · {paper.llm_score}/10"
        lines += ["", f"### [{index}] {sanitize(paper.title)}{score}", ""]
        if paper.reason:
            lines.append(f"🎯 {sanitize(paper.reason)}")
        if paper.tldr:
            lines.append(f"📝 {sanitize(paper.tldr)}")
        elif paper.abstract:
            lines.append(f"📄 {sanitize(_truncate(paper.abstract, 300))}")
        lines.append(f"[arXiv]({paper.url}) · [PDF]({paper.pdf_url})")
    hints = [f"评论 `/deep {first}` 深读"]
    if last > first:
        hints.append(f"`/compare {first} {first + 1}` 对比")
    if last < total:
        hints.append(f"`/more {min(last - len(digest.papers) + 5, 20)}` 查看更多")
    lines += ["", f"<sub>{' · '.join(hints)}</sub>"]
    return "\n".join(lines)


# --- /like、/dislike、/mute、/unmute -------------------------------------------


def feedback_reply(
    paper: Paper,
    signal: Literal["like", "dislike"],
    *,
    note: str | None,
    replaced: bool = False,
    notes: Sequence[str] = (),
) -> str:
    """/like、/dislike 的确认。replaced 表示覆盖了之前相反的一条。"""
    icon, verb = ("👍", "喜欢") if signal == "like" else ("👎", "不喜欢")
    text = f"{icon} 已记下：{verb} {paper_link(paper)}"
    if note:
        text += f"，原因：{code(note, limit=_MAX_NOTE_CHARS)}"
    if replaced:
        text += f"（覆盖了之前的 /{'dislike' if signal == 'like' else 'like'}）"
    lines = [text + "。"]
    if notes:
        lines += ["", f"> ℹ️ {_join(notes)}"]
    lines += ["", "<sub>下一次日报的粗筛就会参考这条反馈；每周汇总进学到的偏好</sub>"]
    return "\n".join(lines)


def mute_reply(
    topic: str, muted: Sequence[str], *, query: str | None = None, already: bool = False
) -> str:
    """/mute 的确认；muted 是屏蔽之后的完整列表。"""
    if already:
        lines = [f"🔇 {code(topic)} 已经屏蔽过了。"]
    else:
        lines = [f"🔇 已屏蔽 {code(topic)}，之后的日报会把这类论文排到后面。"]
        if query:
            lines.append(f"英文检索描述：{sanitize(query)}")
        else:
            lines.append("> ⚠️ 英文检索描述暂时没有生成，下次运行日报时会自动补上。")
    unmute = code(f"/unmute {topic}")
    lines += ["", f"<sub>{muted_topics_text(muted)} · 评论 {unmute} 取消屏蔽</sub>"]
    return "\n".join(lines)


def unmute_reply(topic: str, muted: Sequence[str]) -> str:
    return f"🔊 已取消屏蔽 {code(topic)}。\n\n<sub>{muted_topics_text(muted)}</sub>"


def muted_list_reply(muted: Sequence[str]) -> str:
    """/mute 不带参数时列出当前屏蔽的主题。"""
    if not muted:
        return "🔇 当前没有屏蔽任何主题。评论 `/mute <主题>` 屏蔽，例如 `/mute 医学影像`。"
    lines = [f"🔇 当前屏蔽了 {len(muted)} 个主题：", ""]
    lines += [f"- {code(topic)}" for topic in muted]
    lines += ["", "<sub>评论 `/unmute <主题>` 取消屏蔽</sub>"]
    return "\n".join(lines)


def muted_topics_text(muted: Sequence[str]) -> str:
    if not muted:
        return "当前没有屏蔽任何主题"
    return "当前屏蔽的主题：" + "、".join(code(topic) for topic in muted)


# --- /ask -----------------------------------------------------------------------


def ask_reply(
    question: str,
    answer: str,
    *,
    model: str,
    steps: int,
    max_steps: int,
    tokens: int,
    tool_calls: Sequence[str],
    cited: Sequence[tuple[str, str]],
    note: str | None = None,
) -> str:
    """/ask 的回答：模型输出经过 sanitize_markdown，引用的论文另外给出 arXiv 链接。

    cited 是回答中出现的 arXiv ID 及其标题；链接由 ID 拼出，不用模型写的链接。
    """
    lines = [f"## 💬 /ask · {code(question, limit=80)}", ""]
    if note:
        lines += [f"> ℹ️ {sanitize(note)}", ""]
    lines += [sanitize_markdown(answer), ""]
    papers = [(paper_id, title) for paper_id, title in cited if _ARXIV_ID_RE.match(paper_id)]
    if papers:
        lines += ["**提到的论文**", ""]
        lines += [
            f"- {paper_link(Paper(id=paper_id, title=title, abstract=''))}（{paper_id}）"
            for paper_id, title in papers
        ]
        lines.append("")
    usage = [f"模型 {sanitize(model)}", f"{steps} 步（上限 {max_steps}）", f"token {kilo(tokens)}"]
    if tool_calls:
        usage.append("工具：" + "、".join(tool_calls))
    lines += ["---", f"<sub>{' · '.join(usage)}</sub>"]
    return "\n".join(lines)


def ask_fallback_reply(question: str, *, error: object) -> str:
    """回答没有生成（LLM 调用失败）。"""
    return (
        f"## 💬 /ask · {code(question, limit=80)}\n\n"
        f"> ⚠️ 回答没有生成：{short_error(error)}\n\n"
        "稍后可以再评论一次 `/ask` 重试。"
    )


# --- 公用片段 -------------------------------------------------------------------


def _paper_meta(paper: Paper) -> list[str]:
    meta = []
    if paper.authors:
        authors = ", ".join(paper.authors[:3]) + (" 等" if len(paper.authors) > 3 else "")
        meta.append(f"**作者**：{sanitize(authors)}")
    if paper.categories:
        meta.append(f"**分类**：{', '.join(paper.categories)}")
    links = (
        f"[arXiv]({paper.url}) · [PDF]({paper.pdf_url}) · [HTML](https://arxiv.org/html/{paper.id})"
    )
    return [" · ".join(meta), links] if meta else [links]


def _fulltext_warning(fulltext: FullText) -> list[str]:
    if fulltext.source != "abstract":
        return []
    return [f"> ⚠️ 也没能获取全文{_details(fulltext.notes)}。"]


def _paper_table(papers: Sequence[Paper], fulltexts: Sequence[FullText]) -> list[str]:
    rows = ["| # | 论文 | 全文来源 |", "| --: | --- | --- |"]
    for index, (paper, fulltext) in enumerate(zip(papers, fulltexts, strict=True), start=1):
        rows.append(f"| {index} | {paper_link(paper)} | {SOURCE_NAMES[fulltext.source]} |")
    return rows


def _source_warnings(
    papers: Sequence[Paper], fulltexts: Sequence[FullText], notes: Sequence[str]
) -> list[str]:
    lines = []
    for paper, fulltext in zip(papers, fulltexts, strict=True):
        if fulltext.source == "abstract":
            lines.append(f"> ⚠️ 没能获取 {paper.id} 的全文，只用了摘要{_details(fulltext.notes)}。")
    if notes:
        lines.append(f"> ℹ️ {_join(notes)}")
    return [*lines, ""] if lines else []


def _related_warnings(related: RelatedWorks) -> list[str]:
    if not related.errors:
        return []
    if related.papers:
        return [f"> ℹ️ Semantic Scholar 部分数据没有获取到{_details(related.errors)}。", ""]
    return [
        f"> ⚠️ 没能从 Semantic Scholar 获取相关工作{_details(related.errors)}，"
        "以下对比只依据论文自己的相关工作章节。",
        "",
    ]


def _related_table(related: Sequence[RelatedPaper], *, folded: bool = True) -> list[str]:
    if not related:
        return []
    rows = ["| # | 论文 | 关系 | 年份 | 引用数 |", "| --: | --- | --- | --: | --: |"]
    for index, work in enumerate(related, start=1):
        relation = RELATION_NAMES[work.relation] + (" ⭐" if work.influential else "")
        year = work.year or "—"
        citations = "—" if work.citation_count is None else work.citation_count
        rows.append(f"| R{index} | {_related_link(work)} | {relation} | {year} | {citations} |")
    if not folded:
        return [*rows, ""]
    summary = f"<details><summary>Semantic Scholar 相关工作（{len(related)} 篇）</summary>"
    return [summary, "", *rows, "", "</details>", ""]


def _related_link(work: RelatedPaper) -> str:
    title = " ".join(sanitize(work.title).split()).replace("|", "\\|")
    title = title.replace("[", "\\[").replace("]", "\\]")
    if work.arxiv_id and _ARXIV_ID_RE.match(work.arxiv_id):
        return f"[{title}](https://arxiv.org/abs/{work.arxiv_id})"
    if work.url and work.url.startswith(_S2_URL_PREFIX) and " " not in work.url:
        return f"[{title}]({work.url})"
    return title


def _links(links: Sequence[str]) -> str:
    return "\n".join(f"- <{link}>" for link in links) if links else "没有找到代码链接。"


def _usage(model: str, tokens: int | None) -> str:
    return f"模型 {sanitize(model)}" + (f" · token {kilo(tokens)}" if tokens else "")


def _details(notes: Sequence[str]) -> str:
    return f"（{_join(notes)}）" if notes else ""


def _join(notes: Sequence[str]) -> str:
    return sanitize("；".join(notes))


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"
