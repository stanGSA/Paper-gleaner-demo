"""/compare：多篇论文横向对比，或单篇论文与相关工作对比（docs/architecture.md 3.2）。

- 多篇：几篇论文平分 deep_read.max_input_tokens 的预算，对比问题、方法、实验设置、结果和代码；
- 单篇：目标论文的全文（相关工作章节优先保留）加上 Semantic Scholar 的参考文献和推荐论文。
  Semantic Scholar 失败时只依据论文自己的相关工作章节，由调用方在回复中注明。

和 /deep 一样直接让 deep 档模型输出 Markdown；对比结果不缓存。
"""

from __future__ import annotations

from collections.abc import Mapping

from gleaner.config import DeepReadConfig
from gleaner.llm import LLMClient, LLMOutputError
from gleaner.models import Paper
from gleaner.ranking.rerank import language_name
from gleaner.reader.fulltext import FullText, SectionKind, estimate_tokens, truncate_to_tokens
from gleaner.reader.prompt import (
    PROMPT_MARGIN_TOKENS,
    UNTRUSTED_NOTICE,
    clean_output,
    interests_block,
    paper_block,
    untrusted,
)
from gleaner.sources import RelatedPaper

MAX_COMPARE_PAPERS = 5
MAX_RELATED_ABSTRACT_TOKENS = 150
RELATION_LABELS = {"reference": "参考文献", "recommended": "推荐论文"}
# 单篇对比时，论文自己的相关工作章节和方法、结论同等重要，实验细节次之
RELATED_PRIORITIES: Mapping[SectionKind, int] = {
    "abstract": 0,
    "introduction": 1,
    "related": 1,
    "method": 1,
    "conclusion": 1,
    "experiments": 2,
    "other": 3,
    "appendix": 4,
}

_FORMAT_RULES = (
    "要求：只依据给出的内容，不要编造，没有的信息写「未说明」；数字照抄原文；"
    "方法名、数据集名保留英文；不要使用 HTML 标签。"
)

_MULTI_PROMPT = """\
你是一名科研助理，帮一位研究者横向对比几篇 arXiv 论文。

研究者的兴趣描述：
{interests}

用户消息中有 {count} 个 <paper> 标签，每个是一篇论文的元数据、正文中找到的链接（<links>）\
和正文（<text>，按章节截断过），按 index 属性编号为 [1]、[2] ……{notice}

请用{language}输出 Markdown，依次包含下面四个部分，每部分以三级标题（###）开头，\
标题文字保持不变，不要输出其它标题、开场白或结束语：

### 一句话结论
### 对比总表
一张 Markdown 表格：每篇论文一列，表头写序号加简短的方法名（如「[1] LongKV」）；\
行依次是「问题」「方法」「实验设置」（数据集与基线）「主要结果」「代码」。单元格保持简短，不要换行。\
「代码」一行只能使用对应论文 <links> 中的链接。
### 关键差异
3–5 条列表，说明思路、假设和适用场景上的本质区别。
### 结合你的研究
结合研究者的兴趣描述，建议先读哪篇、哪些做法可以组合借鉴。

{rules}"""

_RELATED_PROMPT = """\
你是一名科研助理，帮一位研究者把一篇 arXiv 论文放到相关工作中比较。

研究者的兴趣描述：
{interests}

用户消息中 <paper> 是目标论文（元数据、链接和按章节截断过的正文）；<related> 中是 \
Semantic Scholar 提供的相关论文，relation 为「参考文献」的是目标论文引用的工作，\
「推荐论文」是相似的工作，只有标题和摘要，按 R1、R2 …… 编号。{notice}

请用{language}输出 Markdown，依次包含下面四个部分，每部分以三级标题（###）开头，\
标题文字保持不变，不要输出其它标题、开场白或结束语：

### 定位
目标论文在研究脉络中的位置：延续了哪些工作，想解决它们的什么问题。
### 与相关工作对比
一张 Markdown 表格，选最相关的 3–6 项工作，列依次为「工作」（编号加简称，如「R2 KVQuant」）\
「关系」「解决的问题」「方法差异」「目标论文的改进或不足」。
### 创新点
目标论文相对已有工作真正新的地方，2–4 条。
### 对你的意义
结合研究者的兴趣描述，说明目标论文和这些相关工作中哪些值得跟进。

如果 <related> 中没有论文，就依据目标论文正文中的相关工作章节比较：\
表格的「工作」写文中提到的方法名，「关系」写「文中提及」。

{rules}"""


def compare_papers(
    papers: list[Paper],
    fulltexts: list[FullText],
    *,
    llm: LLMClient,
    interests: str | None,
    config: DeepReadConfig,
    language: str = "zh",
) -> str:
    """横向对比多篇论文，返回模型输出的 Markdown。LLM 调用失败时抛出 LLMError。"""
    system = _MULTI_PROMPT.format(
        interests=interests_block(interests),
        count=len(papers),
        notice=UNTRUSTED_NOTICE,
        language=language_name(language),
        rules=_FORMAT_RULES,
    )
    budget = config.max_input_tokens - estimate_tokens(system) - PROMPT_MARGIN_TOKENS
    share = budget // len(papers)
    user = "\n\n".join(
        paper_block(paper, fulltext, share, index=index)
        for index, (paper, fulltext) in enumerate(zip(papers, fulltexts, strict=True), start=1)
    )
    return _ask(llm, system, user, config)


def compare_related(
    paper: Paper,
    fulltext: FullText,
    related: list[RelatedPaper],
    *,
    llm: LLMClient,
    interests: str | None,
    config: DeepReadConfig,
    language: str = "zh",
) -> str:
    """把一篇论文与相关工作对比，返回模型输出的 Markdown。LLM 调用失败时抛出 LLMError。"""
    system = _RELATED_PROMPT.format(
        interests=interests_block(interests),
        notice=UNTRUSTED_NOTICE,
        language=language_name(language),
        rules=_FORMAT_RULES,
    )
    works = related_block(related)
    budget = (
        config.max_input_tokens
        - estimate_tokens(system)
        - estimate_tokens(works)
        - PROMPT_MARGIN_TOKENS
    )
    user = paper_block(paper, fulltext, budget, priorities=RELATED_PRIORITIES) + "\n\n" + works
    return _ask(llm, system, user, config)


def related_block(related: list[RelatedPaper]) -> str:
    """Semantic Scholar 返回的相关论文，编号 R1、R2 ……，摘要截断。"""
    if not related:
        return "<related>\n（未能从 Semantic Scholar 获取相关工作）\n</related>"
    works = []
    for index, work in enumerate(related, start=1):
        attributes = f'index="R{index}" relation="{RELATION_LABELS[work.relation]}"'
        if work.year:
            attributes += f' year="{work.year}"'
        if work.citation_count is not None:
            attributes += f' citations="{work.citation_count}"'
        abstract = truncate_to_tokens(work.abstract or "（无摘要）", MAX_RELATED_ABSTRACT_TOKENS)
        works.append(
            f"<work {attributes}>\n标题：{untrusted(work.title)}\n"
            f"摘要：{untrusted(abstract)}\n</work>"
        )
    return "<related>\n" + "\n".join(works) + "\n</related>"


def _ask(llm: LLMClient, system: str, user: str, config: DeepReadConfig) -> str:
    reply = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tier="deep",
        timeout=config.timeout_seconds,
    )
    body = clean_output(reply)
    if not body:
        raise LLMOutputError(f"{llm.model_for('deep')} 只返回了思考过程，没有对比正文")
    return body
