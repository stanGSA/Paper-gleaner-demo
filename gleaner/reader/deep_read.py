"""/deep：深读一篇论文，生成结构化报告（docs/architecture.md 3.3）。

用 deep 档模型，输入是论文全文（按章节优先级截断到 deep_read.max_input_tokens）和用户的
兴趣描述。推理模型不一定支持 JSON mode，这里直接让模型按固定的小标题输出 Markdown。
报告缓存在 gleaner-data 分支的 reports/<arxiv_id>.md：开头是 YAML 元数据，后面是报告正文。
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

import yaml
from pydantic import ValidationError

from gleaner.config import DeepReadConfig
from gleaner.llm import LLMClient, LLMOutputError
from gleaner.models import DeepReport, Paper
from gleaner.ranking.rerank import language_name
from gleaner.reader.fulltext import FullText, estimate_tokens
from gleaner.reader.prompt import (
    PROMPT_MARGIN_TOKENS,
    UNTRUSTED_NOTICE,
    clean_output,
    interests_block,
    paper_block,
)
from gleaner.store import DataStore

REPORT_DIR = "reports"
REPORT_SECTIONS = (
    "一句话总结",
    "问题与动机",
    "方法要点",
    "实验",
    "代码与资源",
    "与你研究的关系",
    "局限与可追问点",
)

_SYSTEM_PROMPT = """\
你是一名科研助理，为一位研究者精读 arXiv 论文，写一份深读报告。

研究者的兴趣描述：
{interests}

用户消息中 <paper> 标签内是论文的元数据、正文中找到的链接（<links>）和正文（<text>，\
可能按章节截断过）。{notice}

请用{language}输出 Markdown，依次包含下面七个部分，每部分以三级标题（###）开头，\
标题文字保持不变，不要输出其它标题、开场白或结束语：

### 一句话总结
一句话说明论文做了什么、效果如何。
### 问题与动机
要解决什么问题，现有方法有什么不足。
### 方法要点
核心思路和关键设计，3–6 条列表。
### 实验
数据集、对比的基线、关键结果（用 Markdown 表格列出主要数字）、消融实验的结论。
### 代码与资源
只能使用 <links> 中列出的链接，说明哪个是官方代码、哪些是数据集或模型；\
没有就写「文中未提供代码链接」。
### 与你研究的关系
结合研究者的兴趣描述，具体说明能借鉴什么、与哪个兴趣点相关；关系不大就直说。
### 局限与可追问点
论文的局限、存疑之处，以及值得进一步追问或验证的问题。

要求：只依据给出的内容，不要编造数字、数据集、基线或链接，没有的信息写「文中未说明」；\
数字照抄原文；方法名、数据集名和指标名保留英文；不要使用 HTML 标签。"""

_ABSTRACT_ONLY = "注意：只拿到了摘要，没有正文。请据此分析，信息不足的部分写「摘要中未提及」。"
_FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)

logger = logging.getLogger(__name__)


def report_path(paper_id: str) -> str:
    return f"{REPORT_DIR}/{paper_id}.md"


def deep_read(
    paper: Paper,
    fulltext: FullText,
    *,
    llm: LLMClient,
    interests: str | None,
    config: DeepReadConfig,
    language: str = "zh",
    now: datetime | None = None,
) -> DeepReport:
    """调用 deep 档模型生成深读报告。LLM 调用失败时抛出 LLMError，由调用方降级。"""
    system = _SYSTEM_PROMPT.format(
        interests=interests_block(interests),
        notice=UNTRUSTED_NOTICE,
        language=language_name(language),
    )
    budget = config.max_input_tokens - estimate_tokens(system) - PROMPT_MARGIN_TOKENS
    user = paper_block(paper, fulltext, budget)
    if fulltext.source == "abstract":
        user += f"\n\n{_ABSTRACT_ONLY}"

    reply = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tier="deep",
        timeout=config.timeout_seconds,
    )
    body = clean_output(reply)
    if not body:
        raise LLMOutputError(f"{llm.model_for('deep')} 只返回了思考过程，没有报告正文")
    return DeepReport(
        paper_id=paper.id,
        title=paper.title,
        model=llm.model_for("deep"),
        source=fulltext.source,
        generated_at=now or datetime.now(UTC),
        body=body,
    )


def dump_report(report: DeepReport) -> str:
    meta = report.model_dump(mode="json", exclude={"body"})
    front_matter = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False)
    return f"---\n{front_matter}---\n\n{report.body.strip()}\n"


def parse_report(text: str) -> DeepReport | None:
    """解析缓存的报告；格式不对（例如被手动改坏）时返回 None。"""
    match = _FRONT_MATTER_RE.match(text.replace("\r\n", "\n"))
    if match is None:
        return None
    try:
        meta = yaml.safe_load(match.group(1))
        return DeepReport.model_validate({**meta, "body": match.group(2).strip()})
    except (yaml.YAMLError, ValidationError, TypeError):
        return None


def load_report(store: DataStore, paper_id: str) -> DeepReport | None:
    text = store.read_text(report_path(paper_id))
    if text is None:
        return None
    report = parse_report(text)
    if report is None or report.paper_id != paper_id:
        logger.warning("%s 格式不正确，忽略缓存", report_path(paper_id))
        return None
    return report


def save_report(store: DataStore, report: DeepReport) -> None:
    store.write_text(report_path(report.paper_id), dump_report(report))
