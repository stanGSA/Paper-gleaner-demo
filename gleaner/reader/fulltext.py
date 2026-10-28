"""论文全文获取与章节切分（docs/architecture.md 3.3）。

按优先级获取：
1. arXiv HTML 版（arxiv.org/html/<id>，由 LaTeXML 生成，保留章节结构）；
2. PDF（arxiv.org/pdf/<id>），用 PyMuPDF 抽取文本，再按标题行切分章节；
3. 都失败时只用摘要，由调用方在回复中注明。

切分后按章节优先级截断到 token 预算：Abstract、Introduction、Method、Experiments、
Conclusion 优先；References、致谢和大部分附录跳过。同时从正文（不含参考文献）中提取
GitHub、GitLab、Hugging Face 链接，作为「代码与资源」的候选。

全文是不可信的外部内容，只作为数据交给 LLM（见 reader/prompt.py）。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit

import httpx
import pymupdf
from selectolax.lexbor import LexborHTMLParser, LexborNode

# token 估算与截断在 llm/prompt.py 中，这里保留原来的导入路径
from gleaner.llm.prompt import TRUNCATION_MARK as TRUNCATION_MARK
from gleaner.llm.prompt import estimate_tokens, truncate_to_tokens
from gleaner.models import Paper
from gleaner.sources.arxiv import ARXIV_RATE_LIMITER, USER_AGENT, RateLimiter

HTML_URL = "https://arxiv.org/html/{id}"
PDF_URL = "https://arxiv.org/pdf/{id}"
MAX_PDF_BYTES = 50 * 1024 * 1024

FulltextSource = Literal["html", "pdf", "abstract"]
SectionKind = Literal[
    "abstract",
    "introduction",
    "method",
    "related",
    "experiments",
    "conclusion",
    "other",
    "appendix",
    "skip",
]

# 截断时各类章节的优先级：数字越小越先保留；不在表中的类型（skip）直接丢弃。
# 正文里标题认不出类型的章节按 method 处理，方法章节常以模型名命名（如「3 LongKV」）
DEFAULT_PRIORITIES: Mapping[SectionKind, int] = {
    "abstract": 0,
    "introduction": 1,
    "method": 1,
    "experiments": 1,
    "conclusion": 1,
    "related": 2,
    "other": 2,
    "appendix": 3,
}

# 预算太少的章节不值得只留一个开头
MIN_SECTION_TOKENS = 80

# 按标题关键词判断章节类型，靠前的规则优先
_KIND_KEYWORDS: list[tuple[SectionKind, tuple[str, ...]]] = [
    ("abstract", ("abstract",)),
    (
        "skip",
        ("reference", "bibliograph", "acknowledg", "checklist", "author contribution", "funding"),
    ),
    ("introduction", ("introduction",)),
    ("related", ("related work", "background", "preliminar", "prior work", "literature")),
    ("conclusion", ("conclusion", "concluding", "limitation", "future work", "discussion")),
    ("experiments", ("experiment", "evaluation", "result", "ablation", "empirical", "benchmark")),
    ("other", ("ethic", "impact", "reproducibility", "societal")),
]
# 附录里只保留这类章节（预算有余时），其余跳过
_USEFUL_APPENDIX = (
    "implementation",
    "experiment",
    "result",
    "ablation",
    "hyperparameter",
    "training",
    "setup",
    "dataset",
)

# 代码与资源链接：GitHub / GitLab 仓库，Hugging Face 模型、数据集和 Space
_LINK_RE = re.compile(
    r"(?<![\w.])(?:https?://)?(?:www\.)?(?:github\.com|gitlab\.com|huggingface\.co)/"
    r"[^\s<>()\[\]{}\"'`|]+",
    re.IGNORECASE,
)
# 这些路径是站点页面而不是仓库
_NOT_REPO = {
    "github.com": {"about", "collections", "explore", "features", "marketplace", "orgs", "topics"},
    "huggingface.co": {"blog", "collections", "docs", "learn", "models", "papers", "tasks"},
}
_PATH_PART_RE = re.compile(r"^[\w.\-]+$")
MAX_LINKS = 10

logger = logging.getLogger(__name__)


class FulltextError(Exception):
    """获取或解析全文失败（调用方据此降级到下一种来源）。"""


@dataclass
class Section:
    title: str
    text: str
    kind: SectionKind


@dataclass
class FullText:
    source: FulltextSource
    sections: list[Section]
    links: list[str] = field(default_factory=list)  # 正文中的代码与资源链接
    notes: list[str] = field(default_factory=list)  # 降级说明，如「HTML：HTTP 404」
    title: str = ""  # 从 HTML 中解析出的标题和摘要，元数据缺失时补用
    abstract: str = ""

    def render(
        self, max_tokens: int, priorities: Mapping[SectionKind, int] = DEFAULT_PRIORITIES
    ) -> str:
        """按优先级挑选章节、截断到约 max_tokens 个 token，按原文顺序输出。

        同一优先级的章节放不下时，用「注水」的方式分配剩余预算：短章节完整保留，
        长章节各自截断保留开头，避免实验或结论整节被挤掉。
        """
        blocks = [f"## {s.title}\n{s.text}" if s.title else s.text for s in self.sections]
        chosen: dict[int, str] = {}
        remaining = max_tokens
        for tier in sorted(set(priorities.values())):
            members = [i for i, s in enumerate(self.sections) if priorities.get(s.kind, -1) == tier]
            costs = [estimate_tokens(blocks[i]) + 1 for i in members]
            if sum(costs) <= remaining:
                chosen.update((i, blocks[i]) for i in members)
                remaining -= sum(costs)
                continue
            for i, share in zip(members, _water_fill(costs, remaining), strict=True):
                if share >= estimate_tokens(blocks[i]) + 1:
                    chosen[i] = blocks[i]
                elif share >= MIN_SECTION_TOKENS:
                    chosen[i] = truncate_to_tokens(blocks[i], share - 1)
            break
        return "\n\n".join(chosen[i] for i in sorted(chosen))


def extract_code_links(texts: Iterable[str]) -> list[str]:
    """从文本或链接地址中找出代码与资源链接，规范成仓库首页地址，按出现顺序去重。"""
    links: dict[str, str] = {}
    for text in texts:
        for match in _LINK_RE.finditer(text):
            link = _normalize_link(match.group(0))
            if link is not None:
                links.setdefault(link.lower(), link)
    return list(links.values())[:MAX_LINKS]


def _normalize_link(url: str) -> str | None:
    url = url.rstrip(".,;:!?*_")
    parsed = urlsplit(url if "://" in url else f"https://{url}")
    host = parsed.netloc.lower().removeprefix("www.")
    parts = [part for part in parsed.path.split("/") if part]
    if not parts or parts[0].lower() in _NOT_REPO.get(host, set()):
        return None
    if host == "gitlab.com":
        # GitLab 支持多级分组；"/-/" 之后是仓库内的页面
        parts = parts[: parts.index("-")] if "-" in parts else parts
        depth = len(parts)
    elif host == "huggingface.co" and parts[0] in {"datasets", "spaces"}:
        depth = 3  # 数据集和 Space 比模型多一级前缀
    else:
        depth = 2  # GitHub 仓库和 Hugging Face 模型都是 <用户>/<名称>
    repo = parts[:depth]
    # 只接受由字母、数字和 ._- 组成的路径，回复里可以原样放进链接
    if len(repo) < max(depth, 2) or not all(_PATH_PART_RE.match(part) for part in repo):
        return None
    repo[-1] = repo[-1].removesuffix(".git")
    return f"https://{host}/" + "/".join(repo)


def classify_section(title: str, *, appendix: bool = False) -> SectionKind:
    """按标题判断章节类型；appendix 表示位于附录中。"""
    name = re.sub(r"^(?:appendix\s+)?(?:[a-z]|\d+|[ivx]+)?[.\s]*\s", "", title.strip().lower())
    name = name or title.strip().lower()
    if appendix or title.strip().lower().startswith("appendix"):
        if any(word in name for word in _USEFUL_APPENDIX):
            return "appendix"
        return "skip"
    for kind, words in _KIND_KEYWORDS:
        if any(word in name for word in words):
            return kind
    return "method"


class FulltextFetcher:
    """依次尝试 arXiv HTML、PDF，都失败时退回摘要。"""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
        limiter: RateLimiter = ARXIV_RATE_LIMITER,
    ) -> None:
        self._client = client
        self._timeout = timeout
        self._limiter = limiter

    def fetch(self, paper: Paper) -> FullText:
        notes: list[str] = []
        try:
            response = self._download(HTML_URL.format(id=paper.id))
            return parse_html(response.text)
        except FulltextError as e:
            logger.info("%s 的 HTML 版不可用：%s", paper.id, e)
            notes.append(f"HTML：{e}")
        try:
            response = self._download(PDF_URL.format(id=paper.id))
            fulltext = parse_pdf(response.content, abstract=paper.abstract)
            fulltext.notes = notes + fulltext.notes
            return fulltext
        except FulltextError as e:
            logger.info("%s 的 PDF 不可用：%s", paper.id, e)
            notes.append(f"PDF：{e}")
        sections = [Section("Abstract", paper.abstract, "abstract")] if paper.abstract else []
        links = extract_code_links([paper.abstract])  # 不少摘要末尾写着代码地址
        return FullText("abstract", sections, links=links, notes=notes)

    def _download(self, url: str) -> httpx.Response:
        self._limiter.wait()
        try:
            if self._client is not None:
                response = self._client.get(url)
            else:
                with httpx.Client(
                    headers={"User-Agent": USER_AGENT},
                    timeout=self._timeout,
                    follow_redirects=True,
                    transport=httpx.HTTPTransport(retries=2),
                ) as client:
                    response = client.get(url)
        except httpx.HTTPError as e:
            raise FulltextError(f"请求失败（{type(e).__name__}）") from e
        if not response.is_success:
            raise FulltextError(f"HTTP {response.status_code}")
        return response


# --- HTML（LaTeXML）------------------------------------------------------------

_DROP_SELECTOR = "script, style, img, svg, button, nav, .ltx_rdf"
_BLOCK_SELECTOR = (
    "p, div.ltx_para, figcaption, li, tr, table.ltx_equation, table.ltx_equationgroup, br"
)
_TOP_SECTION_SELECTOR = "section.ltx_section, section.ltx_appendix, section.ltx_bibliography"
# 分段标记：用一个私用区字符（正文里不会出现），和 HTML 源码中排版用的换行区分开
_BREAK = "\ue000"


def parse_html(html: str) -> FullText:
    """解析 arXiv 的 HTML 全文：标题、摘要和各个一级章节（子章节并入所属章节）。"""
    tree = LexborHTMLParser(html)
    article = tree.css_first("article.ltx_document")
    if article is None:
        raise FulltextError("页面中没有论文正文")

    title_node = article.css_first("h1.ltx_title_document")
    if title_node is not None:
        # LaTeXML 把致谢、通讯作者等出版说明嵌在标题里，提取标题前去掉。
        # 先删除子节点，避免父节点删除后再访问已失效的嵌套节点。
        for note in reversed(title_node.css(".ltx_pubnotes, .ltx_pubnote")):
            note.decompose()
    title = _squash(title_node.text()) if title_node is not None else ""
    for heading in article.css("div.ltx_abstract h6.ltx_title"):
        heading.decompose()
    _prepare(article)

    sections: list[Section] = []
    skipped: set[int] = set()  # 跳过的章节（参考文献等），其中的链接不算本文的代码
    abstract_node = article.css_first("div.ltx_abstract")
    abstract = _node_text(abstract_node) if abstract_node is not None else ""
    if abstract:
        sections.append(Section("Abstract", abstract, "abstract"))
    for node in article.css(_TOP_SECTION_SELECTOR):
        heading = node.css_first("h2.ltx_title")
        name = _squash(heading.text()) if heading is not None else ""
        if heading is not None:
            heading.decompose()
        classes = node.attributes.get("class") or ""
        if "ltx_bibliography" in classes:
            kind: SectionKind = "skip"
        else:
            kind = classify_section(name, appendix="ltx_appendix" in classes)
        if kind == "skip":
            skipped.add(node.mem_id)
        sections.append(Section(name, _node_text(node), kind))

    if len(sections) <= 1:
        # 没有按章节组织（如很短的论文）：整篇作为一节
        body = _node_text(article)
        if not body:
            raise FulltextError("页面中没有论文正文")
        sections = [Section("", body, "method")]

    hrefs = [
        a.attributes.get("href") or ""
        for a in article.css("a[href]")
        if not _has_ancestor(a, skipped, article)
    ]
    texts = [s.text for s in sections if s.kind != "skip"]
    links = extract_code_links([*hrefs, *texts])
    return FullText("html", sections, links=links, title=title, abstract=abstract)


def _has_ancestor(node: LexborNode, ids: set[int], root: LexborNode) -> bool:
    parent = node.parent
    while parent is not None and parent.mem_id != root.mem_id:
        if parent.mem_id in ids:
            return True
        parent = parent.parent
    return False


def _prepare(root: LexborNode) -> None:
    """去掉图片、脚本等，把公式换成 LaTeX 源码，并在块级元素之后插入分段标记。"""
    for node in root.css(_DROP_SELECTOR):
        node.decompose()
    for node in root.css("math"):
        alttext = (node.attributes.get("alttext") or "").strip()
        node.replace_with(f" ${alttext}$ " if alttext else " ")
    for node in root.css("h3.ltx_title, h4.ltx_title, h5.ltx_title, h6.ltx_title"):
        node.insert_before(f"{_BREAK}### ")
        node.insert_after(_BREAK)
    for node in root.css("td, th"):
        node.insert_after(" | ")
    for node in root.css(_BLOCK_SELECTOR):
        node.insert_after(_BREAK)


def _node_text(node: LexborNode) -> str:
    """取出文本：源码中的换行只是排版，按空格处理；分段只看 _prepare 插入的标记。"""
    parts = node.text(deep=True, separator="").split(_BREAK)
    return "\n".join(line for line in (" ".join(part.split()) for part in parts) if line)


# --- PDF -----------------------------------------------------------------------

_NUMBERED_HEADING = re.compile(r"^(\d{1,2})\.?\s+([A-Z][^\n]{1,70})$")
_ROMAN_HEADING = re.compile(r"^([IVX]{1,5})\.\s+([A-Z][^\n]{1,70})$")
_LETTER_HEADING = re.compile(r"^([A-H])\.?\s+([A-Z][^\n]{1,70})$")
_NAMED_HEADING = re.compile(
    r"^(abstract|references|bibliography|acknowledge?ments?|appendix|appendices)\b[\s:.]*$",
    re.IGNORECASE,
)
_APPENDIX_HEADING = re.compile(r"^appendix\b", re.IGNORECASE)
# 页码和 arXiv 侧边的版本戳，例如 "arXiv:2410.12345v1 [cs.CL] 1 Oct 2024"
_NOISE_BLOCK = re.compile(r"^(?:\d{1,4}|arXiv:\d{4}\.\d{4,5}v\d+\b.*)$")
_ROMAN = {"I": 1, "V": 5, "X": 10}


@dataclass
class _PdfBlock:
    text: str
    page: int


def parse_pdf(content: bytes, *, abstract: str = "") -> FullText:
    """用 PyMuPDF 抽取 PDF 文本，按编号标题行（"1 Introduction"、"II. METHOD" 等）切分章节。

    标题之前的部分（题目、作者、单位）丢弃；没有识别出 Abstract 时用元数据中的摘要。
    """
    if not content.startswith(b"%PDF"):
        raise FulltextError("下载的内容不是 PDF")
    if len(content) > MAX_PDF_BYTES:
        raise FulltextError(f"PDF 超过 {MAX_PDF_BYTES // 1024 // 1024} MB")
    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except (RuntimeError, ValueError) as e:  # PyMuPDF 对损坏的文件抛出 RuntimeError 的子类
        raise FulltextError(f"无法解析 PDF（{e}）") from e

    blocks: list[_PdfBlock] = []
    uris: list[tuple[int, str]] = []  # 链接注释（页码，地址）
    with document:
        for page in document:
            for block in page.get_text("blocks"):
                if block[6] == 0 and (text := block[4].strip()):  # 0 表示文本块（1 是图片）
                    blocks.append(_PdfBlock(text, page.number))
            uris.extend((page.number, link["uri"]) for link in page.get_links() if link.get("uri"))

    sections, references_page = _split_pdf_blocks(blocks)
    if not sections or all(s.kind == "skip" for s in sections):
        raise FulltextError("PDF 中没有可用的文本")
    if abstract and not any(s.kind == "abstract" for s in sections):
        sections.insert(0, Section("Abstract", abstract, "abstract"))
    # 参考文献所在页及之后的链接注释多半属于引用的其它工作，不算本文的代码
    annotations = [uri for page, uri in uris if references_page is None or page < references_page]
    texts = [s.text for s in sections if s.kind != "skip"]
    return FullText("pdf", sections, links=extract_code_links([*annotations, *texts]))


def _split_pdf_blocks(blocks: list[_PdfBlock]) -> tuple[list[Section], int | None]:
    """切分章节，同时返回 References 标题所在的页码（没有时为 None）。"""
    sections: list[Section] = []
    references_page: int | None = None
    current: Section | None = None
    body: list[str] = []
    last_number = 0
    last_roman = 0
    last_letter = ""
    appendix = False  # References 之后或遇到 Appendix 标题后，按附录处理

    def flush() -> None:
        if current is not None:
            current.text = "\n".join(body).strip()
            sections.append(current)

    for block in blocks:
        if _NOISE_BLOCK.match(block.text):
            continue
        lines = block.text.split("\n")
        first, rest = lines[0].strip(), lines[1:]
        if first.isdigit() and rest:
            # "1" 和 "Introduction" 可能被拆成两行
            first, rest = f"{first} {rest[0].strip()}", rest[1:]

        heading: str | None = None
        restart = False
        if match := _NAMED_HEADING.match(first):
            heading = match.group(1).title()
        elif (match := _NUMBERED_HEADING.match(first)) and _looks_like_title(match.group(2)):
            number = int(match.group(1))
            # 作者单位常写成 "1 School of ..."，会被误当成第 1 节；
            # 之后遇到 "1 Introduction" 时以它为准，前面误认的章节作为题头丢弃
            restart = number == 1 and last_number > 0 and "introduction" in first.lower()
            if number == last_number + 1 or restart:
                heading, last_number = first, number
        elif (match := _ROMAN_HEADING.match(first)) and _roman(match.group(1)) == last_roman + 1:
            if _looks_like_title(match.group(2)):
                heading, last_roman = first, last_roman + 1
        elif appendix and (match := _LETTER_HEADING.match(first)):
            expected = chr(ord(last_letter) + 1) if last_letter else "A"
            if match.group(1) == expected and _looks_like_title(match.group(2)):
                heading, last_letter = first, expected

        if heading is None:
            if current is not None:
                body.append(_join_lines(block.text))
            continue

        flush()
        if restart:
            sections = [s for s in sections if s.kind == "abstract"]
        if _APPENDIX_HEADING.match(heading):
            appendix = True
        kind = classify_section(heading, appendix=appendix)
        if kind == "skip" and heading.lower().startswith(("reference", "bibliograph")):
            appendix = True
            references_page = block.page if references_page is None else references_page
        current, body = Section(heading, "", kind), []
        if rest:
            body.append(_join_lines("\n".join(rest)))
    flush()

    if not sections:
        # 认不出任何标题：整篇作为一节
        text = "\n".join(_join_lines(b.text) for b in blocks if not _NOISE_BLOCK.match(b.text))
        return ([Section("", text, "method")] if text else []), None
    return sections, references_page


def _looks_like_title(text: str) -> bool:
    words = text.split()
    return len(words) <= 10 and not text.rstrip().endswith((".", ",", ";", ":"))


def _roman(numeral: str) -> int:
    values = [_ROMAN[c] for c in numeral]
    return sum(-v if i + 1 < len(values) and v < values[i + 1] else v for i, v in enumerate(values))


def _join_lines(text: str) -> str:
    """合并 PDF 中被折行的句子，去掉行尾连字符。"""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return " ".join(text.split())


def _squash(text: str) -> str:
    return " ".join(text.split())


def _water_fill(costs: list[int], budget: int) -> list[int]:
    """把 budget 分给各项：需求小的先拿够，剩下的在其余项中平分。"""
    shares = [0] * len(costs)
    remaining = budget
    order = sorted(range(len(costs)), key=costs.__getitem__)
    for position, index in enumerate(order):
        share = min(costs[index], remaining // (len(order) - position))
        shares[index] = share
        remaining -= share
    return shares
