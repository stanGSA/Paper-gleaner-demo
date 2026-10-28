import httpx
import pytest
import respx

from gleaner.models import Paper
from gleaner.reader import FullText, FulltextError, FulltextFetcher, Section
from gleaner.reader.fulltext import (
    MAX_LINKS,
    TRUNCATION_MARK,
    classify_section,
    estimate_tokens,
    extract_code_links,
    parse_html,
    parse_pdf,
    truncate_to_tokens,
)
from gleaner.sources.arxiv import RateLimiter
from tests.conftest import make_pdf

HTML = "https://arxiv.org/html/2610.00001"
PDF = "https://arxiv.org/pdf/2610.00001"
PAPER = Paper(id="2610.00001", title="LongKV", abstract="We compress the KV cache.")

# 一篇小论文的 PDF：题头、误认成标题的作者单位、摘要、正文、页码、参考文献和附录
PDF_PAGES = [
    [
        "LongKV: Hierarchical KV Cache Compression",
        "Alice Zhang1 Bob Li2",
        "1 School of Computer Science",
        "Abstract",
        "We propose a hierarchical KV cache compression scheme.",
        "1 Introduction",
        "Long-context inference is bounded by mem-\nory and the cache grows linearly.",
        "2 Method",
        "We split the cache into three tiers.",
        "1",
    ],
    [
        "arXiv:2610.00001v1 [cs.CL] 22 Oct 2026",
        "3 Experiments",
        "We evaluate on PG-19 and LongBench.",
        "4 Conclusion",
        "LongKV reduces memory by 4x.",
        "References",
        "[1] Someone. KVQuant. 2024.",
        "A Proof of Theorem 1",
        "The proof is straightforward.",
        "B Implementation Details",
        "We train on 8 GPUs.",
    ],
]


def fetcher() -> FulltextFetcher:
    return FulltextFetcher(limiter=RateLimiter(0))


# --- token 估算与截断 -----------------------------------------------------------


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 35) == 10
    assert estimate_tokens("长上下文推理") == 6
    assert estimate_tokens("KV 缓存") == 3  # "KV " 3 个字符约 1 个 token，加上 2 个汉字


def test_truncate_to_tokens() -> None:
    text = " ".join(["word"] * 200)
    assert truncate_to_tokens(text, 1000) == text

    short = truncate_to_tokens(text, 50)
    assert short.endswith(TRUNCATION_MARK)
    assert estimate_tokens(short) <= 50
    assert set(short.removesuffix(TRUNCATION_MARK).split()) == {"word"}  # 不截断单词

    chinese = truncate_to_tokens("论文" * 100, 20)
    assert chinese.endswith(TRUNCATION_MARK)
    assert estimate_tokens(chinese) <= 20


@pytest.mark.parametrize(
    ("title", "appendix", "kind"),
    [
        ("Abstract", False, "abstract"),
        ("1 Introduction", False, "introduction"),
        ("2. Related Work", False, "related"),
        ("II. PRELIMINARIES", False, "related"),
        ("3 LongKV", False, "method"),
        ("4 Experimental Setup", False, "experiments"),
        ("5 Results and Analysis", False, "experiments"),
        ("6 Conclusion and Limitations", False, "conclusion"),
        ("Acknowledgements", False, "skip"),
        ("References", False, "skip"),
        ("Broader Impact", False, "other"),
        ("Appendix A Proof of Theorem 1", False, "skip"),
        ("B Implementation Details", True, "appendix"),
        ("C Additional Proofs", True, "skip"),
    ],
)
def test_classify_section(title: str, appendix: bool, kind: str) -> None:
    assert classify_section(title, appendix=appendix) == kind


def test_extract_code_links() -> None:
    texts = [
        "Code: https://github.com/Foo/Bar.git, mirrored at github.com/foo/bar/tree/main/src.",
        "https://github.com/features/actions and https://github.com/someone",
        "https://gitlab.com/group/sub/repo/-/blob/main/train.py",
        "(https://huggingface.co/datasets/org/data) https://huggingface.co/org/model-7b.",
        "https://huggingface.co/papers/2610.00001 https://someone.github.io/project",
    ]
    assert extract_code_links(texts) == [
        "https://github.com/Foo/Bar",
        "https://gitlab.com/group/sub/repo",
        "https://huggingface.co/datasets/org/data",
        "https://huggingface.co/org/model-7b",
    ]

    many = [f"https://github.com/org/repo{i}" for i in range(MAX_LINKS + 5)]
    assert len(extract_code_links(many)) == MAX_LINKS


# --- 按优先级渲染 ---------------------------------------------------------------


def long_text(word: str, tokens: int) -> str:
    return " ".join([word] * int(tokens * 3.5 / (len(word) + 1)))


def test_render_skips_low_priority_sections() -> None:
    fulltext = FullText(
        "html",
        [
            Section("Abstract", "Short abstract.", "abstract"),
            Section("1 Introduction", "Intro.", "introduction"),
            Section("2 Related Work", "Related.", "related"),
            Section("References", "[1] A paper.", "skip"),
            Section("A Implementation Details", "Details.", "appendix"),
        ],
    )
    assert fulltext.render(10_000) == (
        "## Abstract\nShort abstract.\n\n## 1 Introduction\nIntro.\n\n"
        "## 2 Related Work\nRelated.\n\n## A Implementation Details\nDetails."
    )
    # 提高相关工作的优先级（单篇 /compare），预算只够放下第一档
    boosted = fulltext.render(20, priorities={"abstract": 0, "related": 0, "introduction": 1})
    assert boosted == "## Abstract\nShort abstract.\n\n## 2 Related Work\nRelated."


def test_render_shares_budget_within_a_tier() -> None:
    fulltext = FullText(
        "html",
        [
            Section("Abstract", "Short abstract.", "abstract"),
            Section("1 Introduction", long_text("intro", 2000), "introduction"),
            Section("3 Method", long_text("method", 2000), "method"),
            Section("4 Experiments", long_text("result", 2000), "experiments"),
            Section("5 Conclusion", "We conclude.", "conclusion"),
            Section("2 Related Work", long_text("related", 500), "related"),
        ],
    )

    text = fulltext.render(1500)

    assert estimate_tokens(text) <= 1500
    # 短的结论完整保留，三个长章节各自截断保留开头，低优先级的相关工作被挤掉
    assert "## 5 Conclusion\nWe conclude." in text
    for title in ("1 Introduction", "3 Method", "4 Experiments"):
        assert f"## {title}\n" in text
    assert text.count(TRUNCATION_MARK) == 3
    assert "Related Work" not in text
    assert text.index("Introduction") < text.index("Method") < text.index("Conclusion")


# --- HTML -----------------------------------------------------------------------


def test_parse_html(html_text: str) -> None:
    fulltext = parse_html(html_text)

    assert fulltext.source == "html"
    assert fulltext.title == (
        "LongKV: Hierarchical KV Cache Compression for Million-Token Inference"
    )
    assert fulltext.abstract.startswith("We propose a hierarchical KV cache compression scheme")
    assert [(s.title, s.kind) for s in fulltext.sections] == [
        ("Abstract", "abstract"),
        ("1 Introduction", "introduction"),
        ("2 Related Work", "related"),
        ("3 LongKV", "method"),
        ("4 Experiments", "experiments"),
        ("5 Conclusion", "conclusion"),
        ("Acknowledgements", "skip"),
        ("References", "skip"),
        ("Appendix A Proof of Theorem 1", "skip"),
        ("Appendix B Implementation Details", "appendix"),
    ]
    sections = {s.title: s.text for s in fulltext.sections}
    # 公式换成 LaTeX 源码；源码中的换行不拆开句子
    assert sections["1 Introduction"].startswith(
        "Long-context inference is bounded by memory: the KV cache grows as $O(n)$ "
        "with the sequence length.\n"
    )
    assert sections["3 LongKV"] == (
        "We split the cache into three tiers by attention entropy.\n"
        "### 3.1 Entropy-based Tiering\nEach head is assigned a tier."
    )
    table = "Table 1: Perplexity on PG-19.\nMethod | PPL |\nFull KV | 8.21 |"
    assert table in sections["4 Experiments"]

    rendered = fulltext.render(30_000)
    assert "References" not in rendered
    assert "Jensen" not in rendered
    assert "## Appendix B Implementation Details" in rendered

    # 参考文献中的仓库链接属于其它工作，不算本文的代码
    assert fulltext.links == [
        "https://github.com/example/longkv",
        "https://huggingface.co/example/longkv-7b",
    ]


@pytest.mark.parametrize(
    "notes",
    [
        '<span class="ltx_pubnotes"><span class="ltx_pubnote">Thanks: funding</span></span>',
        '<span class="ltx_pubnote">Thanks: corresponding author</span>',
        '<span class="ltx_pubnotes">Accepted for presentation</span>',
        '<span class="ltx_pubnote">Thanks: author</span>'
        '<span class="ltx_pubnote">Thanks: funding</span>',
    ],
)
def test_parse_html_ignores_publication_notes_in_title(html_text: str, notes: str) -> None:
    html = html_text.replace("</h1>", f"{notes}</h1>", 1)

    assert parse_html(html).title == (
        "LongKV: Hierarchical KV Cache Compression for Million-Token Inference"
    )


@pytest.mark.parametrize(
    "html",
    [
        "<html><body><p>No HTML for this paper.</p></body></html>",
        '<html><body><article class="ltx_document"></article></body></html>',
    ],
)
def test_parse_html_without_body(html: str) -> None:
    with pytest.raises(FulltextError, match="没有论文正文"):
        parse_html(html)


# --- PDF ------------------------------------------------------------------------


def test_parse_pdf() -> None:
    fulltext = parse_pdf(make_pdf(PDF_PAGES), abstract="Metadata abstract.")

    assert fulltext.source == "pdf"
    assert [(s.title, s.kind) for s in fulltext.sections] == [
        ("Abstract", "abstract"),
        ("1 Introduction", "introduction"),
        ("2 Method", "method"),
        ("3 Experiments", "experiments"),
        ("4 Conclusion", "conclusion"),
        ("References", "skip"),
        ("A Proof of Theorem 1", "skip"),
        ("B Implementation Details", "appendix"),
    ]
    sections = {s.title: s.text for s in fulltext.sections}
    assert sections["Abstract"] == "We propose a hierarchical KV cache compression scheme."
    # 折行合并、去掉连字符；页码和 arXiv 版本戳被丢弃
    assert sections["1 Introduction"] == (
        "Long-context inference is bounded by memory and the cache grows linearly."
    )
    assert sections["2 Method"] == "We split the cache into three tiers."
    assert "arXiv" not in sections["3 Experiments"]
    assert "School" not in fulltext.render(30_000)


def test_parse_pdf_links() -> None:
    pages = [
        ["1 Introduction", "Code: https://github.com/example/longkv.", "2 Method", "Text."],
        ["References", "[1] https://github.com/other/kvquant"],
    ]
    content = make_pdf(pages, links=("https://huggingface.co/datasets/example/pg19-long",))

    fulltext = parse_pdf(content)

    assert fulltext.links == [
        "https://huggingface.co/datasets/example/pg19-long",
        "https://github.com/example/longkv",
    ]


def test_parse_pdf_uses_metadata_abstract_and_split_heading() -> None:
    pages = [["A Paper", "1\nIntroduction", "Intro text.", "2 Method", "Method text."]]

    fulltext = parse_pdf(make_pdf(pages), abstract="Metadata abstract.")

    assert [(s.title, s.text) for s in fulltext.sections] == [
        ("Abstract", "Metadata abstract."),
        ("1 Introduction", "Intro text."),
        ("2 Method", "Method text."),
    ]


def test_parse_pdf_without_headings() -> None:
    fulltext = parse_pdf(make_pdf([["Just some text.", "More text."]]))
    assert [(s.title, s.kind, s.text) for s in fulltext.sections] == [
        ("", "method", "Just some text.\nMore text.")
    ]


@pytest.mark.parametrize("content", [b"<html>Not found</html>", b"%PDF-1.7 broken"])
def test_parse_pdf_rejects_invalid_content(content: bytes) -> None:
    with pytest.raises(FulltextError):
        parse_pdf(content)


# --- 获取与降级 -----------------------------------------------------------------


@respx.mock
def test_fetch_prefers_html(html_text: str) -> None:
    respx.get(HTML).mock(return_value=httpx.Response(200, text=html_text))
    pdf = respx.get(PDF)

    fulltext = fetcher().fetch(PAPER)

    assert fulltext.source == "html"
    assert fulltext.notes == []
    assert pdf.call_count == 0
    assert respx.calls.last.request.headers["User-Agent"].startswith("paper-gleaner/")


@respx.mock
def test_fetch_falls_back_to_pdf() -> None:
    respx.get(HTML).mock(return_value=httpx.Response(404))
    respx.get(PDF).mock(return_value=httpx.Response(200, content=make_pdf(PDF_PAGES)))

    fulltext = fetcher().fetch(PAPER)

    assert fulltext.source == "pdf"
    assert fulltext.notes == ["HTML：HTTP 404"]


@respx.mock
def test_fetch_falls_back_to_pdf_when_html_has_no_body() -> None:
    respx.get(HTML).mock(return_value=httpx.Response(200, text="<p>No HTML</p>"))
    respx.get(PDF).mock(return_value=httpx.Response(200, content=make_pdf(PDF_PAGES)))

    fulltext = fetcher().fetch(PAPER)

    assert fulltext.source == "pdf"
    assert fulltext.notes == ["HTML：页面中没有论文正文"]


@respx.mock
def test_fetch_falls_back_to_abstract() -> None:
    respx.get(HTML).mock(side_effect=httpx.ConnectError("boom"))
    respx.get(PDF).mock(return_value=httpx.Response(503))

    fulltext = fetcher().fetch(PAPER)

    assert fulltext.source == "abstract"
    assert fulltext.notes == ["HTML：请求失败（ConnectError）", "PDF：HTTP 503"]
    assert [(s.title, s.text) for s in fulltext.sections] == [
        ("Abstract", "We compress the KV cache.")
    ]


@respx.mock
def test_abstract_fallback_keeps_code_links() -> None:
    respx.get(HTML).mock(return_value=httpx.Response(404))
    respx.get(PDF).mock(return_value=httpx.Response(404))
    paper = PAPER.model_copy(update={"abstract": "Code: https://github.com/example/longkv."})

    assert fetcher().fetch(paper).links == ["https://github.com/example/longkv"]


@respx.mock
def test_fetch_waits_between_arxiv_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("gleaner.sources.arxiv.time.sleep", sleeps.append)
    respx.get(HTML).mock(return_value=httpx.Response(404))
    respx.get(PDF).mock(return_value=httpx.Response(404))

    FulltextFetcher(limiter=RateLimiter(3.0)).fetch(PAPER)

    assert len(sleeps) == 1
    assert 2.5 < sleeps[0] <= 3.0
