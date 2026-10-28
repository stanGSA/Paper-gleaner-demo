import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from gleaner.agent import (
    Command,
    CommandError,
    Context,
    UnknownCommandError,
    execute,
    latest_digest_day,
    parse_command,
    parse_ref,
)
from gleaner.agent.commands import resolve_papers
from gleaner.config import Config
from gleaner.llm import LLMError
from gleaner.models import Digest
from gleaner.pipelines.daily import digest_path
from gleaner.profile import load_feedback
from gleaner.profile.muted import load_events, load_muted
from gleaner.reader import load_report, report_path
from gleaner.store import DataStore
from tests.conftest import (
    DEEP_REPORT,
    DIGEST_DAY,
    interact_digest,
    interact_llm,
    interact_services,
    make_pdf,
    user_prompt,
)

NOW = datetime(2026, 10, 26, 2, 0, tzinfo=UTC)
ARXIV_API = "https://export.arxiv.org/api/query"
S2 = "https://api.semanticscholar.org"


def html_url(paper_id: str) -> str:
    return f"https://arxiv.org/html/{paper_id}"


def pdf_url(paper_id: str) -> str:
    return f"https://arxiv.org/pdf/{paper_id}"


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    store = DataStore(tmp_path / "data")
    store.write_model(digest_path(DIGEST_DAY), interact_digest())
    return store


@pytest.fixture
def config(tmp_path: Path) -> Config:
    interests = tmp_path / "interests.md"
    interests.write_text("我研究长上下文推理。", encoding="utf-8")
    return Config.model_validate({"profile": {"interests_file": str(interests)}})


def context(config: Config, store: DataStore, handler=interact_llm, **fields):
    services, fake = interact_services(handler)
    values = {"day": DIGEST_DAY, "issue_number": 1, "now": NOW} | fields
    return Context(config, services, store, **values), fake


def run(text: str, config: Config, store: DataStore, **fields):
    ctx, fake = context(config, store, **fields)
    return execute(parse_command(text), ctx), fake


def mock_fulltext(html_text: str, *paper_ids: str) -> None:
    for paper_id in paper_ids:
        respx.get(html_url(paper_id)).mock(return_value=httpx.Response(200, text=html_text))


def api_feed(*entries: tuple[str, str]) -> str:
    items = "".join(
        f"<entry><id>http://arxiv.org/abs/{paper_id}v1</id><title>{title}</title>"
        f"<summary>Abstract of {title}.</summary><author><name>Erin Liu</name></author>"
        '<category term="cs.CV"/></entry>'
        for paper_id, title in entries
    )
    return f'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">{items}</feed>'


# --- 解析 -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "command"),
    [
        ("/deep 3", Command("deep", ("3",))),
        ("/Deep 3 --force\r\n顺便说一句", Command("deep", ("3",), force=True)),
        ("/deep -f 2610.00001v2", Command("deep", ("2610.00001v2",), force=True)),
        ("/compare 3,5、7  9", Command("compare", ("3", "5", "7", "9"))),
        ("/compare 3， 5", Command("compare", ("3", "5"))),
        ("/more", Command("more")),
        ("  /more 10", Command("more", ("10",))),
        ("/help", Command("help")),
    ],
)
def test_parse_command(text: str, command: Command) -> None:
    assert parse_command(text) == command


@pytest.mark.parametrize(
    ("text", "name"), [("/summarize 今天哪篇好？", "summarize"), ("/", ""), ("/deep3", "deep3")]
)
def test_unknown_command(text: str, name: str) -> None:
    with pytest.raises(UnknownCommandError) as info:
        parse_command(text)
    assert info.value.name == name


def test_command_text() -> None:
    assert Command("deep", ("3",), force=True).text == "/deep 3 --force"


@pytest.mark.parametrize(
    ("text", "name", "rest"),
    [
        ("/like 3 有开源代码，方法简单", "like", "3 有开源代码，方法简单"),
        ("/Dislike,7、纯 benchmark\n第二行不算", "dislike", "7、纯 benchmark"),
        ("/mute  Medical Imaging ", "mute", "Medical Imaging"),
        ("/unmute 医学影像", "unmute", "医学影像"),
        ("/mute", "mute", ""),
    ],
)
def test_parse_keeps_raw_text(text: str, name: str, rest: str) -> None:
    command = parse_command(text)
    assert (command.name, command.rest) == (name, rest)


@pytest.mark.parametrize(
    ("token", "ref"),
    [
        ("3", 3),
        ("12", 12),
        ("2410.12345", "2410.12345"),
        ("2410.12345v2", "2410.12345"),
        ("arXiv:2410.1234", "2410.1234"),
        ("https://arxiv.org/abs/2410.12345v3", "2410.12345"),
        ("http://www.arxiv.org/abs/2410.12345#comments", "2410.12345"),
        ("arxiv.org/pdf/2410.12345.pdf", "2410.12345"),
        ("https://arxiv.org/pdf/2410.12345v1", "2410.12345"),
        ("https://arxiv.org/html/2410.12345v1/", "2410.12345"),
        ("<https://arxiv.org/abs/2410.12345>", "2410.12345"),
    ],
)
def test_parse_ref(token: str, ref: int | str) -> None:
    assert parse_ref(token) == ref


@pytest.mark.parametrize("token", ["0", "foo", "2410.123", "https://example.com/abs/2410.12345"])
def test_parse_ref_rejects(token: str) -> None:
    with pytest.raises(CommandError):
        parse_ref(token)


def test_bad_ref_is_quoted_as_code() -> None:
    # 用户输入放进行内代码，其中的 @ 提及和 HTML 都不会生效
    with pytest.raises(CommandError) as info:
        parse_ref("@someone`<b>`")
    assert "`@someone'<b>'`" in str(info.value)


# --- 解析论文 -------------------------------------------------------------------


def test_resolve_by_index_and_id(config: Config, store: DataStore) -> None:
    ctx, _ = context(config, store)
    papers, notes = resolve_papers([1, 4, "2610.00002", 1], ctx)
    # 序号 4 是第 2 篇候选；重复的论文只保留一次
    assert [p.id for p in papers] == ["2610.00001", "2610.00011", "2610.00002"]
    assert notes == []


@pytest.mark.parametrize(
    ("index", "message"),
    [
        (6, "序号 6 超出范围：2026-10-25 的日报推荐了 2 篇，另有 3 篇候选（序号 3–5，"),
        (99, "序号 99 超出范围"),
    ],
)
def test_index_out_of_range(config: Config, store: DataStore, index: int, message: str) -> None:
    ctx, _ = context(config, store)
    with pytest.raises(CommandError, match=message):
        resolve_papers([index], ctx)


def test_index_out_of_range_without_candidates(config: Config, tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    store.write_model(digest_path(DIGEST_DAY), interact_digest(others=[]))
    ctx, _ = context(config, store)
    with pytest.raises(CommandError) as info:
        resolve_papers([3], ctx)
    assert str(info.value) == "序号 3 超出范围：2026-10-25 的日报推荐了 2 篇"


def test_index_requires_daily_issue(config: Config, store: DataStore) -> None:
    ctx, _ = context(config, store, day=None)
    with pytest.raises(CommandError, match="不是日报"):
        resolve_papers([1], ctx)


def test_index_requires_saved_digest(config: Config, store: DataStore) -> None:
    ctx, _ = context(config, store, day=date(2026, 10, 20))
    with pytest.raises(CommandError, match="history/digests/2026-10-20.json"):
        resolve_papers([1], ctx)


def test_index_on_replaced_issue(config: Config, store: DataStore) -> None:
    # 同一天的日报用 --force 重新发布在 #3，旧 Issue #1 的序号不再可靠
    store.write_model(digest_path(DIGEST_DAY), interact_digest(issue_number=3))
    ctx, _ = context(config, store)
    with pytest.raises(CommandError, match="已重新发布在 #3"):
        resolve_papers([1], ctx)


@respx.mock
def test_resolve_unknown_id_through_arxiv_api(config: Config, store: DataStore) -> None:
    route = respx.get(ARXIV_API).mock(
        return_value=httpx.Response(200, text=api_feed(("2609.99999", "Old Paper")))
    )
    ctx, _ = context(config, store, day=None)  # 不在日报 Issue 下也能用 arXiv ID

    papers, notes = resolve_papers(["2609.99999", "2609.99999"], ctx)

    assert [(p.id, p.title, p.authors) for p in papers] == [
        ("2609.99999", "Old Paper", ["Erin Liu"])
    ]
    assert notes == []
    assert route.calls.last.request.url.params["id_list"] == "2609.99999"


@respx.mock
def test_resolve_id_missing_on_arxiv(config: Config, store: DataStore) -> None:
    respx.get(ARXIV_API).mock(return_value=httpx.Response(200, text=api_feed()))
    ctx, _ = context(config, store)
    with pytest.raises(CommandError, match="arXiv 上找不到 `2609.99999`"):
        resolve_papers(["2609.99999"], ctx)


@respx.mock
def test_resolve_when_arxiv_api_is_down(config: Config, store: DataStore) -> None:
    respx.get(ARXIV_API).mock(return_value=httpx.Response(503))
    ctx, _ = context(config, store)

    papers, notes = resolve_papers(["2609.99999"], ctx)

    assert [(p.id, p.title, p.abstract) for p in papers] == [("2609.99999", "2609.99999", "")]
    assert notes == ["arXiv API 暂时不可用，论文信息从全文中获取"]


def test_latest_digest_day(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    assert latest_digest_day(store) is None
    for day in ("2026-10-23", "2026-10-25", "not-a-date"):
        store.write_text(f"history/digests/{day}.json", "{}")
    assert latest_digest_day(store) == date(2026, 10, 25)


# --- /deep ----------------------------------------------------------------------


@respx.mock
def test_deep_generates_and_caches_report(config: Config, store: DataStore, html_text: str) -> None:
    mock_fulltext(html_text, "2610.00001")

    reply, fake = run("/deep 1", config, store)

    assert reply.startswith(
        "## 🔍 深读 · [LongKV: Hierarchical KV Cache Compression](https://arxiv.org/abs/2610.00001)"
    )
    assert "**作者**：Alice Zhang, Bob Li · **分类**：cs.CL" in reply
    assert DEEP_REPORT in reply
    assert "全文来源：arXiv HTML · 模型 deepseek-reasoner · 2026-10-26 生成 · token 120" in reply
    assert "<!-- gleaner:paper" not in reply
    assert "我研究长上下文推理。" in fake.requests[0]["messages"][0]["content"]

    report = load_report(store, "2610.00001")
    assert report is not None and report.body == DEEP_REPORT
    feedback = store.read_jsonl("feedback/2026-10.jsonl")
    assert [(f["paper_id"], f["signal"], f["weight"], f["source"]) for f in feedback] == [
        ("2610.00001", "deep", 0.5, "command")
    ]


@respx.mock
def test_deep_returns_cached_report(config: Config, store: DataStore, html_text: str) -> None:
    mock_fulltext(html_text, "2610.00001")
    run("/deep 1", config, store)
    html = respx.get(html_url("2610.00001"))
    calls = html.call_count

    reply, fake = run("/deep 2610.00001v2", config, store)

    assert fake.requests == []
    assert html.call_count == calls
    assert DEEP_REPORT in reply
    assert "来自缓存，评论 `/deep 2610.00001 --force` 重新生成" in reply
    assert len(store.read_jsonl("feedback/2026-10.jsonl")) == 2


@respx.mock
def test_deep_force_regenerates(config: Config, store: DataStore, html_text: str) -> None:
    mock_fulltext(html_text, "2610.00001")
    run("/deep 1", config, store)

    reply, fake = run("/deep 1 --force", config, store, handler=lambda request: "### 新报告")

    assert len(fake.requests) == 1
    assert "### 新报告" in reply
    assert "来自缓存" not in reply
    assert load_report(store, "2610.00001").body == "### 新报告"


@respx.mock
def test_deep_falls_back_to_abstract(config: Config, store: DataStore) -> None:
    respx.get(html_url("2610.00001")).mock(return_value=httpx.Response(404))
    respx.get(pdf_url("2610.00001")).mock(return_value=httpx.Response(503))

    reply, fake = run("/deep 1", config, store)

    assert "> ⚠️ 没能获取全文，以下分析仅基于摘要（HTML：HTTP 404；PDF：HTTP 503）。" in reply
    assert "全文来源：仅摘要" in reply
    assert "只拿到了摘要" in user_prompt(fake.requests[0])
    assert load_report(store, "2610.00001").source == "abstract"


@respx.mock
def test_deep_reads_pdf_when_html_is_missing(config: Config, store: DataStore) -> None:
    respx.get(html_url("2610.00001")).mock(return_value=httpx.Response(404))
    pdf = make_pdf([["1 Introduction", "PDF intro text.", "2 Method", "PDF method text."]])
    respx.get(pdf_url("2610.00001")).mock(return_value=httpx.Response(200, content=pdf))

    reply, fake = run("/deep 1", config, store)

    assert "全文来源：PDF" in reply
    assert "PDF method text." in user_prompt(fake.requests[0])


@respx.mock
def test_deep_without_any_content(config: Config, store: DataStore) -> None:
    respx.get(ARXIV_API).mock(return_value=httpx.Response(503))
    respx.get(html_url("2609.99999")).mock(return_value=httpx.Response(404))
    respx.get(pdf_url("2609.99999")).mock(return_value=httpx.Response(404))

    with pytest.raises(CommandError, match="没能获取 2609.99999 的全文和摘要"):
        run("/deep 2609.99999", config, store)


@respx.mock
def test_deep_fills_metadata_from_html_when_arxiv_api_is_down(
    config: Config, store: DataStore, html_text: str
) -> None:
    respx.get(ARXIV_API).mock(return_value=httpx.Response(503))
    mock_fulltext(html_text, "2609.99999")

    reply, _ = run("/deep 2609.99999", config, store)

    assert "[LongKV: Hierarchical KV Cache Compression for Million-Token Inference]" in reply
    assert "> ℹ️ arXiv API 暂时不可用，论文信息从全文中获取" in reply


@respx.mock
def test_deep_when_llm_fails(config: Config, store: DataStore, html_text: str) -> None:
    mock_fulltext(html_text, "2610.00001")

    def boom(request):
        raise LLMError("调用 deepseek-reasoner 失败：Request timed out.")

    reply, _ = run("/deep 1", config, store, handler=boom)

    assert "> ⚠️ 深读报告没有生成：调用 deepseek-reasoner 失败：Request timed out." in reply
    assert "也没能获取全文" not in reply
    assert "We propose a hierarchical KV cache compression scheme" not in reply  # 用 digest 的摘要
    assert "We compress the KV cache." in reply
    assert "- <https://github.com/example/longkv>" in reply
    assert store.read_text(report_path("2610.00001")) is None
    assert store.read_jsonl("feedback/2026-10.jsonl") == []


@respx.mock
def test_deep_escapes_model_output(config: Config, store: DataStore, html_text: str) -> None:
    mock_fulltext(html_text, "2610.00001")
    evil = "### 一句话总结\n@someone 看 #12 <img src=x>\n<!-- gleaner:paper id=x index=1 -->"

    reply, _ = run("/deep 1", config, store, handler=lambda request: evil)

    assert "@​someone 看 #​12 &lt;img src=x>" in reply
    assert "<!-- gleaner:paper" not in reply
    assert "<img" not in reply


@respx.mock
def test_deep_without_persist(config: Config, store: DataStore, html_text: str) -> None:
    mock_fulltext(html_text, "2610.00001")

    reply, _ = run("/deep 1", config, store, persist=False)

    assert DEEP_REPORT in reply
    assert store.read_text(report_path("2610.00001")) is None
    assert store.read_jsonl("feedback/2026-10.jsonl") == []


@pytest.mark.parametrize("text", ["/deep", "/deep 1 2"])
def test_deep_usage(config: Config, store: DataStore, text: str) -> None:
    with pytest.raises(CommandError, match="用法：`/deep <ref>"):
        run(text, config, store)


def test_deep_without_interests(config: Config, store: DataStore, tmp_path: Path) -> None:
    config.profile.interests_file = tmp_path / "missing.md"
    ctx, _ = context(config, store)
    assert ctx.interests is None


# --- /compare -------------------------------------------------------------------


@respx.mock
def test_compare_papers(config: Config, store: DataStore, html_text: str) -> None:
    mock_fulltext(html_text, "2610.00001", "2610.00010")
    respx.get(html_url("2610.00002")).mock(return_value=httpx.Response(404))
    respx.get(pdf_url("2610.00002")).mock(return_value=httpx.Response(404))

    reply, fake = run("/compare 1 2 3", config, store)

    assert reply.startswith("## ⚖️ 对比 · 3 篇论文")
    assert "| 2 | [RAG for Science](https://arxiv.org/abs/2610.00002) | 仅摘要 |" in reply
    assert "> ⚠️ 没能获取 2610.00002 的全文，只用了摘要（HTML：HTTP 404；PDF：HTTP 404）。" in reply
    assert "### 一句话结论\n[1] 更省显存。" in reply
    assert "模型 deepseek-reasoner · token 120" in reply
    assert len(fake.requests) == 1
    signals = [(f["paper_id"], f["signal"]) for f in store.read_jsonl("feedback/2026-10.jsonl")]
    assert signals == [
        ("2610.00001", "compare"),
        ("2610.00002", "compare"),
        ("2610.00010", "compare"),
    ]


@respx.mock
def test_compare_when_llm_fails(config: Config, store: DataStore, html_text: str) -> None:
    mock_fulltext(html_text, "2610.00001", "2610.00002")

    def boom(request):
        raise LLMError("额度用完")

    reply, _ = run("/compare 1 2", config, store, handler=boom)

    assert "> ⚠️ 对比没有生成：额度用完" in reply
    assert "**[2] RAG for Science**" in reply
    assert store.read_jsonl("feedback/2026-10.jsonl") == []


def test_compare_usage(config: Config, store: DataStore) -> None:
    with pytest.raises(CommandError, match="用法：`/compare"):
        run("/compare", config, store)
    with pytest.raises(CommandError, match="一次最多对比 5 篇"):
        run("/compare 1 2 3 4 5 1", config, store)


def s2_references() -> dict:
    cited = {
        "title": "KVQuant",
        "abstract": "Quantize the cache.",
        "year": 2024,
        "externalIds": {"ArXiv": "2401.18079"},
        "citationCount": 120,
        "url": "https://www.semanticscholar.org/paper/abc",
    }
    return {"data": [{"isInfluential": True, "citedPaper": cited}]}


@respx.mock
def test_compare_with_related_work(config: Config, store: DataStore, html_text: str) -> None:
    mock_fulltext(html_text, "2610.00001")
    respx.get(f"{S2}/graph/v1/paper/arXiv:2610.00001/references").mock(
        return_value=httpx.Response(200, json=s2_references())
    )
    respx.get(f"{S2}/recommendations/v1/papers/forpaper/arXiv:2610.00001").mock(
        return_value=httpx.Response(200, json={"recommendedPapers": []})
    )

    reply, fake = run("/compare 1", config, store)

    assert reply.startswith("## ⚖️ 与相关工作对比 · [LongKV")
    assert "### 定位\n延续了 R1。" in reply
    assert "<details><summary>Semantic Scholar 相关工作（1 篇）</summary>" in reply
    assert (
        "| R1 | [KVQuant](https://arxiv.org/abs/2401.18079) | 参考文献 ⭐ | 2024 | 120 |" in reply
    )
    assert "⚠️" not in reply
    assert 'relation="参考文献"' in user_prompt(fake.requests[0])
    assert store.read_jsonl("feedback/2026-10.jsonl")[0]["signal"] == "compare"


@respx.mock
def test_compare_when_semantic_scholar_fails(
    config: Config, store: DataStore, html_text: str
) -> None:
    mock_fulltext(html_text, "2610.00001")
    respx.get(f"{S2}/graph/v1/paper/arXiv:2610.00001/references").mock(
        return_value=httpx.Response(429)
    )
    respx.get(f"{S2}/recommendations/v1/papers/forpaper/arXiv:2610.00001").mock(
        return_value=httpx.Response(404)
    )

    reply, fake = run("/compare 2610.00001", config, store)

    assert "> ⚠️ 没能从 Semantic Scholar 获取相关工作（参考文献：Semantic Scholar 限流" in reply
    assert "以下对比只依据论文自己的相关工作章节" in reply
    assert "<details>" not in reply
    assert "### 定位" in reply
    assert "未能从 Semantic Scholar 获取相关工作" in user_prompt(fake.requests[0])


@respx.mock
def test_compare_related_when_llm_fails(config: Config, store: DataStore, html_text: str) -> None:
    mock_fulltext(html_text, "2610.00001")
    respx.get(f"{S2}/graph/v1/paper/arXiv:2610.00001/references").mock(
        return_value=httpx.Response(200, json=s2_references())
    )
    respx.get(f"{S2}/recommendations/v1/papers/forpaper/arXiv:2610.00001").mock(
        return_value=httpx.Response(200, json={"recommendedPapers": []})
    )

    def boom(request):
        raise LLMError("超时")

    reply, _ = run("/compare 1", config, store, handler=boom)

    assert "> ⚠️ 对比没有生成：超时" in reply
    assert "| R1 | [KVQuant](https://arxiv.org/abs/2401.18079) |" in reply


# --- /more ----------------------------------------------------------------------


def test_more(config: Config, store: DataStore) -> None:
    reply, fake = run("/more", config, store)

    assert reply.startswith("## 📋 更多候选 · 2026-10-25")
    assert "这份日报推荐了 2 篇，另有 3 篇打过分但没有推送，下面是第 3–5 篇：" in reply
    assert "### [3] Candidate 0 · 5/10" in reply
    assert "🎯 候选 0 的理由\n📝 候选 0 的一句话摘要" in reply
    assert "### [5] Candidate 2" in reply
    assert "评论 `/deep 3` 深读 · `/compare 3 4` 对比" in reply
    assert "/more 8" not in reply  # 已经全部列出
    assert fake.requests == []


def test_more_with_count(config: Config, store: DataStore) -> None:
    reply, _ = run("/more 1", config, store)
    assert "下面是第 3–3 篇" in reply
    assert "### [4]" not in reply
    assert "`/more 6` 查看更多" in reply


def test_more_without_candidates(config: Config, tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    store.write_model(digest_path(DIGEST_DAY), interact_digest(others=[]))
    reply, _ = run("/more", config, store)
    assert reply == "📭 2026-10-25 的日报没有其它候选。"


@pytest.mark.parametrize("text", ["/more abc", "/more 1 2"])
def test_more_usage(config: Config, store: DataStore, text: str) -> None:
    with pytest.raises(CommandError, match="用法：`/more"):
        run(text, config, store)


def test_more_escapes_candidates(config: Config, tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    digest = interact_digest()
    digest.others[0].title = "<b>Bold</b> @team"
    store.write_model(digest_path(DIGEST_DAY), Digest.model_validate(digest.model_dump()))

    reply, _ = run("/more 1", config, store)

    assert "### [3] &lt;b&gt;Bold&lt;/b&gt; @​team · 5/10" in reply


def test_help(config: Config, store: DataStore) -> None:
    reply, _ = run("/help", config, store)
    implemented = ("/deep <ref>", "/like <ref> [原因]", "/dislike <ref>", "/mute", "/unmute")
    for name in (*implemented, "/ask <问题>"):
        assert f"`{name}" in reply


@respx.mock
def test_deep_without_llm_or_full_text(config: Config, store: DataStore) -> None:
    respx.get(html_url("2610.00001")).mock(return_value=httpx.Response(404))
    respx.get(pdf_url("2610.00001")).mock(return_value=httpx.Response(404))
    digest = interact_digest()
    digest.papers[0].abstract = "We compress it. Code: https://github.com/example/longkv."
    store.write_model(digest_path(DIGEST_DAY), digest)

    def no_key():
        raise LLMError("未设置 LLM_API_KEY")

    ctx, _ = context(config, store)
    ctx.services.llm_factory = no_key

    reply = execute(parse_command("/deep 1"), ctx)

    assert "> ⚠️ 深读报告没有生成：未设置 LLM_API_KEY" in reply
    assert "> ⚠️ 也没能获取全文（HTML：HTTP 404；PDF：HTTP 404）。" in reply
    assert "- <https://github.com/example/longkv>" in reply


# --- /like、/dislike ------------------------------------------------------------


def feedback_rows(store: DataStore) -> list[tuple]:
    return [
        (r.paper_id, r.signal, r.weight, r.note, r.source)
        for r in load_feedback(store, DIGEST_DAY + timedelta(days=1))
    ]


def test_like_with_reason(config: Config, store: DataStore) -> None:
    reply, fake = run("/like 1 有开源代码，方法简单", config, store)

    assert fake.requests == []  # 不调用 LLM
    assert reply.startswith(
        "👍 已记下：喜欢 [LongKV: Hierarchical KV Cache Compression]"
        "(https://arxiv.org/abs/2610.00001)，原因：`有开源代码，方法简单`。"
    )
    assert "下一次日报的粗筛就会参考这条反馈" in reply
    assert feedback_rows(store) == [("2610.00001", "like", 2.0, "有开源代码，方法简单", "command")]
    saved = load_feedback(store, DIGEST_DAY + timedelta(days=1))[0]
    assert (saved.title, saved.abstract, saved.created_at) == (
        "LongKV: Hierarchical KV Cache Compression",
        "We compress the KV cache.",
        NOW,
    )


def test_dislike_without_reason(config: Config, store: DataStore) -> None:
    reply, _ = run("/dislike 5", config, store)  # 序号 5 是 /more 里的第 3 篇候选

    assert reply.startswith("👎 已记下：不喜欢 [Candidate 2](https://arxiv.org/abs/2610.00012)。")
    assert "原因" not in reply
    assert feedback_rows(store) == [("2610.00012", "dislike", -2.0, None, "command")]


def test_dislike_replaces_like(config: Config, store: DataStore) -> None:
    run("/like 2", config, store)
    reply, _ = run("/dislike 2：纯 benchmark", config, store)

    assert "原因：`纯 benchmark`（覆盖了之前的 /like）。" in reply
    assert [row[1] for row in feedback_rows(store)] == ["like", "dislike"]


def test_like_escapes_reason(config: Config, store: DataStore) -> None:
    reply, _ = run("/like 1 @team 看 #12 <img src=x> `code`", config, store)
    # 原因放进行内代码：@ 提及、#123 和 HTML 都不会生效
    assert "原因：`@team 看 #12 <img src=x> 'code'`。" in reply


@pytest.mark.parametrize("text", ["/like 9 很好", "/dislike 99"])
def test_like_index_out_of_range(config: Config, store: DataStore, text: str) -> None:
    with pytest.raises(CommandError, match="超出范围"):
        run(text, config, store)
    assert feedback_rows(store) == []


@pytest.mark.parametrize("text", ["/like", "/dislike   "])
def test_like_usage(config: Config, store: DataStore, text: str) -> None:
    with pytest.raises(CommandError, match="用法：`/(like|dislike) <ref> \\[原因\\]`"):
        run(text, config, store)


@respx.mock
def test_like_by_arxiv_id(config: Config, store: DataStore) -> None:
    respx.get(ARXIV_API).mock(
        return_value=httpx.Response(200, text=api_feed(("2609.99999", "Old Paper")))
    )

    reply, _ = run("/like https://arxiv.org/abs/2609.99999v2 经典", config, store, day=None)

    assert "[Old Paper](https://arxiv.org/abs/2609.99999)" in reply
    assert feedback_rows(store)[0][:2] == ("2609.99999", "like")


@respx.mock
def test_like_when_arxiv_api_is_down(config: Config, store: DataStore) -> None:
    respx.get(ARXIV_API).mock(return_value=httpx.Response(503))

    reply, _ = run("/like 2609.99999 经典方法", config, store)

    assert "[2609.99999](https://arxiv.org/abs/2609.99999)，原因：`经典方法`。" in reply
    assert "> ℹ️ arXiv API 暂时不可用，只记下了论文 ID：这条反馈不参与粗筛" in reply
    assert "从全文中获取" not in reply
    assert feedback_rows(store)[0][:4] == ("2609.99999", "like", 2.0, "经典方法")


def test_like_without_persist(config: Config, store: DataStore) -> None:
    reply, _ = run("/like 1", config, store, persist=False)
    assert reply.startswith("👍 已记下")
    assert feedback_rows(store) == []


# --- /mute、/unmute -------------------------------------------------------------

MUTE_QUERY = "Deep learning methods for medical image segmentation and diagnosis."


def describer(request) -> str:
    return json.dumps({"query": MUTE_QUERY})


def test_mute_generates_and_caches_english_query(config: Config, store: DataStore) -> None:
    reply, fake = run("/mute 医学影像", config, store, handler=describer)

    assert len(fake.requests) == 1
    assert fake.requests[0]["model"] == "deepseek-chat"
    assert "<topic>\n医学影像\n</topic>" in user_prompt(fake.requests[0])
    assert reply.startswith("🔇 已屏蔽 `医学影像`，之后的日报会把这类论文排到后面。")
    assert f"英文检索描述：{MUTE_QUERY}" in reply
    assert "当前屏蔽的主题：`医学影像` · 评论 `/unmute 医学影像` 取消屏蔽" in reply
    [event] = load_events(store)
    assert (event.action, event.topic, event.query, event.model, event.created_at) == (
        "mute",
        "医学影像",
        MUTE_QUERY,
        "deepseek-chat",
        NOW,
    )


def test_mute_twice(config: Config, store: DataStore) -> None:
    run("/mute Medical Imaging", config, store, handler=describer)

    reply, fake = run("/mute   medical   IMAGING ", config, store, handler=describer)

    assert fake.requests == []
    assert reply.startswith("🔇 `Medical Imaging` 已经屏蔽过了。")
    assert len(load_events(store)) == 1


def test_mute_when_llm_fails(config: Config, store: DataStore) -> None:
    def boom(request):
        raise LLMError("额度用完")

    reply, _ = run("/mute 医学影像", config, store, handler=boom)

    assert "> ⚠️ 英文检索描述暂时没有生成，下次运行日报时会自动补上。" in reply
    [topic] = load_muted(store)
    assert (topic.topic, topic.query) == ("医学影像", None)


def test_mute_lists_topics(config: Config, store: DataStore) -> None:
    empty, _ = run("/mute", config, store)
    assert empty.startswith("🔇 当前没有屏蔽任何主题。")

    run("/mute 医学影像", config, store, handler=describer)
    run("/mute speech <b>", config, store, handler=describer)
    reply, fake = run("/mute", config, store)

    assert fake.requests == []
    assert "🔇 当前屏蔽了 2 个主题：\n\n- `医学影像`\n- `speech <b>`" in reply


def test_mute_rejects_long_topics(config: Config, store: DataStore) -> None:
    with pytest.raises(CommandError, match="主题太长（51 个字）"):
        run("/mute " + "长" * 51, config, store, handler=describer)


def test_unmute(config: Config, store: DataStore) -> None:
    run("/mute 医学影像", config, store, handler=describer)
    later = NOW + timedelta(minutes=1)

    reply, _ = run("/unmute 医学影像", config, store, now=later)

    assert reply == "🔊 已取消屏蔽 `医学影像`。\n\n<sub>当前没有屏蔽任何主题</sub>"
    assert [e.action for e in load_events(store)] == ["mute", "unmute"]
    assert load_muted(store) == []

    # 再次屏蔽：回放 mute → unmute → mute
    run("/mute 医学影像", config, store, handler=describer, now=later + timedelta(minutes=1))
    assert [m.topic for m in load_muted(store)] == ["医学影像"]


def test_unmute_unknown_topic_lists_current(config: Config, store: DataStore) -> None:
    run("/mute 医学影像", config, store, handler=describer)

    with pytest.raises(CommandError) as info:
        run("/unmute 语音", config, store)

    assert str(info.value) == "`语音` 不在屏蔽列表中。当前屏蔽的主题：`医学影像`"
    with pytest.raises(CommandError, match="用法：`/unmute <主题>`。当前屏蔽的主题：`医学影像`"):
        run("/unmute", config, store)


def test_mute_without_persist(config: Config, store: DataStore) -> None:
    reply, _ = run("/mute 医学影像", config, store, handler=describer, persist=False)
    assert reply.startswith("🔇 已屏蔽")
    assert load_events(store) == []
