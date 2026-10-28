from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx

from gleaner.agent import Context
from gleaner.agent.tools import MAX_RESULT_TOKENS, TOOLS, ToolError
from gleaner.config import Config
from gleaner.llm.prompt import estimate_tokens
from gleaner.store import DataStore
from tests.conftest import (
    CACHED_REPORT,
    DIGEST_DAY,
    interact_services,
    seed_agent_data,
)

NOW = datetime(2026, 10, 26, 2, 0, tzinfo=UTC)
ARXIV_API = "https://export.arxiv.org/api/query"
S2 = "https://api.semanticscholar.org"
HTML = "https://arxiv.org/html/2610.00001"


@pytest.fixture
def ctx(tmp_path: Path) -> Context:
    interests = tmp_path / "interests.md"
    interests.write_text("我研究单源域泛化的目标检测。", encoding="utf-8")
    config = Config.model_validate({"profile": {"interests_file": str(interests)}})
    store = DataStore(tmp_path / "data")
    seed_agent_data(store)
    services, _ = interact_services()
    return Context(config, services, store, day=DIGEST_DAY, issue_number=1, persist=False, now=NOW)


def call(ctx: Context, name: str, **arguments) -> tuple[str, dict[str, str]]:
    found: dict[str, str] = {}
    return TOOLS[name].call(ctx, arguments, found), found


def api_feed(*entries: tuple[str, str]) -> str:
    items = "".join(
        f"<entry><id>http://arxiv.org/abs/{paper_id}v1</id><title>{title}</title>"
        f"<summary>Abstract of {title}.</summary><published>2026-10-01T00:00:00Z</published>"
        f'<author><name>Erin Liu</name></author><category term="cs.CV"/></entry>'
        for paper_id, title in entries
    )
    return f'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">{items}</feed>'


# --- get_digest -----------------------------------------------------------------


def test_get_digest_defaults_to_current_issue(ctx: Context) -> None:
    text, found = call(ctx, "get_digest")

    assert text.startswith("2026-10-25 的日报：推荐 2 篇，另有 3 篇")
    assert "[1] 2610.00001 · 9/10 · LongKV: Hierarchical KV Cache Compression" in text
    assert "[3] 2610.00010 · 5/10 · Candidate 0\n    理由：候选 0 的理由" in text
    assert found["2610.00002"] == "RAG for Science"


def test_get_digest_by_date(ctx: Context) -> None:
    text, _ = call(ctx, "get_digest", date="2026-10-20")
    assert "[1] 2610.00100 · 8/10 · Domain Generalization for Object Detectors" in text


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"date": "2026-13-01"}, "参数不合法：date"),
        ({"date": "yesterday"}, "日期必须写成 YYYY-MM-DD"),
        ({"day": "2026-10-25"}, "参数不合法：day"),  # 不认识的参数
        ({"date": "2026-10-01"}, "没有 2026-10-01 的日报，最近一份是 2026-10-25"),
    ],
)
def test_get_digest_rejects_bad_arguments(ctx: Context, arguments: dict, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        TOOLS["get_digest"].call(ctx, arguments)


# --- get_paper ------------------------------------------------------------------


def test_get_paper_by_index_with_cached_report(ctx: Context) -> None:
    text, found = call(ctx, "get_paper", ref="2")

    assert "arXiv ID：2610.00002\n标题：RAG for Science" in text
    assert "已有深读报告（2026-10-25，deepseek-reasoner）" in text
    assert CACHED_REPORT in text
    assert found == {"2610.00002": "RAG for Science"}


@respx.mock
def test_get_paper_outside_digest_uses_arxiv_api(ctx: Context) -> None:
    respx.get(ARXIV_API).mock(
        return_value=httpx.Response(200, text=api_feed(("2401.00001", "Outside Paper")))
    )

    text, _ = call(ctx, "get_paper", ref="https://arxiv.org/abs/2401.00001v2")

    assert "标题：Outside Paper" in text and "摘要：Abstract of Outside Paper." in text


def test_get_paper_index_out_of_range(ctx: Context) -> None:
    with pytest.raises(ToolError, match="序号 9 超出范围"):
        call(ctx, "get_paper", ref="9")


# --- read_fulltext --------------------------------------------------------------


@respx.mock
def test_read_fulltext(ctx: Context, html_text: str) -> None:
    respx.get(HTML).mock(return_value=httpx.Response(200, text=html_text))

    text, found = call(ctx, "read_fulltext", arxiv_id="2610.00001")

    assert "全文来源：arXiv HTML 全文" in text
    assert "章节：" in text
    assert found["2610.00001"].startswith("LongKV")
    assert estimate_tokens(text) <= MAX_RESULT_TOKENS


@respx.mock
def test_read_fulltext_selected_sections(ctx: Context, html_text: str) -> None:
    respx.get(HTML).mock(return_value=httpx.Response(200, text=html_text))

    whole, _ = call(ctx, "read_fulltext", arxiv_id="2610.00001")
    method, _ = call(ctx, "read_fulltext", arxiv_id="2610.00001", sections=["method"])
    missing, _ = call(ctx, "read_fulltext", arxiv_id="2610.00001", sections=["nonexistent"])

    assert len(method) < len(whole)
    assert "没有找到标题含有 nonexistent 的章节" in missing


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"arxiv_id": "abc"}, "无法识别"),
        ({"arxiv_id": "3"}, "这里要写 arXiv ID，不是序号"),
        ({"arxiv_id": "2610.00001", "sections": ["x" * 81]}, "章节名太长"),
        ({}, "arxiv_id"),
    ],
)
def test_read_fulltext_validates_arguments(ctx: Context, arguments: dict, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        TOOLS["read_fulltext"].call(ctx, arguments)


# --- search_arxiv ---------------------------------------------------------------


@respx.mock
def test_search_arxiv(ctx: Context) -> None:
    route = respx.get(ARXIV_API).mock(
        return_value=httpx.Response(200, text=api_feed(("2410.11111", "DG Detector")))
    )

    text, found = call(ctx, "search_arxiv", query='domain "generalization" AND detector 检测')

    query = route.calls.last.request.url.params["search_query"]
    assert query == "all:domain AND all:generalization AND all:detector"
    assert route.calls.last.request.url.params["sortBy"] == "relevance"
    assert "- 2410.11111 · DG Detector（2026-10-01）" in text
    assert found == {"2410.11111": "DG Detector"}


def test_search_arxiv_needs_english_terms(ctx: Context) -> None:
    text, _ = call(ctx, "search_arxiv", query="单源域泛化")
    assert "只支持英文关键词" in text


@respx.mock
def test_search_arxiv_failure_is_a_tool_error(ctx: Context) -> None:
    respx.get(ARXIV_API).mock(return_value=httpx.Response(503))
    with pytest.raises(ToolError, match="外部服务暂时不可用"):
        call(ctx, "search_arxiv", query="detector")


def test_query_length_is_limited(ctx: Context) -> None:
    with pytest.raises(ToolError, match="参数不合法：query"):
        call(ctx, "search_history", query="x" * 201)
    with pytest.raises(ToolError, match="查询不能为空"):
        call(ctx, "search_history", query="   ")


# --- related_works --------------------------------------------------------------


@respx.mock
def test_related_works(ctx: Context) -> None:
    cited = {
        "title": "KVQuant",
        "abstract": "Quantize KV cache.",
        "year": 2024,
        "externalIds": {"ArXiv": "2401.18079"},
        "citationCount": 120,
    }
    respx.get(f"{S2}/graph/v1/paper/arXiv:2610.00001/references").mock(
        return_value=httpx.Response(
            200, json={"data": [{"isInfluential": True, "citedPaper": cited}]}
        )
    )
    respx.get(f"{S2}/recommendations/v1/papers/forpaper/arXiv:2610.00001").mock(
        return_value=httpx.Response(429)
    )

    text, found = call(ctx, "related_works", arxiv_id="2610.00001")

    assert "- KVQuant（参考文献，2024，被引 120，arXiv 2401.18079）" in text
    assert "部分数据没有获取到：推荐论文" in text
    assert found == {"2401.18079": "KVQuant"}


# --- get_profile、search_history -----------------------------------------------


def test_get_profile(ctx: Context) -> None:
    text, _ = call(ctx, "get_profile")

    assert "我研究单源域泛化的目标检测。" in text
    assert "- 偏好有开源代码的工作" in text
    assert "屏蔽的主题：医学影像" in text


def test_search_history_over_digests_and_feedback(ctx: Context) -> None:
    text, found = call(ctx, "search_history", query="domain generalization detector")

    first = text.split("\n- ")[1]
    assert first.startswith("2610.00100 · Domain Generalization for Object Detectors")
    assert "推送：2026-10-20 第 1 篇（推荐，8/10）" in first
    assert "反馈：/like（原因：有开源代码）" in first
    assert "2610.00100" in found


def test_search_history_finds_feedback_only_papers(ctx: Context) -> None:
    text, _ = call(ctx, "search_history", query="style randomization")
    assert "- 2609.00900 · Style Randomization\n  反馈：/dislike" in text


def test_search_history_without_match(ctx: Context) -> None:
    text, found = call(ctx, "search_history", query="quantum chemistry")
    assert text.startswith("历史中没有与 quantum chemistry 匹配的论文") and found == {}


def test_tool_schemas_are_compact() -> None:
    schema = TOOLS["get_digest"].schema()
    assert schema == {
        "properties": {
            "date": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "description": TOOLS["get_digest"].params.model_fields["date"].description,
            }
        },
        "required": [],
    }
