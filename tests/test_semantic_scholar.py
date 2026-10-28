import httpx
import pytest
import respx

from gleaner.sources import SemanticScholar

API = "https://api.semanticscholar.org"
REFERENCES = f"{API}/graph/v1/paper/arXiv:2610.00001/references"
RECOMMENDATIONS = f"{API}/recommendations/v1/papers/forpaper/arXiv:2610.00001"


def s2_paper(title: str | None, *, arxiv: str | None = None, citations: int = 0, **extra) -> dict:
    return {
        "paperId": "abc",
        "title": title,
        "abstract": f"Abstract of {title}.",
        "year": 2025,
        "externalIds": {"ArXiv": arxiv} if arxiv else {"DOI": "10.1/x"},
        "citationCount": citations,
        "url": "https://www.semanticscholar.org/paper/abc",
        **extra,
    }


def client(**kwargs) -> SemanticScholar:
    return SemanticScholar(backoff_seconds=0, **kwargs)


@respx.mock
def test_references_sorted_by_influence_and_citations() -> None:
    route = respx.get(REFERENCES).mock(
        return_value=httpx.Response(
            200,
            json={
                "offset": 0,
                "data": [
                    {"isInfluential": False, "citedPaper": s2_paper("Popular", citations=900)},
                    {"isInfluential": True, "citedPaper": s2_paper("Key", arxiv="2401.00001")},
                    {"isInfluential": False, "citedPaper": s2_paper(None)},  # 没有标题，跳过
                    {"isInfluential": None, "citedPaper": s2_paper("Minor", citations=3)},
                ],
            },
        )
    )

    papers = client(api_key="s2-key").references("2610.00001")

    assert [p.title for p in papers] == ["Key", "Popular", "Minor"]
    assert papers[0].influential is True
    assert papers[0].arxiv_id == "2401.00001"
    assert papers[0].relation == "reference"
    assert papers[1].arxiv_id is None
    request = route.calls.last.request
    assert request.headers["x-api-key"] == "s2-key"
    assert request.url.params["fields"].startswith("isInfluential,title,abstract")


@respx.mock
def test_recommendations() -> None:
    respx.get(RECOMMENDATIONS).mock(
        return_value=httpx.Response(
            200, json={"recommendedPapers": [s2_paper("Similar", arxiv="2609.11111")]}
        )
    )

    papers = client().recommendations("2610.00001")

    assert [(p.title, p.relation, p.year) for p in papers] == [("Similar", "recommended", 2025)]
    assert "x-api-key" not in respx.calls.last.request.headers


@respx.mock
def test_related_works_merges_and_deduplicates() -> None:
    respx.get(REFERENCES).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [{"citedPaper": s2_paper(f"Ref {i}", citations=10 - i)} for i in range(5)]
                + [{"citedPaper": s2_paper("Shared", arxiv="2609.11111")}]
            },
        )
    )
    respx.get(RECOMMENDATIONS).mock(
        return_value=httpx.Response(
            200,
            json={
                "recommendedPapers": [
                    s2_paper("Shared again", arxiv="2609.11111"),
                    s2_paper("Itself", arxiv="2610.00001"),
                    s2_paper("New"),
                ]
            },
        )
    )

    related = client().related_works("2610.00001", max_references=3, max_recommendations=2)

    assert [(p.title, p.relation) for p in related.papers] == [
        ("Ref 0", "reference"),
        ("Ref 1", "reference"),
        ("Ref 2", "reference"),
        ("Shared again", "recommended"),
        ("New", "recommended"),
    ]
    assert related.errors == []


@respx.mock
def test_related_works_degrades_when_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("gleaner.sources.semantic_scholar.time.sleep", sleeps.append)
    references = respx.get(REFERENCES).mock(
        return_value=httpx.Response(429, headers={"Retry-After": "5"})
    )
    respx.get(RECOMMENDATIONS).mock(
        return_value=httpx.Response(200, json={"recommendedPapers": [s2_paper("Similar")]})
    )

    related = client().related_works("2610.00001")

    assert references.call_count == 2  # 重试一次
    assert sleeps == [5.0]
    assert [p.title for p in related.papers] == ["Similar"]
    assert related.errors == ["参考文献：Semantic Scholar 限流（HTTP 429）"]


@respx.mock
def test_related_works_when_paper_is_unknown() -> None:
    respx.get(REFERENCES).mock(return_value=httpx.Response(404, json={"error": "not found"}))
    respx.get(RECOMMENDATIONS).mock(side_effect=httpx.ConnectTimeout("slow"))

    related = client().related_works("2610.00001")

    assert related.papers == []
    assert related.errors == [
        "参考文献：Semantic Scholar 尚未收录这篇论文",
        "推荐论文：请求失败（ConnectTimeout）",
    ]


@respx.mock
def test_unexpected_response_shape() -> None:
    respx.get(RECOMMENDATIONS).mock(
        return_value=httpx.Response(200, json={"recommendedPapers": "oops"})
    )
    related = client().related_works("2610.00001", max_references=0)
    assert "推荐论文：响应格式不符合预期" in related.errors
