from datetime import UTC, datetime

import httpx
import pytest
import respx

from gleaner.config import ArxivConfig
from gleaner.sources import ArxivSource, SourceError
from gleaner.sources.arxiv import RateLimiter, parse_api, parse_rss

FEED_URL = "https://rss.arxiv.org/rss/cs.CL+cs.LG"


def test_parse_rss_fields(rss_bytes: bytes) -> None:
    papers = parse_rss(rss_bytes)
    assert [p.id for p in papers] == [
        "2610.00001",
        "2610.00002",
        "2610.00003",
        "2509.12345",
        "2508.54321",
    ]

    first = papers[0]
    assert first.title == "LongKV: Hierarchical KV Cache Compression for Million-Token Inference"
    assert first.abstract == (
        "We propose a hierarchical KV cache compression scheme "
        "that keeps perplexity nearly unchanged."
    )
    assert first.authors == ["Alice Zhang", "Bob Li", "Carol Wang", "David Chen"]
    assert first.categories == ["cs.CL", "cs.LG"]
    assert first.announce_type == "new"
    assert first.announced == datetime(2026, 10, 22, 4, 0, tzinfo=UTC)
    assert first.url == "https://arxiv.org/abs/2610.00001"
    assert first.pdf_url == "https://arxiv.org/pdf/2610.00001"

    assert [p.announce_type for p in papers[2:]] == ["cross", "replace", "replace-cross"]


def test_parse_rss_deduplicates(rss_bytes: bytes) -> None:
    text = rss_bytes.decode()
    first_item = text[text.index("<item>") : text.index("</item>") + len("</item>")]
    doubled = text.replace(first_item, first_item * 2, 1).encode()
    assert len(parse_rss(doubled)) == len(parse_rss(rss_bytes))


def test_parse_empty_feed(empty_rss_bytes: bytes) -> None:
    assert parse_rss(empty_rss_bytes) == []


@pytest.mark.parametrize(
    "content",
    [
        b"<html><body>Service Unavailable",
        b"<html><body><p>Down for maintenance</p></body></html>",
    ],
)
def test_parse_non_feed_raises(content: bytes) -> None:
    with pytest.raises(SourceError):
        parse_rss(content)


@respx.mock
def test_fetch_latest_default_filters(rss_bytes: bytes) -> None:
    respx.get(FEED_URL).mock(return_value=httpx.Response(200, content=rss_bytes))
    source = ArxivSource(ArxivConfig(categories=["cs.CL", "cs.LG"]))

    papers = source.fetch_latest()

    # 默认包含 new 和 cross，不包含 replace
    assert [p.announce_type for p in papers] == ["new", "new", "cross"]


@respx.mock
def test_fetch_latest_with_replace_without_cross(rss_bytes: bytes) -> None:
    respx.get(FEED_URL).mock(return_value=httpx.Response(200, content=rss_bytes))
    config = ArxivConfig(
        categories=["cs.CL", "cs.LG"], include_cross_list=False, include_replace=True
    )

    papers = ArxivSource(config).fetch_latest()

    assert [p.announce_type for p in papers] == ["new", "new", "replace", "replace-cross"]


@respx.mock
def test_fetch_latest_http_error() -> None:
    respx.get(FEED_URL).mock(return_value=httpx.Response(503))
    source = ArxivSource(ArxivConfig(categories=["cs.CL", "cs.LG"]))

    with pytest.raises(SourceError, match="503"):
        source.fetch_latest()


@respx.mock
def test_fetch_latest_sends_user_agent(rss_bytes: bytes) -> None:
    route = respx.get(FEED_URL).mock(return_value=httpx.Response(200, content=rss_bytes))

    ArxivSource(ArxivConfig(categories=["cs.CL", "cs.LG"])).fetch_latest()

    assert route.calls.last.request.headers["User-Agent"].startswith("paper-gleaner/")


API_URL = "https://export.arxiv.org/api/query"


def test_parse_api(api_bytes: bytes) -> None:
    papers = parse_api(api_bytes)

    assert [p.id for p in papers] == ["2610.00001", "2610.00002"]
    first = papers[0]
    assert first.title == "LongKV: Hierarchical KV Cache Compression for Million-Token Inference"
    assert first.abstract == (
        "We propose a hierarchical KV cache compression scheme "
        "that keeps perplexity nearly unchanged."
    )
    assert first.authors == ["Alice Zhang", "Bob Li"]
    assert first.categories == ["cs.CL", "cs.LG"]
    assert first.announce_type is None
    assert first.announced == datetime(2026, 10, 23, 17, 59, 59, tzinfo=UTC)


def test_parse_api_garbage_raises() -> None:
    with pytest.raises(SourceError, match="arXiv API"):
        parse_api(b"<html><body>Rate exceeded</body></html>")


@respx.mock
def test_fetch_recent(api_bytes: bytes) -> None:
    route = respx.get(API_URL).mock(return_value=httpx.Response(200, content=api_bytes))
    source = ArxivSource(ArxivConfig(categories=["cs.CL", "cs.LG"]))

    papers = source.fetch_recent(20)

    assert len(papers) == 2
    params = route.calls.last.request.url.params
    assert params["search_query"] == "cat:cs.CL OR cat:cs.LG"
    assert params["sortBy"] == "submittedDate"
    assert params["sortOrder"] == "descending"
    assert params["max_results"] == "20"


@respx.mock
def test_fetch_recent_http_error() -> None:
    respx.get(API_URL).mock(return_value=httpx.Response(503))
    with pytest.raises(SourceError, match="抓取 arXiv API 失败"):
        ArxivSource(ArxivConfig()).fetch_recent(20)


@respx.mock
def test_fetch_by_ids(api_bytes: bytes) -> None:
    route = respx.get(API_URL).mock(return_value=httpx.Response(200, content=api_bytes))

    papers = ArxivSource(ArxivConfig()).fetch_by_ids(["2610.00001", "2610.00002"])

    assert [p.id for p in papers] == ["2610.00001", "2610.00002"]
    params = route.calls.last.request.url.params
    assert params["id_list"] == "2610.00001,2610.00002"
    assert params["max_results"] == "2"
    assert "search_query" not in params


def test_fetch_by_ids_without_ids_makes_no_request() -> None:
    assert ArxivSource(ArxivConfig()).fetch_by_ids([]) == []


def test_parse_api_skips_error_entries() -> None:
    # 格式不对的 ID 会得到一条错误条目，它的 id 里也带着像 arXiv ID 的数字
    content = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query: id_list=2610.123456</title>
  <id>http://arxiv.org/api/abc</id>
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format_for_2610.123456</id>
    <title>Error</title>
    <summary>incorrect id format for 2610.123456</summary>
  </entry>
</feed>"""
    assert parse_api(content) == []


@respx.mock
def test_api_requests_are_spaced(api_bytes: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("gleaner.sources.arxiv.time.sleep", sleeps.append)
    respx.get(API_URL).mock(return_value=httpx.Response(200, content=api_bytes))
    source = ArxivSource(ArxivConfig(), limiter=RateLimiter(3.0))

    source.fetch_by_ids(["2610.00001"])
    source.fetch_recent(20)

    assert len(sleeps) == 1  # 第一次请求不等待，之后与上一次至少间隔 3 秒
    assert 2.5 < sleeps[0] <= 3.0
