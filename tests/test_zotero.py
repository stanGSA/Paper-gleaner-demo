"""Zotero Web API v3 的读取、缓存、去重和选取。

响应按官方文档手写：条目和 collection 都是 {key, version, library, links, meta, data}，
多对象请求带 Total-Results 和 Last-Modified-Version 响应头。
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr

from gleaner.config import ZoteroConfig
from gleaner.models import ZoteroStats
from gleaner.render.markdown import zotero_line
from gleaner.sources.zotero import (
    ZoteroClient,
    ZoteroConfigError,
    ZoteroItem,
    ZoteroLibrary,
    deduplicate,
    parse_item,
    resolve_collections,
    select_items,
)

API = "https://api.zotero.org/users/1234567"
GROUP_API = "https://api.zotero.org/groups/7654321"
KEY = "zotero-api-key-xyz"
BASE = datetime(2026, 10, 1, tzinfo=UTC)


def item(
    key: str,
    title: str,
    *,
    days: int = 0,
    item_type: str = "journalArticle",
    collections: list[str] | None = None,
    abstract: str = "",
    **data: Any,
) -> dict[str, Any]:
    """一个条目：days 越大加入文库越晚。"""
    added = (BASE + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "key": key,
        "version": 10,
        "library": {"type": "user", "id": 1234567, "name": "me", "links": {}},
        "links": {"self": {"href": f"{API}/items/{key}", "type": "application/json"}},
        "meta": {"creatorSummary": "Zhang et al.", "numChildren": 1},
        "data": {
            "key": key,
            "version": 10,
            "itemType": item_type,
            "title": title,
            "creators": [{"creatorType": "author", "firstName": "San", "lastName": "Zhang"}],
            "abstractNote": abstract,
            "date": "2026",
            "url": "",
            "extra": "",
            "tags": [],
            "collections": collections or [],
            "relations": {},
            "dateAdded": added,
            "dateModified": added,
            **data,
        },
    }


def collection(key: str, name: str, parent: str | bool = False) -> dict[str, Any]:
    return {
        "key": key,
        "version": 5,
        "library": {"type": "user", "id": 1234567, "name": "me", "links": {}},
        "links": {},
        "meta": {"numCollections": 0, "numItems": 1},
        "data": {
            "key": key,
            "version": 5,
            "name": name,
            "parentCollection": parent,
            "relations": {},
        },
    }


def page(items: list[dict], *, total: int | None = None, version: int = 100) -> httpx.Response:
    headers = {
        "Total-Results": str(len(items) if total is None else total),
        "Last-Modified-Version": str(version),
        "Zotero-API-Version": "3",
    }
    return httpx.Response(200, json=items, headers=headers)


def paged(items: list[dict], *, version: int = 100):
    """按请求的 start、limit 返回对应的一页。"""

    def respond(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params.get("start", 0))
        limit = int(request.url.params.get("limit", 25))
        return page(items[start : start + limit], total=len(items), version=version)

    return respond


def config(**fields: Any) -> ZoteroConfig:
    values = {"enabled": True} | fields
    zotero = ZoteroConfig.model_validate(values)
    zotero.library_id = SecretStr("7654321" if fields.get("library_type") == "group" else "1234567")
    zotero.api_key = SecretStr(KEY)
    return zotero


def library(tmp_path: Path, **fields: Any) -> ZoteroLibrary:
    zotero = config(**fields)
    client = ZoteroClient(
        zotero.library_type,
        zotero.library_id.get_secret_value(),
        KEY,
        backoff_seconds=0,
        sleep=lambda seconds: None,
    )
    return ZoteroLibrary(zotero, client=client, cache_dir=tmp_path / "zotero")


def probe(version: int = 100) -> httpx.Response:
    return page([item("PROBE001", "x")], total=1, version=version)


# --- 读取整个文库 ---------------------------------------------------------------


@respx.mock
def test_reads_whole_library_with_pagination(tmp_path: Path) -> None:
    items = [item(f"ITEM{i:04d}", f"Paper {i}", days=i) for i in range(150)]
    route = respx.get(f"{API}/items/top").mock(side_effect=paged(items))

    result = library(tmp_path, max_items=500).load()

    # 第一次是检查版本的请求，之后两页
    pages = [call.request.url.params for call in route.calls[1:]]
    assert [(p["start"], p["limit"]) for p in pages] == [("0", "100"), ("100", "100")]
    assert all(p["itemType"] == "-attachment" for p in pages)
    headers = route.calls[1].request.headers
    assert headers["Zotero-API-Key"] == KEY and headers["Zotero-API-Version"] == "3"
    assert result.stats == ZoteroStats(used=150, total=150, source="api")
    assert result.papers[0].title == "Paper 149"  # 最新加入的在前
    assert result.warnings == []


@respx.mock
def test_group_library_prefix(tmp_path: Path) -> None:
    route = respx.get(f"{GROUP_API}/items/top").mock(
        side_effect=paged([item("ITEM0001", "Group paper")])
    )

    result = library(tmp_path, library_type="group").load()

    assert route.called
    assert [p.title for p in result.papers] == ["Group paper"]


def test_skips_attachments_notes_trash_and_untitled() -> None:
    raw = [
        item("ATTACH01", "paper.pdf", item_type="attachment"),
        item("NOTE0001", "", item_type="note", note="<p>my note</p>"),
        item("ANNOT001", "", item_type="annotation"),
        item("TRASH001", "Trashed paper", deleted=1),
        item("CHILD001", "Child", parentItem="PARENT01"),
        item("EMPTY001", "   "),
        item("KEEP0001", "Kept paper", abstract="An abstract."),
    ]

    parsed = [parse_item(entry, set()) for entry in raw]

    assert [p.key for p in parsed if p is not None] == ["KEEP0001"]


def test_extracts_doi_and_arxiv_id() -> None:
    preprint = parse_item(
        item("PRE00001", "P", item_type="preprint", archiveID="arXiv:2410.12345"), set()
    )
    by_url = parse_item(item("URL00001", "P", url="https://arxiv.org/abs/2410.12345v2"), set())
    by_doi = parse_item(item("DOI00001", "P", DOI="https://doi.org/10.1145/3290605.3300857"), set())
    by_extra = parse_item(
        item("EXT00001", "P", extra="DOI: 10.1000/XYZ.1\narXiv: 2401.00001"), set()
    )

    assert preprint.arxiv_id == by_url.arxiv_id == "2410.12345"
    assert by_doi.doi == "10.1145/3290605.3300857"
    assert (by_extra.doi, by_extra.arxiv_id) == ("10.1000/xyz.1", "2401.00001")


def test_deduplicates_by_doi_arxiv_and_title() -> None:
    def parsed(entry: dict, groups: set[str]) -> ZoteroItem:
        result = parse_item(entry, groups)
        assert result is not None
        return result

    items = [
        parsed(item("A0000001", "Same DOI, newer", days=5, DOI="10.1145/abc"), {"C1"}),
        parsed(
            item("A0000002", "Same DOI, older", days=1, DOI="10.1145/ABC", abstract="Old."), {"C2"}
        ),
        parsed(
            item("B0000001", "From arXiv", days=4, url="https://arxiv.org/abs/2410.00001"), set()
        ),
        parsed(
            item(
                "B0000002",
                "Preprint copy",
                days=2,
                item_type="preprint",
                archiveID="arXiv:2410.00001",
            ),
            set(),
        ),
        parsed(item("C0000001", "Attention Is All You Need!", days=3), set()),
        parsed(item("C0000002", "attention is all you need", days=0), {"C3"}),
        parsed(item("D0000001", "Unique", days=6), set()),
    ]

    result = deduplicate(items)

    assert [i.key for i in result] == ["D0000001", "A0000001", "B0000001", "C0000001"]
    same_doi = result[1]
    assert same_doi.collections == ["C1", "C2"]  # 合并所属的 collection
    assert same_doi.abstract == "Old."  # 最新的那条没有摘要时用重复项的
    assert result[3].collections == ["C3"]


# --- collection -----------------------------------------------------------------

COLLECTIONS = [
    collection("DETECT01", "Detection"),
    collection("SUBDET01", "Domain generalization", parent="DETECT01"),
    collection("SUBSUB01", "Style transfer", parent="SUBDET01"),
    collection("OTHER001", "Reading"),
    collection("DUPNAME1", "Misc"),
    collection("DUPNAME2", "Misc", parent="OTHER001"),
]


def test_resolve_collections_by_name_or_key() -> None:
    assert resolve_collections(["Detection"], COLLECTIONS, True) == [
        "DETECT01",
        "SUBDET01",
        "SUBSUB01",
    ]
    assert resolve_collections(["Detection"], COLLECTIONS, False) == ["DETECT01"]
    assert resolve_collections(["other001", "detection"], COLLECTIONS, False) == [
        "OTHER001",
        "DETECT01",
    ]


def test_duplicate_collection_names_must_use_key() -> None:
    with pytest.raises(ZoteroConfigError, match="有 2 个名为「Misc」的 collection") as info:
        resolve_collections(["Misc"], COLLECTIONS, True)
    assert "DUPNAME1、DUPNAME2" in str(info.value) and "改用 key" in str(info.value)

    with pytest.raises(ZoteroConfigError, match="找不到名称或 key 为「Nope」"):
        resolve_collections(["Nope"], COLLECTIONS, True)


@respx.mock
def test_reads_selected_collections_with_subcollections(tmp_path: Path) -> None:
    respx.get(f"{API}/items/top").mock(return_value=probe())
    respx.get(f"{API}/collections").mock(side_effect=paged(COLLECTIONS))
    top = respx.get(f"{API}/collections/DETECT01/items/top").mock(
        side_effect=paged([item("P0000001", "Detector", days=1, collections=["DETECT01"])])
    )
    sub = respx.get(f"{API}/collections/SUBDET01/items/top").mock(
        side_effect=paged(
            [item("P0000002", "DG detector", days=2, collections=["SUBDET01", "OTHER001"])]
        )
    )
    subsub = respx.get(f"{API}/collections/SUBSUB01/items/top").mock(
        side_effect=paged(
            [item("P0000001", "Detector", days=1, collections=["DETECT01", "SUBSUB01"])]
        )
    )
    other = respx.get(f"{API}/collections/OTHER001/items/top")

    result = library(tmp_path, collections=["Detection"]).load()

    assert top.called and sub.called and subsub.called and not other.called
    assert result.stats.total == 2  # 同时在两个 collection 里的只算一次
    assert [p.title for p in result.papers] == ["DG detector", "Detector"]


@respx.mock
def test_without_subcollections(tmp_path: Path) -> None:
    respx.get(f"{API}/items/top").mock(return_value=probe())
    respx.get(f"{API}/collections").mock(side_effect=paged(COLLECTIONS))
    respx.get(f"{API}/collections/DETECT01/items/top").mock(
        side_effect=paged([item("P0000001", "Detector")])
    )
    sub = respx.get(f"{API}/collections/SUBDET01/items/top")

    result = library(tmp_path, collections=["Detection"], include_subcollections=False).load()

    assert not sub.called
    assert result.stats.total == 1


@respx.mock
def test_duplicate_collection_name_skips_zotero(tmp_path: Path) -> None:
    respx.get(f"{API}/items/top").mock(return_value=probe())
    respx.get(f"{API}/collections").mock(side_effect=paged(COLLECTIONS))

    result = library(tmp_path, collections=["Misc"]).load()

    assert result.papers == []
    assert result.stats.source == "none"
    [warning] = result.warnings
    assert warning.startswith("Zotero 配置有误，这次不使用 Zotero：有 2 个名为「Misc」")


# --- 选取锚点 -------------------------------------------------------------------


def zitem(key: str, days: int, collections: list[str]) -> ZoteroItem:
    return ZoteroItem(
        key=key, title=key, date_added=BASE + timedelta(days=days), collections=collections
    )


def test_select_round_robin_newest_first() -> None:
    items = [
        zitem("A1", 10, ["A"]),
        zitem("A2", 9, ["A"]),
        zitem("A3", 8, ["A"]),
        zitem("A4", 7, ["A"]),
        zitem("B1", 6, ["B"]),
        zitem("B2", 5, ["B"]),
        zitem("AB", 4, ["A", "B"]),
        zitem("U1", 3, []),
        zitem("U2", 1, []),
    ]

    chosen = select_items(items, 5)

    # 组按最近加入的时间排序：A（10 天）、B（6 天）、未分组（3 天）；轮流各取最新的一篇
    assert [i.key for i in chosen] == ["A1", "B1", "U1", "A2", "B2"]
    assert [i.key for i in select_items(items, 100)] == [
        "A1", "B1", "U1", "A2", "B2", "U2", "A3", "AB", "A4"
    ]  # fmt: skip
    # 属于两个 collection 的 AB 只出现一次
    assert [i.key for i in select_items(items, 100)].count("AB") == 1


@respx.mock
def test_limit_reports_used_and_total(tmp_path: Path) -> None:
    items = [item(f"ITEM{i:04d}", f"Paper {i}", days=i) for i in range(7)]
    respx.get(f"{API}/items/top").mock(side_effect=paged(items))

    result = library(tmp_path, max_items=3).load()

    assert result.stats == ZoteroStats(used=3, total=7, source="api")
    assert [p.title for p in result.papers] == ["Paper 6", "Paper 5", "Paper 4"]
    assert zotero_line(result.stats) == "Zotero：使用 3 / 7 篇"
    assert zotero_line(ZoteroStats(used=5, total=5)) == "Zotero：使用 5 篇"


# --- 缓存与降级 -----------------------------------------------------------------


@respx.mock
def test_not_modified_uses_cache(tmp_path: Path) -> None:
    items = [item("ITEM0001", "Cached paper", abstract="About detection.")]
    respx.get(f"{API}/items/top").mock(side_effect=paged(items, version=321))
    library(tmp_path).load()
    cache_files = list((tmp_path / "zotero").glob("*.json"))
    assert len(cache_files) == 1

    respx.reset()
    route = respx.get(f"{API}/items/top").mock(return_value=httpx.Response(304))

    result = library(tmp_path).load()

    [call] = route.calls  # 只发了一次带版本号的检查请求
    assert call.request.headers["If-Modified-Since-Version"] == "321"
    assert result.stats.source == "cache"
    assert [(p.title, p.abstract) for p in result.papers] == [("Cached paper", "About detection.")]


@respx.mock
def test_changed_library_is_refetched(tmp_path: Path) -> None:
    respx.get(f"{API}/items/top").mock(side_effect=paged([item("ITEM0001", "Old")], version=1))
    library(tmp_path).load()

    respx.reset()
    respx.get(f"{API}/items/top").mock(
        side_effect=paged([item("ITEM0002", "New", days=1), item("ITEM0001", "Old")], version=2)
    )
    result = library(tmp_path).load()

    assert result.stats.source == "api"
    assert [p.title for p in result.papers] == ["New", "Old"]


@respx.mock
def test_changed_scope_ignores_cache(tmp_path: Path) -> None:
    respx.get(f"{API}/items/top").mock(side_effect=paged([item("ITEM0001", "Old")]))
    library(tmp_path).load()

    respx.reset()
    route = respx.get(f"{API}/items/top").mock(return_value=httpx.Response(304))
    respx.get(f"{API}/collections").mock(side_effect=paged(COLLECTIONS))
    respx.get(f"{API}/collections/OTHER001/items/top").mock(side_effect=paged([]))
    library(tmp_path, collections=["Reading"], include_subcollections=False).load()

    assert "If-Modified-Since-Version" not in route.calls[0].request.headers


@respx.mock
def test_failure_falls_back_to_cache(tmp_path: Path) -> None:
    respx.get(f"{API}/items/top").mock(side_effect=paged([item("ITEM0001", "Cached")]))
    library(tmp_path).load()

    respx.reset()
    respx.get(f"{API}/items/top").mock(side_effect=httpx.ConnectError(f"cannot reach {API}"))

    result = library(tmp_path).load()

    assert [p.title for p in result.papers] == ["Cached"]
    assert result.stats.source == "stale"
    [warning] = result.warnings
    assert warning.startswith("读取 Zotero 失败，改用 ")
    assert "请求 Zotero 失败（ConnectError）" in warning
    assert "1234567" not in warning and "Cached" not in warning
    assert "读取失败，用的是上次的缓存" in zotero_line(result.stats)


@respx.mock
def test_failure_without_cache_skips(tmp_path: Path) -> None:
    respx.get(f"{API}/items/top").mock(
        return_value=httpx.Response(403, text=f"Invalid key {KEY} for /users/1234567")
    )

    result = library(tmp_path).load()

    assert result.papers == []
    assert result.stats.source == "none"
    [warning] = result.warnings
    assert "又没有缓存，这次不使用 Zotero" in warning
    assert "HTTP 403" in warning and "检查 ZOTERO_KEY" in warning
    assert KEY not in warning and "1234567" not in warning
    assert zotero_line(result.stats).startswith(
        "⚠️ Zotero 这次没有使用：读取 Zotero 失败（HTTP 403）"
    )


def test_missing_credentials(tmp_path: Path) -> None:
    zotero = ZoteroConfig(enabled=True)

    result = ZoteroLibrary(zotero, cache_dir=tmp_path).load()

    assert result.warnings == [
        "Zotero 配置有误，这次不使用 Zotero：缺少环境变量 ZOTERO_ID、ZOTERO_KEY（GitHub Secret）"
    ]


@respx.mock
def test_backoff_and_retry_after(tmp_path: Path) -> None:
    waits: list[float] = []
    clock = [0.0]

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        clock[0] += seconds

    client = ZoteroClient(
        "user", "1234567", KEY, backoff_seconds=0, sleep=sleep, clock=lambda: clock[0]
    )
    respx.get(f"{API}/items/top").mock(
        side_effect=[
            httpx.Response(200, json=[], headers={"Last-Modified-Version": "9", "Backoff": "5"}),
            httpx.Response(429, headers={"Retry-After": "3"}),
            page([item("ITEM0001", "After waiting")]),
        ]
    )

    assert client.library_version() == 9
    items = client.top_items()

    assert waits == [5.0, 3.0]  # 先等够 Backoff，再按 Retry-After 重试
    assert [entry["key"] for entry in items] == ["ITEM0001"]


@respx.mock
def test_long_retry_after_gives_up(tmp_path: Path) -> None:
    respx.get(f"{API}/items/top").mock(
        return_value=httpx.Response(503, headers={"Retry-After": "600"})
    )

    result = library(tmp_path).load()

    assert result.stats.source == "none"
    assert "Zotero 限流或暂时不可用（HTTP 503）" in result.warnings[0]


@respx.mock
def test_cache_is_local_json_only(tmp_path: Path) -> None:
    respx.get(f"{API}/items/top").mock(side_effect=paged([item("ITEM0001", "Private title")]))
    zotero = library(tmp_path)
    zotero.load()

    saved = json.loads(zotero.cache_path.read_text(encoding="utf-8"))
    assert saved["version"] == 100 and saved["items"][0]["title"] == "Private title"
    assert KEY not in zotero.cache_path.read_text(encoding="utf-8")
