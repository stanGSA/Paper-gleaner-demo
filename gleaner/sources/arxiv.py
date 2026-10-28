"""arXiv 来源：通过 RSS 获取当天公告的论文。

arXiv 的 RSS（rss.arxiv.org）每个公告日更新一次，条目里已包含标题、摘要、作者、
分类和公告类型（new / cross / replace / replace-cross），日报所需的元数据都在其中。
周末和节假日没有公告时，RSS 不含任何条目；这时试运行改用 arXiv API 取最近提交的论文。
按 ID 查询（/deep、/compare 中不在日报里的论文）也走 arXiv API。

arXiv 要求 API 请求间隔至少 3 秒；同一进程内对 arxiv.org 的请求（API、HTML 全文、PDF）
共用一个限速器 ARXIV_RATE_LIMITER。
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from time import struct_time
from typing import Any

import feedparser
import httpx

from gleaner import __version__
from gleaner.config import ArxivConfig
from gleaner.models import Paper
from gleaner.sources.base import SourceError

RSS_URL = "https://rss.arxiv.org/rss/{categories}"
API_URL = "https://export.arxiv.org/api/query"
USER_AGENT = f"paper-gleaner/{__version__} (+https://github.com/stanGSA/Paper-gleaner)"

_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")
# API 条目的 id 是论文链接；出错时是 .../api/errors#...，其中可能也带着一串像 ID 的数字
_API_ENTRY_ID_RE = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})(?:v\d+)?$")
# 摘要前缀形如 "arXiv:2410.12345v1 Announce Type: new \nAbstract: ..."
_DESCRIPTION_RE = re.compile(
    r"^\s*arXiv:\S+\s+Announce Type:\s*(?P<type>\S+)\s*Abstract:\s*(?P<abstract>.*)$",
    re.DOTALL,
)
_REPLACE_TYPES = {"replace", "replace-cross"}
# 关键词检索：arXiv API 的检索语法只认 ASCII 词，其余字符（中文、引号、括号、布尔运算符）都去掉
_QUERY_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")
MAX_QUERY_TERMS = 12


class RateLimiter:
    """保证相邻两次请求之间至少间隔 interval 秒。"""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            delay = self._last + self.interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self._last = time.monotonic()


ARXIV_RATE_LIMITER = RateLimiter(3.0)


class ArxivSource:
    name = "arxiv"

    def __init__(
        self,
        config: ArxivConfig,
        *,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
        limiter: RateLimiter = ARXIV_RATE_LIMITER,
    ) -> None:
        self.config = config
        self._client = client
        self._timeout = timeout
        self._limiter = limiter

    @property
    def feed_url(self) -> str:
        return RSS_URL.format(categories="+".join(self.config.categories))

    def fetch_latest(self) -> list[Paper]:
        papers = parse_rss(self._download(self.feed_url, what="arXiv RSS"))
        return [paper for paper in papers if self._wanted(paper)]

    def fetch_recent(self, limit: int) -> list[Paper]:
        """所关注分类中最近提交的 limit 篇论文，不区分公告日。"""
        params = {
            "search_query": " OR ".join(f"cat:{c}" for c in self.config.categories),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": limit,
        }
        return parse_api(self._api(params))

    def search(self, query: str, limit: int = 10) -> list[Paper]:
        """按关键词检索 arXiv（各个词都要出现在标题、摘要等字段中），按相关度排序。

        只取查询中的英文单词和数字，最多 MAX_QUERY_TERMS 个；一个都没有时返回空列表。
        """
        terms = _QUERY_TERM_RE.findall(query)[:MAX_QUERY_TERMS]
        terms = [term for term in terms if term.upper() not in {"AND", "OR", "ANDNOT"}]
        if not terms:
            return []
        params = {
            "search_query": " AND ".join(f"all:{term}" for term in terms),
            "sortBy": "relevance",
            "sortOrder": "descending",
            "max_results": limit,
        }
        return parse_api(self._api(params))

    def fetch_by_ids(self, ids: list[str]) -> list[Paper]:
        """按 arXiv ID（不带版本号）查询元数据；arXiv 上不存在的 ID 不会出现在结果里。"""
        if not ids:
            return []
        params = {"id_list": ",".join(ids), "max_results": len(ids)}
        return parse_api(self._api(params))

    def _api(self, params: dict[str, Any]) -> bytes:
        self._limiter.wait()
        return self._download(API_URL, what="arXiv API", params=params)

    def _download(self, url: str, *, what: str, params: dict[str, Any] | None = None) -> bytes:
        try:
            if self._client is not None:
                response = self._client.get(url, params=params)
            else:
                with httpx.Client(
                    headers={"User-Agent": USER_AGENT},
                    timeout=self._timeout,
                    follow_redirects=True,
                    transport=httpx.HTTPTransport(retries=3),
                ) as client:
                    response = client.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise SourceError(f"抓取 {what} 失败：{e}") from e
        return response.content

    def _wanted(self, paper: Paper) -> bool:
        if paper.announce_type == "cross":
            return self.config.include_cross_list
        if paper.announce_type in _REPLACE_TYPES:
            return self.config.include_replace
        return True


def parse_rss(content: bytes) -> list[Paper]:
    """解析 arXiv RSS，按出现顺序返回去重后的论文。"""
    return _unique(_parse_rss_entry(entry) for entry in _parse_feed(content, "arXiv RSS"))


def parse_api(content: bytes) -> list[Paper]:
    """解析 arXiv API 返回的 Atom，按出现顺序返回去重后的论文。"""
    return _unique(_parse_api_entry(entry) for entry in _parse_feed(content, "arXiv API 响应"))


def _parse_feed(content: bytes, what: str) -> list[Any]:
    feed = feedparser.parse(content)
    # 维护页等非 feed 内容不能当作"没有论文"
    if not feed.entries and (feed.bozo or not feed.version):
        raise SourceError(f"无法解析 {what}：{feed.get('bozo_exception') or '不是 feed 内容'}")
    return feed.entries


def _unique(papers: Iterable[Paper | None]) -> list[Paper]:
    result: dict[str, Paper] = {}
    for paper in papers:
        if paper is not None and paper.id not in result:
            result[paper.id] = paper
    return list(result.values())


def _parse_api_entry(entry: Any) -> Paper | None:
    # 出错时 API 返回一条 id 不是论文链接的条目，这里会被跳过
    match = _API_ENTRY_ID_RE.search(entry.get("id", "").strip())
    if match is None or not entry.get("title", "").strip():
        return None
    return Paper(
        id=match.group(1),
        title=_squash(entry.get("title", "")),
        abstract=_squash(entry.get("summary", "")),
        authors=[_squash(a["name"]) for a in entry.get("authors", []) if a.get("name")],
        categories=[tag["term"] for tag in entry.get("tags", []) if tag.get("term")],
        announced=_to_datetime(entry.get("published_parsed")),
    )


def _parse_rss_entry(entry: Any) -> Paper | None:
    match = _ID_RE.search(entry.get("link", "")) or _ID_RE.search(entry.get("id", ""))
    if match is None:
        return None

    description = entry.get("summary", "")
    announce_type = entry.get("arxiv_announce_type")
    if parsed := _DESCRIPTION_RE.match(description):
        description = parsed["abstract"]
        announce_type = announce_type or parsed["type"]

    return Paper(
        id=match.group(1),
        title=_squash(entry.get("title", "")),
        abstract=_squash(description),
        authors=[name for name in (_squash(n) for n in entry.get("author", "").split(",")) if name],
        categories=[tag["term"] for tag in entry.get("tags", []) if tag.get("term")],
        announce_type=announce_type,
        announced=_to_datetime(entry.get("published_parsed")),
    )


def _squash(text: str) -> str:
    return " ".join(text.split())


def _to_datetime(value: struct_time | None) -> datetime | None:
    if value is None:
        return None
    # feedparser 已把时间换算为 UTC
    return datetime(*value[:6], tzinfo=UTC)
