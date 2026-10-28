"""Semantic Scholar：按 arXiv ID 获取参考文献和推荐论文，供 /compare 单篇对比相关工作使用。

没有 API Key（环境变量 S2_API_KEY）时与所有匿名用户共享限额，经常返回 429。
这里只重试一次，仍然失败就交给调用方降级：报告里注明缺少相关工作部分，指令照常完成。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gleaner.sources.arxiv import USER_AGENT
from gleaner.sources.base import SourceError

API_URL = "https://api.semanticscholar.org"
PAPER_FIELDS = "title,abstract,year,externalIds,citationCount,url"
_RETRY_STATUS = {429, 500, 502, 503, 504}

logger = logging.getLogger(__name__)
M = TypeVar("M", bound=BaseModel)
Relation = Literal["reference", "recommended"]


class SemanticScholarError(SourceError):
    """调用 Semantic Scholar API 失败。"""


class RelatedPaper(BaseModel):
    title: str
    abstract: str | None = None
    year: int | None = None
    arxiv_id: str | None = None
    url: str | None = None
    citation_count: int | None = None
    relation: Relation
    influential: bool = False  # Semantic Scholar 判断为「有影响力的引用」


@dataclass
class RelatedWorks:
    papers: list[RelatedPaper] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # 失败的请求及原因，用于在报告中说明


class _S2Paper(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    abstract: str | None = None
    year: int | None = None
    external_ids: dict[str, Any] | None = Field(None, alias="externalIds")
    citation_count: int | None = Field(None, alias="citationCount")
    url: str | None = None


class _Reference(BaseModel):
    influential: bool | None = Field(None, alias="isInfluential")
    cited_paper: _S2Paper = Field(alias="citedPaper")


class _References(BaseModel):
    data: list[_Reference] | None = None


class _Recommendations(BaseModel):
    papers: list[_S2Paper] | None = Field(None, alias="recommendedPapers")


class SemanticScholar:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: httpx.Client | None = None,
        timeout: float = 20.0,
        max_attempts: int = 2,
        backoff_seconds: float = 2.0,
    ) -> None:
        headers = {"User-Agent": USER_AGENT}
        if api_key:
            headers["x-api-key"] = api_key
        self._client = client or httpx.Client(
            base_url=API_URL, headers=headers, timeout=timeout, follow_redirects=True
        )
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds

    def references(self, arxiv_id: str, *, limit: int = 100) -> list[RelatedPaper]:
        """论文的参考文献，有影响力的引用和被引多的排在前面。"""
        response = self._get(
            f"/graph/v1/paper/arXiv:{arxiv_id}/references",
            {"fields": f"isInfluential,{PAPER_FIELDS}", "limit": limit},
            _References,
        )
        papers = [
            _related(item.cited_paper, "reference", influential=bool(item.influential))
            for item in response.data or []
        ]
        papers = [p for p in papers if p is not None]
        return sorted(papers, key=lambda p: (p.influential, p.citation_count or 0), reverse=True)

    def recommendations(self, arxiv_id: str, *, limit: int = 10) -> list[RelatedPaper]:
        """Semantic Scholar 根据这篇论文推荐的相似论文。"""
        response = self._get(
            f"/recommendations/v1/papers/forpaper/arXiv:{arxiv_id}",
            {"fields": PAPER_FIELDS, "limit": limit},
            _Recommendations,
        )
        papers = [_related(p, "recommended") for p in response.papers or []]
        return [p for p in papers if p is not None]

    def related_works(
        self, arxiv_id: str, *, max_references: int = 6, max_recommendations: int = 4
    ) -> RelatedWorks:
        """参考文献和推荐论文各取一部分、去重；某一类获取失败时记下原因，不抛出异常。"""
        result = RelatedWorks()
        lookups = [
            ("参考文献", self.references, max_references),
            ("推荐论文", self.recommendations, max_recommendations),
        ]
        seen = {arxiv_id}
        for what, lookup, limit in lookups:
            if limit <= 0:
                continue
            try:
                papers = lookup(arxiv_id)
            except SemanticScholarError as e:
                logger.warning("获取 %s 的%s失败：%s", arxiv_id, what, e)
                result.errors.append(f"{what}：{e}")
                continue
            added = 0
            for paper in papers:
                key = paper.arxiv_id or paper.title.lower()
                if key in seen or added >= limit:
                    continue
                seen.add(key)
                result.papers.append(paper)
                added += 1
        return result

    def _get(self, url: str, params: dict[str, Any], model: type[M]) -> M:
        for attempt in range(self.max_attempts):
            try:
                response = self._client.get(url, params=params)
            except httpx.HTTPError as e:
                raise SemanticScholarError(f"请求失败（{type(e).__name__}）") from e
            if response.status_code in _RETRY_STATUS and attempt < self.max_attempts - 1:
                delay = self._retry_delay(response, attempt)
                logger.info("Semantic Scholar 返回 %d，%.0f 秒后重试", response.status_code, delay)
                time.sleep(delay)
                continue
            if response.status_code == 404:
                raise SemanticScholarError("Semantic Scholar 尚未收录这篇论文")
            if response.status_code == 429:
                raise SemanticScholarError("Semantic Scholar 限流（HTTP 429）")
            if not response.is_success:
                raise SemanticScholarError(f"HTTP {response.status_code}")
            try:
                return model.model_validate_json(response.content)
            except ValidationError as e:
                raise SemanticScholarError("响应格式不符合预期") from e
        raise AssertionError("unreachable")

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("retry-after", "")
        if retry_after.isdigit():
            return min(float(retry_after), 30.0)
        return self.backoff_seconds * 2**attempt


def _related(
    paper: _S2Paper, relation: Relation, *, influential: bool = False
) -> RelatedPaper | None:
    if not paper.title:
        return None
    arxiv_id = (paper.external_ids or {}).get("ArXiv")
    return RelatedPaper(
        title=" ".join(paper.title.split()),
        abstract=" ".join(paper.abstract.split()) if paper.abstract else None,
        year=paper.year,
        arxiv_id=arxiv_id if isinstance(arxiv_id, str) else None,
        url=paper.url,
        citation_count=paper.citation_count,
        relation=relation,
        influential=influential,
    )
