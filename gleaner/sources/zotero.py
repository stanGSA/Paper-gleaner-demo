"""Zotero 文库：可选的兴趣锚点（docs/architecture.md 3.1.1 的 A4）。

用 httpx 直接调用 Zotero Web API v3（https://www.zotero.org/support/dev/web_api/v3/basics）：

- 认证：请求头 Zotero-API-Key，并用 Zotero-API-Version: 3 固定接口版本；
- 个人文库的路径前缀是 /users/<ZOTERO_ID>，群组文库是 /groups/<ZOTERO_ID>；
- 整个文库读 <前缀>/items/top；限定 collection 时逐个读 <前缀>/collections/<key>/items/top。
  这个接口不包含子 collection 里的条目，所以子 collection 也要逐个读；子 collection 由
  <前缀>/collections 返回的 parentCollection 推出；
- 分页：每页最多 100 条，用 start 翻页，总数在响应头 Total-Results 里；
- 回收站：条目接口默认不返回回收站里的条目（includeTrashed 默认为 0），不需要额外参数；
  回收站条目的 data 里带 "deleted": 1，这里再按它过滤一遍；
- 附件和笔记：请求时加 itemType=-attachment，返回后再按 itemType 去掉笔记、附件和标注；
- 限流：任何响应都可能带 Backoff: <秒>，之后的请求要等够这么久；429 和 503 可能带 Retry-After。
  要等的时间超过 MAX_WAIT_SECONDS 时放弃，改用缓存，不拖长日报；
- 变化检测：多对象请求的响应头 Last-Modified-Version 是文库当前的版本。带上
  If-Modified-Since-Version: <缓存的版本> 请求时，文库没有变化会返回 304，直接用缓存；
  有变化时整体重新拉取，不做逐条增量。

重复项按 DOI、arXiv ID、规范化后的标题依次判断。选取锚点时按加入文库的时间（dateAdded）
从新到旧，在各 collection 之间轮流取（不属于任何 collection 的算一组，属于多个的只算一次），
最多 max_items 篇。

隐私：文库内容只缓存在本机（GitHub Actions 中由 actions/cache 保存
~/.cache/paper-gleaner/zotero），不写进 gleaner-data；运行摘要只显示数量，
错误信息里也不出现文献标题、ZOTERO_ID 和 ZOTERO_KEY。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from gleaner.config import ZoteroConfig
from gleaner.models import Paper, ZoteroStats
from gleaner.redaction import protect, redact
from gleaner.sources.arxiv import USER_AGENT
from gleaner.sources.base import SourceError

API_URL = "https://api.zotero.org"
PAGE_SIZE = 100
MAX_WAIT_SECONDS = 60.0
SKIPPED_TYPES = {"attachment", "note", "annotation"}
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "paper-gleaner" / "zotero"
UNFILED = ""  # 不属于任何 collection 的条目算作一组
_RETRY_STATUS = {429, 503}
_ARXIV_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:\s*|10\.48550/arxiv\.)(\d{4}\.\d{4,5})", re.IGNORECASE
)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/\S+)", re.IGNORECASE)
_TITLE_NOISE = re.compile(r"[\W_]+")

logger = logging.getLogger(__name__)


class ZoteroError(SourceError):
    """读取 Zotero 失败（网络、认证、限流等）。信息不含 ZOTERO_ID、ZOTERO_KEY 和文献内容。"""


class ZoteroConfigError(ZoteroError):
    """配置有误：缺少 ZOTERO_ID / ZOTERO_KEY，或者 collection 找不到、重名。"""


class ZoteroItem(BaseModel):
    """文库中一篇有效的文献。只保存在本机缓存里。"""

    key: str
    title: str
    abstract: str = ""
    date_added: datetime | None = None
    collections: list[str] = Field(
        default_factory=list
    )  # 所属的 collection（限定范围时只算范围内的）
    doi: str | None = None
    arxiv_id: str | None = None


class LibraryCache(BaseModel):
    """缓存文件的内容：某个读取范围内去重后的全部有效文献，按加入时间从新到旧。"""

    scope: str  # 文库类型、ID 和 collection 配置的指纹，配置改了缓存就作废
    version: int  # 拉取时的文库版本（Last-Modified-Version）
    fetched_at: datetime
    items: list[ZoteroItem] = Field(default_factory=list)


@dataclass
class ZoteroResult:
    papers: list[Paper] = field(default_factory=list)  # 选中的锚点（标题 + 摘要）
    stats: ZoteroStats = field(default_factory=ZoteroStats)
    warnings: list[str] = field(default_factory=list)


class ZoteroClient:
    """Zotero Web API v3 的最小封装：读 collection 列表和顶层条目，处理分页、限流和 304。"""

    def __init__(
        self,
        library_type: str,
        library_id: str,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
        max_attempts: int = 3,
        backoff_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.prefix = f"/{'groups' if library_type == 'group' else 'users'}/{library_id}"
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self._secrets = [library_id, api_key]
        self._sleep = sleep
        self._clock = clock
        self._not_before = 0.0  # Backoff 要求的最早下次请求时间
        protect(library_id, api_key)
        self._client = client or httpx.Client(
            base_url=API_URL,
            timeout=timeout,
            transport=httpx.HTTPTransport(retries=2),
        )
        self._headers = {
            "Zotero-API-Key": api_key,
            "Zotero-API-Version": "3",
            "User-Agent": USER_AGENT,
        }

    def library_version(self, since: int | None = None) -> int | None:
        """文库当前的版本；带上 since 且文库从那以后没有变化（304）时返回 None。"""
        headers = {"If-Modified-Since-Version": str(since)} if since is not None else {}
        response = self._get(f"{self.prefix}/items/top", {"limit": 1}, headers)
        if response.status_code == 304:
            return None
        return _library_version(response)

    def collections(self) -> list[dict[str, Any]]:
        """文库中的全部 collection（含各级子 collection）。"""
        return self._paginate(f"{self.prefix}/collections", {})

    def top_items(self, collection: str | None = None) -> list[dict[str, Any]]:
        """整个文库（或某个 collection，不含子 collection）的顶层条目，不含附件。"""
        path = (
            f"{self.prefix}/collections/{collection}/items/top"
            if collection
            else (f"{self.prefix}/items/top")
        )
        return self._paginate(path, {"itemType": "-attachment"})

    def _paginate(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        while True:
            response = self._get(path, {**params, "limit": PAGE_SIZE, "start": len(results)})
            batch = _json_list(response)
            results.extend(batch)
            total = _int_header(response, "Total-Results")
            if not batch or total is None or len(results) >= total:
                return results

    def _get(
        self, path: str, params: dict[str, Any], headers: dict[str, str] | None = None
    ) -> httpx.Response:
        for attempt in range(self.max_attempts):
            self._wait_for_backoff()
            try:
                response = self._client.get(
                    path, params=params, headers={**self._headers, **(headers or {})}
                )
            except httpx.HTTPError as e:
                # httpx 的异常信息带着请求地址，地址里有 ZOTERO_ID：只保留类型
                raise ZoteroError(f"请求 Zotero 失败（{type(e).__name__}）") from None
            if (backoff := _float_header(response, "Backoff")) is not None:
                self._not_before = self._clock() + backoff
            if response.status_code not in _RETRY_STATUS:
                return self._check(response)
            delay = _float_header(response, "Retry-After")
            delay = self.backoff_seconds * 2**attempt if delay is None else delay
            if attempt == self.max_attempts - 1 or delay > MAX_WAIT_SECONDS:
                break
            logger.info("Zotero 返回 %d，%.0f 秒后重试", response.status_code, delay)
            self._sleep(delay)
        raise ZoteroError(f"Zotero 限流或暂时不可用（HTTP {response.status_code}），稍后再试")

    def _wait_for_backoff(self) -> None:
        delay = self._not_before - self._clock()
        if delay > MAX_WAIT_SECONDS:
            raise ZoteroError(f"Zotero 要求暂停 {delay:.0f} 秒再请求（Backoff），这次不再等待")
        if delay > 0:
            self._sleep(delay)

    def _check(self, response: httpx.Response) -> httpx.Response:
        status = response.status_code
        if response.is_success or status == 304:
            return response
        if status in {401, 403}:
            reason = "没有权限：检查 ZOTERO_KEY 是否有效、是否允许读取这个文库"
        elif status == 404:
            reason = "找不到文库或 collection：检查 ZOTERO_ID 和 profile.zotero.library_type"
        else:
            reason = redact(" ".join(response.text.split())[:120], self._secrets)
        raise ZoteroError(f"读取 Zotero 失败（HTTP {status}）：{reason}")


class ZoteroLibrary:
    """读取文库并选出锚点：文库没有变化时用缓存；读取失败时用旧缓存，没有缓存就跳过。"""

    def __init__(
        self,
        config: ZoteroConfig,
        *,
        client: ZoteroClient | None = None,
        cache_dir: Path | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.config = config
        self._client = client
        default_dir = os.environ.get("GLEANER_ZOTERO_CACHE", "").strip() or DEFAULT_CACHE_DIR
        self.cache_dir = cache_dir or Path(default_dir)
        self._now = now

    @property
    def scope(self) -> str:
        """读取范围的指纹：文库类型、ID 和 collection 配置。"""
        config = self.config
        library_id = config.library_id.get_secret_value() if config.library_id else ""
        parts = [
            config.library_type,
            library_id,
            json.dumps(sorted(config.collections), ensure_ascii=False),
            str(config.include_subcollections),
        ]
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / f"library-{self.scope[:16]}.json"

    def load(self) -> ZoteroResult:
        """不抛出异常：失败时降级，并在 warnings 里说明。"""
        cached = self._read_cache()
        try:
            library, source = self._fetch(cached)
        except ZoteroConfigError as e:
            stats = ZoteroStats(source="none", error=str(e))
            return ZoteroResult(stats=stats, warnings=[f"Zotero 配置有误，这次不使用 Zotero：{e}"])
        except ZoteroError as e:
            if cached is None:
                stats = ZoteroStats(source="none", error=str(e))
                warning = f"读取 Zotero 失败，又没有缓存，这次不使用 Zotero：{e}"
                return ZoteroResult(stats=stats, warnings=[warning])
            when = cached.fetched_at.strftime("%Y-%m-%d %H:%M")
            library, source = cached, "stale"
            warnings = [f"读取 Zotero 失败，改用 {when}（UTC）拉取的缓存：{e}"]
            return self._result(library, source, warnings, error=str(e))
        return self._result(library, source, [])

    def _result(
        self, library: LibraryCache, source: str, warnings: list[str], error: str | None = None
    ) -> ZoteroResult:
        chosen = select_items(library.items, self.config.max_items)
        papers = [
            Paper(id=f"zotero:{item.key}", title=item.title, abstract=item.abstract)
            for item in chosen
        ]
        stats = ZoteroStats(used=len(papers), total=len(library.items), source=source, error=error)
        return ZoteroResult(papers, stats, warnings)

    def _fetch(self, cached: LibraryCache | None) -> tuple[LibraryCache, str]:
        client = self._client or self._build_client()
        version = client.library_version(cached.version if cached else None)
        if version is None and cached is not None:
            return cached, "cache"

        if self.config.collections:
            scope = resolve_collections(
                self.config.collections, client.collections(), self.config.include_subcollections
            )
            found: dict[str, tuple[dict[str, Any], set[str]]] = {}
            for key in scope:
                for item in client.top_items(key):
                    entry = found.setdefault(_item_key(item), (item, set()))
                    entry[1].add(key)
        else:
            found = {
                _item_key(item): (item, set((item.get("data") or {}).get("collections") or []))
                for item in client.top_items()
            }
        parsed = [parse_item(item, groups) for item, groups in found.values()]
        items = deduplicate([item for item in parsed if item is not None])
        library = LibraryCache(
            scope=self.scope, version=version or 0, fetched_at=self._now(), items=items
        )
        self._write_cache(library)
        return library, "api"

    def _build_client(self) -> ZoteroClient:
        config = self.config
        missing = [
            name
            for name, value in (("ZOTERO_ID", config.library_id), ("ZOTERO_KEY", config.api_key))
            if value is None
        ]
        if missing:
            raise ZoteroConfigError(f"缺少环境变量 {'、'.join(missing)}（GitHub Secret）")
        assert config.library_id is not None and config.api_key is not None
        return ZoteroClient(
            config.library_type,
            config.library_id.get_secret_value(),
            config.api_key.get_secret_value(),
        )

    def _read_cache(self) -> LibraryCache | None:
        try:
            cached = LibraryCache.model_validate_json(self.cache_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValidationError, ValueError) as e:
            logger.warning("Zotero 缓存不可用，忽略：%s", type(e).__name__)
            return None
        return cached if cached.scope == self.scope else None

    def _write_cache(self, library: LibraryCache) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_path.with_suffix(".tmp")
            temporary.write_text(library.model_dump_json(), encoding="utf-8")
            temporary.replace(self.cache_path)
        except OSError as e:
            # 缓存写不进去只影响下次是否要重新拉取
            logger.warning("写入 Zotero 缓存失败：%s", type(e).__name__)


def resolve_collections(
    wanted: list[str], collections: list[dict[str, Any]], include_subcollections: bool
) -> list[str]:
    """把配置中的名称或 key 解析成 collection key；需要时加上各级子 collection。"""
    by_key: dict[str, dict[str, Any]] = {}
    for collection in collections:
        data = collection.get("data") or {}
        if key := data.get("key") or collection.get("key"):
            by_key[key] = data
    selected: list[str] = []
    for entry in (item.strip() for item in wanted):
        if entry.upper() in by_key and re.fullmatch(r"[A-Za-z0-9]{8}", entry):
            selected.append(entry.upper())
            continue
        matches = [key for key, data in by_key.items() if (data.get("name") or "").strip() == entry]
        if not matches:
            folded = entry.casefold()
            matches = [
                key
                for key, data in by_key.items()
                if (data.get("name") or "").strip().casefold() == folded
            ]
        if not matches:
            raise ZoteroConfigError(f"找不到名称或 key 为「{entry}」的 collection")
        if len(matches) > 1:
            keys = "、".join(sorted(matches))
            raise ZoteroConfigError(
                f"有 {len(matches)} 个名为「{entry}」的 collection（key：{keys}），"
                "请在 profile.zotero.collections 中改用 key"
            )
        selected.append(matches[0])
    if include_subcollections:
        children: dict[str, list[str]] = {}
        for key, data in by_key.items():
            if parent := data.get("parentCollection"):
                children.setdefault(parent, []).append(key)
        queue = deque(selected)
        while queue:
            for child in sorted(children.get(queue.popleft(), [])):
                if child not in selected:
                    selected.append(child)
                    queue.append(child)
    return list(dict.fromkeys(selected))


def parse_item(raw: dict[str, Any], groups: set[str]) -> ZoteroItem | None:
    """把 API 返回的条目转成 ZoteroItem；附件、笔记、回收站条目和没有标题的返回 None。"""
    data = raw.get("data") or {}
    if data.get("itemType") in SKIPPED_TYPES or data.get("deleted") or data.get("parentItem"):
        return None
    title = " ".join(str(data.get("title") or "").split())
    if not title:
        return None
    extra = str(data.get("extra") or "")
    doi = _normalize_doi(str(data.get("DOI") or "")) or _doi_from_extra(extra)
    sources = [data.get("archiveID"), data.get("url"), data.get("DOI"), extra]
    arxiv = next((m.group(1) for s in sources if s and (m := _ARXIV_RE.search(str(s)))), None)
    return ZoteroItem(
        key=_item_key(raw),
        title=title,
        abstract=" ".join(str(data.get("abstractNote") or "").split()),
        date_added=_parse_time(data.get("dateAdded")),
        collections=sorted(groups),
        doi=doi,
        arxiv_id=arxiv,
    )


def deduplicate(items: list[ZoteroItem]) -> list[ZoteroItem]:
    """去掉重复项，按 DOI、arXiv ID、规范化后的标题依次判断。

    保留最新加入文库的那一条，合并所属的 collection；它没有摘要时用重复项的摘要补上。
    返回结果按加入时间从新到旧排列。
    """
    kept: list[ZoteroItem] = []
    index: dict[tuple[str, str], int] = {}
    for item in sorted(items, key=_newest_first):
        keys = [
            (kind, value)
            for kind, value in (
                ("doi", item.doi),
                ("arxiv", item.arxiv_id),
                ("title", _TITLE_NOISE.sub("", item.title.casefold())),
            )
            if value
        ]
        match = next((index[key] for key in keys if key in index), None)
        if match is None:
            match = len(kept)
            kept.append(item.model_copy(deep=True))
        else:
            existing = kept[match]
            existing.collections = sorted(set(existing.collections) | set(item.collections))
            existing.abstract = existing.abstract or item.abstract
        for key in keys:
            index.setdefault(key, match)
    return kept


def select_items(items: list[ZoteroItem], max_items: int) -> list[ZoteroItem]:
    """选出最多 max_items 篇：每个 collection 内从新到旧，各 collection 之间轮流取。

    不属于任何 collection 的条目算作一组；属于多个 collection 的只取一次。
    最近加过文献的 collection 排在前面。
    """
    groups: dict[str, deque[ZoteroItem]] = {}
    for item in sorted(items, key=_newest_first):
        for key in item.collections or [UNFILED]:
            groups.setdefault(key, deque()).append(item)
    selected: list[ZoteroItem] = []
    taken: set[str] = set()
    queues = list(groups.values())
    while len(selected) < max_items and any(queues):
        for queue in queues:
            while queue and queue[0].key in taken:
                queue.popleft()
            if queue:
                item = queue.popleft()
                taken.add(item.key)
                selected.append(item)
                if len(selected) >= max_items:
                    break
    return selected


def _newest_first(item: ZoteroItem) -> tuple[bool, float, str]:
    timestamp = item.date_added.timestamp() if item.date_added else 0.0
    return item.date_added is None, -timestamp, item.key


def _item_key(raw: dict[str, Any]) -> str:
    return str(raw.get("key") or (raw.get("data") or {}).get("key") or "")


def _normalize_doi(value: str) -> str | None:
    match = _DOI_RE.search(value)
    return match.group(1).rstrip(".,;").lower() if match else None


def _doi_from_extra(extra: str) -> str | None:
    for line in extra.splitlines():
        name, _, value = line.partition(":")
        if name.strip().lower() == "doi":
            return _normalize_doi(value)
    return None


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _json_list(response: httpx.Response) -> list[dict[str, Any]]:
    try:
        data = response.json()
    except ValueError:
        raise ZoteroError("Zotero 返回的内容不是 JSON") from None
    if not isinstance(data, list):
        raise ZoteroError("Zotero 返回的不是列表")
    return [item for item in data if isinstance(item, dict)]


def _library_version(response: httpx.Response) -> int:
    version = _int_header(response, "Last-Modified-Version")
    if version is None:
        raise ZoteroError("Zotero 的响应缺少 Last-Modified-Version")
    return version


def _int_header(response: httpx.Response, name: str) -> int | None:
    value = response.headers.get(name, "").strip()
    return int(value) if value.isdigit() else None


def _float_header(response: httpx.Response, name: str) -> float | None:
    value = response.headers.get(name, "").strip()
    try:
        return max(float(value), 0.0) if value else None
    except ValueError:
        return None
