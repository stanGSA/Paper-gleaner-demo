"""去重与关键词过滤。"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

from gleaner.config import FiltersConfig
from gleaner.models import Paper
from gleaner.store import DataStore

SEEN_DIR = "history/seen"
SEEN_LOOKBACK_MONTHS = 3


@dataclass
class FilterResult:
    papers: list[Paper]
    # 命中 include_keywords 的论文：跳过粗筛，直接进入精排
    forced_ids: set[str] = field(default_factory=set)
    duplicates: int = 0
    excluded: int = 0


def load_seen(store: DataStore, today: date, months: int = SEEN_LOOKBACK_MONTHS) -> set[str]:
    """读取最近几个月处理过的论文 ID。

    不包含今天的记录，这样同一天强制重跑时，今天的论文不会被当成重复。
    """
    seen = set()
    for month in recent_months(today, months):
        for record in store.read_jsonl(f"{SEEN_DIR}/{month}.jsonl"):
            if record.get("date") != today.isoformat() and "id" in record:
                seen.add(record["id"])
    return seen


def record_seen(store: DataStore, papers: Iterable[Paper], today: date) -> None:
    store.append_jsonl(
        f"{SEEN_DIR}/{today:%Y-%m}.jsonl",
        ({"id": paper.id, "date": today.isoformat()} for paper in papers),
    )


def apply_filters(papers: list[Paper], *, seen: set[str], config: FiltersConfig) -> FilterResult:
    """去掉处理过的论文和命中黑名单的论文，标记命中白名单的论文。

    同时命中黑名单和白名单时以黑名单为准。
    """
    include = compile_keywords(config.include_keywords)
    exclude = compile_keywords(config.exclude_keywords)
    result = FilterResult(papers=[])

    for paper in papers:
        if paper.id in seen:
            result.duplicates += 1
            continue
        text = f"{paper.title}\n{paper.abstract}"
        if exclude and exclude.search(text):
            result.excluded += 1
            continue
        if include and include.search(text):
            result.forced_ids.add(paper.id)
        result.papers.append(paper)
    return result


def compile_keywords(keywords: list[str]) -> re.Pattern[str] | None:
    """把关键词编译成一个忽略大小写、按整词匹配的正则（"RAG" 不会匹配 "storage"）。"""
    words = [re.escape(word.strip()) for word in keywords if word.strip()]
    if not words:
        return None
    # 用前后不是单词字符代替 \b，以支持 "C++" 这类以符号结尾的关键词
    return re.compile(rf"(?<!\w)(?:{'|'.join(words)})(?!\w)", re.IGNORECASE)


def recent_months(today: date, months: int) -> list[str]:
    """今天所在的月份及之前的 months - 1 个月，格式为 YYYY-MM，从近到远。"""
    year, month = today.year, today.month
    result = []
    for _ in range(months):
        result.append(f"{year:04d}-{month:02d}")
        year, month = (year - 1, 12) if month == 1 else (year, month - 1)
    return result
