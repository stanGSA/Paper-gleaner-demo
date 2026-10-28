from datetime import date
from pathlib import Path

from gleaner.config import FiltersConfig
from gleaner.filters import apply_filters, compile_keywords, load_seen, record_seen
from gleaner.models import Paper
from gleaner.store import DataStore


def paper(id: str, title: str = "A paper", abstract: str = "Something.") -> Paper:
    return Paper(id=id, title=title, abstract=abstract)


def test_seen_round_trip_excludes_today(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    record_seen(store, [paper("1"), paper("2")], date(2026, 9, 30))
    record_seen(store, [paper("3")], date(2026, 10, 23))
    record_seen(store, [paper("4")], date(2026, 10, 24))

    assert load_seen(store, date(2026, 10, 24)) == {"1", "2", "3"}
    assert (tmp_path / "history/seen/2026-10.jsonl").is_file()


def test_seen_lookback_crosses_year(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    record_seen(store, [paper("old")], date(2025, 10, 15))
    record_seen(store, [paper("dec")], date(2025, 12, 31))

    assert load_seen(store, date(2026, 1, 5), months=3) == {"dec"}
    assert load_seen(store, date(2026, 1, 5), months=4) == {"old", "dec"}


def test_apply_filters() -> None:
    papers = [
        paper("dup"),
        paper("rag", title="Better RAG for science"),
        paper("storage", abstract="Efficient storage of KV cache."),
        paper("medical", title="Medical imaging with RAG"),
        paper("plain"),
    ]
    config = FiltersConfig(include_keywords=["rag"], exclude_keywords=["medical imaging"])

    result = apply_filters(papers, seen={"dup"}, config=config)

    assert [p.id for p in result.papers] == ["rag", "storage", "plain"]
    assert result.forced_ids == {"rag"}  # "storage" 不算命中 "rag"
    assert result.duplicates == 1
    assert result.excluded == 1  # 同时命中黑白名单时以黑名单为准


def test_apply_filters_without_keywords() -> None:
    result = apply_filters([paper("a"), paper("b")], seen=set(), config=FiltersConfig())
    assert [p.id for p in result.papers] == ["a", "b"]
    assert result.forced_ids == set()


def test_compile_keywords() -> None:
    assert compile_keywords([]) is None
    assert compile_keywords(["  "]) is None

    pattern = compile_keywords(["C++", "long context"])
    assert pattern is not None
    assert pattern.search("Rewriting C++ code")
    assert not pattern.search("Rewriting C++x code")
    assert pattern.search("LONG CONTEXT reasoning")
    assert not pattern.search("longcontext")
