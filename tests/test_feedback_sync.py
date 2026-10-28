from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from gleaner.models import Digest, ScoredPaper
from gleaner.notify import GitHubClient
from gleaner.pipelines.daily import digest_path, load_recent_digests
from gleaner.pipelines.feedback import sync_reactions
from gleaner.profile import aggregate_feedback, load_feedback
from gleaner.store import DataStore

API = "https://api.github.com/repos/o/r"
TODAY = date(2026, 10, 26)
NOW = datetime(2026, 10, 26, 1, 30, tzinfo=UTC)
TZ = "Asia/Shanghai"


def reaction(id: int, login: str, content: str, *, type: str = "User") -> dict:
    """与 GET /repos/{owner}/{repo}/issues/comments/{id}/reactions 返回的数组元素同样的结构。"""
    return {
        "id": id,
        "node_id": f"REA_{id}",
        "user": {"login": login, "id": 1000 + id, "type": type, "site_admin": False},
        "content": content,
        "created_at": "2026-10-25T12:00:00Z",
    }


def digest(day: date, comment_ids: dict[str, int], issue: int = 1) -> Digest:
    papers = [
        ScoredPaper(id=paper_id, title=f"Title {paper_id}", abstract=f"Abstract {paper_id}.")
        for paper_id in comment_ids
    ]
    return Digest(
        day=day,
        generated_at=NOW,
        papers=papers,
        issue_number=issue,
        comment_ids=comment_ids,
    )


DIGEST = digest(date(2026, 10, 25), {"2610.00001": 101, "2610.00002": 102})


def mock_reactions(comment_id: int, *reactions: dict, status: int = 200) -> respx.Route:
    return respx.get(f"{API}/issues/comments/{comment_id}/reactions").mock(
        return_value=httpx.Response(status, json=list(reactions) if status == 200 else {})
    )


def sync(store: DataStore, digests=(DIGEST,), *, owner: str = "owner", now: datetime = NOW):
    reader = GitHubClient("ghs_token", "o/r", backoff_seconds=0)
    return sync_reactions(
        store, list(digests), reader, owner=owner, today=TODAY, now=now, timezone=TZ
    )


def records(store: DataStore) -> list[tuple[str, str, float]]:
    return [(r.paper_id, r.signal, r.weight) for r in load_feedback(store, TODAY)]


@respx.mock
def test_syncs_owner_reactions(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    route = mock_reactions(101, reaction(1, "owner", "+1"), reaction(2, "owner", "eyes"))
    mock_reactions(102, reaction(3, "owner", "-1"))

    result = sync(store)

    assert (result.comments, result.records, result.error) == (2, 2, None)
    assert records(store) == [("2610.00001", "up", 1.0), ("2610.00002", "down", -1.0)]
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer ghs_token"
    assert request.url.params["per_page"] == "100"
    saved = load_feedback(store, TODAY)[0]
    assert (saved.title, saved.abstract, saved.source) == (
        "Title 2610.00001",
        "Abstract 2610.00001.",
        "reaction",
    )
    assert saved.created_at == NOW


@respx.mock
def test_repeated_sync_does_not_duplicate(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    mock_reactions(101, reaction(1, "owner", "+1"))
    mock_reactions(102)

    sync(store)
    again = sync(store, now=NOW + timedelta(days=1))

    assert again.records == 0
    assert records(store) == [("2610.00001", "up", 1.0)]
    [item] = aggregate_feedback(load_feedback(store, TODAY))
    assert item.total == 1.0


@respx.mock
def test_revoked_and_switched_reactions(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    first = mock_reactions(101, reaction(1, "owner", "+1"))
    second = mock_reactions(102, reaction(2, "owner", "+1"))
    sync(store)

    # 第一篇的 👍 被撤销，第二篇改成了 👎
    first.mock(return_value=httpx.Response(200, json=[]))
    second.mock(return_value=httpx.Response(200, json=[reaction(3, "owner", "-1")]))
    result = sync(store, now=NOW + timedelta(days=1))

    assert result.records == 2
    assert records(store)[2:] == [("2610.00001", "up", 0.0), ("2610.00002", "down", -1.0)]
    totals = {i.paper_id: i.total for i in aggregate_feedback(load_feedback(store, TODAY))}
    assert totals == {"2610.00001": 0.0, "2610.00002": -1.0}


@respx.mock
def test_both_thumbs_cancel_out(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    first = mock_reactions(101, reaction(1, "owner", "+1"))
    mock_reactions(102, reaction(2, "owner", "+1"), reaction(3, "owner", "-1"))
    sync(store)
    assert records(store) == [("2610.00001", "up", 1.0)]  # 两个都点了：状态为 0，不写记录

    first.mock(
        return_value=httpx.Response(
            200, json=[reaction(1, "owner", "+1"), reaction(4, "owner", "-1")]
        )
    )
    sync(store, now=NOW + timedelta(days=1))
    assert records(store)[-1] == ("2610.00001", "up", 0.0)


@respx.mock
def test_ignores_everyone_but_the_owner(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    mock_reactions(
        101,
        reaction(1, "collaborator", "+1"),  # 协作者
        reaction(2, "github-actions[bot]", "-1", type="Bot"),  # 机器人
        reaction(3, "owner-fan", "+1"),
        {"id": 4, "user": None, "content": "+1"},  # 已注销的用户
    )
    mock_reactions(102, reaction(5, "owner", "heart"), reaction(6, "owner", "rocket"))

    result = sync(store)

    assert (result.comments, result.records) == (2, 0)
    assert records(store) == []


@respx.mock
def test_owner_is_case_insensitive(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    mock_reactions(101, reaction(1, "StanGSA", "+1"))
    mock_reactions(102)

    sync(store, owner="stanGSA")

    assert records(store) == [("2610.00001", "up", 1.0)]


@respx.mock
def test_paginates_reactions(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    page_one = [reaction(i, f"user{i}", "+1") for i in range(100)]
    route = respx.get(f"{API}/issues/comments/101/reactions").mock(
        side_effect=[
            httpx.Response(200, json=page_one),
            httpx.Response(200, json=[reaction(200, "owner", "-1")]),
        ]
    )
    mock_reactions(102)

    sync(store)

    assert [call.request.url.params["page"] for call in route.calls] == ["1", "2"]
    assert records(store) == [("2610.00001", "down", -1.0)]


@respx.mock
def test_merges_reactions_across_digests(tmp_path: Path) -> None:
    # 同一篇论文出现在两份日报里，在其中任意一条评论上点的表情都算
    store = DataStore(tmp_path)
    older = digest(date(2026, 10, 22), {"2610.00001": 90}, issue=0)
    mock_reactions(90)
    mock_reactions(101, reaction(1, "owner", "+1"))
    mock_reactions(102)

    sync(store, digests=(older, DIGEST))

    assert records(store) == [("2610.00001", "up", 1.0)]


@respx.mock
def test_rate_limit_keeps_what_was_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gleaner.notify.github_issue.time.sleep", lambda seconds: None)
    store = DataStore(tmp_path)
    mock_reactions(101, reaction(1, "owner", "+1"))
    limited = respx.get(f"{API}/issues/comments/102/reactions").mock(
        return_value=httpx.Response(
            403,
            json={"message": "You have exceeded a secondary rate limit."},
            headers={"x-ratelimit-remaining": "0"},
        )
    )

    result = sync(store)

    assert limited.call_count == 3  # GitHubClient 按限流重试过
    assert result.records == 1
    assert result.comments == 1
    assert "HTTP 403" in result.error
    assert records(store) == [("2610.00001", "up", 1.0)]


@respx.mock
def test_network_error_is_reported(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    respx.get(f"{API}/issues/comments/101/reactions").mock(side_effect=httpx.ConnectError("boom"))

    result = sync(store)

    assert result.error == "ConnectError：boom"
    assert result.records == 0


@respx.mock
def test_skips_deleted_comments(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    mock_reactions(101, status=404)
    mock_reactions(102, reaction(1, "owner", "-1"))

    result = sync(store)

    assert result.error is None
    assert result.comments == 1
    assert records(store) == [("2610.00002", "down", -1.0)]


def test_load_recent_digests(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    for day in (date(2026, 10, 19), date(2026, 10, 20), date(2026, 10, 26)):
        store.write_model(digest_path(day), digest(day, {}))
    store.write_text(digest_path(date(2026, 10, 23)), "{broken")

    days = [d.day for d in load_recent_digests(store, TODAY, 7)]

    assert days == [date(2026, 10, 20), date(2026, 10, 26)]
