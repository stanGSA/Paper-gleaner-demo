import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from gleaner.models import Feedback, FeedbackStats, Paper
from gleaner.profile import (
    Memory,
    MutedTopic,
    aggregate_feedback,
    command_feedback,
    load_feedback,
    load_memory,
    record_feedback,
)
from gleaner.profile.memory import MAX_NOTE_CHARS, feedback_path, latest_reactions
from gleaner.store import DataStore

NOW = datetime(2026, 10, 26, 2, 0, tzinfo=UTC)
TZ = "Asia/Shanghai"
PAPER = Paper(id="2610.00001", title="LongKV", abstract="We compress the KV cache.")


def fb(signal: str, weight: float, hours: float = 0, **fields) -> Feedback:
    """距离 NOW hours 小时之前的一条反馈。"""
    values = {
        "paper_id": PAPER.id,
        "title": PAPER.title,
        "abstract": PAPER.abstract,
        "signal": signal,
        "weight": weight,
        "source": "reaction" if signal in ("up", "down") else "command",
        "created_at": NOW - timedelta(hours=hours),
    } | fields
    return Feedback(**values)


def one(records: list[Feedback]):
    [item] = aggregate_feedback(records)
    return item


# --- 读写 -----------------------------------------------------------------------


def test_record_and_load_round_trip(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    # UTC 10 月 31 日 17:00 已是北京时间 11 月 1 日，落在 11 月的文件里
    late = datetime(2026, 10, 31, 17, 0, tzinfo=UTC)
    records = [fb("like", 2, note="有开源代码"), fb("deep", 0.5, created_at=late)]

    record_feedback(store, records, timezone=TZ)

    assert feedback_path(late, TZ) == "feedback/2026-11.jsonl"
    assert len(store.read_jsonl("feedback/2026-10.jsonl")) == 1
    assert len(store.read_jsonl("feedback/2026-11.jsonl")) == 1
    loaded = load_feedback(store, date(2026, 11, 2))
    assert loaded == records
    assert "有开源代码" in (tmp_path / "feedback/2026-10.jsonl").read_text(encoding="utf-8")


def test_load_sorts_by_time_and_skips_bad_lines(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    later, earlier = fb("dislike", -2, hours=1), fb("like", 2, hours=5)
    # union 合并后行序可能被打乱；手改的记录可能不带时区、可能写错字段
    lines = [
        later.model_dump_json(),
        json.dumps({"paper_id": "x", "signal": "love"}),
        "[1, 2]",
        earlier.model_dump_json(),
        json.dumps(fb("up", 1).model_dump(mode="json") | {"created_at": "2026-10-26T03:00:00"}),
    ]
    store.write_text("feedback/2026-10.jsonl", "\n".join(lines) + "\n")

    loaded = load_feedback(store, date(2026, 10, 26))

    assert [r.signal for r in loaded] == ["like", "dislike", "up"]
    assert loaded[-1].created_at == datetime(2026, 10, 26, 3, 0, tzinfo=UTC)


def test_load_reads_only_recent_months(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    store.append_jsonl("feedback/2025-10.jsonl", [fb("like", 2).model_dump(mode="json")])
    store.append_jsonl("feedback/2025-11.jsonl", [fb("dislike", -2).model_dump(mode="json")])

    assert [r.signal for r in load_feedback(store, date(2026, 10, 26))] == ["dislike"]


def test_command_feedback_weights_and_notes() -> None:
    like = command_feedback(PAPER, "like", now=NOW, note="  有开源代码，\n 方法简单 ")
    assert (like.signal, like.weight, like.source) == ("like", 2.0, "command")
    assert like.note == "有开源代码， 方法简单"
    assert command_feedback(PAPER, "dislike", now=NOW).weight == -2.0
    assert command_feedback(PAPER, "deep", now=NOW).weight == 0.5
    assert command_feedback(PAPER, "like", now=NOW, note="   ").note is None
    assert len(command_feedback(PAPER, "like", now=NOW, note="长" * 500).note) == MAX_NOTE_CHARS


# --- 聚合 -----------------------------------------------------------------------


def test_repeated_reaction_counts_once() -> None:
    # 同一个 👍 每天都会被同步到；即使重复写入，也只取最新状态
    item = one([fb("up", 1, hours=48), fb("up", 1, hours=24), fb("up", 1)])
    assert (item.reaction, item.total) == (1.0, 1.0)


def test_revoked_reaction() -> None:
    item = one([fb("up", 1, hours=24), fb("up", 0)])
    assert (item.reaction, item.total) == (0.0, 0.0)
    assert item.signals == []


def test_reaction_switches_to_thumbs_down() -> None:
    item = one([fb("up", 1, hours=24), fb("down", -1)])
    assert item.total == -1.0
    assert item.signals == ["👎"]


def test_thumbs_up_and_dislike() -> None:
    item = one([fb("up", 1, hours=24), fb("dislike", -2, note="纯 benchmark")])
    assert (item.reaction, item.command, item.total) == (1.0, -2.0, -1.0)
    assert item.signals == ["👍", "/dislike"]
    assert item.notes == ["纯 benchmark"]


def test_latest_command_wins_and_resets_notes() -> None:
    item = one(
        [
            fb("like", 2, hours=3, note="有代码"),
            fb("like", 2, hours=2, note="有代码"),
            fb("dislike", -2, hours=1, note="实验不可信"),
        ]
    )
    assert item.command == -2.0
    assert item.notes == ["实验不可信"]

    item = one([fb("like", 2, hours=2, note="有代码"), fb("like", 2, hours=1, note="方法简单")])
    assert item.notes == ["有代码", "方法简单"]


def test_implicit_feedback_counts_once() -> None:
    item = one([fb("deep", 0.5, hours=3), fb("deep", 0.5, hours=2), fb("compare", 0.5)])
    assert (item.implicit, item.total) == (0.5, 0.5)
    assert item.signals == ["/deep", "/compare"]


def test_all_signals_add_up() -> None:
    item = one([fb("deep", 0.5, hours=3), fb("down", -1, hours=2), fb("like", 2, hours=1)])
    assert item.total == 1.5
    assert item.signals == ["👎", "/like", "/deep"]


def test_order_is_by_time_not_by_input() -> None:
    item = one([fb("dislike", -2, hours=1), fb("like", 2, hours=5)])
    assert item.command == -2.0


def test_metadata_prefers_records_with_abstract() -> None:
    # arXiv API 不可用时 /like 只记下了 ID；之后的表情记录带有完整的标题和摘要
    item = one(
        [
            fb("up", 1, hours=5),
            fb("like", 2, hours=1, title=PAPER.id, abstract=""),
        ]
    )
    assert (item.title, item.abstract) == (PAPER.title, PAPER.abstract)
    assert item.has_text

    bare = one([fb("like", 2, title=PAPER.id, abstract="")])
    assert not bare.has_text
    assert bare.paper.title == PAPER.id


def test_groups_by_paper() -> None:
    items = aggregate_feedback(
        [fb("like", 2, hours=2), fb("down", -1, hours=1, paper_id="2610.00002"), fb("deep", 0.5)]
    )
    assert [(i.paper_id, i.total) for i in items] == [("2610.00001", 2.5), ("2610.00002", -1.0)]


def test_updated_at_ignores_revoked_signals() -> None:
    item = one([fb("like", 2, hours=48), fb("up", 1, hours=24), fb("up", 0)])
    assert item.updated_at == NOW - timedelta(hours=48)

    neutral = one([fb("up", 1, hours=24), fb("up", 0)])
    assert neutral.updated_at == NOW


def test_strength_and_time_decay() -> None:
    like = one([fb("like", 2)])
    assert like.strength() == 1.0
    assert one([fb("up", 1)]).strength() == 0.5
    assert one([fb("deep", 0.5)]).strength() == 0.25
    assert one([fb("like", 2, hours=1), fb("up", 1), fb("deep", 0.5)]).strength() == 1.0

    assert like.weight(NOW, 90) == pytest.approx(1.0)
    assert like.weight(NOW + timedelta(days=90), 90) == pytest.approx(0.5)
    assert like.weight(NOW + timedelta(days=180), 90) == pytest.approx(0.25)
    assert like.weight(NOW + timedelta(days=30), 30) == pytest.approx(0.5)
    assert like.weight(NOW - timedelta(days=1), 90) == pytest.approx(1.0)  # 时钟偏差不放大


def test_latest_reactions() -> None:
    records = [
        fb("up", 1, hours=48),
        fb("like", 2, hours=30),
        fb("up", 0, hours=24),
        fb("down", -1, hours=1, paper_id="2610.00002"),
        fb("deep", 0.5, paper_id="2610.00003"),
    ]
    assert latest_reactions(records) == {"2610.00001": 0.0, "2610.00002": -1.0}


# --- 粗筛用到的记忆 -------------------------------------------------------------


def test_memory_anchors_and_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    feedback = aggregate_feedback(
        [
            fb("up", 1, hours=48),  # 2610.00001：👍 + /like = +3
            fb("like", 2, hours=24),
            fb("down", -1, paper_id="2610.00002"),
            fb("deep", 0.5, paper_id="2610.00003"),
            fb("up", 1, hours=5, paper_id="2610.00004"),  # 撤销了：不参与
            fb("up", 0, paper_id="2610.00004"),
            fb("dislike", -2, paper_id="2610.00005", title="2610.00005", abstract=""),
        ]
    )
    memory = Memory(feedback, [MutedTopic("医学影像", None, NOW)])

    positive = memory.positive(NOW, 90)
    assert [(item.paper_id, round(weight, 3)) for item, weight in positive] == [
        ("2610.00001", 0.992),  # 一天前，衰减了一点
        ("2610.00003", 0.25),
    ]
    assert [item.paper_id for item, _ in memory.negative(NOW, 90)] == ["2610.00002"]
    assert memory.stats(NOW, 90) == FeedbackStats(
        up=1, down=1, like=1, dislike=1, implicit=1, positive=2, negative=1, muted=1
    )

    monkeypatch.setattr("gleaner.profile.memory.MAX_FEEDBACK_ANCHORS", 1)
    assert [item.paper_id for item, _ in memory.positive(NOW, 90)] == ["2610.00001"]


def test_load_memory(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    record_feedback(store, [fb("like", 2)], timezone=TZ)
    store.append_jsonl(
        "profile/muted.jsonl",
        [{"action": "mute", "topic": "医学影像", "query": "Medical imaging."}],
    )

    memory = load_memory(store, date(2026, 10, 26))

    assert [item.paper_id for item in memory.feedback] == ["2610.00001"]
    assert [(m.topic, m.query) for m in memory.muted] == [("医学影像", "Medical imaging.")]
