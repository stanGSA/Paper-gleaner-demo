import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from gleaner.config import Config
from gleaner.filters import record_seen
from gleaner.llm import LLMError
from gleaner.models import DeepReport, Digest, Paper, ScoredPaper, ZoteroStats
from gleaner.notify import GitHubError, IssueRef
from gleaner.notify.base import NotifyError
from gleaner.pipelines.daily import digest_path, run_daily
from gleaner.profile import (
    LEARNED_PREFERENCES_PATH,
    ProfileError,
    command_feedback,
    load_feedback,
    load_muted,
    record_feedback,
)
from gleaner.profile.muted import MUTED_PATH
from gleaner.reader import FullText, Section, load_report, save_report
from gleaner.sources.arxiv import parse_rss
from gleaner.sources.zotero import ZoteroResult
from gleaner.store import DataStore
from tests.conftest import (
    DEEP_REPORT,
    FIXTURES,
    PIPELINE_VOCABULARY,
    FakeNotifier,
    KeywordEmbedder,
    fake_llm,
    pipeline_llm,
)

TODAY = date(2026, 10, 26)
SCORES = {"2610.00001": 9, "2610.00002": 7, "2610.00003": 4}


class FakeSource:
    name = "fake"

    def __init__(self, papers: list[Paper], recent: list[Paper] | None = None) -> None:
        self.papers = papers
        self.recent = recent or []
        self.calls = 0
        self.recent_limits: list[int] = []

    def fetch_latest(self) -> list[Paper]:
        self.calls += 1
        return self.papers

    def fetch_recent(self, limit: int) -> list[Paper]:
        self.recent_limits.append(limit)
        return self.recent[:limit]


class FakePublisher:
    def __init__(self, existing: IssueRef | None = None) -> None:
        self.existing = existing
        self.published: list[Digest] = []
        self.repaired: list[Digest] = []
        self.reports: list[dict] = []  # 每次 publish / repair 收到的自动深读报告
        self.closed = 0

    def find_existing(self, day: date) -> IssueRef | None:
        return self.existing

    def publish(self, digest: Digest, reports=None) -> Digest:
        self.published.append(digest)
        self.reports.append(dict(reports or {}))
        return digest.model_copy(update={"issue_number": 7, "issue_url": "https://x/7"})

    def repair(self, digest: Digest, reports=None) -> Digest:
        self.repaired.append(digest)
        self.reports.append(dict(reports or {}))
        ids = {p.id: 900 + i for i, p in enumerate(digest.papers)} | digest.comment_ids
        return digest.model_copy(update={"comment_ids": ids})

    def close_stale(self, now: datetime | None = None) -> list[int]:
        self.closed += 1
        return []


class FakeReader:
    """代替 GitHubClient 读取评论表情；error 不为空时每次都抛出。"""

    def __init__(self, reactions: dict[int, list[dict]] | None = None, error=None) -> None:
        self.reactions = reactions or {}
        self.error = error
        self.calls: list[int] = []

    def list_reactions(self, comment_id: int) -> list[dict[str, Any]]:
        self.calls.append(comment_id)
        if self.error is not None:
            raise self.error
        return self.reactions.get(comment_id, [])


def thumbs(login: str, content: str) -> dict[str, Any]:
    return {"id": 1, "user": {"login": login, "type": "User"}, "content": content}


def seed_yesterday(store: DataStore) -> None:
    """昨天的日报：一篇论文，评论 ID 为 501。"""
    paper = ScoredPaper(id="2610.00009", title="Yesterday", abstract="KV cache again.")
    yesterday = Digest(
        day=date(2026, 10, 25),
        generated_at=datetime(2026, 10, 25, 1, 30, tzinfo=UTC),
        papers=[paper],
        issue_number=6,
        comment_ids={paper.id: 501},
    )
    store.write_model(digest_path(yesterday.day), yesterday)


@pytest.fixture
def config(tmp_path: Path) -> Config:
    interests = tmp_path / "interests.md"
    interests.write_text("我研究长上下文推理和 RAG。", encoding="utf-8")
    # 自动深读另有专门的测试；这里关掉，避免请求 arXiv 全文
    return Config.model_validate(
        {"profile": {"interests_file": str(interests)}, "deep_read": {"auto_top_m": 0}}
    )


@pytest.fixture
def papers(rss_bytes: bytes) -> list[Paper]:
    # 与 ArxivSource 默认过滤一致：new 和 cross，不含 replace
    return [p for p in parse_rss(rss_bytes) if p.announce_type in {"new", "cross"}]


def run(config: Config, store: DataStore, papers: list[Paper], scores=SCORES, **kwargs):
    llm, fake = fake_llm(pipeline_llm(scores))
    source = FakeSource(papers, recent=kwargs.pop("recent", None))
    result = run_daily(
        config,
        store=store,
        llm=llm,
        embedder=KeywordEmbedder(PIPELINE_VOCABULARY),
        today=kwargs.pop("today", TODAY),
        source=source,
        now=datetime(2026, 10, 26, 1, 30, tzinfo=UTC),
        **kwargs,
    )
    return result, fake, source


def test_preview_without_publisher(config: Config, papers: list[Paper], tmp_path: Path) -> None:
    store = DataStore(tmp_path / "data")

    result, fake, _ = run(config, store, papers)

    assert result.status == "preview"
    digest = result.digest
    assert [p.id for p in digest.papers] == ["2610.00001", "2610.00002"]
    assert [p.id for p in digest.others] == ["2610.00003"]
    assert digest.papers[0].reason == "2610.00001 与长上下文推理相关"
    assert digest.overview == "今天以长上下文推理为主，建议先读 [1]。"
    assert digest.issue_number is None

    stats = digest.stats
    assert (stats.fetched, stats.candidates, stats.selected) == (3, 3, 2)
    assert stats.llm_calls == len(fake.requests) == 3  # 兴趣拆解 + 1 批精排 + 概览
    assert stats.prompt_tokens == 300

    assert store.read_model(digest_path(TODAY), Digest) == digest
    assert [r["id"] for r in store.read_jsonl("history/seen/2026-10.jsonl")] == [
        "2610.00001",
        "2610.00002",
        "2610.00003",
    ]
    assert store.read_jsonl("stats/2026-10.jsonl")[0]["status"] == "preview"
    assert store.read_text("profile/anchors.json") is not None


def test_publishes_and_records_issue(config: Config, papers: list[Paper], tmp_path: Path) -> None:
    store = DataStore(tmp_path / "data")
    publisher = FakePublisher()

    result, _, _ = run(config, store, papers, publisher=publisher)

    assert result.status == "published"
    assert len(publisher.published) == 1
    assert publisher.closed == 1
    saved = store.read_model(digest_path(TODAY), Digest)
    assert saved.issue_number == 7
    assert store.read_jsonl("stats/2026-10.jsonl")[0]["issue"] == 7


def test_skips_when_already_published(config: Config, papers: list[Paper], tmp_path: Path) -> None:
    store = DataStore(tmp_path / "data")
    existing = IssueRef(3, "https://x/3")

    result, fake, source = run(config, store, papers, publisher=FakePublisher(existing))

    assert result.status == "already_published"
    assert result.existing == existing
    assert source.calls == 0
    assert fake.requests == []
    assert not (tmp_path / "data").exists()


def test_rerun_repairs_missing_comments(
    config: Config, papers: list[Paper], tmp_path: Path
) -> None:
    store = DataStore(tmp_path / "data")
    first, _, _ = run(config, store, papers, publisher=FakePublisher())
    saved = first.digest.model_copy(update={"comment_ids": {"2610.00001": 11}})
    store.write_model(digest_path(TODAY), saved)
    publisher = FakePublisher(IssueRef(7, "https://x/7"))

    result, fake, source = run(config, store, papers, publisher=publisher)

    assert result.status == "repaired"
    assert publisher.repaired == [saved]
    assert source.calls == 0
    assert fake.requests == []
    assert store.read_model(digest_path(TODAY), Digest).comment_ids == {
        "2610.00001": 11,
        "2610.00002": 901,
    }


def test_rerun_without_missing_comments(
    config: Config, papers: list[Paper], tmp_path: Path
) -> None:
    store = DataStore(tmp_path / "data")
    first, _, _ = run(config, store, papers, publisher=FakePublisher())
    complete = first.digest.model_copy(update={"comment_ids": {"2610.00001": 11, "2610.00002": 12}})
    store.write_model(digest_path(TODAY), complete)

    result, _, _ = run(config, store, papers, publisher=FakePublisher(IssueRef(7, "https://x/7")))

    assert result.status == "already_published"
    assert result.digest == complete


def test_rerun_ignores_digest_of_another_issue(
    config: Config, papers: list[Paper], tmp_path: Path
) -> None:
    store = DataStore(tmp_path / "data")
    run(config, store, papers, publisher=FakePublisher())  # 保存的 digest 对应 Issue #7
    publisher = FakePublisher(IssueRef(8, "https://x/8"))

    result, _, _ = run(config, store, papers, publisher=publisher)

    assert result.status == "already_published"
    assert publisher.repaired == []


def test_force_republishes(config: Config, papers: list[Paper], tmp_path: Path) -> None:
    publisher = FakePublisher(IssueRef(3, "https://x/3"))
    result, _, _ = run(config, DataStore(tmp_path), papers, publisher=publisher, force=True)
    assert result.status == "published"


# --- 推送渠道 -------------------------------------------------------------------


def test_notifies_after_publishing(config: Config, papers: list[Paper], tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    email = FakeNotifier("email")
    feishu = FakeNotifier("feishu", NotifyError("飞书返回错误 19021：签名校验失败"))
    wecom = FakeNotifier("wecom")

    result, _, _ = run(
        config, store, papers, publisher=FakePublisher(), notifiers=[email, feishu, wecom]
    )

    # 某个渠道失败不影响其它渠道和日报
    assert result.status == "published"
    [(sent, test)] = email.sent
    assert sent.issue_url == "https://x/7" and test is False
    assert len(wecom.sent) == 1
    saved = store.read_model(digest_path(TODAY), Digest)
    assert {name: r.status for name, r in saved.notifications.items()} == {
        "email": "sent",
        "feishu": "failed",
        "wecom": "sent",
    }
    assert saved.notifications["feishu"].error == "飞书返回错误 19021：签名校验失败"
    assert result.notified == saved.notifications
    assert result.warnings == ["飞书推送失败：飞书返回错误 19021：签名校验失败"]


def test_fallback_run_only_resends_failed_channels(
    config: Config, papers: list[Paper], tmp_path: Path
) -> None:
    store = DataStore(tmp_path)
    email, feishu = FakeNotifier("email"), FakeNotifier("feishu", NotifyError("限流"))
    first, _, _ = run(config, store, papers, publisher=FakePublisher(), notifiers=[email, feishu])
    complete = first.digest.model_copy(update={"comment_ids": {"2610.00001": 1, "2610.00002": 2}})
    store.write_model(digest_path(TODAY), complete)

    email2, feishu2, wecom2 = FakeNotifier("email"), FakeNotifier("feishu"), FakeNotifier("wecom")
    publisher = FakePublisher(IssueRef(7, "https://x/7"))
    result, fake, source = run(
        config, store, papers, publisher=publisher, notifiers=[email2, feishu2, wecom2]
    )

    assert result.status == "repaired"
    assert source.calls == 0 and fake.requests == []
    assert email2.sent == [] and wecom2.sent == []  # 成功过的、上次没启用的都不发
    assert len(feishu2.sent) == 1
    assert result.notified["feishu"].status == "sent"
    saved = store.read_model(digest_path(TODAY), Digest)
    assert saved.notifications["feishu"].status == "sent"
    assert saved.notifications["email"] == first.digest.notifications["email"]

    third, _, _ = run(
        config, store, papers, publisher=publisher, notifiers=[FakeNotifier("feishu")]
    )
    assert third.status == "already_published"


def test_notifies_without_github_issue(config: Config, papers: list[Paper], tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    wecom = FakeNotifier("wecom", NotifyError("企业微信返回错误 45009"))

    result, _, _ = run(config, store, papers, notifiers=[wecom])

    assert result.status == "published"
    [(sent, _)] = wecom.sent
    assert sent.issue_url is None and sent.issue_number is None

    # 兜底运行：不重新生成，只补发失败的渠道
    retry = FakeNotifier("wecom")
    second, fake, source = run(config, store, papers, notifiers=[retry])
    assert second.status == "repaired"
    assert source.calls == 0 and fake.requests == []
    assert len(retry.sent) == 1

    third, _, source = run(config, store, papers, notifiers=[FakeNotifier("wecom")])
    assert third.status == "already_published" and source.calls == 0


def test_no_notification_without_relevant_papers(
    config: Config, papers: list[Paper], tmp_path: Path
) -> None:
    email = FakeNotifier("email")
    low = {paper_id: 3 for paper_id in SCORES}

    result, _, _ = run(
        config,
        DataStore(tmp_path),
        papers,
        scores=low,
        publisher=FakePublisher(),
        notifiers=[email],
    )

    assert result.status == "nothing_relevant"
    assert email.sent == []
    assert result.digest.notifications == {}


def test_no_notification_without_announcements(config: Config, tmp_path: Path) -> None:
    email = FakeNotifier("email")
    result, _, _ = run(config, DataStore(tmp_path), [], notifiers=[email], sample_size=20)

    assert result.status == "no_announcements"
    assert email.sent == []


def test_no_announcements(config: Config, tmp_path: Path) -> None:
    publisher = FakePublisher()
    result, fake, _ = run(config, DataStore(tmp_path / "data"), [], publisher=publisher)

    assert result.status == "no_announcements"
    assert fake.requests == []
    assert publisher.published == []
    assert not (tmp_path / "data").exists()


def test_nothing_relevant(config: Config, papers: list[Paper], tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    publisher = FakePublisher()
    low = {paper_id: 3 for paper_id in SCORES}

    result, fake, _ = run(config, store, papers, scores=low, publisher=publisher)

    assert result.status == "nothing_relevant"
    assert publisher.published == []
    assert len(result.digest.others) == 3
    assert len(fake.requests) == 2  # 没有入选论文时不生成概览
    assert store.read_model(digest_path(TODAY), Digest) is not None


def test_papers_are_not_repeated_on_later_days(
    config: Config, papers: list[Paper], tmp_path: Path
) -> None:
    store = DataStore(tmp_path)
    run(config, store, papers)

    same_day, _, _ = run(config, store, papers)
    next_day, _, _ = run(config, store, papers, today=date(2026, 10, 27))

    assert same_day.digest.stats.duplicates == 0  # 当天强制重跑不算重复
    assert next_day.status == "nothing_relevant"
    assert next_day.digest.stats.duplicates == 3
    assert next_day.digest.stats.candidates == 0


def test_missing_interests_fails_before_fetching(papers: list[Paper], tmp_path: Path) -> None:
    config = Config.model_validate({"profile": {"interests_file": str(tmp_path / "nope.md")}})
    with pytest.raises(ProfileError):
        run(config, DataStore(tmp_path), papers)


def test_dry_run_samples_recent_papers_without_announcements(
    config: Config, papers: list[Paper], tmp_path: Path
) -> None:
    store = DataStore(tmp_path)
    record_seen(store, papers, date(2026, 10, 23))  # 最近的论文都处理过，也照样参与试运行

    result, fake, source = run(config, store, [], recent=papers, sample_size=20)

    assert result.status == "preview"
    assert result.sampled is True
    assert source.recent_limits == [20]
    assert [p.id for p in result.digest.papers] == ["2610.00001", "2610.00002"]
    assert result.digest.stats.duplicates == 0
    assert len(fake.requests) == 3
    # 抽样的论文不记为已处理，不影响之后的去重
    assert store.read_jsonl("history/seen/2026-10.jsonl") == [
        {"id": p.id, "date": "2026-10-23"} for p in papers
    ]


def test_no_sampling_when_publishing(config: Config, papers: list[Paper], tmp_path: Path) -> None:
    result, _, source = run(
        config, DataStore(tmp_path), [], recent=papers, sample_size=20, publisher=FakePublisher()
    )

    assert result.status == "no_announcements"
    assert source.recent_limits == []


# --- 第 0 步：表情同步 -----------------------------------------------------------


def test_syncs_reactions_first(config: Config, papers: list[Paper], tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    seed_yesterday(store)
    reader = FakeReader({501: [thumbs("Owner", "+1"), thumbs("someone", "-1")]})

    result, _, _ = run(config, store, papers, reactions=reader, owner="owner")

    assert reader.calls == [501]
    assert (result.sync.records, result.sync.error, result.warnings) == (1, None, [])
    assert [(r.paper_id, r.signal) for r in load_feedback(store, TODAY)] == [("2610.00009", "up")]
    assert result.digest.stats.feedback.synced == 1


def test_sync_failure_does_not_block_the_digest(
    config: Config, papers: list[Paper], tmp_path: Path
) -> None:
    store = DataStore(tmp_path)
    seed_yesterday(store)
    reader = FakeReader(error=GitHubError("查询表情失败（HTTP 403）：rate limit", 403))

    result, _, _ = run(config, store, papers, reactions=reader, owner="owner")

    assert result.status == "preview"
    assert [p.id for p in result.digest.papers] == ["2610.00001", "2610.00002"]
    assert result.digest.stats.feedback.sync_error == "查询表情失败（HTTP 403）：rate limit"
    [warning] = result.warnings
    assert "表情同步没有完成" in warning and "rate limit" in warning
    assert load_feedback(store, TODAY) == []


def test_syncs_even_when_already_published(
    config: Config, papers: list[Paper], tmp_path: Path
) -> None:
    store = DataStore(tmp_path)
    seed_yesterday(store)
    reader = FakeReader({501: [thumbs("owner", "-1")]})
    publisher = FakePublisher(IssueRef(3, "https://x/3"))

    result, _, source = run(
        config, store, papers, publisher=publisher, reactions=reader, owner="owner"
    )

    assert result.status == "already_published"
    assert source.calls == 0
    assert result.sync.records == 1


def test_skips_sync_without_reader(config: Config, papers: list[Paper], tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    seed_yesterday(store)

    result, _, _ = run(config, store, papers)

    assert (result.sync.records, result.sync.error) == (0, None)
    assert load_feedback(store, TODAY) == []


def test_reads_phase2_digest() -> None:
    # Phase 2 保存的 digest（没有粗筛分项和反馈统计字段）仍能读取，新字段取默认值
    text = (FIXTURES / "digest_phase2.json").read_text(encoding="utf-8")
    digest = Digest.model_validate_json(text)

    assert digest.issue_number == 1
    assert digest.comment_ids == {"2610.19540": 6388934280, "2610.04922": 6388934591}
    paper = digest.papers[0]
    assert paper.coarse_score == 0.8093
    assert (paper.interest_score, paper.positive_score, paper.negative_score) == (None, None, None)
    assert digest.stats.feedback.empty
    assert digest.notifications == {}  # Phase 4 的推送结果字段取默认值


# --- 反馈参与粗筛 ---------------------------------------------------------------


def with_describer(scores=SCORES, *, fail: bool = False):
    """在 pipeline_llm 之外再处理屏蔽主题的英文描述。"""
    base = pipeline_llm(scores)

    def handler(request: dict[str, Any]) -> str:
        if "要屏蔽的一个主题" in request["messages"][0]["content"]:
            if fail:
                raise LLMError("额度用完")
            return json.dumps({"query": "Medical imaging with deep networks."})
        return base(request)

    return handler


def test_feedback_shapes_coarse_scores(config: Config, papers: list[Paper], tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    liked = Paper(id="2609.00001", title="Fast attention", abstract="Attention kernels, attention.")
    disliked = Paper(id="2609.00002", title="Retrieval benchmark", abstract="Retrieval.")
    record_feedback(
        store,
        [
            command_feedback(liked, "like", now=datetime(2026, 10, 20, tzinfo=UTC)),
            command_feedback(disliked, "dislike", now=datetime(2026, 10, 21, tzinfo=UTC)),
        ],
        timezone=config.timezone,
    )

    result, _, _ = run(config, store, papers)

    by_id = {p.id: p for p in [*result.digest.papers, *result.digest.others]}
    attention, rag = by_id["2610.00003"], by_id["2610.00002"]
    assert attention.positive_score > rag.positive_score
    assert rag.negative_score > attention.negative_score
    assert attention.coarse_score == pytest.approx(
        attention.interest_score + attention.positive_score - attention.negative_score, abs=1e-3
    )
    feedback = result.digest.stats.feedback
    assert (feedback.like, feedback.dislike, feedback.positive, feedback.negative) == (1, 1, 1, 1)
    saved = store.read_model(digest_path(TODAY), Digest)
    assert saved.papers[0].interest_score is not None


def test_backfills_muted_topic_descriptions(
    config: Config, papers: list[Paper], tmp_path: Path
) -> None:
    store = DataStore(tmp_path)
    store.append_jsonl(MUTED_PATH, [{"action": "mute", "topic": "医学影像"}])  # 手写的，没有描述
    llm, fake = fake_llm(with_describer())

    result = run_daily(
        config,
        store=store,
        llm=llm,
        embedder=KeywordEmbedder(PIPELINE_VOCABULARY),
        today=TODAY,
        source=FakeSource(papers),
        now=datetime(2026, 10, 26, 1, 30, tzinfo=UTC),
    )

    assert len(fake.requests) == 4  # 兴趣拆解 + 屏蔽主题描述 + 精排 + 概览
    assert [m.query for m in load_muted(store)] == ["Medical imaging with deep networks."]
    assert result.digest.stats.feedback.muted == 1
    assert result.warnings == []


def test_muted_description_failure_is_a_warning(
    config: Config, papers: list[Paper], tmp_path: Path
) -> None:
    store = DataStore(tmp_path)
    store.append_jsonl(MUTED_PATH, [{"action": "mute", "topic": "医学影像"}])
    llm, _ = fake_llm(with_describer(fail=True))

    result = run_daily(
        config,
        store=store,
        llm=llm,
        embedder=KeywordEmbedder(PIPELINE_VOCABULARY),
        today=TODAY,
        source=FakeSource(papers),
    )

    assert result.status == "preview"
    [warning] = result.warnings
    assert "屏蔽主题「医学影像」的英文描述没有生成" in warning


def test_rerank_sees_preferences_and_muted_topics(
    config: Config, papers: list[Paper], tmp_path: Path
) -> None:
    store = DataStore(tmp_path)
    store.write_text(LEARNED_PREFERENCES_PATH, "<!-- gleaner:note x -->\n### 偏好\n- 有开源代码")
    store.append_jsonl(MUTED_PATH, [{"action": "mute", "topic": "医学影像", "query": "Medical."}])

    _, fake, _ = run(config, store, papers)

    [rerank_request] = [r for r in fake.requests if "<paper id=" in r["messages"][-1]["content"]]
    system = rerank_request["messages"][0]["content"]
    assert "<preferences>\n### 偏好\n- 有开源代码\n</preferences>" in system
    assert "<muted>\n- 医学影像\n</muted>" in system
    assert "gleaner:note" not in system


# --- 自动深读 -------------------------------------------------------------------


class FakeFetcher:
    """代替 FulltextFetcher：返回一段固定的正文；sections 为空表示连摘要都没有。"""

    def __init__(self, *, empty: bool = False, error: Exception | None = None) -> None:
        self.empty = empty
        self.error = error
        self.calls: list[str] = []

    def fetch(self, paper: Paper) -> FullText:
        self.calls.append(paper.id)
        if self.error is not None:
            raise self.error
        sections = (
            [] if self.empty else [Section("Introduction", "We compress KV cache.", "introduction")]
        )
        return FullText("html", sections, links=["https://github.com/example/longkv"])


def deep_handler(scores=SCORES, *, fail: bool = False):
    """在 pipeline_llm 之外再处理深读。"""
    base = pipeline_llm(scores)

    def handler(request: dict[str, Any]) -> str:
        if "深读报告" in request["messages"][0]["content"]:
            if fail:
                raise LLMError("调用 deepseek-reasoner 失败：Request timed out.")
            return DEEP_REPORT
        return base(request)

    return handler


def run_deep(config: Config, store: DataStore, papers: list[Paper], *, handler=None, **kwargs):
    llm, fake = fake_llm(handler or deep_handler(), **kwargs.pop("llm", {}))
    fetcher = kwargs.pop("fetcher", FakeFetcher())
    result = run_daily(
        config,
        store=store,
        llm=llm,
        embedder=KeywordEmbedder(PIPELINE_VOCABULARY),
        today=TODAY,
        source=FakeSource(papers),
        fetcher=fetcher,
        now=datetime(2026, 10, 26, 1, 30, tzinfo=UTC),
        **kwargs,
    )
    return result, fake, fetcher


def with_auto_deep_read(config: Config, top_m: int = 1) -> Config:
    return config.model_copy(
        update={"deep_read": config.deep_read.model_copy(update={"auto_top_m": top_m})}
    )


def deep_requests(fake) -> list[dict]:
    return [r for r in fake.requests if "深读报告" in r["messages"][0]["content"]]


def test_auto_deep_read_attaches_report_and_caches(
    config: Config, papers: list[Paper], tmp_path: Path
) -> None:
    store = DataStore(tmp_path)
    publisher = FakePublisher()

    result, fake, fetcher = run_deep(
        with_auto_deep_read(config), store, papers, publisher=publisher
    )

    assert result.status == "published"
    assert fetcher.calls == ["2610.00001"]  # 只深读排在第一的那篇
    [request] = deep_requests(fake)
    assert request["model"] == "deepseek-reasoner"
    [reports] = publisher.reports
    assert list(reports) == ["2610.00001"]
    assert reports["2610.00001"].body == DEEP_REPORT
    cached = load_report(store, "2610.00001")
    assert cached is not None and cached.body == DEEP_REPORT  # 之后 /deep 直接返回缓存
    digest = store.read_model(digest_path(TODAY), Digest)
    assert digest.deep_reads == ["2610.00001"]
    assert digest.stats.deep_read == 1
    assert digest.stats.llm_calls == len(fake.requests) == 4  # 深读的 token 计入运行统计
    assert result.warnings == []


def test_auto_deep_read_reuses_cache(config: Config, papers: list[Paper], tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    report = DeepReport(
        paper_id="2610.00001",
        title="LongKV",
        model="deepseek-reasoner",
        source="pdf",
        generated_at=datetime(2026, 10, 20, tzinfo=UTC),
        body="### 一句话总结\n缓存的报告。",
    )
    save_report(store, report)
    publisher = FakePublisher()

    result, fake, fetcher = run_deep(
        with_auto_deep_read(config), store, papers, publisher=publisher
    )

    assert fetcher.calls == [] and deep_requests(fake) == []
    assert publisher.reports[0]["2610.00001"].body == "### 一句话总结\n缓存的报告。"
    assert result.digest.deep_reads == ["2610.00001"]


def test_auto_deep_read_disabled_makes_no_calls(
    config: Config, papers: list[Paper], tmp_path: Path
) -> None:
    result, fake, fetcher = run_deep(config, DataStore(tmp_path), papers)  # auto_top_m=0

    assert fetcher.calls == [] and deep_requests(fake) == []
    assert result.digest.deep_reads == [] and result.digest.stats.deep_read == 0


def test_auto_deep_read_llm_failure_degrades(
    config: Config, papers: list[Paper], tmp_path: Path
) -> None:
    store = DataStore(tmp_path)
    publisher = FakePublisher()

    result, _, _ = run_deep(
        with_auto_deep_read(config),
        store,
        papers,
        handler=deep_handler(fail=True),
        publisher=publisher,
    )

    assert result.status == "published"
    assert publisher.reports == [{}]
    assert result.digest.deep_reads == []
    [note] = result.digest.stats.deep_read_notes
    assert note.startswith("第 1 篇自动深读失败：调用 deepseek-reasoner 失败")
    assert result.warnings == [note]
    assert load_report(store, "2610.00001") is None


@pytest.mark.parametrize(
    ("fetcher", "reason"),
    [
        (FakeFetcher(empty=True), "没能获取全文和摘要"),
        (FakeFetcher(error=RuntimeError("corrupt pdf")), "获取全文失败（RuntimeError）"),
    ],
)
def test_auto_deep_read_fulltext_failure_degrades(
    config: Config, papers: list[Paper], tmp_path: Path, fetcher: FakeFetcher, reason: str
) -> None:
    result, fake, _ = run_deep(
        with_auto_deep_read(config), DataStore(tmp_path), papers, fetcher=fetcher
    )

    assert result.status == "preview"
    assert deep_requests(fake) == []
    assert result.digest.stats.deep_read_notes == [f"第 1 篇没有自动深读：{reason}"]


def test_auto_deep_read_skipped_over_budget(
    config: Config, papers: list[Paper], tmp_path: Path
) -> None:
    # 前面的调用每次记 120 token；剩余预算不够 max_input_tokens + 输出时跳过
    result, fake, fetcher = run_deep(
        with_auto_deep_read(config),
        DataStore(tmp_path),
        papers,
        llm={"daily_token_budget": 20_000},
    )

    assert fetcher.calls == [] and deep_requests(fake) == []
    [note] = result.digest.stats.deep_read_notes
    assert "第 1 篇没有自动深读：今天已用 360 token" in note
    assert "超出 llm.daily_token_budget（20.0k）" in note
    assert result.warnings == [note]


def test_fallback_run_reattaches_cached_report(
    config: Config, papers: list[Paper], tmp_path: Path
) -> None:
    store = DataStore(tmp_path)
    first, _, _ = run_deep(with_auto_deep_read(config), store, papers, publisher=FakePublisher())
    # 第一篇的评论（附深读报告）上次没发出去
    saved = first.digest.model_copy(update={"comment_ids": {"2610.00002": 12}})
    store.write_model(digest_path(TODAY), saved)
    publisher = FakePublisher(IssueRef(7, "https://x/7"))

    result, fake, fetcher = run_deep(
        with_auto_deep_read(config), store, papers, publisher=publisher
    )

    assert result.status == "repaired"
    assert fake.requests == [] and fetcher.calls == []  # 不重新生成
    assert publisher.reports[0]["2610.00001"].body == DEEP_REPORT


# --- Zotero ---------------------------------------------------------------------


class FakeZotero:
    def __init__(self, result: ZoteroResult) -> None:
        self.result = result
        self.calls = 0

    def load(self) -> ZoteroResult:
        self.calls += 1
        return self.result


def test_zotero_anchors_join_coarse_ranking(
    config: Config, papers: list[Paper], tmp_path: Path
) -> None:
    store = DataStore(tmp_path)
    private = "Secret Zotero Paper About Attention Kernels"
    library = ZoteroResult(
        papers=[Paper(id="zotero:ABCD2345", title=private, abstract="Attention, attention.")],
        stats=ZoteroStats(used=1, total=1342, source="api"),
    )

    result, _, _ = run(config, store, papers, zotero=FakeZotero(library), publisher=FakePublisher())

    by_id = {p.id: p for p in [*result.digest.papers, *result.digest.others]}
    attention = by_id["2610.00003"]
    assert attention.zotero_score > by_id["2610.00002"].zotero_score
    assert result.digest.stats.zotero == ZoteroStats(used=1, total=1342, source="api")
    # 文库内容不写进 gleaner-data：只有数量
    for path in tmp_path.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert private not in text and "ABCD2345" not in text, path


def test_zotero_failure_is_a_warning(config: Config, papers: list[Paper], tmp_path: Path) -> None:
    library = ZoteroResult(
        stats=ZoteroStats(source="none", error="请求 Zotero 失败（ConnectError）"),
        warnings=[
            "读取 Zotero 失败，又没有缓存，这次不使用 Zotero：请求 Zotero 失败（ConnectError）"
        ],
    )

    result, _, _ = run(config, DataStore(tmp_path), papers, zotero=FakeZotero(library))

    assert result.status == "preview"
    assert result.warnings == library.warnings
    assert all(p.zotero_score is None for p in result.digest.papers)
    assert result.digest.stats.zotero.source == "none"


def test_without_zotero_stats_are_empty(
    config: Config, papers: list[Paper], tmp_path: Path
) -> None:
    result, _, _ = run(config, DataStore(tmp_path), papers)
    assert result.digest.stats.zotero is None
    assert all(p.zotero_score is None for p in result.digest.papers)
