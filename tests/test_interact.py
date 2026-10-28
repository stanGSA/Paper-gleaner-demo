import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from gleaner.config import Config
from gleaner.notify import GitHubError
from gleaner.pipelines.daily import digest_path
from gleaner.pipelines.interact import (
    CommentEvent,
    EventError,
    handle_comment,
    ignore_reason,
    issue_day,
    load_event,
    report_failure,
)
from gleaner.reader import report_path
from gleaner.store import DataBranchError, DataStore
from tests.conftest import (
    DEEP_REPORT,
    DIGEST_DAY,
    FIXTURES,
    interact_digest,
    interact_services,
)

NOW = datetime(2026, 10, 26, 2, 0, tzinfo=UTC)
HTML = "https://arxiv.org/html/2610.00001"


class FakeReplier:
    def __init__(self, *, fail_comments: bool = False, fail_reactions: bool = False) -> None:
        self.comments: list[tuple[int, str]] = []
        self.reactions: list[tuple[int, str]] = []
        self.fail_comments = fail_comments
        self.fail_reactions = fail_reactions

    def create_comment(self, issue_number: int, body: str) -> int:
        if self.fail_comments:
            raise GitHubError("发表评论失败（HTTP 502）")
        self.comments.append((issue_number, body))
        return 100 + len(self.comments)

    def add_reaction(self, comment_id: int, content: str) -> None:
        if self.fail_reactions:
            raise GitHubError("添加表情失败（HTTP 403）")
        self.reactions.append((comment_id, content))

    @property
    def bodies(self) -> list[str]:
        return [body for _, body in self.comments]


class FakeBranch:
    def __init__(self, root: Path, *, fail_push: bool = False) -> None:
        self.store = DataStore(root)
        self.store.write_model(digest_path(DIGEST_DAY), interact_digest())
        self.checkouts = 0
        self.commits: list[str] = []
        self.fail_push = fail_push

    def checkout(self) -> DataStore:
        self.checkouts += 1
        return self.store

    def commit_and_push(self, message: str) -> bool:
        if self.fail_push:
            raise DataBranchError("推送 gleaner-data 失败：rejected")
        self.commits.append(message)
        return True


def payload(body: str = "/deep 1", **changes: Any) -> dict[str, Any]:
    data = json.loads((FIXTURES / "issue_comment_event.json").read_text(encoding="utf-8"))
    data["comment"]["body"] = body
    for key, value in changes.items():  # issue__title=... 改 data["issue"]["title"]
        *sections, field = key.split("__")
        target = data
        for section in sections:
            target = target[section]
        target[field] = value
    return data


def event(body: str = "/deep 1", **changes: Any) -> CommentEvent:
    return CommentEvent.from_payload(payload(body, **changes))


@pytest.fixture
def config(tmp_path: Path) -> Config:
    interests = tmp_path / "interests.md"
    interests.write_text("我研究长上下文推理。", encoding="utf-8")
    return Config.model_validate({"profile": {"interests_file": str(interests)}})


def handle(config: Config, branch: FakeBranch, comment: CommentEvent, **kwargs):
    replier = kwargs.pop("replier", FakeReplier())
    services, fake = interact_services(kwargs.pop("handler", None) or _default_handler)
    result = handle_comment(
        comment,
        config=config,
        replier=replier,
        branch=branch,
        services=kwargs.pop("services", services),
        now=NOW,
        pause_seconds=0,
        **kwargs,
    )
    return result, replier, fake


def _default_handler(request: dict[str, Any]) -> str:
    return DEEP_REPORT


# --- 事件 -----------------------------------------------------------------------


def test_load_event(tmp_path: Path) -> None:
    comment = load_event(FIXTURES / "issue_comment_event.json")

    assert comment.repository == "o/r"
    assert comment.issue_number == 1
    assert comment.labels == ["gleaner", "daily"]
    assert comment.is_pull_request is False
    assert comment.comment_id == 9001
    assert comment.comment_body == "/deep 1"
    assert comment.author == "owner"
    assert comment.author_association == "OWNER"


def test_load_event_errors(tmp_path: Path) -> None:
    with pytest.raises(EventError, match="找不到事件文件"):
        load_event(tmp_path / "missing.json")

    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(EventError, match="不是合法的 JSON"):
        load_event(broken)

    push = tmp_path / "push.json"
    push.write_text(json.dumps({"ref": "refs/heads/main"}), encoding="utf-8")
    with pytest.raises(EventError, match="不是 issue_comment 事件"):
        load_event(push)


@pytest.mark.parametrize(
    ("comment", "reason"),
    [
        (event(action="edited"), "只处理新评论"),
        (event(issue__pull_request={"url": "https://x"}), "Pull Request"),
        (event("请帮我 /deep 1"), "不是以 / 开头"),
        (event(comment__author_association="NONE"), "不是仓库所有者或协作者"),
        (event(comment__author_association="CONTRIBUTOR"), "不是仓库所有者或协作者"),
        (event(issue__labels=[{"name": "bug"}]), "没有 gleaner 标签"),
    ],
)
def test_ignored_events(config: Config, tmp_path: Path, comment: CommentEvent, reason: str) -> None:
    assert reason in ignore_reason(comment)
    branch = FakeBranch(tmp_path)

    result, replier, _ = handle(config, branch, comment)

    assert result.status == "ignored"
    assert replier.comments == replier.reactions == []
    assert branch.checkouts == 0


@pytest.mark.parametrize(
    ("body", "title", "day"),
    [
        ("<!-- gleaner:digest day=2026-10-25 -->", "随便", date(2026, 10, 25)),
        ("正文被人改过了", "📚 Paper Gleaner · 2026-10-24 · 6 篇", date(2026, 10, 24)),
        ("<!-- gleaner:digest day=2026-13-45 -->", "欢迎使用", None),
        ("", "👋 欢迎使用 Paper Gleaner", None),
    ],
)
def test_issue_day(body: str, title: str, day: date | None) -> None:
    assert issue_day(event(issue__body=body, issue__title=title)) == day


# --- 处理流程 -------------------------------------------------------------------


@respx.mock
def test_deep_flow(config: Config, tmp_path: Path, html_text: str) -> None:
    respx.get(HTML).mock(return_value=httpx.Response(200, text=html_text))
    branch = FakeBranch(tmp_path)

    result, replier, fake = handle(config, branch, event("/deep 1"))

    assert result.status == "done"
    assert result.reply_ids == [101]
    assert replier.reactions == [(9001, "eyes"), (9001, "rocket")]
    [(issue, body)] = replier.comments
    assert issue == 1
    assert DEEP_REPORT in body
    assert len(fake.requests) == 1
    assert branch.commits == ["data: /deep from issue 1"]
    assert branch.store.read_text(report_path("2610.00001")) is not None
    assert "数据已提交到 gleaner-data" in result.message


def test_like_flow(config: Config, tmp_path: Path) -> None:
    branch = FakeBranch(tmp_path)

    result, replier, fake = handle(config, branch, event("/like 1 有开源代码"))

    assert result.status == "done"
    assert replier.bodies[0].startswith("👍 已记下：喜欢 [LongKV")
    assert replier.reactions == [(9001, "eyes"), (9001, "rocket")]
    assert fake.requests == []
    assert branch.commits == ["data: /like from issue 1"]  # 提交说明里不写用户写的原因
    [record] = branch.store.read_jsonl("feedback/2026-10.jsonl")
    assert (record["signal"], record["weight"], record["note"]) == ("like", 2.0, "有开源代码")


def test_mute_list_commits_nothing(config: Config, tmp_path: Path) -> None:
    class NoChanges(FakeBranch):
        def commit_and_push(self, message: str) -> bool:
            return False

    branch = NoChanges(tmp_path)
    result, replier, _ = handle(config, branch, event("/mute"))

    assert result.status == "done"
    assert replier.bodies[0].startswith("🔇 当前没有屏蔽任何主题")
    assert "数据已提交" not in result.message


def test_unmute_unknown_topic_is_rejected(config: Config, tmp_path: Path) -> None:
    result, replier, _ = handle(config, FakeBranch(tmp_path), event("/unmute 医学影像"))

    assert result.status == "rejected"
    assert replier.bodies[0].startswith("⚠️ `医学影像` 不在屏蔽列表中。当前没有屏蔽任何主题")
    assert replier.reactions[-1] == (9001, "confused")


def test_help_does_not_touch_data(config: Config, tmp_path: Path) -> None:
    branch = FakeBranch(tmp_path)

    result, replier, _ = handle(config, branch, event("/help"))

    assert result.status == "done"
    assert "`/deep <ref>`" in replier.bodies[0]
    assert branch.checkouts == 0
    assert branch.commits == []
    assert replier.reactions == [(9001, "eyes"), (9001, "rocket")]


def test_unknown_command_replies_with_help(config: Config, tmp_path: Path) -> None:
    result, replier, _ = handle(config, FakeBranch(tmp_path), event("/summarize 哪篇最好？"))

    assert result.status == "rejected"
    assert replier.bodies[0].startswith("⚠️ 不认识的指令 `/summarize`。")
    assert "`/compare <ref> <ref> ...`" in replier.bodies[0]
    assert replier.reactions == [(9001, "eyes"), (9001, "confused")]


def test_command_error_replies_with_reason(config: Config, tmp_path: Path) -> None:
    branch = FakeBranch(tmp_path)

    result, replier, _ = handle(config, branch, event("/deep 9"))

    assert result.status == "rejected"
    assert replier.bodies == [
        "⚠️ 序号 9 超出范围：2026-10-25 的日报推荐了 2 篇，另有 3 篇候选（序号 3–5，"
        "评论 `/more` 查看）\n\n评论 `/help` 查看可用指令。"
    ]
    assert replier.reactions[-1] == (9001, "confused")
    assert branch.commits == []


def test_unexpected_error_is_reported(config: Config, tmp_path: Path) -> None:
    services, _ = interact_services()

    class BrokenFetcher:
        def fetch(self, paper):
            raise RuntimeError("bug <script>")

    services.fetcher = BrokenFetcher()  # type: ignore[assignment]

    result, replier, _ = handle(config, FakeBranch(tmp_path), event("/deep 1"), services=services)

    assert result.status == "failed"
    assert replier.bodies[0].startswith("⚠️ 处理指令时出错（RuntimeError）：bug &lt;script&gt;")
    assert replier.reactions[-1] == (9001, "confused")


def test_checkout_failure_is_reported(config: Config, tmp_path: Path) -> None:
    class BrokenBranch(FakeBranch):
        def checkout(self) -> DataStore:
            raise DataBranchError("无法访问远端 origin")

    result, replier, _ = handle(config, BrokenBranch(tmp_path), event("/more"))

    assert result.status == "failed"
    assert "DataBranchError" in replier.bodies[0]
    assert "无法访问远端 origin" in replier.bodies[0]


def test_long_reply_is_split(config: Config, tmp_path: Path, html_text: str) -> None:
    long_report = "\n\n".join(f"### 第 {i} 部分\n" + "很长的内容。" * 4000 for i in range(5))

    with respx.mock:
        respx.get(HTML).mock(return_value=httpx.Response(200, text=html_text))
        result, replier, _ = handle(
            config, FakeBranch(tmp_path), event("/deep 1"), handler=lambda request: long_report
        )

    assert result.status == "done"
    assert len(replier.comments) == len(result.reply_ids) >= 2
    assert all(len(body) <= 65_536 for body in replier.bodies)
    assert replier.bodies[0].endswith("<sub>（未完，见下一条评论）</sub>")
    assert replier.bodies[1].startswith(f"<sub>（续 2/{len(replier.bodies)}）</sub>")


def test_reply_failure(config: Config, tmp_path: Path) -> None:
    replier = FakeReplier(fail_comments=True)
    result, _, _ = handle(config, FakeBranch(tmp_path), event("/more"), replier=replier)
    assert result.status == "failed"
    assert "回复发表失败" in result.message
    assert replier.reactions[-1] == (9001, "confused")


def test_reaction_failure_is_not_fatal(config: Config, tmp_path: Path) -> None:
    replier = FakeReplier(fail_reactions=True)
    result, _, _ = handle(config, FakeBranch(tmp_path), event("/more"), replier=replier)
    assert result.status == "done"
    assert len(replier.comments) == 1


@respx.mock
def test_push_failure_keeps_the_reply(config: Config, tmp_path: Path, html_text: str) -> None:
    respx.get(HTML).mock(return_value=httpx.Response(200, text=html_text))
    branch = FakeBranch(tmp_path, fail_push=True)

    result, replier, _ = handle(config, branch, event("/deep 1"))

    assert result.status == "done"
    assert len(replier.comments) == 1
    assert "提交 gleaner-data 失败" in result.message
    assert replier.reactions[-1] == (9001, "rocket")


@respx.mock
def test_dry_run_does_not_commit(config: Config, tmp_path: Path, html_text: str) -> None:
    respx.get(HTML).mock(return_value=httpx.Response(200, text=html_text))
    branch = FakeBranch(tmp_path)

    result, _, _ = handle(config, branch, event("/deep 1"), persist=False)

    assert result.status == "done"
    assert branch.commits == []
    assert branch.store.read_text(report_path("2610.00001")) is None


def test_report_failure(config: Config) -> None:
    replier = FakeReplier()

    result = report_failure(event("/deep 1"), replier, "配置有误", pause_seconds=0)

    assert result.status == "failed"
    assert replier.bodies == ["⚠️ 配置有误\n\n评论 `/help` 查看可用指令。"]
    assert replier.reactions == [(9001, "confused")]
    ignored = report_failure(event("hello"), replier, "配置有误", pause_seconds=0)
    assert ignored.status == "ignored"
    assert len(replier.comments) == 1


# --- /ask -----------------------------------------------------------------------


class CommentingReplier(FakeReplier):
    """还能读取 Issue 的评论（和 GitHubClient.list_comments 返回的格式相同）。"""

    def __init__(self, comments: list[dict[str, Any]]) -> None:
        super().__init__()
        self.existing = comments
        self.listed: list[int] = []

    def list_comments(self, issue_number: int) -> list[dict[str, Any]]:
        self.listed.append(issue_number)
        return self.existing


def test_ask_reads_issue_comments_as_context(config: Config, tmp_path: Path) -> None:
    replier = CommentingReplier(
        [
            {
                "id": 1,
                "user": {"login": "owner"},
                "author_association": "OWNER",
                "body": "关注检测",
            },
            {"id": 2, "user": {"login": "x"}, "author_association": "NONE", "body": "陌生人的话"},
            {"id": 9001, "user": {"login": "owner"}, "author_association": "OWNER", "body": "/ask"},
        ]
    )
    prompts: list[str] = []

    def handler(request: dict[str, Any]) -> str:
        prompts.append(request["messages"][1]["content"])
        if "每一步只做一件事" in request["messages"][0]["content"]:
            return json.dumps({"action": "answer"})
        return "第 1 篇最合适（2610.00001）。"

    result, replier, _ = handle(
        config,
        FakeBranch(tmp_path),
        event("/ask 哪篇最适合检测？\n顺便看看第 2 篇"),
        replier=replier,
        handler=handler,
    )

    assert result.status == "done"
    assert replier.listed == [1]
    assert "<question>\n哪篇最适合检测？\n顺便看看第 2 篇\n</question>" in prompts[0]
    assert "owner（OWNER）：关注检测" in prompts[0] and "陌生人的话" not in prompts[0]
    [body] = replier.bodies
    assert body.startswith("## 💬 /ask · `哪篇最适合检测？ 顺便看看第 2 篇`")
    assert "- [LongKV: Hierarchical KV Cache Compression](https://arxiv.org/abs/2610.00001)" in body
    assert replier.reactions == [(9001, "eyes"), (9001, "rocket")]
