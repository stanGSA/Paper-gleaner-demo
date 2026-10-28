"""MCP Server：用真实的 git 裸仓库作为远端（参考 tests/test_store.py）。"""

import asyncio
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx

from gleaner import mcp_server
from gleaner.agent.tools import ToolError
from gleaner.config import Config
from gleaner.mcp_server import GleanerMCP, build_server
from gleaner.models import Paper
from gleaner.profile import command_feedback, record_feedback
from gleaner.store import DataBranch, DataStore
from tests.conftest import (
    CACHED_REPORT,
    COMPARE_BODY,
    DEEP_REPORT,
    git,
    interact_services,
    seed_agent_data,
)

NOW = datetime(2026, 10, 26, 2, 0, tzinfo=UTC)
HTML = "https://arxiv.org/html/{}"
TOOL_NAMES = {
    "get_digest",
    "search_history",
    "deep_read",
    "compare",
    "give_feedback",
    "get_profile",
}


@pytest.fixture
def config(tmp_path: Path) -> Config:
    interests = tmp_path / "interests.md"
    interests.write_text("我研究长上下文推理。", encoding="utf-8")
    return Config.model_validate({"profile": {"interests_file": str(interests)}})


@pytest.fixture
def seeded(clone: Callable[[str], Path]) -> None:
    """远端的 gleaner-data 上已有日报、反馈和报告缓存。"""
    seed_repo = clone("data-seed")
    seed = DataBranch(seed_repo, seed_repo / ".gleaner-data", backoff_seconds=0)
    seed_agent_data(seed.checkout())
    seed.commit_and_push("data: seed")


def make_gleaner(config: Config, repo: Path, branch: DataBranch | None = None):
    services, fake = interact_services()
    branch = branch or DataBranch(repo, repo / ".gleaner-data", backoff_seconds=0)
    return GleanerMCP(config, branch, services, now=lambda: NOW), fake


def remote_feedback(remote: Path) -> list[dict]:
    text = git("show", "gleaner-data:feedback/2026-10.jsonl", cwd=remote)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --- 只读工具 -------------------------------------------------------------------


def test_read_tools(config: Config, clone, seeded) -> None:
    gleaner, fake = make_gleaner(config, clone("work"))

    digest = gleaner.get_digest()
    assert digest.startswith("2026-10-25 的日报：推荐 2 篇")
    assert "[1] 2610.00001 · 9/10 · LongKV" in digest
    assert "2026-10-20 的日报" in gleaner.get_digest("2026-10-20")
    history = gleaner.search_history("domain generalization")
    assert "2610.00100 · Domain Generalization for Object Detectors" in history
    assert "屏蔽的主题：医学影像" in gleaner.get_profile()
    assert fake.requests == []

    with pytest.raises(ToolError, match="日期必须写成 YYYY-MM-DD"):
        gleaner.get_digest("10/25")


def test_deep_read_prefers_cache(config: Config, clone, seeded) -> None:
    gleaner, fake = make_gleaner(config, clone("work"))

    text = gleaner.deep_read("2610.00002")

    assert CACHED_REPORT.split("\n")[1] in text
    assert "来自缓存" in text
    assert fake.requests == []


@respx.mock
def test_deep_read_generates_without_writing(
    config: Config, clone, seeded, git_remote: Path, html_text: str
) -> None:
    respx.get(HTML.format("2610.00001")).mock(return_value=httpx.Response(200, text=html_text))
    repo = clone("work")
    gleaner, fake = make_gleaner(config, repo)
    before = git("rev-parse", "gleaner-data", cwd=git_remote)

    text = gleaner.deep_read("1")  # 也可以写最近一份日报中的序号

    assert DEEP_REPORT.split("\n")[1] in text
    assert len(fake.requests) == 1
    assert not (repo / ".gleaner-data" / "reports" / "2610.00001.md").exists()  # 只读
    assert git("rev-parse", "gleaner-data", cwd=git_remote) == before


@respx.mock
def test_compare(config: Config, clone, seeded, html_text: str) -> None:
    for paper_id in ("2610.00001", "2610.00002"):
        respx.get(HTML.format(paper_id)).mock(return_value=httpx.Response(200, text=html_text))
    gleaner, _ = make_gleaner(config, clone("work"))

    text = gleaner.compare(["2610.00001", "2"])

    assert text.startswith("## ⚖️ 对比 · 2 篇论文")
    assert COMPARE_BODY.split("\n")[1] in text
    with pytest.raises(ToolError, match="至少写一篇"):
        gleaner.compare([])
    with pytest.raises(ToolError, match="最多对比 5 篇"):
        gleaner.compare([str(i) for i in range(1, 7)])


def test_falls_back_to_local_data_when_sync_fails(config: Config, clone, seeded) -> None:
    repo = clone("work")
    gleaner, _ = make_gleaner(config, repo)
    gleaner.get_digest()  # 先同步一次，本地有了数据
    git("remote", "set-url", "origin", str(repo.parent / "gone.git"), cwd=repo)

    text = gleaner.get_digest()

    assert text.startswith("（同步 gleaner-data 失败，以下使用本地已有的数据：")
    assert "2026-10-25 的日报" in text
    with pytest.raises(ToolError, match="同步 gleaner-data 失败，这条反馈没有保存"):
        gleaner.give_feedback("2610.00001", "like")


def test_no_local_data_and_no_remote_access(config: Config, clone) -> None:
    repo = clone("work")
    git("remote", "set-url", "origin", str(repo.parent / "gone.git"), cwd=repo)
    gleaner, _ = make_gleaner(config, repo)

    with pytest.raises(ToolError, match="读取 gleaner-data 失败"):
        gleaner.get_digest()


# --- give_feedback --------------------------------------------------------------


def test_give_feedback_commits_and_pushes(config: Config, clone, seeded, git_remote: Path) -> None:
    gleaner, _ = make_gleaner(config, clone("work"))

    text = gleaner.give_feedback("2610.00100v2", "like", "  检测器可以直接用 #12 @someone ")

    assert text.startswith(
        "✓ 已记下：喜欢 2610.00100《Domain Generalization for Object Detectors》"
    )
    assert "已提交并推送到 gleaner-data" in text
    records = [r for r in remote_feedback(git_remote) if r["created_at"].startswith("2026-10-26")]
    [record] = records
    assert (record["paper_id"], record["signal"], record["weight"]) == ("2610.00100", "like", 2.0)
    assert record["source"] == "command"  # 与 /like 相同，聚合规则不变
    assert record["note"] == "检测器可以直接用 #12 @someone"
    assert record["abstract"].startswith("Single-source domain generalization")
    message = git("log", "-1", "--format=%B", "gleaner-data", cwd=git_remote)
    assert message.strip() == "data: /like via MCP"  # 提交说明里不写原因


@respx.mock
def test_give_feedback_for_unknown_paper_uses_arxiv_api(
    config: Config, clone, seeded, git_remote: Path
) -> None:
    feed = (
        '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>'
        "<id>http://arxiv.org/abs/2401.00001v1</id><title>Outside</title>"
        "<summary>Outside abstract.</summary></entry></feed>"
    )
    respx.get("https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(200, text=feed)
    )
    gleaner, _ = make_gleaner(config, clone("work"))

    text = gleaner.give_feedback("2401.00001", "dislike")

    assert text.startswith("✓ 已记下：不喜欢 2401.00001《Outside》。")
    assert remote_feedback(git_remote)[-1]["abstract"] == "Outside abstract."


class RacingBranch(DataBranch):
    """在提交之前，另一个 clone 先往同一个 jsonl 推送了一条反馈。"""

    def __init__(self, *args, rival: Callable[[], None], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rival = rival

    def commit_and_push(self, message: str) -> bool:
        self.rival()
        return super().commit_and_push(message)


def test_give_feedback_rebases_when_push_is_rejected(
    config: Config, clone, seeded, git_remote: Path
) -> None:
    rival_repo = clone("rival")

    def rival_push() -> None:
        rival = DataBranch(rival_repo, rival_repo / ".gleaner-data", backoff_seconds=0)
        store = rival.checkout()
        paper = Paper(id="2610.00002", title="RAG for Science", abstract="A RAG benchmark.")
        record_feedback(store, [command_feedback(paper, "dislike", now=NOW)], timezone="UTC")
        rival.commit_and_push("data: rival feedback")

    repo = clone("work")
    branch = RacingBranch(repo, repo / ".gleaner-data", backoff_seconds=0, rival=rival_push)
    gleaner, _ = make_gleaner(config, repo, branch)

    text = gleaner.give_feedback("2610.00001", "like")

    assert text.startswith("✓ 已记下")
    signals = [(r["paper_id"], r["signal"]) for r in remote_feedback(git_remote)]
    # jsonl 用 union 合并：两边的记录都在
    assert ("2610.00002", "dislike") in signals and ("2610.00001", "like") in signals


def test_give_feedback_reports_push_failure(
    config: Config, clone, seeded, git_remote: Path
) -> None:
    repo = clone("work")
    gleaner, _ = make_gleaner(config, repo)
    gleaner.get_digest()
    before = git("rev-parse", "gleaner-data", cwd=git_remote)
    hook = git_remote / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\necho 'permission denied' >&2\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)

    with pytest.raises(ToolError, match="这条反馈没有保存：推送 gleaner-data 失败") as info:
        gleaner.give_feedback("2610.00001", "like", "原因不会出现在提交里")

    assert "git push" in str(info.value)
    assert git("rev-parse", "gleaner-data", cwd=git_remote) == before
    local = DataStore(repo / ".gleaner-data").read_jsonl("feedback/2026-10.jsonl")
    assert not [r for r in local if r["paper_id"] == "2610.00001"]  # 本地那次提交已丢掉


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("2610.00001", "up"), "signal 只能是 like 或 dislike"),
        (("not-an-id", "like"), "无法识别"),
        (("3", "like"), "这里要写 arXiv ID"),
    ],
)
def test_give_feedback_validates_arguments(
    config: Config, clone, arguments: tuple, message: str
) -> None:
    gleaner, _ = make_gleaner(config, clone("work"))
    with pytest.raises(ToolError, match=message):
        gleaner.give_feedback(*arguments)


# --- MCP 协议层 -----------------------------------------------------------------


def test_server_registers_tools(config: Config, clone, seeded) -> None:
    from mcp.server.mcpserver.exceptions import ToolError as MCPToolError

    gleaner, _ = make_gleaner(config, clone("work"))
    server = build_server(gleaner)

    async def scenario():
        tools = {tool.name: tool for tool in await server.list_tools()}
        digest = await server.call_tool("get_digest", {})
        with pytest.raises(MCPToolError, match="signal 只能是 like 或 dislike|like"):
            await server.call_tool("give_feedback", {"arxiv_id": "2610.00001", "signal": "meh"})
        return tools, digest

    tools, digest = asyncio.run(scenario())

    assert set(tools) == TOOL_NAMES
    assert tools["get_digest"].annotations.read_only_hint is True
    assert tools["give_feedback"].annotations.read_only_hint is False
    assert tools["give_feedback"].input_schema["required"] == ["arxiv_id", "signal"]
    assert digest.is_error is False
    assert digest.content[0].text.startswith("2026-10-25 的日报")


def test_main_without_mcp_extra(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", None)  # import 时抛出 ImportError

    with pytest.raises(SystemExit) as info:
        mcp_server.main([])

    assert info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # stdout 是 MCP 协议的通道，不打印任何东西
    assert "uv run --extra mcp gleaner-mcp" in captured.err
