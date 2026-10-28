import json
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest
import respx
from typer.testing import CliRunner

from gleaner import __version__, cli
from gleaner.cli import _report, app
from gleaner.llm import LLMClient, LLMError
from gleaner.models import Digest, Paper, ScoredPaper
from gleaner.notify import IssueRef
from gleaner.pipelines.daily import DailyResult, digest_path
from gleaner.profile import command_feedback, record_feedback
from gleaner.store import DataBranch
from tests.conftest import (
    DEEP_REPORT,
    DIGEST_DAY,
    FIXTURES,
    PIPELINE_VOCABULARY,
    FakeOpenAI,
    KeywordEmbedder,
    git,
    interact_digest,
    interact_llm,
    pipeline_llm,
)

runner = CliRunner()
RSS_URL = "https://rss.arxiv.org/rss/cs.CL+cs.LG"
API = "https://api.github.com/repos/o/r"
SCORES = {"2610.00001": 9, "2610.00002": 7, "2610.00003": 4}
# 自动深读会请求 arXiv 全文，CLI 测试中关掉（另有专门的测试）
REPO_CONFIG = "sources:\n  arxiv:\n    categories: [cs.CL, cs.LG]\ndeep_read:\n  auto_top_m: 0\n"
ENV_VARS = [
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_PROVIDER",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "GITHUB_TOKEN",
    "GITHUB_REPOSITORY",
    "GITHUB_REPOSITORY_OWNER",
    "GITHUB_API_URL",
    "GITHUB_SERVER_URL",
    "GITHUB_ACTIONS",
    "GITHUB_STEP_SUMMARY",
    "GITHUB_EVENT_PATH",
    "S2_API_KEY",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "SMTP_HOST",
    "SMTP_PORT",
    "SMTP_USER",
    "SMTP_PASSWORD",
    "EMAIL_TO",
    "FEISHU_WEBHOOK",
    "FEISHU_SECRET",
    "WECOM_WEBHOOK",
    "ZOTERO_ID",
    "ZOTERO_KEY",
    "GLEANER_ZOTERO_CACHE",
]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None) -> Path:
    """一个干净的工作目录，只含指定分类的 config.yaml。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("sources:\n  arxiv:\n    categories: [cs.CL, cs.LG]\n")
    return tmp_path


@pytest.fixture
def repo(clone: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch, clean_env: None) -> Path:
    """以一个 git 仓库为工作目录，配置齐全，LLM 和嵌入模型换成假的。"""
    path = clone("repo")
    monkeypatch.chdir(path)
    (path / "config.yaml").write_text(REPO_CONFIG)
    (path / "profile").mkdir()
    (path / "profile" / "interests.md").write_text("我研究长上下文推理。", encoding="utf-8")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setattr(
        cli, "LLMClient", lambda config: LLMClient(config, client=FakeOpenAI(pipeline_llm(SCORES)))
    )
    monkeypatch.setattr(cli, "build_embedder", lambda config: KeywordEmbedder(PIPELINE_VOCABULARY))
    monkeypatch.setattr("gleaner.notify.github_issue.time.sleep", lambda seconds: None)
    return path


def github_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_test")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")


def data_branch_exists(remote: Path) -> bool:
    return bool(git("ls-remote", "--heads", str(remote), "gleaner-data", cwd=remote))


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


@respx.mock
def test_daily_dry_run(repo: Path, rss_bytes: bytes, git_remote: Path) -> None:
    respx.get(RSS_URL).mock(return_value=httpx.Response(200, content=rss_bytes))

    result = runner.invoke(app, ["daily", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "今天以长上下文推理为主" in result.output
    assert "[1] 9/10  2610.00001  LongKV" in result.output
    assert "🎯 2610.00001 与长上下文推理相关" in result.output
    assert "[2] 7/10  2610.00002" in result.output
    assert "2610.00003" not in result.output  # 低于 min_score
    assert "抓取 3 篇" in result.output
    assert not data_branch_exists(git_remote)  # dry-run 不提交数据分支


@respx.mock
def test_daily_quickstart_skips_optional_services(
    repo: Path, rss_bytes: bytes, git_remote: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = REPO_CONFIG.replace("auto_top_m: 0", "auto_top_m: 1") + (
        "profile:\n  zotero: {enabled: true}\nnotify:\n  email: {enabled: true}\n"
    )
    (repo / "config.yaml").write_text(original, encoding="utf-8")
    respx.get(RSS_URL).mock(return_value=httpx.Response(200, content=rss_bytes))

    def forbidden(*args, **kwargs):
        pytest.fail("quickstart must skip optional services")

    monkeypatch.setattr(cli, "ZoteroLibrary", forbidden)
    monkeypatch.setattr(cli, "build_notifiers", forbidden)
    monkeypatch.setattr("gleaner.pipelines.daily.FulltextFetcher", forbidden)
    result = runner.invoke(app, ["daily", "--dry-run", "--quickstart"])

    assert result.exit_code == 0, result.output
    assert "快速开始" in result.output
    assert "[1] 9/10" in result.output
    assert (repo / "config.yaml").read_text(encoding="utf-8") == original
    assert not data_branch_exists(git_remote)


@respx.mock
def test_daily_publishes_issue_and_data(
    repo: Path,
    rss_bytes: bytes,
    git_remote: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    github_env(monkeypatch)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    respx.get(RSS_URL).mock(return_value=httpx.Response(200, content=rss_bytes))
    respx.get(f"{API}/issues").mock(return_value=httpx.Response(200, json=[]))
    respx.post(f"{API}/labels").mock(return_value=httpx.Response(201, json={}))
    respx.post(f"{API}/issues").mock(
        return_value=httpx.Response(
            201, json={"number": 5, "html_url": "https://github.com/o/r/issues/5"}
        )
    )
    comments = respx.post(f"{API}/issues/5/comments").mock(
        return_value=httpx.Response(201, json={"id": 100})
    )

    result = runner.invoke(app, ["daily"])

    assert result.exit_code == 0, result.output
    assert "✓ 已发布：https://github.com/o/r/issues/5" in result.output
    assert comments.call_count == 2
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    saved = git("show", f"gleaner-data:history/digests/{today}.json", cwd=git_remote)
    assert '"issue_number": 5' in saved
    assert f"## Paper Gleaner · {today}" in summary.read_text(encoding="utf-8")


@respx.mock
def test_daily_skips_when_already_published(
    repo: Path, git_remote: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    github_env(monkeypatch)
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    existing = {
        "number": 4,
        "title": f"📚 Paper Gleaner · {today} · 10 篇",
        "html_url": "https://github.com/o/r/issues/4",
        "created_at": "2026-10-26T01:30:00Z",
    }
    respx.get(f"{API}/issues").mock(return_value=httpx.Response(200, json=[existing]))
    rss = respx.get(RSS_URL)

    result = runner.invoke(app, ["daily"])

    assert result.exit_code == 0, result.output
    assert "今天的日报已经发布：https://github.com/o/r/issues/4" in result.output
    assert rss.call_count == 0
    assert not data_branch_exists(git_remote)


def seed_data(clone: Callable[[str], Path], *digests: Digest) -> None:
    """在远端的 gleaner-data 分支上放几份日报。"""
    seed_repo = clone("data-seed")
    seed = DataBranch(seed_repo, seed_repo / ".gleaner-data", backoff_seconds=0)
    store = seed.checkout()
    for digest in digests:
        store.write_model(digest_path(digest.day), digest)
    seed.commit_and_push("data: seed digests")


def todays_digest(**fields) -> Digest:
    """今天（按配置的时区）已发布的日报，评论都在。"""
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    comment_ids = {"2610.00001": 801, "2610.00002": 802}
    return interact_digest(**({"day": today, "comment_ids": comment_ids} | fields))


def reactions_json(*items: tuple[str, str]) -> list[dict]:
    return [
        {"id": i, "user": {"login": login, "type": "User"}, "content": content}
        for i, (login, content) in enumerate(items, start=1)
    ]


@respx.mock
def test_daily_dry_run_syncs_reactions_without_committing(
    repo: Path,
    clone: Callable[[str], Path],
    rss_bytes: bytes,
    git_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_data(clone, todays_digest())
    github_env(monkeypatch)
    monkeypatch.setenv("GITHUB_REPOSITORY_OWNER", "Owner")
    before = git("rev-parse", "gleaner-data", cwd=git_remote)
    respx.get(RSS_URL).mock(return_value=httpx.Response(200, content=rss_bytes))
    first = respx.get(f"{API}/issues/comments/801/reactions").mock(
        return_value=httpx.Response(200, json=reactions_json(("owner", "+1"), ("bob", "-1")))
    )
    respx.get(f"{API}/issues/comments/802/reactions").mock(
        return_value=httpx.Response(200, json=reactions_json(("bob", "+1")))
    )

    result = runner.invoke(app, ["daily", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert first.called
    assert "· 同步了 1 条表情反馈" in result.output
    assert "[1] 9/10  2610.00001" in result.output
    # 终端输出里有粗筛分数的分项和反馈统计，用来观察反馈是否生效
    assert "    📊 粗筛 " in result.output
    assert "+ 正反馈 " in result.output
    assert "反馈：👍 1 · 👎 0" in result.output
    assert "本次同步表情 1 条" in result.output
    assert git("rev-parse", "gleaner-data", cwd=git_remote) == before  # dry-run 不写数据


@respx.mock
def test_daily_dry_run_warns_without_github_env(repo: Path, rss_bytes: bytes) -> None:
    respx.get(RSS_URL).mock(return_value=httpx.Response(200, content=rss_bytes))

    result = runner.invoke(app, ["daily", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert result.output.startswith("! 没有 GITHUB_TOKEN 和 GITHUB_REPOSITORY，跳过表情同步")


@respx.mock
def test_daily_commits_synced_reactions_when_already_published(
    repo: Path,
    clone: Callable[[str], Path],
    git_remote: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = todays_digest(issue_number=4)
    seed_data(clone, digest)
    github_env(monkeypatch)  # 没有 GITHUB_REPOSITORY_OWNER 时从 GITHUB_REPOSITORY 取出 owner
    existing = {
        "number": 4,
        "title": f"📚 Paper Gleaner · {digest.day.isoformat()} · 2 篇",
        "html_url": "https://github.com/o/r/issues/4",
        "created_at": f"{digest.day.isoformat()}T01:30:00Z",
    }
    respx.get(f"{API}/issues").mock(return_value=httpx.Response(200, json=[existing]))
    respx.get(f"{API}/issues/comments/801/reactions").mock(
        return_value=httpx.Response(200, json=reactions_json(("O", "-1")))
    )
    respx.get(f"{API}/issues/comments/802/reactions").mock(
        return_value=httpx.Response(200, json=[])
    )

    result = runner.invoke(app, ["daily"])

    assert result.exit_code == 0, result.output
    assert "今天的日报已经发布" in result.output
    message = git("log", "-1", "--format=%s", "gleaner-data", cwd=git_remote)
    assert message == f"data: sync reactions {digest.day.isoformat()}"
    month = digest.day.strftime("%Y-%m")
    saved = git("show", f"gleaner-data:feedback/{month}.jsonl", cwd=git_remote)
    assert '"signal": "down"' in saved


# --- 推送渠道 -------------------------------------------------------------------

FEISHU = "https://open.feishu.cn/open-apis/bot/v2/hook/feishu-token-123"
WECOM = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=wecom-key-456"
CHANNELS_CONFIG = REPO_CONFIG + "notify:\n  feishu: {enabled: true}\n  wecom: {enabled: true}\n"


def channel_env(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (path / "config.yaml").write_text(CHANNELS_CONFIG)
    monkeypatch.setenv("FEISHU_WEBHOOK", FEISHU)
    monkeypatch.setenv("FEISHU_SECRET", "sign-secret")
    monkeypatch.setenv("WECOM_WEBHOOK", WECOM)


@respx.mock
def test_notify_test_sends_to_enabled_channels(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channel_env(workdir, monkeypatch)
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    feishu = respx.post(FEISHU).mock(return_value=httpx.Response(200, json={"code": 0}))
    wecom = respx.post(WECOM).mock(
        return_value=httpx.Response(200, json={"errcode": 93000, "errmsg": "invalid webhook url"})
    )

    result = runner.invoke(app, ["notify-test"])

    assert result.exit_code == 1, result.output
    assert "✓ 飞书：测试消息已发送，请查收" in result.output
    assert "✗ 企业微信：企业微信返回错误 93000：Webhook 地址无效" in result.output
    card = json.loads(feishu.calls.last.request.content)["card"]
    assert card["header"]["title"]["content"].startswith("【测试】📚 Paper Gleaner · ")
    assert "https://github.com/o/r/issues?q=label%3Adaily" in json.dumps(card)
    assert wecom.called
    assert "feishu-token-123" not in result.output and "wecom-key-456" not in result.output

    only = runner.invoke(app, ["notify-test", "--channel", "feishu"])
    assert only.exit_code == 0, only.output
    assert "企业微信" not in only.output


def test_notify_test_reports_disabled_and_unknown_channels(workdir: Path) -> None:
    result = runner.invoke(app, ["notify-test"])
    assert result.exit_code == 1
    assert "没有启用任何推送渠道" in result.output

    result = runner.invoke(app, ["notify-test", "--channel", "email"])
    assert result.exit_code == 1
    assert "✗ 邮件没有启用：在 config.yaml 中设置 notify.email.enabled: true" in result.output

    result = runner.invoke(app, ["notify-test", "--channel", "sms"])
    assert result.exit_code == 1
    assert "不认识的渠道：sms" in result.output


def test_notify_test_reports_missing_secrets(workdir: Path) -> None:
    (workdir / "config.yaml").write_text("notify:\n  feishu: {enabled: true}\n")

    result = runner.invoke(app, ["notify-test"])

    assert result.exit_code == 1
    assert "✗ 飞书已启用但没有配置好，这次不发送：缺少环境变量 FEISHU_WEBHOOK" in result.output


@respx.mock
def test_daily_dry_run_does_not_notify(
    repo: Path, rss_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    channel_env(repo, monkeypatch)
    respx.get(RSS_URL).mock(return_value=httpx.Response(200, content=rss_bytes))
    feishu = respx.post(FEISHU)
    wecom = respx.post(WECOM)

    result = runner.invoke(app, ["daily", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "[1] 9/10  2610.00001" in result.output
    assert not feishu.called and not wecom.called
    assert "推送：" not in result.output


@respx.mock
def test_daily_notifies_after_publishing(
    repo: Path, rss_bytes: bytes, git_remote: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channel_env(repo, monkeypatch)
    github_env(monkeypatch)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    respx.get(RSS_URL).mock(return_value=httpx.Response(200, content=rss_bytes))
    respx.get(f"{API}/issues").mock(return_value=httpx.Response(200, json=[]))
    respx.post(f"{API}/labels").mock(return_value=httpx.Response(201, json={}))
    respx.post(f"{API}/issues").mock(
        return_value=httpx.Response(
            201, json={"number": 5, "html_url": "https://github.com/o/r/issues/5"}
        )
    )
    respx.post(f"{API}/issues/5/comments").mock(return_value=httpx.Response(201, json={"id": 1}))
    feishu = respx.post(FEISHU).mock(return_value=httpx.Response(200, json={"code": 0}))
    respx.post(WECOM).mock(side_effect=httpx.ConnectError(f"failed to connect {WECOM}"))

    result = runner.invoke(app, ["daily"])

    assert result.exit_code == 0, result.output
    assert "✓ 已发布：https://github.com/o/r/issues/5" in result.output
    assert "· 推送：飞书 ✓ · 企业微信 ✗" in result.output
    assert "::warning::企业微信推送失败：请求企业微信机器人失败" in result.output
    assert "wecom-key-456" not in result.output
    card = json.loads(feishu.calls.last.request.content)["card"]
    assert "https://github.com/o/r/issues/5" in json.dumps(card)
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    saved = git("show", f"gleaner-data:history/digests/{today}.json", cwd=git_remote)
    assert '"feishu": {\n      "status": "sent"' in saved
    assert "wecom-key-456" not in saved


# --- 反馈与偏好 -----------------------------------------------------------------

PREFERENCES_REPLY = {
    "markdown": "### 偏好\n- 偏好有开源代码的工作（1 次 /like 提到）",
    "changes": ["新增一条"],
}


def seed_feedback(clone: Callable[[str], Path]) -> None:
    """gleaner-data 上的反馈和屏蔽主题。"""
    seed_repo = clone("data-seed")
    seed = DataBranch(seed_repo, seed_repo / ".gleaner-data", backoff_seconds=0)
    store = seed.checkout()
    now = datetime.now(ZoneInfo("UTC"))
    liked = Paper(id="2610.00001", title="LongKV", abstract="We compress the KV cache.")
    disliked = Paper(id="2610.00002", title="RAG Bench", abstract="A benchmark.")
    bare = Paper(id="2610.00003", title="2610.00003", abstract="")
    record_feedback(
        store,
        [
            command_feedback(liked, "like", now=now, note="有开源代码"),
            command_feedback(disliked, "dislike", now=now, note="纯 benchmark"),
            command_feedback(bare, "deep", now=now),
        ],
        timezone="Asia/Shanghai",
    )
    store.append_jsonl(
        "profile/muted.jsonl",
        [
            {"action": "mute", "topic": "医学影像", "query": "Medical imaging."},
            {"action": "mute", "topic": "语音"},
        ],
    )
    seed.commit_and_push("data: seed feedback")


def use_llm(monkeypatch: pytest.MonkeyPatch, handler) -> FakeOpenAI:
    fake = FakeOpenAI(handler)
    monkeypatch.setattr(cli, "LLMClient", lambda config: LLMClient(config, client=fake))
    return fake


def test_feedback_command(repo: Path, clone: Callable[[str], Path], git_remote: Path) -> None:
    seed_feedback(clone)
    before = git("rev-parse", "gleaner-data", cwd=git_remote)

    result = runner.invoke(app, ["feedback", "--days", "7"])

    assert result.exit_code == 0, result.output
    output = result.output
    assert "反馈（最近 7 天，时间衰减半衰期 90 天）：共 3 篇论文" in output
    assert "正反馈（A2） 2 篇：" in output
    assert "  +2.0  权重 1.00  2610.00001  LongKV\n        /like · 原因：有开源代码" in output
    assert "/deep · 只有 ID，不参与粗筛" in output
    assert "负反馈（A3） 1 篇：\n  -2.0  权重 1.00  2610.00002  RAG Bench" in output
    assert "屏蔽主题 2 个：\n  医学影像 —— Medical imaging.\n  语音 —— （还没有英文描述" in output
    assert "（还没有，运行 gleaner summarize-preferences" in output
    assert git("rev-parse", "gleaner-data", cwd=git_remote) == before


def test_summarize_preferences_commits(
    repo: Path,
    clone: Callable[[str], Path],
    git_remote: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_feedback(clone)
    fake = use_llm(monkeypatch, lambda request: json.dumps(PREFERENCES_REPLY, ensure_ascii=False))
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    result = runner.invoke(app, ["summarize-preferences"])

    assert result.exit_code == 0, result.output
    assert "✓ 已把 2 篇论文的新反馈总结进 profile/learned_preferences.md。" in result.output
    assert "· 新增一条" in result.output
    assert "- 偏好有开源代码的工作（1 次 /like 提到）" in result.output
    assert "我研究长上下文推理。" in fake.requests[0]["messages"][-1]["content"]
    saved = git("show", "gleaner-data:profile/learned_preferences.md", cwd=git_remote)
    assert saved.endswith("- 偏好有开源代码的工作（1 次 /like 提到）")
    assert git("log", "-1", "--format=%s", "gleaner-data", cwd=git_remote) == (
        "data: update learned preferences"
    )
    assert "```markdown\n<!-- gleaner:preferences" in summary.read_text(encoding="utf-8")

    again = runner.invoke(app, ["summarize-preferences"])
    assert "上次总结之后没有新反馈，不调用 LLM" in again.output
    assert len(fake.requests) == 1


def test_summarize_preferences_dry_run(
    repo: Path, clone: Callable[[str], Path], git_remote: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_feedback(clone)
    use_llm(monkeypatch, lambda request: json.dumps(PREFERENCES_REPLY, ensure_ascii=False))
    before = git("rev-parse", "gleaner-data", cwd=git_remote)

    result = runner.invoke(app, ["summarize-preferences", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "试运行：会把 2 篇论文的新反馈总结进 profile/learned_preferences.md，没有提交。" in (
        result.output
    )
    assert git("rev-parse", "gleaner-data", cwd=git_remote) == before


def test_summarize_preferences_without_feedback(repo: Path, monkeypatch) -> None:
    fake = use_llm(monkeypatch, lambda request: "{}")

    result = runner.invoke(app, ["summarize-preferences"])

    assert result.exit_code == 0, result.output
    assert "上次总结之后没有新反馈，不调用 LLM" in result.output
    assert fake.requests == []


def test_summarize_preferences_llm_failure(
    repo: Path, clone: Callable[[str], Path], git_remote: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_feedback(clone)

    def boom(request):
        raise LLMError("调用 deepseek-chat 失败：Request timed out.")

    use_llm(monkeypatch, boom)
    before = git("rev-parse", "gleaner-data", cwd=git_remote)

    result = runner.invoke(app, ["summarize-preferences"])

    assert result.exit_code == 1
    assert "✗ 偏好总结失败，profile/learned_preferences.md 保持不变" in result.output
    assert git("rev-parse", "gleaner-data", cwd=git_remote) == before


def test_daily_requires_github_env(repo: Path) -> None:
    result = runner.invoke(app, ["daily"])
    assert result.exit_code == 1
    assert "GITHUB_TOKEN" in result.output
    assert "--dry-run" in result.output


def test_daily_requires_llm_key(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY")
    monkeypatch.setattr(cli, "LLMClient", LLMClient)
    result = runner.invoke(app, ["daily", "--dry-run"])
    assert result.exit_code == 1
    assert "LLM_API_KEY" in result.output


def test_daily_requires_interests(repo: Path) -> None:
    (repo / "profile" / "interests.md").unlink()
    result = runner.invoke(app, ["daily", "--dry-run"])
    assert result.exit_code == 1
    assert "interests.example.md" in result.output


@respx.mock
def test_daily_without_announcements(
    repo: Path, empty_rss_bytes: bytes, git_remote: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    github_env(monkeypatch)
    respx.get(f"{API}/issues").mock(return_value=httpx.Response(200, json=[]))
    respx.get(RSS_URL).mock(return_value=httpx.Response(200, content=empty_rss_bytes))

    result = runner.invoke(app, ["daily"])

    assert result.exit_code == 0, result.output
    assert "今天没有新公告" in result.output
    assert not data_branch_exists(git_remote)


@respx.mock
def test_dry_run_without_announcements_uses_recent_papers(
    repo: Path, empty_rss_bytes: bytes, api_bytes: bytes
) -> None:
    respx.get(RSS_URL).mock(return_value=httpx.Response(200, content=empty_rss_bytes))
    api = respx.get("https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(200, content=api_bytes)
    )

    result = runner.invoke(app, ["daily", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "改用最近提交的 2 篇论文试运行" in result.output
    assert "[1] 9/10  2610.00001  LongKV" in result.output
    assert api.calls.last.request.url.params["max_results"] == "20"


@respx.mock
def test_daily_network_error(repo: Path) -> None:
    respx.get(RSS_URL).mock(return_value=httpx.Response(403))

    result = runner.invoke(app, ["daily", "--dry-run"])

    assert result.exit_code == 1
    assert "抓取 arXiv RSS 失败" in result.output


def test_config_error_is_reported(workdir: Path) -> None:
    (workdir / "config.yaml").write_text("ranking:\n  top_n: 5\n")
    result = runner.invoke(app, ["check"])
    assert result.exit_code == 1
    assert "ranking.top_n" in result.output


def test_check_reports_missing_items(workdir: Path) -> None:
    result = runner.invoke(app, ["check"])
    assert result.exit_code == 1
    assert "✗ 兴趣描述" in result.output
    assert "✗ LLM_API_KEY" in result.output


def test_check_passes_when_complete(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (workdir / "profile").mkdir()
    (workdir / "profile" / "interests.md").write_text("长上下文推理", encoding="utf-8")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")

    result = runner.invoke(app, ["check"])

    assert result.exit_code == 0, result.output
    assert "✓ 兴趣描述" in result.output
    assert "✓ LLM_API_KEY" in result.output
    assert "cs.CL, cs.LG" in result.output


def test_check_reports_channels(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (workdir / "profile").mkdir()
    (workdir / "profile" / "interests.md").write_text("长上下文推理", encoding="utf-8")
    (workdir / "config.yaml").write_text(
        "notify:\n  github_issue: {enabled: false}\n"
        "  email: {enabled: true}\n  feishu: {enabled: true}\n"
    )
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    webhook = "https://open.feishu.cn/open-apis/bot/v2/hook/token-abc"
    monkeypatch.setenv("FEISHU_WEBHOOK", webhook)
    monkeypatch.setenv("WECOM_WEBHOOK", "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=k1")
    monkeypatch.setenv("SMTP_HOST", "smtp.qq.com")

    result = runner.invoke(app, ["check"])

    assert result.exit_code == 1, result.output
    assert "· GitHub Issue：未启用（其它渠道照样推送" in result.output
    assert "✗ 邮件：已启用，但缺少环境变量 SMTP_USER、SMTP_PASSWORD、EMAIL_TO" in result.output
    assert "✓ 飞书：已设置 Webhook；未设置 FEISHU_SECRET" in result.output
    assert "· 企业微信：未启用（相关 Secret 已设置，把 notify.wecom.enabled 改为 true" in (
        result.output
    )
    assert "gleaner notify-test" in result.output
    assert "token-abc" not in result.output and "k1" not in result.output

    monkeypatch.setenv("SMTP_USER", "me@qq.com")
    monkeypatch.setenv("SMTP_PASSWORD", "auth-code")
    monkeypatch.setenv("EMAIL_TO", "a@example.com,b@example.com")
    monkeypatch.setenv("FEISHU_SECRET", "sign")
    result = runner.invoke(app, ["check"])

    assert result.exit_code == 0, result.output
    assert "✓ 邮件：SMTP smtp.qq.com:465（SSL），收件人 2 个" in result.output
    assert "✓ 飞书：已设置 Webhook，已开启签名校验" in result.output
    assert "me@qq.com" not in result.output and "a@example.com" not in result.output


def test_falls_back_to_example_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.example.yaml").write_text("")

    result = runner.invoke(app, ["check"])

    assert "暂用 config.example.yaml" in result.output


def test_report_warns_about_missing_comments() -> None:
    digest = Digest(
        day=date(2026, 10, 26),
        generated_at=datetime.now(ZoneInfo("UTC")),
        papers=[ScoredPaper(id="2610.00001", title="P", abstract="A", llm_score=8)],
        issue_number=5,
        issue_url="https://github.com/o/r/issues/5",
    )
    published = _report(DailyResult("published", digest), missing=1)
    assert "✓ 已发布：https://github.com/o/r/issues/5" in published
    assert "还有 1 篇论文的评论没有发出去" in published

    existing = IssueRef(5, "https://github.com/o/r/issues/5")
    repaired = _report(DailyResult("repaired", digest, existing))
    assert repaired == "✓ 已补发缺失的论文评论：https://github.com/o/r/issues/5"


# --- 指令交互 -------------------------------------------------------------------

HTML_URL = "https://arxiv.org/html/2610.00001"


@pytest.fixture
def interact_repo(
    repo: Path, clone: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
) -> Path:
    """repo 之外再准备好 gleaner-data 分支上的日报，LLM 换成深读、对比用的假模型。"""
    seed_repo = clone("data-seed")  # "seed" 已被 git_remote 占用
    seed = DataBranch(seed_repo, seed_repo / ".gleaner-data", backoff_seconds=0)
    seed.checkout().write_model(digest_path(DIGEST_DAY), interact_digest())
    seed.commit_and_push("data: seed digest")
    monkeypatch.setattr(
        cli, "LLMClient", lambda config: LLMClient(config, client=FakeOpenAI(interact_llm))
    )
    return repo


def event_file(tmp_path: Path, body: str) -> Path:
    data = json.loads((FIXTURES / "issue_comment_event.json").read_text(encoding="utf-8"))
    data["comment"]["body"] = body
    path = tmp_path / "event.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def remote_files(remote: Path) -> list[str]:
    return git("ls-tree", "-r", "--name-only", "gleaner-data", cwd=remote).split()


def test_handle_comment_dry_run(interact_repo: Path, tmp_path: Path, git_remote: Path) -> None:
    before = git("rev-parse", "gleaner-data", cwd=git_remote)

    result = runner.invoke(
        app, ["handle-comment", "--event", str(event_file(tmp_path, "/more 2")), "--dry-run"]
    )

    assert result.exit_code == 0, result.output
    assert "· 给评论 9001 加 eyes" in result.output
    assert "----- 回复 Issue 1（第 1 条）-----" in result.output
    assert "### [3] Candidate 0" in result.output
    assert "### [4] Candidate 1" in result.output
    assert "· 给评论 9001 加 rocket" in result.output
    assert "✓ 完成 `/more 2`" in result.output
    assert git("rev-parse", "gleaner-data", cwd=git_remote) == before


@respx.mock
def test_handle_comment_posts_reply_and_caches_report(
    interact_repo: Path,
    tmp_path: Path,
    git_remote: Path,
    html_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    github_env(monkeypatch)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file(tmp_path, "/deep 1")))
    respx.get(HTML_URL).mock(return_value=httpx.Response(200, text=html_text))
    reactions = respx.post(f"{API}/issues/comments/9001/reactions").mock(
        return_value=httpx.Response(201, json={})
    )
    comments = respx.post(f"{API}/issues/1/comments").mock(
        return_value=httpx.Response(201, json={"id": 555})
    )

    result = runner.invoke(app, ["handle-comment"])

    assert result.exit_code == 0, result.output
    assert [json.loads(c.request.content)["content"] for c in reactions.calls] == [
        "eyes",
        "rocket",
    ]
    assert DEEP_REPORT in json.loads(comments.calls[0].request.content)["body"]
    files = remote_files(git_remote)
    assert "reports/2610.00001.md" in files
    assert any(f.startswith("feedback/") for f in files)  # 按当前月份分片
    assert "## Paper Gleaner · Issue 1" in summary.read_text(encoding="utf-8")


@respx.mock
def test_handle_comment_reports_config_errors(
    interact_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    github_env(monkeypatch)
    (interact_repo / "config.yaml").write_text("ranking:\n  top_n: 5\n")
    respx.post(f"{API}/issues/comments/9001/reactions").mock(
        return_value=httpx.Response(201, json={})
    )
    comments = respx.post(f"{API}/issues/1/comments").mock(
        return_value=httpx.Response(201, json={"id": 555})
    )

    result = runner.invoke(app, ["handle-comment", "--event", str(event_file(tmp_path, "/help"))])

    assert result.exit_code == 1
    body = json.loads(comments.calls[0].request.content)["body"]
    assert body.startswith("⚠️ 配置有误，请检查 config.yaml：")
    assert "ranking.top_n" in body


def test_handle_comment_requires_event(repo: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["handle-comment", "--dry-run"])
    assert result.exit_code == 1
    assert "GITHUB_EVENT_PATH" in result.output

    missing = runner.invoke(app, ["handle-comment", "--event", str(tmp_path / "x.json")])
    assert missing.exit_code == 1
    assert "找不到事件文件" in missing.output


def test_handle_comment_requires_github_token(repo: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["handle-comment", "--event", str(event_file(tmp_path, "/help"))])
    assert result.exit_code == 1
    assert "GITHUB_TOKEN" in result.output


@respx.mock
def test_local_deep(interact_repo: Path, git_remote: Path, html_text: str) -> None:
    respx.get(HTML_URL).mock(return_value=httpx.Response(200, text=html_text))

    result = runner.invoke(app, ["deep", "1"])  # 按最近一份日报解析序号

    assert result.exit_code == 0, result.output
    assert "## 🔍 深读 · [LongKV" in result.output
    assert DEEP_REPORT in result.output
    assert "reports/2610.00001.md" not in remote_files(git_remote)


@respx.mock
def test_local_compare(interact_repo: Path, html_text: str) -> None:
    respx.get(HTML_URL).mock(return_value=httpx.Response(200, text=html_text))
    respx.get("https://arxiv.org/html/2610.00002").mock(
        return_value=httpx.Response(200, text=html_text)
    )

    result = runner.invoke(app, ["compare", "1", "2610.00002", "--day", "2026-10-25"])

    assert result.exit_code == 0, result.output
    assert "## ⚖️ 对比 · 2 篇论文" in result.output


def test_local_ask(interact_repo: Path, git_remote: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request):
        if "每一步只做一件事" in request["messages"][0]["content"]:
            if len(request["messages"]) == 2:
                return json.dumps({"action": "call_tool", "tool": "get_digest", "arguments": {}})
            return json.dumps({"action": "answer"})
        return "推荐先读 2610.00001。"

    use_llm(monkeypatch, handler)
    before = git("rev-parse", "gleaner-data", cwd=git_remote)

    result = runner.invoke(app, ["ask", "今天先读哪篇？"])

    assert result.exit_code == 0, result.output
    assert "## 💬 /ask · `今天先读哪篇？`" in result.output
    assert "推荐先读 2610.00001。" in result.output
    assert "2 步（上限 8）" in result.output and "工具：get_digest" in result.output
    assert git("rev-parse", "gleaner-data", cwd=git_remote) == before  # 不写数据


def test_digest_command(repo: Path, clone: Callable[[str], Path], git_remote: Path) -> None:
    older = interact_digest(day=date(2026, 10, 20), overview=None, others=[])
    seed_data(clone, interact_digest(overview="今天以长上下文为主。"), older)
    before = git("rev-parse", "gleaner-data", cwd=git_remote)

    latest = runner.invoke(app, ["digest"])

    assert latest.exit_code == 0, latest.output
    output = latest.output
    assert output.startswith("📚 2026-10-25 的日报 · 推荐 2 篇 · https://github.com/o/r/issues/1")
    assert "今天以长上下文为主。" in output
    assert "[1] 9/10  2610.00001  LongKV" in output
    assert "    https://arxiv.org/abs/2610.00001" in output
    assert "另有 3 篇候选没有推送，加 --all 查看。" in output
    assert "Candidate 0" not in output

    everything = runner.invoke(app, ["digest", "--all"])
    assert "其余 3 篇打过分但没有推送的候选：" in everything.output
    assert "[3] 5/10  2610.00010  Candidate 0" in everything.output
    assert "    🎯 候选 0 的理由" in everything.output

    older_day = runner.invoke(app, ["digest", "--day", "2026-10-20"])
    assert older_day.output.startswith("📚 2026-10-20 的日报")
    assert "加 --all" not in older_day.output

    missing = runner.invoke(app, ["digest", "--day", "2026-10-01"])
    assert missing.exit_code == 1
    assert "没有 2026-10-01 的日报，最近一份是 2026-10-25" in missing.output
    assert git("rev-parse", "gleaner-data", cwd=git_remote) == before


def test_digest_marks_cached_reports_and_works_offline(
    repo: Path, clone: Callable[[str], Path]
) -> None:
    seed_repo = clone("data-seed")
    seed = DataBranch(seed_repo, seed_repo / ".gleaner-data", backoff_seconds=0)
    store = seed.checkout()
    store.write_model(digest_path(DIGEST_DAY), interact_digest())
    store.write_text("reports/2610.00002.md", "---\npaper_id: 2610.00002\n---\n报告")
    seed.commit_and_push("data: seed")

    result = runner.invoke(app, ["digest"])
    assert "    🔍 已有深读报告：gleaner deep 2610.00002 直接读取" in result.output

    git("remote", "set-url", "origin", str(repo.parent / "gone.git"), cwd=repo)
    offline = runner.invoke(app, ["digest"])
    assert offline.exit_code == 0, offline.output
    assert "! 同步 gleaner-data 失败，使用本地已有的数据" in offline.output
    assert "📚 2026-10-25 的日报" in offline.output


def test_digest_without_any_data(repo: Path) -> None:
    result = runner.invoke(app, ["digest"])
    assert result.exit_code == 1
    assert "gleaner-data 上还没有日报" in result.output


def test_local_deep_reports_bad_refs(interact_repo: Path) -> None:
    result = runner.invoke(app, ["deep", "9"])
    assert result.exit_code == 1
    assert "序号 9 超出范围" in result.output


@respx.mock
def test_daily_dry_run_auto_deep_reads_first_paper(
    repo: Path, rss_bytes: bytes, html_text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    (repo / "config.yaml").write_text(REPO_CONFIG.replace("auto_top_m: 0", "auto_top_m: 1"))
    base = pipeline_llm(SCORES)
    use_llm(
        monkeypatch,
        lambda request: (
            DEEP_REPORT if "深读报告" in request["messages"][0]["content"] else base(request)
        ),
    )
    respx.get(RSS_URL).mock(return_value=httpx.Response(200, content=rss_bytes))
    html = respx.get(HTML_URL).mock(return_value=httpx.Response(200, text=html_text))

    result = runner.invoke(app, ["daily", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert html.called
    assert "    🔍 已自动深读：报告附在评论里，并写入 reports/2610.00001.md" in result.output
    assert "自动深读 1 篇，报告附在对应论文的评论里" in result.output
    assert "LLM 调用 4 次" in result.output


ZOTERO_API = "https://api.zotero.org/users/1234567"


@respx.mock
def test_daily_dry_run_with_zotero(
    repo: Path, rss_bytes: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (repo / "config.yaml").write_text(
        REPO_CONFIG + "profile:\n  zotero: {enabled: true, max_items: 2}\n"
    )
    monkeypatch.setenv("ZOTERO_ID", "1234567")
    monkeypatch.setenv("ZOTERO_KEY", "zotero-key-xyz")
    cache = tmp_path / "zotero-cache"
    monkeypatch.setenv("GLEANER_ZOTERO_CACHE", str(cache))
    items = [
        {
            "key": f"ITEM000{i}",
            "version": 3,
            "data": {
                "key": f"ITEM000{i}",
                "itemType": "journalArticle",
                "title": f"My library paper on attention {i}",
                "abstractNote": "Attention kernels.",
                "collections": [],
                "dateAdded": f"2026-10-0{i + 1}T00:00:00Z",
            },
        }
        for i in range(3)
    ]
    headers = {"Total-Results": "3", "Last-Modified-Version": "42"}
    respx.get(f"{ZOTERO_API}/items/top").mock(
        return_value=httpx.Response(200, json=items, headers=headers)
    )
    respx.get(RSS_URL).mock(return_value=httpx.Response(200, content=rss_bytes))

    result = runner.invoke(app, ["daily", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Zotero：使用 2 / 3 篇" in result.output
    assert "+ Zotero " in result.output
    assert "My library paper" not in result.output
    assert "zotero-key-xyz" not in result.output and "1234567" not in result.output
    assert list(cache.glob("library-*.json"))
    data = repo / ".gleaner-data"
    for path in data.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            assert "My library paper" not in path.read_text(encoding="utf-8"), path


def test_check_reports_zotero(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (workdir / "config.yaml").write_text(
        "profile:\n  zotero:\n    enabled: true\n    library_type: group\n"
        "    collections: [Detection]\n"
    )
    result = runner.invoke(app, ["check"])
    assert "✗ Zotero：已启用，但缺少环境变量 ZOTERO_ID、ZOTERO_KEY" in result.output

    monkeypatch.setenv("ZOTERO_ID", "7654321")
    monkeypatch.setenv("ZOTERO_KEY", "zotero-key-xyz")
    result = runner.invoke(app, ["check"])
    assert (
        "✓ Zotero：群组文库，范围：collection Detection（含子 collection），最多取 200 篇作为锚点"
        in result.output
    )
    assert "7654321" not in result.output
