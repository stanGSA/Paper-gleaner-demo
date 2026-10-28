import json
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from gleaner import cli
from gleaner.config import Config
from gleaner.notify import GitHubClient
from gleaner.onboarding import (
    LABEL,
    MARKER,
    SetupProblem,
    SetupReport,
    config_problems,
    inspect_setup,
    quickstart_config,
    setup_body,
    sync_setup_issue,
)

API = "https://api.github.com/repos/o/r"
runner = CliRunner()


@pytest.fixture
def setup_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    for name in (
        "LLM_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "LLM_PROVIDER",
        "SMTP_PORT",
        "EMAIL_TO",
        "GITHUB_STEP_SUMMARY",
    ):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "config.example.yaml").write_text("", encoding="utf-8")
    return tmp_path


def ready_report(setup_dir: Path, monkeypatch: pytest.MonkeyPatch) -> SetupReport:
    (setup_dir / "profile").mkdir(exist_ok=True)
    (setup_dir / "profile/interests.md").write_text("研究高效推理。", encoding="utf-8")
    monkeypatch.setenv("LLM_API_KEY", "private-api-key")
    return inspect_setup()


def client() -> GitHubClient:
    return GitHubClient("private-github-token", "o/r", backoff_seconds=0)


def existing(report: SetupReport, *, state: str = "open") -> dict:
    return {
        "number": 7,
        "html_url": "https://github.com/o/r/issues/7",
        "body": setup_body(report),
        "state": state,
    }


def test_missing_and_comment_only_interests_are_rejected(setup_dir: Path) -> None:
    report = inspect_setup()
    assert not report.ready
    assert {p.code for p in report.problems} == {"interests", "llm_key"}
    (setup_dir / "profile").mkdir()
    (setup_dir / "profile/interests.md").write_text("<!-- 填写研究方向 -->\n", encoding="utf-8")
    assert "interests" in {p.code for p in inspect_setup().problems}
    checked = runner.invoke(cli.app, ["check"])
    assert checked.exit_code == 1
    assert "为空" in checked.output


def test_example_fallback_is_allowed_and_dotenv_is_loaded(setup_dir: Path) -> None:
    (setup_dir / ".env").write_text("LLM_API_KEY=private-dotenv-key\n", encoding="utf-8")
    (setup_dir / "profile").mkdir()
    (setup_dir / "profile/interests.md").write_text("我研究 RAG。", encoding="utf-8")
    report = inspect_setup()
    assert report.ready and report.fallback
    assert "private-dotenv-key" not in setup_body(report)


@pytest.mark.parametrize(
    "text", ["llm: [private-config-secret", "llm:\n  api_key: private-config-secret\n"]
)
def test_bad_config_never_exposes_values(setup_dir: Path, text: str) -> None:
    (setup_dir / "config.yaml").write_text(text, encoding="utf-8")
    report = inspect_setup()
    assert not report.ready
    assert report.problems[0].code == "config"
    assert "private-config-secret" not in setup_body(report)


def test_quickstart_preserves_original_and_skips_optional_config(
    setup_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ready_report(setup_dir, monkeypatch).config
    assert config is not None
    config.notify.email.enabled = True
    config.profile.zotero.enabled = True
    assert {p.code for p in config_problems(config)} == {"email", "zotero"}
    first = quickstart_config(config)
    assert not config_problems(first)
    assert first.deep_read.auto_top_m == 0
    assert config.deep_read.auto_top_m == 1
    assert config.notify.email.enabled and config.profile.zotero.enabled
    assert first.notify.github_issue.enabled


@respx.mock
def test_creates_one_dedicated_issue_then_noops(setup_dir: Path) -> None:
    report = inspect_setup()
    listing = respx.get(f"{API}/issues").mock(
        side_effect=[httpx.Response(200, json=[]), httpx.Response(200, json=[existing(report)])]
    )
    labels = respx.post(f"{API}/labels").mock(return_value=httpx.Response(201, json={}))
    create = respx.post(f"{API}/issues").mock(
        return_value=httpx.Response(201, json=existing(report))
    )
    assert sync_setup_issue(client(), report).number == 7
    assert sync_setup_issue(client(), report).number == 7
    assert create.call_count == labels.call_count == 1
    payload = json.loads(create.calls.last.request.content)
    assert payload["labels"] == [LABEL]
    assert MARKER in payload["body"]
    assert listing.calls.last.request.url.params["state"] == "all"
    assert listing.calls.last.request.url.params["labels"] == ""


@respx.mock
def test_updates_closes_and_reopens_same_issue(
    setup_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = inspect_setup()
    ready = ready_report(setup_dir, monkeypatch)
    changed = SetupReport(
        config=Config(), problems=[SetupProblem("config", "新的问题", "请修改配置")]
    )
    respx.get(f"{API}/issues").mock(
        side_effect=[
            httpx.Response(200, json=[existing(missing)]),
            httpx.Response(200, json=[existing(ready, state="closed")]),
            httpx.Response(200, json=[existing(missing)]),
        ]
    )
    update = respx.patch(f"{API}/issues/7").mock(return_value=httpx.Response(200, json={}))
    sync_setup_issue(client(), ready)
    sync_setup_issue(client(), missing)
    sync_setup_issue(client(), changed)
    payloads = [json.loads(call.request.content) for call in update.calls]
    assert [p["state"] for p in payloads] == ["closed", "open", "open"]
    assert payloads[0]["state_reason"] == "completed"
    assert payloads[1]["state_reason"] == "reopened"
    assert "新的问题" in payloads[2]["body"]


@respx.mock
def test_finds_issue_beyond_first_page(setup_dir: Path) -> None:
    report = inspect_setup()
    listing = respx.get(f"{API}/issues").mock(
        side_effect=[
            httpx.Response(200, json=[{"number": n, "body": "unrelated"} for n in range(100)]),
            httpx.Response(200, json=[existing(report)]),
        ]
    )
    assert sync_setup_issue(client(), report).number == 7
    assert [c.request.url.params["page"] for c in listing.calls] == ["1", "2"]


@respx.mock
def test_keeps_oldest_issue_and_closes_legacy_duplicates(setup_dir: Path) -> None:
    report = inspect_setup()
    older = existing(report)
    newer = {**existing(report), "number": 8}
    respx.get(f"{API}/issues").mock(return_value=httpx.Response(200, json=[newer, older]))
    close = respx.patch(f"{API}/issues/8").mock(return_value=httpx.Response(200, json={}))
    assert sync_setup_issue(client(), report).number == 7
    assert json.loads(close.calls.last.request.content)["state"] == "closed"


@respx.mock
def test_ready_without_existing_issue_does_not_create_one(
    setup_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = ready_report(setup_dir, monkeypatch)
    respx.get(f"{API}/issues").mock(return_value=httpx.Response(200, json=[]))
    assert sync_setup_issue(client(), report) is None


@respx.mock
def test_disabled_issues_falls_back_to_safe_summary(
    setup_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = setup_dir / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GITHUB_TOKEN", "private-github-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    respx.get(f"{API}/issues").mock(
        return_value=httpx.Response(403, json={"message": "private-api-error-secret"})
    )
    result = runner.invoke(cli.app, ["setup", "--github"])
    assert result.exit_code == 1
    assert "HTTP 403" in result.output
    assert "LLM_API_KEY" in summary.read_text(encoding="utf-8")
    assert "private-api-error-secret" not in result.output
    assert "private-api-error-secret" not in summary.read_text(encoding="utf-8")


def test_dry_run_never_contacts_github_or_loads_models(
    setup_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("preflight must not contact GitHub or load models")

    monkeypatch.setattr(cli, "_github_client", forbidden)
    monkeypatch.setattr(cli, "LLMClient", forbidden)
    monkeypatch.setattr(cli, "build_embedder", forbidden)
    assert runner.invoke(cli.app, ["setup", "--github", "--dry-run"]).exit_code == 1
    assert runner.invoke(cli.app, ["daily", "--dry-run"]).exit_code == 1
    ready_report(setup_dir, monkeypatch)
    assert runner.invoke(cli.app, ["setup", "--github", "--dry-run"]).exit_code == 0
