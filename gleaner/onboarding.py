"""Shared, offline setup checks and an idempotent GitHub welcome Issue.

Public guidance contains only fixed instructions and setting names. Never publish
configuration values, interest text, or the original parsing/API exception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from gleaner.config import (
    EXAMPLE_CONFIG_PATH,
    Config,
    ConfigError,
    load_config,
    resolve_config_path,
)
from gleaner.notify import GitHubClient, IssueRef
from gleaner.notify.base import CHANNEL_LABELS, CHANNELS, missing_settings
from gleaner.profile import ProfileError, load_interests

MARKER = "<!-- gleaner:onboarding -->"
LABEL = "gleaner-setup"
TITLE = "👋 欢迎使用 Paper Gleaner"


@dataclass(frozen=True)
class SetupProblem:
    code: str
    message: str
    remedy: str
    blocking: bool = True


@dataclass
class SetupReport:
    path: Path | None = None
    config: Config | None = None
    problems: list[SetupProblem] = field(default_factory=list)
    fallback: bool = False

    @property
    def ready(self) -> bool:
        return self.config is not None and not any(p.blocking for p in self.problems)


def quickstart_config(config: Config) -> Config:
    """Use a copy: skip optional work for the first digest, without editing YAML."""
    result = config.model_copy(deep=True)
    result.deep_read.auto_top_m = 0
    result.profile.zotero.enabled = False
    for channel in CHANNELS:
        getattr(result.notify, channel).enabled = False
    return result


def config_problems(config: Config) -> list[SetupProblem]:
    """Validate required inputs before model loading, with optional-service warnings."""
    problems = []
    try:
        load_interests(config.profile.interests_file)
    except (ProfileError, OSError, UnicodeError):
        problems.append(
            SetupProblem(
                "interests",
                "兴趣描述不存在、为空或无法读取",
                "把 profile/interests.example.md 复制为 profile/interests.md，"
                "用几句话写下研究方向；若自定义路径，检查 profile.interests_file。"
                "只包含 HTML 注释的文件也算空文件。",
            )
        )
    if config.llm.api_key is None:
        problems.append(
            SetupProblem(
                "llm_key",
                "LLM_API_KEY：未设置",
                "在 Settings → Secrets and variables → Actions 添加 LLM_API_KEY。"
                "本地运行可将密钥放在仓库根目录的 .env。"
                "使用其他服务商时，同时设置 LLM_BASE_URL 和两档模型名。",
            )
        )
    for channel in CHANNELS:
        if getattr(config.notify, channel).enabled and (
            missing := missing_settings(config.notify, channel)
        ):
            problems.append(
                SetupProblem(
                    channel,
                    f"{CHANNEL_LABELS[channel]}：缺少 {'、'.join(missing)}",
                    "补齐对应 Actions Secrets，或在 config.yaml 中关闭这个可选渠道；"
                    "也可以勾选 quickstart，先运行基础日报。",
                    blocking=False,
                )
            )
    zotero = config.profile.zotero
    if zotero.enabled:
        missing = [
            name
            for name, value in (("ZOTERO_ID", zotero.library_id), ("ZOTERO_KEY", zotero.api_key))
            if value is None
        ]
        if missing:
            problems.append(
                SetupProblem(
                    "zotero",
                    f"Zotero：缺少 {'、'.join(missing)}",
                    "补齐对应 Actions Secrets，或设置 profile.zotero.enabled: false；"
                    "quickstart 会跳过 Zotero。",
                    blocking=False,
                )
            )
    return problems


def inspect_setup(path: Path | None = None, *, quickstart: bool = False) -> SetupReport:
    report = SetupReport()
    try:
        report.path = resolve_config_path(path)
        report.fallback = path is None and report.path == EXAMPLE_CONFIG_PATH
        config = load_config(report.path)
    except (ConfigError, OSError, UnicodeError):
        report.problems.append(
            SetupProblem(
                "config",
                "配置文件缺失或无法通过校验",
                "从 config.example.yaml 创建 config.yaml，检查 YAML 缩进和字段名。"
                "密钥只能放在 Actions Secrets 或本地 .env；"
                "运行 gleaner check 查看具体配置错误。",
            )
        )
        return report
    report.config = quickstart_config(config) if quickstart else config
    report.problems = config_problems(report.config)
    return report


def setup_body(report: SetupReport) -> str:
    lines = ["## 👋 欢迎使用 Paper Gleaner", ""]
    if not report.problems:
        lines += ["✅ 配置预检查已通过，可以运行 Daily digest。", ""]
    else:
        lines += [
            "请按下面的清单完成配置，再到 Actions → Daily digest → Run workflow 重新运行。",
            "只检查配置，不会为此下载嵌入模型或调用大模型。",
            "",
        ]
        for problem in report.problems:
            kind = "必需" if problem.blocking else "可选，不阻止基础日报"
            lines += [f"- [ ] **{problem.message}**（{kind}）", f"  {problem.remedy}", ""]
    if report.fallback:
        lines += [
            "> 当前暂用 config.example.yaml 的默认设置。允许试跑；"
            "建议在自己的 Fork 中提交 config.yaml，明确选择分类和模型。",
            "",
        ]
    lines += [
        "首次使用建议勾选 **quickstart**：跳过自动深读、Zotero 和额外通知，先生成日报。",
        "`dry_run` 只验证流程，不会发布日报或引导 Issue。",
        "",
        "若 Issues 未启用，请在 Settings → General → Features 中启用；"
        "Actions 需要 `contents: write` 和 `issues: write` 权限。",
        "API Key 是否真实可用、模型名是否正确，可在本地运行 `gleaner check --ping-llm` 验证。",
        "",
        MARKER,
    ]
    return "\n".join(lines) + "\n"


def sync_setup_issue(github: GitHubClient, report: SetupReport) -> IssueRef | None:
    """Update/reopen the same marked Issue; close it when all findings are fixed."""
    # GitHub's label-filtered listing can lag immediately after creation. Use
    # the unfiltered repository listing and the body marker instead, including
    # closed Issues. Workflow concurrency serializes writes across Actions runs.
    issues = sorted(
        (
            issue
            for issue in github.list_issues([], state="all")
            if MARKER in (issue.get("body") or "")
        ),
        key=lambda issue: issue["number"],
    )
    existing = issues[0] if issues else None
    # Heal duplicates left by older versions or interrupted concurrent clients.
    for duplicate in issues[1:]:
        if duplicate.get("state") != "closed":
            github.close_issue(duplicate["number"])
    if existing is None and not report.problems:
        return None
    body = setup_body(report)
    if existing is None:
        github.ensure_label(LABEL, "d4c5f9", "Paper Gleaner 配置引导")
        return github.create_issue(TITLE, body, [LABEL])
    state = "open" if report.problems else "closed"
    if existing.get("body") != body or existing.get("state") != state:
        github.update_issue(existing["number"], body=body, state=state)
    return IssueRef(existing["number"], existing["html_url"])
