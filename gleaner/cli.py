"""命令行入口。GitHub Actions、MCP Server 和 Claude Code Skill 都通过它调用核心能力。"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

import httpx
import typer
from pydantic import ValidationError

from gleaner import __version__
from gleaner.agent import Command, CommandError, Context, Services, execute, latest_digest_day
from gleaner.config import (
    EXAMPLE_CONFIG_PATH,
    Config,
    ConfigError,
    load_config,
    resolve_config_path,
)
from gleaner.llm import LLMClient, LLMError
from gleaner.models import Digest, ScoredPaper
from gleaner.notify import GitHubClient, GitHubError, IssuePublisher
from gleaner.notify.base import (
    CHANNEL_LABELS,
    CHANNELS,
    DEFAULT_SMTP_PORT,
    ChannelName,
    missing_settings,
    sample_digest,
)
from gleaner.notify.dispatch import build_notifiers, results_line, send_all
from gleaner.onboarding import (
    config_problems,
    inspect_setup,
    quickstart_config,
    setup_body,
    sync_setup_issue,
)
from gleaner.pipelines.daily import DailyResult, digest_path, run_daily
from gleaner.pipelines.interact import (
    CommentEvent,
    EventError,
    InteractResult,
    handle_comment,
    load_event,
    report_failure,
)
from gleaner.profile import (
    LEARNED_PREFERENCES_PATH,
    Memory,
    PreferencesError,
    ProfileError,
    SummaryResult,
    load_interests,
    load_memory,
    summarize_preferences,
)
from gleaner.profile.memory import FEEDBACK_LOOKBACK_MONTHS
from gleaner.ranking import EmbeddingError, build_embedder
from gleaner.reader import FulltextFetcher, report_path
from gleaner.render import sanitize, stats_line
from gleaner.render.replies import code, short_error
from gleaner.sources import ArxivSource, SemanticScholar, SourceError
from gleaner.sources.zotero import ZoteroLibrary
from gleaner.store import DataBranch, DataBranchError, DataStore

app = typer.Typer(
    help="Paper Gleaner：fork 即用、零服务器的论文追踪 Agent。",
    no_args_is_help=True,
    add_completion=False,
)

DEFAULT_DATA_DIR = Path(".gleaner-data")
# 试运行时如果没有新公告（周末、节假日），改用最近提交的这么多篇论文
DRY_RUN_SAMPLE_SIZE = 20

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config", "-c", help="配置文件路径，默认依次查找 config.yaml、config.example.yaml"
    ),
]
DataDirOption = Annotated[
    Path, typer.Option("--data-dir", help="gleaner-data 分支的本地 worktree 目录")
]
DayOption = Annotated[
    datetime | None,
    typer.Option(
        "--day",
        formats=["%Y-%m-%d"],
        help="按哪一天的日报解析序号，默认最近一份",
    ),
]


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="输出调试日志")] = False,
) -> None:
    _configure_stdio()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _configure_stdio() -> None:
    """Use UTF-8 on Windows consoles while preserving redirected streams."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(OSError, ValueError):
                stream.reconfigure(encoding="utf-8")


@app.command()
def version() -> None:
    """显示版本号。"""
    typer.echo(__version__)


@app.command()
def check(
    config_path: ConfigOption = None,
    ping_llm: Annotated[
        bool, typer.Option("--ping-llm", help="向两档模型各发一次极短请求，验证 API Key 和模型名")
    ] = False,
) -> None:
    """检查配置是否完整、可用。"""
    path, config = _load(config_path)
    problems = 0

    typer.echo(f"✓ 配置文件：{path}")

    interests = config.profile.interests_file
    required_problems = {p.code: p for p in config_problems(config) if p.blocking}
    if "interests" not in required_problems:
        typer.echo(f"✓ 兴趣描述：{interests}")
    else:
        typer.echo(
            f"✗ 兴趣描述：{required_problems['interests'].message}，"
            "请从 profile/interests.example.md 复制一份并改写"
        )
        problems += 1

    if config.llm.api_key is not None:
        typer.echo("✓ LLM_API_KEY：已设置")
    else:
        typer.echo(
            "✗ LLM_API_KEY：未设置，请在仓库 Settings → Secrets and variables → Actions 中添加"
        )
        problems += 1

    models = config.llm.models
    typer.echo(f"· LLM：{config.llm.base_url}（fast={models.fast}，deep={models.deep}）")
    typer.echo(f"· arXiv 分类：{', '.join(config.sources.arxiv.categories)}")
    channel_lines, channel_problems = _channel_status(config)
    typer.echo("\n".join(channel_lines))
    problems += channel_problems
    zotero_line, zotero_problem = _zotero_status(config)
    typer.echo(zotero_line)
    problems += zotero_problem

    if ping_llm and config.llm.api_key is not None:
        client = LLMClient(config.llm)
        for tier in ("fast", "deep"):
            model = client.model_for(tier)
            try:
                client.chat([{"role": "user", "content": "请只回复 OK"}], tier=tier)
            except LLMError as e:
                typer.echo(f"✗ LLM {tier}（{model}）：{e}")
                problems += 1
            else:
                typer.echo(f"✓ LLM {tier}（{model}）：可用")

    if problems:
        raise typer.Exit(1)


@app.command()
def setup(
    config_path: ConfigOption = None,
    github: Annotated[bool, typer.Option("--github", help="同步 GitHub 配置引导 Issue")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="只检查，不访问或修改 GitHub")] = False,
    quickstart: Annotated[
        bool, typer.Option("--quickstart", help="仅检查首次运行所需配置")
    ] = False,
) -> None:
    """离线预检查；可在 Actions 中创建、更新或关闭同一条配置引导 Issue。"""
    report = inspect_setup(config_path, quickstart=quickstart)
    body = setup_body(report)
    typer.echo(body)
    if github and not dry_run:
        try:
            issue = sync_setup_issue(_github_client(), report)
            if issue is not None:
                note = f"配置引导：{issue.url}"
                typer.echo(note)
                body += f"\n{note}\n"
        except (ConfigError, GitHubError, httpx.HTTPError) as error:
            # GitHub 的错误正文可能包含敏感输入，不在公开日志中回显。
            status = (
                f"（HTTP {error.status}）"
                if isinstance(error, GitHubError) and error.status
                else ""
            )
            note = (
                f"⚠️ 无法同步配置引导 Issue{status}；请检查 Issues、GITHUB_TOKEN 和写入权限。"
                "配置清单见上方及 Actions Summary。"
            )
            typer.echo(note, err=True)
            body += f"\n{note}\n"
    if summary_file := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_file, "a", encoding="utf-8") as stream:
            stream.write(body)
    if not report.ready:
        raise typer.Exit(1)


def _channel_status(config: Config) -> tuple[list[str], int]:
    """各推送渠道的配置状态（不发送消息，验证请用 gleaner notify-test）。"""
    notify = config.notify
    lines = [
        "· GitHub Issue：已启用"
        if notify.github_issue.enabled
        else "· GitHub Issue：未启用（其它渠道照样推送，但不带 Issue 链接，也不能评论交互）"
    ]
    problems = 0
    for channel in CHANNELS:
        label = CHANNEL_LABELS[channel]
        missing = missing_settings(notify, channel)
        if not getattr(notify, channel).enabled:
            configured = "（相关 Secret 已设置，" if not missing else ""
            if configured:
                configured += f"把 notify.{channel}.enabled 改为 true 即可启用）"
            lines.append(f"· {label}：未启用{configured}")
        elif missing:
            lines.append(f"✗ {label}：已启用，但缺少环境变量 {'、'.join(missing)}（GitHub Secret）")
            problems += 1
        else:
            lines.append(f"✓ {label}：{_channel_detail(config, channel)}")
    if any(getattr(notify, channel).enabled for channel in CHANNELS):
        lines.append("  运行 gleaner notify-test 给已启用的渠道各发一条测试消息")
    return lines, problems


def _zotero_status(config: Config) -> tuple[str, int]:
    """Zotero 的配置状态（不请求 Zotero；collection 名称在运行日报时才解析）。"""
    zotero = config.profile.zotero
    missing = [
        name
        for name, value in (("ZOTERO_ID", zotero.library_id), ("ZOTERO_KEY", zotero.api_key))
        if value is None
    ]
    if not zotero.enabled:
        return "· Zotero：未启用", 0
    if missing:
        return f"✗ Zotero：已启用，但缺少环境变量 {'、'.join(missing)}（GitHub Secret）", 1
    library = "群组文库" if zotero.library_type == "group" else "个人文库"
    scope = "整个文库"
    if zotero.collections:
        scope = "collection " + "、".join(zotero.collections)
        scope += "（含子 collection）" if zotero.include_subcollections else "（不含子 collection）"
    return f"✓ Zotero：{library}，范围：{scope}，最多取 {zotero.max_items} 篇作为锚点", 0


def _channel_detail(config: Config, channel: str) -> str:
    notify = config.notify
    if channel == "email":
        email = notify.email
        port = email.smtp_port or DEFAULT_SMTP_PORT
        mode = "SSL" if port == 465 else "STARTTLS"
        return f"SMTP {email.smtp_host}:{port}（{mode}），收件人 {len(email.to)} 个"
    if channel == "feishu":
        if notify.feishu.secret is None:
            return (
                "已设置 Webhook；未设置 FEISHU_SECRET，"
                "建议在机器人的安全设置中开启「签名校验」并把密钥配置为 Secret"
            )
        return "已设置 Webhook，已开启签名校验"
    return f"已设置 Webhook，每次最多发 {notify.wecom.max_papers} 篇"


# 日报运行中可预期的失败：打印原因后以退出码 1 结束，不打印调用栈
_DAILY_ERRORS = (
    ConfigError,
    ProfileError,
    LLMError,
    EmbeddingError,
    SourceError,
    GitHubError,
    DataBranchError,
    httpx.HTTPError,
)


@app.command()
def daily(
    config_path: ConfigOption = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="完整运行，但不发布 Issue、不提交数据分支")
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="当天日报已存在时仍重新生成")] = False,
    quickstart: Annotated[
        bool, typer.Option("--quickstart", help="首次运行：跳过自动深读、Zotero 和额外通知")
    ] = False,
    data_dir: DataDirOption = DEFAULT_DATA_DIR,
) -> None:
    """运行每日推送流水线：抓取、筛选、精排，并发布到 GitHub Issue。"""
    _, config = _load(config_path)
    if quickstart:
        config = quickstart_config(config)
        typer.echo("· 快速开始：跳过自动深读、Zotero 和额外通知；配置文件保持不变")
    if problems := [p for p in config_problems(config) if p.blocking]:
        for problem in problems:
            typer.echo(f"✗ {problem.message}：{problem.remedy}", err=True)
        raise typer.Exit(1)
    today = datetime.now(ZoneInfo(config.timezone)).date()
    branch = DataBranch(Path.cwd(), data_dir)
    try:
        publisher = None if dry_run else _build_publisher(config)
        # dry-run 不推送：不创建渠道
        notifiers, notify_problems = ([], []) if dry_run else build_notifiers(config.notify)
        reactions, owner, skipped = _reaction_access(publisher)
        llm = LLMClient(config.llm)
        result = run_daily(
            config,
            store=branch.checkout(),
            llm=llm,
            embedder=build_embedder(config),
            today=today,
            publisher=publisher,
            reactions=reactions,
            owner=owner,
            notifiers=notifiers,
            zotero=ZoteroLibrary(config.profile.zotero) if config.profile.zotero.enabled else None,
            force=force,
            sample_size=DRY_RUN_SAMPLE_SIZE if dry_run else None,
        )
        if not dry_run:
            if result.status not in {"already_published", "no_announcements"}:
                branch.commit_and_push(f"data: daily digest {today.isoformat()}")
            elif result.sync.records:
                branch.commit_and_push(f"data: sync reactions {today.isoformat()}")
    except _DAILY_ERRORS as e:
        typer.echo(f"✗ {e}", err=True)
        raise typer.Exit(1) from e

    missing = _missing_comments(result, config)
    warnings = ([skipped] if skipped else []) + notify_problems + result.warnings
    report = _report(result, missing, warnings)
    typer.echo(report)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        # 在 Actions 运行页面显示为黄色警告
        if missing:
            typer.echo(f"::warning::{missing} 篇论文的评论没有发出去，{_REPAIR_HINT}")
        for warning in warnings:
            typer.echo(f"::warning::{warning}")
    if summary_file := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(f"## Paper Gleaner · {today.isoformat()}\n\n{report}\n")


@app.command("notify-test")
def notify_test(
    channels: Annotated[
        list[str] | None,
        typer.Option(
            "--channel",
            help="只测试这个渠道（email、feishu、wecom），可以写多次；默认测试全部已启用的渠道",
        ),
    ] = None,
    config_path: ConfigOption = None,
) -> None:
    """给已启用的推送渠道（邮件、飞书、企业微信）各发一条测试消息，验证配置。"""
    _, config = _load(config_path)
    selected: list[ChannelName] | None = None
    if channels:
        unknown = [name for name in channels if name not in CHANNELS]
        if unknown:
            typer.echo(f"✗ 不认识的渠道：{'、'.join(unknown)}（可选 {'、'.join(CHANNELS)}）")
            raise typer.Exit(1)
        selected = [name for name in CHANNELS if name in channels]
        disabled = [name for name in selected if not getattr(config.notify, name).enabled]
        for name in disabled:
            label = CHANNEL_LABELS[name]
            typer.echo(f"✗ {label}没有启用：在 config.yaml 中设置 notify.{name}.enabled: true")
        if disabled:
            raise typer.Exit(1)
    notifiers, problems = build_notifiers(config.notify, selected)
    for problem in problems:
        typer.echo(f"✗ {problem}")
    if not notifiers and not problems:
        typer.echo("✗ 没有启用任何推送渠道：在 config.yaml 的 notify 中启用 email、feishu 或 wecom")
        raise typer.Exit(1)

    today = datetime.now(ZoneInfo(config.timezone)).date()
    results = send_all(
        sample_digest(today, _issues_url()), notifiers, now=datetime.now(UTC), test=True
    )
    for name, outcome in results.items():
        label = CHANNEL_LABELS[name]
        if outcome.status == "sent":
            typer.echo(f"✓ {label}：测试消息已发送，请查收")
        else:
            typer.echo(f"✗ {label}：{outcome.error}")
    if problems or any(outcome.status == "failed" for outcome in results.values()):
        raise typer.Exit(1)


def _issues_url() -> str | None:
    """测试消息里的「去 Issue 讨论」链接：在 Actions 中指向本仓库的日报列表。"""
    repository = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if not repository:
        return None
    server = os.environ.get("GITHUB_SERVER_URL", "").strip() or "https://github.com"
    return f"{server}/{repository}/issues?q=label%3Adaily"


@app.command("handle-comment")
def handle_comment_command(
    config_path: ConfigOption = None,
    event_path: Annotated[
        Path | None,
        typer.Option("--event", help="issue_comment 事件的 JSON 文件，默认读取 GITHUB_EVENT_PATH"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="只把回复打印到终端：不发评论、不加表情、不写数据分支"),
    ] = False,
    data_dir: DataDirOption = DEFAULT_DATA_DIR,
) -> None:
    """处理 Issue 评论中的指令（由 interact.yml 调用）。"""
    if event_path is None and (env_path := os.environ.get("GITHUB_EVENT_PATH", "").strip()):
        event_path = Path(env_path)
    if event_path is None:
        typer.echo(
            "✗ 没有事件文件：请用 --event 指定，或在 GitHub Actions 中运行（GITHUB_EVENT_PATH）"
        )
        raise typer.Exit(1)
    try:
        event = load_event(event_path)
        replier = _EchoReplier() if dry_run else _github_client(event.repository)
    except (EventError, ConfigError) as e:
        typer.echo(f"✗ {e}", err=True)
        raise typer.Exit(1) from e

    try:
        config = load_config(resolve_config_path(config_path))
    except ConfigError as e:
        result = report_failure(event, replier, f"配置有误，请检查 config.yaml：{short_error(e)}")
    else:
        result = handle_comment(
            event,
            config=config,
            replier=replier,
            branch=DataBranch(Path.cwd(), data_dir),
            services=build_services(config),
            persist=not dry_run,
            pause_seconds=0 if dry_run else 1.0,
        )

    summary = _interact_summary(event, result)
    typer.echo(summary)
    if summary_file := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(f"## Paper Gleaner · Issue {event.issue_number}\n\n{summary}\n")
    if result.status == "failed":
        raise typer.Exit(1)


@app.command("digest")
def digest_command(
    day: Annotated[
        datetime | None,
        typer.Option("--day", formats=["%Y-%m-%d"], help="哪一天的日报，默认最近一份"),
    ] = None,
    show_all: Annotated[bool, typer.Option("--all", help="同时列出打过分但没有推送的候选")] = False,
    config_path: ConfigOption = None,
    data_dir: DataDirOption = DEFAULT_DATA_DIR,
) -> None:
    """打印某一天推送的论文（默认最近一份），读取 gleaner-data，不修改。"""
    _load(config_path)
    store = _read_data(data_dir)
    target = day.date() if day else latest_digest_day(store)
    if target is None:
        typer.echo("✗ gleaner-data 上还没有日报", err=True)
        raise typer.Exit(1)
    try:
        digest = store.read_model(digest_path(target), Digest)
    except ValidationError as e:
        typer.echo(f"✗ {digest_path(target)} 格式不正确：{e}", err=True)
        raise typer.Exit(1) from e
    if digest is None:
        latest = latest_digest_day(store)
        hint = f"，最近一份是 {latest.isoformat()}" if latest else ""
        typer.echo(f"✗ 没有 {target.isoformat()} 的日报{hint}", err=True)
        raise typer.Exit(1)
    typer.echo(_format_digest(digest, store, show_all=show_all))


@app.command("deep")
def deep_command(
    ref: Annotated[str, typer.Argument(help="日报序号、arXiv ID 或 arXiv 链接")],
    force: Annotated[bool, typer.Option("--force", help="忽略缓存，重新生成报告")] = False,
    day: DayOption = None,
    config_path: ConfigOption = None,
    data_dir: DataDirOption = DEFAULT_DATA_DIR,
) -> None:
    """在本地深读一篇论文，把报告打印到终端（不发评论、不写数据分支）。"""
    _run_local(Command("deep", (ref,), force=force), config_path, data_dir, day)


@app.command("compare")
def compare_command(
    refs: Annotated[list[str], typer.Argument(help="两篇以上横向对比；一篇时与相关工作对比")],
    day: DayOption = None,
    config_path: ConfigOption = None,
    data_dir: DataDirOption = DEFAULT_DATA_DIR,
) -> None:
    """在本地对比论文，把结果打印到终端（不发评论、不写数据分支）。"""
    _run_local(Command("compare", tuple(refs)), config_path, data_dir, day)


@app.command("ask")
def ask_command(
    question: Annotated[str, typer.Argument(help="问题，例如「今天哪篇的方法最适合我的检测器？」")],
    day: DayOption = None,
    config_path: ConfigOption = None,
    data_dir: DataDirOption = DEFAULT_DATA_DIR,
) -> None:
    """在本地向 Agent 提问（同 /ask），回答打印到终端（不发评论、不写数据分支）。"""
    _run_local(Command("ask", rest=question), config_path, data_dir, day)


@app.command("summarize-preferences")
def summarize_preferences_command(
    config_path: ConfigOption = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="只打印更新后的内容，不提交数据分支")
    ] = False,
    data_dir: DataDirOption = DEFAULT_DATA_DIR,
) -> None:
    """把近期反馈增量总结进 gleaner-data 上的 learned_preferences.md（由 weekly.yml 调用）。"""
    _, config = _load(config_path)
    try:
        interests = load_interests(config.profile.interests_file)
    except ProfileError as e:
        typer.echo(f"! {e}；这次总结不参考兴趣描述", err=True)
        interests = None
    branch = DataBranch(Path.cwd(), data_dir)
    try:
        result = summarize_preferences(
            branch.checkout(),
            llm_factory=lambda: LLMClient(config.llm),
            interests=interests,
            now=datetime.now(UTC),
            timezone=config.timezone,
            language=config.language,
        )
    except (LLMError, PreferencesError, DataBranchError) as e:
        typer.echo(f"✗ 偏好总结失败，{LEARNED_PREFERENCES_PATH} 保持不变：{e}", err=True)
        raise typer.Exit(1) from e

    if result.status == "updated" and not dry_run:
        try:
            branch.commit_and_push("data: update learned preferences")
        except DataBranchError as e:
            # 只有 weekly 写这个文件；冲突通常是这期间有人在 gleaner-data 上手动改了它
            typer.echo(
                f"✗ {LEARNED_PREFERENCES_PATH} 没有提交：{e}\n"
                "如果这期间你在 gleaner-data 上改过这个文件，你的修改已保留，"
                "重新运行一次即可在它的基础上更新。",
                err=True,
            )
            raise typer.Exit(1) from e

    report = _summary_report(result, dry_run)
    typer.echo(report)
    if summary_file := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(f"## Paper Gleaner · 偏好总结\n\n{_summary_markdown(result, dry_run)}\n")


@app.command("feedback")
def feedback_command(
    days: Annotated[
        int | None, typer.Option("--days", min=1, help="只看最近 N 天里有变化的论文")
    ] = None,
    config_path: ConfigOption = None,
    data_dir: DataDirOption = DEFAULT_DATA_DIR,
) -> None:
    """查看聚合后的反馈、当前屏蔽的主题和学到的偏好（读取 gleaner-data，不修改）。"""
    _, config = _load(config_path)
    now = datetime.now(UTC)
    try:
        store = DataBranch(Path.cwd(), data_dir).checkout()
    except DataBranchError as e:
        typer.echo(f"✗ {e}", err=True)
        raise typer.Exit(1) from e
    memory = load_memory(store, now.astimezone(ZoneInfo(config.timezone)).date())
    typer.echo(_format_memory(memory, now, config.ranking.feedback_half_life_days, days))


def _read_data(data_dir: Path) -> DataStore:
    """同步 gleaner-data 后返回它的目录；同步失败但本地已有数据时用本地的，并给出提示。"""
    branch = DataBranch(Path.cwd(), data_dir)
    try:
        return branch.checkout()
    except DataBranchError as e:
        if not (data_dir / ".git").exists():
            typer.echo(f"✗ {e}", err=True)
            raise typer.Exit(1) from e
        typer.echo(f"! 同步 gleaner-data 失败，使用本地已有的数据：{e}", err=True)
        return DataStore(data_dir)


def _format_digest(digest: Digest, store: DataStore, *, show_all: bool) -> str:
    title = f"📚 {digest.day.isoformat()} 的日报 · 推荐 {len(digest.papers)} 篇"
    if digest.issue_url:
        title += f" · {digest.issue_url}"
    lines = [title]
    if digest.overview:
        lines.append(f"\n{digest.overview}")
    for index, paper in enumerate(digest.papers, start=1):
        lines += ["", _format_paper(index, paper), f"    {paper.url}"]
        if store.read_text(report_path(paper.id)) is not None:
            lines.append(f"    🔍 已有深读报告：gleaner deep {paper.id} 直接读取")
    if digest.others and show_all:
        lines.append(f"\n其余 {len(digest.others)} 篇打过分但没有推送的候选：")
        for index, paper in enumerate(digest.others, start=len(digest.papers) + 1):
            lines += ["", _format_paper(index, paper), f"    {paper.url}"]
    elif digest.others:
        lines.append(f"\n另有 {len(digest.others)} 篇候选没有推送，加 --all 查看。")
    return "\n".join(lines)


def _run_local(
    command: Command, config_path: Path | None, data_dir: Path, day: datetime | None
) -> None:
    _, config = _load(config_path)
    try:
        store = DataBranch(Path.cwd(), data_dir).checkout()
        ctx = Context(
            config,
            build_services(config),
            store,
            day=day.date() if day else latest_digest_day(store),
            persist=False,
        )
        typer.echo(execute(command, ctx))
    except (CommandError, DataBranchError) as e:
        typer.echo(f"✗ {e}", err=True)
        raise typer.Exit(1) from e


def build_services(config: Config) -> Services:
    """命令行和 MCP Server 使用的外部服务；S2_API_KEY 从环境变量读取。"""
    api_key = os.environ.get("S2_API_KEY", "").strip() or None
    return Services(
        arxiv=ArxivSource(config.sources.arxiv),
        fetcher=FulltextFetcher(),
        s2=SemanticScholar(api_key),
        llm_factory=lambda: LLMClient(config.llm),
    )


class _EchoReplier:
    """试运行时代替 GitHub：把回复打印到终端。"""

    def __init__(self) -> None:
        self._next_id = 0

    def create_comment(self, issue_number: int, body: str) -> int:
        self._next_id += 1
        typer.echo(f"----- 回复 Issue {issue_number}（第 {self._next_id} 条）-----\n{body}\n")
        return self._next_id

    def add_reaction(self, comment_id: int, content: str) -> None:
        typer.echo(f"· 给评论 {comment_id} 加 {content}")


def _interact_summary(event: CommentEvent, result: InteractResult) -> str:
    command = result.command.text if result.command else event.comment_body.split("\n")[0]
    labels = {"ignored": "跳过", "done": "✓ 完成", "rejected": "! 未执行", "failed": "✗ 失败"}
    return f"{labels[result.status]} {code(command)}：{result.message}"


def _load(config_path: Path | None) -> tuple[Path, Config]:
    try:
        path = resolve_config_path(config_path)
        if path == EXAMPLE_CONFIG_PATH and config_path is None:
            typer.echo(f"! 未找到 config.yaml，暂用 {EXAMPLE_CONFIG_PATH}", err=True)
        return path, load_config(path)
    except ConfigError as e:
        typer.echo(f"✗ 配置错误：{e}", err=True)
        raise typer.Exit(1) from e


def _build_publisher(config: Config) -> IssuePublisher | None:
    if not config.notify.github_issue.enabled:
        return None
    return IssuePublisher(_github_client(), config.notify.github_issue)


def _reaction_access(
    publisher: IssuePublisher | None,
) -> tuple[GitHubClient | None, str | None, str | None]:
    """表情同步用的 GitHub 客户端和仓库所有者；无法同步时第三项是跳过的原因。

    只统计仓库所有者的表情：优先取 GITHUB_REPOSITORY_OWNER，没有时（如本地运行）从
    GITHUB_REPOSITORY 中取出 owner。
    """
    if publisher is not None:
        github = publisher.github
    else:
        try:
            github = _github_client()
        except ConfigError:
            return None, None, "没有 GITHUB_TOKEN 和 GITHUB_REPOSITORY，跳过表情同步"
    owner = os.environ.get("GITHUB_REPOSITORY_OWNER", "").strip()
    owner = owner or github.repository.partition("/")[0].strip()
    if not owner:
        return None, None, "无法确定仓库所有者（GITHUB_REPOSITORY_OWNER），跳过表情同步"
    return github, owner, None


def _github_client(repository: str = "") -> GitHubClient:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repository = os.environ.get("GITHUB_REPOSITORY", "").strip() or repository
    if not token or not repository:
        raise ConfigError(
            "访问 GitHub 需要 GITHUB_TOKEN 和 GITHUB_REPOSITORY 环境变量"
            "（GitHub Actions 中自动提供）；本地试运行请加 --dry-run"
        )
    api_url = os.environ.get("GITHUB_API_URL", "").strip() or "https://api.github.com"
    return GitHubClient(token, repository, api_url=api_url)


_REPAIR_HINT = "不加 --force 重新运行（或等 11:30 的兜底运行）会自动补发"


def _missing_comments(result: DailyResult, config: Config) -> int:
    """已发布但评论没发出去的论文数。"""
    digest = result.digest
    if digest is None or digest.issue_number is None:
        return 0
    if not config.notify.github_issue.per_paper_comment:
        return 0
    return sum(1 for paper in digest.papers if paper.id not in digest.comment_ids)


def _report(result: DailyResult, missing: int = 0, warnings: Sequence[str] = ()) -> str:
    notes = "\n".join(f"! {warning}" for warning in warnings)
    if result.sync.records:
        notes = "\n".join(filter(None, [f"· 同步了 {result.sync.records} 条表情反馈", notes]))
    body = _report_body(result, missing)
    return f"{notes}\n{body}" if notes else body


def _report_body(result: DailyResult, missing: int) -> str:
    warning = f"! 还有 {missing} 篇论文的评论没有发出去，{_REPAIR_HINT}。" if missing else ""
    url = result.existing.url if result.existing else ""
    if not url and result.digest is not None:
        url = result.digest.issue_url or ""
    pushed = f"· {results_line(result.notified)}" if result.notified else ""
    if result.status == "already_published":
        head = f"今天的日报已经发布：{url}" if url else "今天的日报已经推送过。"
        lines = [head, warning, "如需重新生成，请加 --force。"]
        return "\n".join(line for line in lines if line)
    if result.status == "repaired":
        lines = []
        if result.repaired_comments or not result.notified:
            lines.append(f"✓ 已补发缺失的论文评论：{url}")
        if result.notified:
            lines.append(f"· 补发上次失败的渠道：{results_line(result.notified)}")
        return "\n".join(line for line in [*lines, warning] if line)
    if result.status == "no_announcements":
        return "今天没有新公告（arXiv 周末和节假日不更新）。"

    digest = result.digest
    assert digest is not None
    lines = []
    if result.sampled:
        count = digest.stats.fetched
        lines.append(f"今天没有新公告，改用最近提交的 {count} 篇论文试运行（只用于检查配置）。")
    if result.status == "published":
        if digest.issue_url:
            lines.append(f"✓ 已发布：{digest.issue_url}")
        else:
            lines.append("✓ 已推送（没有启用 GitHub Issue）")
        lines += [line for line in (pushed, warning) if line]
    elif result.status == "nothing_relevant":
        lines.append("今天没有达到 min_score 的论文，未发布日报。")
    if digest.stats.degraded:
        lines.append("! LLM 精排没有成功，以下按向量相似度排序。")
    if digest.overview:
        lines.append(f"\n{digest.overview}")
    for index, paper in enumerate(digest.papers, start=1):
        lines.append("\n" + _format_paper(index, paper))
        if paper.id in digest.deep_reads:
            lines.append(f"    🔍 已自动深读：报告附在评论里，并写入 reports/{paper.id}.md")
    lines.append("\n" + stats_line(digest.stats).replace("<br>", ""))
    return "\n".join(lines).lstrip("\n")


def _format_paper(index: int, paper: ScoredPaper) -> str:
    score = "—" if paper.llm_score is None else f"{paper.llm_score}/10"
    lines = [f"[{index}] {score}  {paper.id}  {paper.title}"]
    if paper.reason:
        lines.append(f"    🎯 {paper.reason}")
    if paper.tldr:
        lines.append(f"    📝 {paper.tldr}")
    if breakdown := score_breakdown(paper):
        lines.append(f"    📊 {breakdown}")
    return "\n".join(lines)


def score_breakdown(paper: ScoredPaper) -> str:
    """粗筛分数的分项，如「粗筛 0.952 = 兴趣 0.850 + 正反馈 0.120 + Zotero 0.040 − 负向 0.058」。"""
    if paper.coarse_score is None or paper.interest_score is None:
        return ""
    text = f"粗筛 {paper.coarse_score:.3f} = 兴趣 {paper.interest_score:.3f}"
    if paper.positive_score is not None:
        text += f" + 正反馈 {paper.positive_score:.3f}"
    if paper.zotero_score is not None:
        text += f" + Zotero {paper.zotero_score:.3f}"
    if paper.negative_score is not None:
        text += f" − 负向 {paper.negative_score:.3f}"
    return text


def _summary_report(result: SummaryResult, dry_run: bool) -> str:
    lines = _summary_lines(result, dry_run)
    if result.text is not None:
        lines += ["", "-----", result.text.rstrip()]
    return "\n".join(lines)


def _summary_markdown(result: SummaryResult, dry_run: bool) -> str:
    """写进 Actions 运行摘要：模型输出是外部内容，说明文字转义，正文放进代码块。"""
    lines = [sanitize(line) for line in _summary_lines(result, dry_run)]
    if result.text is not None:
        fence = "`" * max([3, *(len(run) + 1 for run in re.findall(r"`+", result.text))])
        lines += ["", f"{fence}markdown", result.text.rstrip(), fence]
    return "\n".join(lines)


def _summary_lines(result: SummaryResult, dry_run: bool) -> list[str]:
    if result.status == "no_new_feedback":
        if result.new_records:
            return [
                f"上次总结之后新增的 {result.new_records} 条反馈都已撤销或相互抵消，"
                f"不调用 LLM，{LEARNED_PREFERENCES_PATH} 保持不变。"
            ]
        return [f"上次总结之后没有新反馈，不调用 LLM，{LEARNED_PREFERENCES_PATH} 保持不变。"]
    verb = "试运行：会把" if dry_run else "✓ 已把"
    line = f"{verb} {result.papers} 篇论文的新反馈总结进 {LEARNED_PREFERENCES_PATH}"
    if result.omitted:
        line += f"（另有 {result.omitted} 篇超出输入上限，没有放进 prompt）"
    return [line + ("，没有提交。" if dry_run else "。"), *(f"· {c}" for c in result.changes)]


def _format_memory(memory: Memory, now: datetime, half_life_days: float, days: int | None) -> str:
    feedback = memory.feedback
    if days is not None:
        since = now - timedelta(days=days)
        feedback = [item for item in feedback if item.updated_at and item.updated_at >= since]
    window = f"最近 {days} 天" if days is not None else f"最近 {FEEDBACK_LOOKBACK_MONTHS} 个月"
    lines = [f"反馈（{window}，时间衰减半衰期 {half_life_days} 天）：共 {len(feedback)} 篇论文"]
    for title, items in (
        ("正反馈（A2）", [item for item in feedback if item.total > 0]),
        ("负反馈（A3）", [item for item in feedback if item.total < 0]),
    ):
        items.sort(key=lambda item: item.weight(now, half_life_days), reverse=True)
        lines.append(f"\n{title} {len(items)} 篇：")
        for item in items:
            weight = item.weight(now, half_life_days)
            paper_title = item.title if len(item.title) <= 70 else item.title[:70] + "…"
            lines.append(f"  {item.total:+.1f}  权重 {weight:.2f}  {item.paper_id}  {paper_title}")
            detail = "、".join(item.signals)
            if item.notes:
                detail += " · 原因：" + "；".join(item.notes)
            if not item.has_text:
                detail += " · 只有 ID，不参与粗筛"
            lines.append(f"        {detail}")
    neutral = [item.paper_id for item in feedback if item.total == 0]
    if neutral:
        lines.append(f"\n中性 {len(neutral)} 篇（表情已撤销或反馈相互抵消）：{'、'.join(neutral)}")

    lines.append(f"\n屏蔽主题 {len(memory.muted)} 个：")
    for topic in memory.muted:
        query = topic.query or "（还没有英文描述，下次运行日报时生成）"
        lines.append(f"  {topic.topic} —— {query}")

    lines.append(f"\n学到的偏好（{LEARNED_PREFERENCES_PATH}）：")
    if memory.preferences:
        lines += [f"  {line}" if line else "" for line in memory.preferences.splitlines()]
    else:
        lines.append("  （还没有，运行 gleaner summarize-preferences 或等每周的 Weekly 生成）")
    return "\n".join(lines)
