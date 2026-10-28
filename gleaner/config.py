"""配置加载与校验。

配置来自两处：
- ``config.yaml``：可以公开的设置，提交在用户自己的 fork 中；
- 环境变量：API Key 等密钥，在 GitHub Actions 中由 Secrets 注入。
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

DEFAULT_CONFIG_PATH = Path("config.yaml")
EXAMPLE_CONFIG_PATH = Path("config.example.yaml")

# 只能来自环境变量（GitHub Secrets）的配置项：配置文件中的位置 → 环境变量。
# 除了模型中实际存在的字段，也列出几个容易写错的名字，给出更明确的提示
_SECRET_FIELDS: dict[tuple[str, ...], str] = {
    ("llm", "api_key"): "LLM_API_KEY",
    ("ranking", "embedding", "api_key"): "EMBEDDING_API_KEY",
    ("notify", "email", "smtp_host"): "SMTP_HOST",
    ("notify", "email", "smtp_port"): "SMTP_PORT",
    ("notify", "email", "smtp_user"): "SMTP_USER",
    ("notify", "email", "smtp_password"): "SMTP_PASSWORD",
    ("notify", "email", "password"): "SMTP_PASSWORD",
    ("notify", "email", "to"): "EMAIL_TO",
    ("notify", "feishu", "webhook"): "FEISHU_WEBHOOK",
    ("notify", "feishu", "secret"): "FEISHU_SECRET",
    ("notify", "wecom", "webhook"): "WECOM_WEBHOOK",
    ("profile", "zotero", "library_id"): "ZOTERO_ID",
    ("profile", "zotero", "id"): "ZOTERO_ID",
    ("profile", "zotero", "api_key"): "ZOTERO_KEY",
    ("profile", "zotero", "key"): "ZOTERO_KEY",
}
_EMAIL_SEPARATOR = re.compile(r"[\s,;，；]+")


class ConfigError(Exception):
    """配置缺失或不合法。"""


class _Section(BaseModel):
    # 拒绝未知字段，拼错的配置项会直接报错，而不是被悄悄忽略
    model_config = ConfigDict(extra="forbid")


class ArxivConfig(_Section):
    categories: list[str] = Field(default_factory=lambda: ["cs.CL", "cs.LG", "cs.AI"], min_length=1)
    include_cross_list: bool = True
    include_replace: bool = False


class SourcesConfig(_Section):
    arxiv: ArxivConfig = Field(default_factory=ArxivConfig)


class ZoteroConfig(_Section):
    """Zotero 文库作为可选的兴趣锚点（docs 3.1.1 的 A4）。

    文库内容只缓存在本机（Actions 缓存）里，不写进 gleaner-data。
    """

    enabled: bool = False
    # 个人文库为 user，群组文库为 group；ZOTERO_ID 相应地填用户 ID 或群组 ID
    library_type: Literal["user", "group"] = "user"
    # 限定范围的 collection，写名称或 key；为空时使用整个文库
    collections: list[str] = Field(default_factory=list)
    include_subcollections: bool = True
    # 最多取多少篇作为锚点：按加入文库的时间从新到旧，在各 collection 之间轮流取
    max_items: int = Field(200, ge=1)
    # 只从环境变量 ZOTERO_ID / ZOTERO_KEY 读取
    library_id: SecretStr | None = Field(default=None, exclude=True)
    api_key: SecretStr | None = Field(default=None, exclude=True)


class ProfileConfig(_Section):
    interests_file: Path = Path("profile/interests.md")
    zotero: ZoteroConfig = Field(default_factory=ZoteroConfig)


class FiltersConfig(_Section):
    include_keywords: list[str] = Field(default_factory=list)
    exclude_keywords: list[str] = Field(default_factory=list)


class RankingWeights(_Section):
    interest: float = 1.0
    positive: float = 0.6
    zotero: float = 0.5
    negative: float = 0.8


class EmbeddingConfig(_Section):
    provider: Literal["local", "api"] = "local"
    model: str = "BAAI/bge-small-en-v1.5"
    # provider 为 api 时使用；未设置时沿用 LLM 的 base_url 和 API Key
    base_url: str | None = None
    # 只从环境变量 EMBEDDING_API_KEY 读取
    api_key: SecretStr | None = Field(default=None, exclude=True)


class RankingConfig(_Section):
    coarse_top_k: int = Field(50, ge=1)
    final_top_n: int = Field(10, ge=1)
    min_score: int = Field(6, ge=0, le=10)
    weights: RankingWeights = Field(default_factory=RankingWeights)
    feedback_half_life_days: int = Field(90, ge=1)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    @model_validator(mode="after")
    def _top_n_within_top_k(self) -> RankingConfig:
        if self.final_top_n > self.coarse_top_k:
            raise ValueError("final_top_n 不能大于 coarse_top_k")
        return self


class DeepReadConfig(_Section):
    auto_top_m: int = Field(1, ge=0)
    max_input_tokens: int = Field(30_000, ge=1_000)
    # 深读、对比单次 LLM 调用的超时：输入长达几万 token，推理模型往往要一两分钟
    timeout_seconds: float = Field(180, gt=0)


class AskConfig(_Section):
    """/ask 的工具调用循环（docs 3.2）。"""

    max_steps: int = Field(8, ge=1, le=20)
    # 单条 /ask 指令的 token 上限（各步输入加输出，按服务商返回的用量累计）；
    # 接近上限时停止调用工具，基于已有信息作答
    max_tokens: int = Field(60_000, ge=10_000)


class LLMModels(_Section):
    fast: str = "deepseek-chat"
    deep: str = "deepseek-reasoner"


class LLMConfig(_Section):
    base_url: str = "https://api.deepseek.com"
    models: LLMModels = Field(default_factory=LLMModels)
    daily_token_budget: int | None = Field(500_000, ge=1)
    timeout_seconds: float = Field(60, gt=0)
    max_retries: int = Field(3, ge=0)
    # 只从环境变量 LLM_API_KEY 读取，不允许写在配置文件里
    api_key: SecretStr | None = Field(default=None, exclude=True)


class GithubIssueNotifyConfig(_Section):
    enabled: bool = True
    per_paper_comment: bool = True
    close_after_days: int | None = Field(7, ge=1)


class EmailNotifyConfig(_Section):
    enabled: bool = False
    subject_prefix: str = "📚 Paper Gleaner"
    # 以下只从环境变量读取：SMTP_HOST、SMTP_PORT（默认 465）、SMTP_USER、SMTP_PASSWORD、
    # EMAIL_TO（多个地址用逗号分隔）。收件人地址也算隐私：公开仓库的 Actions 日志任何人都能看
    smtp_host: str | None = Field(default=None, exclude=True)
    smtp_port: int | None = Field(default=None, exclude=True)
    smtp_user: SecretStr | None = Field(default=None, exclude=True)
    smtp_password: SecretStr | None = Field(default=None, exclude=True)
    to: list[str] = Field(default_factory=list, exclude=True)


class FeishuNotifyConfig(_Section):
    enabled: bool = False
    max_papers: int = Field(10, ge=1)  # 卡片中最多列出的篇数（请求体还受飞书 20 KB 的限制）
    # 只从环境变量 FEISHU_WEBHOOK / FEISHU_SECRET 读取；Webhook 地址本身就是密钥
    webhook: SecretStr | None = Field(default=None, exclude=True)
    secret: SecretStr | None = Field(default=None, exclude=True)


class WecomNotifyConfig(_Section):
    enabled: bool = False
    max_papers: int = Field(5, ge=1)  # markdown 消息最多 4096 字节，只发前几篇和链接
    # 只从环境变量 WECOM_WEBHOOK 读取
    webhook: SecretStr | None = Field(default=None, exclude=True)


class NotifyConfig(_Section):
    github_issue: GithubIssueNotifyConfig = Field(default_factory=GithubIssueNotifyConfig)
    email: EmailNotifyConfig = Field(default_factory=EmailNotifyConfig)
    feishu: FeishuNotifyConfig = Field(default_factory=FeishuNotifyConfig)
    wecom: WecomNotifyConfig = Field(default_factory=WecomNotifyConfig)


class Config(_Section):
    language: str = "zh"
    timezone: str = "Asia/Shanghai"
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    deep_read: DeepReadConfig = Field(default_factory=DeepReadConfig)
    ask: AskConfig = Field(default_factory=AskConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    notify: NotifyConfig = Field(default_factory=NotifyConfig)

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as e:
            raise ValueError(f"未知时区：{value}") from e
        return value


def resolve_config_path(path: Path | None = None) -> Path:
    """确定要使用的配置文件：显式指定 > config.yaml > config.example.yaml。"""
    if path is not None:
        return path
    for candidate in (DEFAULT_CONFIG_PATH, EXAMPLE_CONFIG_PATH):
        if candidate.is_file():
            return candidate
    raise ConfigError(f"找不到 {DEFAULT_CONFIG_PATH}，请从 {EXAMPLE_CONFIG_PATH} 复制一份")


def load_config(path: Path, env: Mapping[str, str] | None = None) -> Config:
    """读取并校验配置文件，再用环境变量补充密钥。"""
    if env is None:
        # Load a repo-local .env for local development; explicit process variables win.
        load_dotenv(dotenv_path=path.parent / ".env", override=False)
        env = os.environ

    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ConfigError(f"配置文件不存在：{path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} 不是合法的 YAML：{e}") from e

    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} 的顶层必须是键值对")
    _reject_secrets(raw)

    try:
        config = Config.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(_format_validation_error(path, e)) from e

    provider = env.get("LLM_PROVIDER", "").strip().lower()
    if provider not in {"", "deepseek", "openai"}:
        raise ConfigError("LLM_PROVIDER 必须是 deepseek 或 openai")

    # GitHub Actions 中未配置的 Secret 会以空字符串传入，按未设置处理。
    # LLM_* 保持兼容；本地 .env 也可按供应商分别使用 *_API_KEY / *_BASE_URL。
    provider_prefix = provider.upper() if provider else "DEEPSEEK"
    key_names = ["LLM_API_KEY", f"{provider_prefix}_API_KEY"]
    base_url_names = ["LLM_BASE_URL", f"{provider_prefix}_BASE_URL"]
    if not provider:
        key_names += ["OPENAI_API_KEY"]
        base_url_names += ["OPENAI_BASE_URL"]

    if api_key := _first_env(env, key_names):
        config.llm.api_key = SecretStr(api_key)
    if base_url := _first_env(env, base_url_names):
        config.llm.base_url = base_url
    model_prefix = provider.upper() if provider else "DEEPSEEK"
    for tier in ("fast", "deep"):
        model_names = [f"LLM_{tier.upper()}_MODEL", f"{model_prefix}_{tier.upper()}_MODEL"]
        if model := _first_env(env, model_names):
            setattr(config.llm.models, tier, model)

    if api_key := _first_env(env, ["EMBEDDING_API_KEY"]):
        config.ranking.embedding.api_key = SecretStr(api_key)
    if base_url := _first_env(env, ["EMBEDDING_BASE_URL"]):
        config.ranking.embedding.base_url = base_url
    _load_channel_secrets(config.notify, env)
    zotero = config.profile.zotero
    if library_id := _first_env(env, ["ZOTERO_ID"]):
        zotero.library_id = SecretStr(library_id)
    if api_key := _first_env(env, ["ZOTERO_KEY"]):
        zotero.api_key = SecretStr(api_key)
    return config


def _reject_secrets(raw: dict[str, Any]) -> None:
    """配置文件里写了密钥就报错：config.yaml 提交在公开的 fork 中。"""
    for path, env_name in _SECRET_FIELDS.items():
        node: Any = raw
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict) and path[-1] in node:
            raise ConfigError(
                f"不要把密钥写进配置文件：{'.'.join(path)} 请删掉，"
                f"改用环境变量 {env_name}（GitHub Secret）"
            )


def _load_channel_secrets(notify: NotifyConfig, env: Mapping[str, str]) -> None:
    email = notify.email
    if host := _first_env(env, ["SMTP_HOST"]):
        email.smtp_host = host
    if port := _first_env(env, ["SMTP_PORT"]):
        if not port.isdigit() or not 0 < int(port) < 65536:
            raise ConfigError("SMTP_PORT 必须是端口号，如 465（SSL）或 587（STARTTLS）")
        email.smtp_port = int(port)
    if user := _first_env(env, ["SMTP_USER"]):
        email.smtp_user = SecretStr(user)
    if password := _first_env(env, ["SMTP_PASSWORD"]):
        email.smtp_password = SecretStr(password)
    if recipients := _first_env(env, ["EMAIL_TO"]):
        email.to = [item for item in _EMAIL_SEPARATOR.split(recipients) if item]
        if invalid := [item for item in email.to if "@" not in item]:
            # 不在报错信息里写出地址本身：这条信息可能出现在公开的 Actions 日志里
            raise ConfigError(f"EMAIL_TO 中有 {len(invalid)} 个不是邮箱地址，多个地址用逗号分隔")
    if webhook := _first_env(env, ["FEISHU_WEBHOOK"]):
        notify.feishu.webhook = SecretStr(webhook)
    if secret := _first_env(env, ["FEISHU_SECRET"]):
        notify.feishu.secret = SecretStr(secret)
    if webhook := _first_env(env, ["WECOM_WEBHOOK"]):
        notify.wecom.webhook = SecretStr(webhook)


def _first_env(env: Mapping[str, str], names: list[str]) -> str:
    for name in names:
        if value := env.get(name, "").strip():
            return value
    return ""


def _format_validation_error(path: Path, error: ValidationError) -> str:
    lines = [f"{path} 校验失败："]
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "(顶层)"
        lines.append(f"  - {location}: {item['msg']}")
    return "\n".join(lines)
