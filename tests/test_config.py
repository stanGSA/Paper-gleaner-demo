from pathlib import Path

import pytest

from gleaner.config import (
    Config,
    ConfigError,
    load_config,
    resolve_config_path,
)

REPO_ROOT = Path(__file__).parent.parent


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_example_config_matches_defaults() -> None:
    # 示例配置应与代码中的默认值保持一致，改了一边要记得改另一边
    assert load_config(REPO_ROOT / "config.example.yaml", env={}) == Config()


def test_empty_file_uses_defaults(tmp_path: Path) -> None:
    assert load_config(write(tmp_path, ""), env={}) == Config()


def test_partial_config_merges_with_defaults(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, "sources:\n  arxiv:\n    categories: [cs.CV]\n"), env={})
    assert config.sources.arxiv.categories == ["cs.CV"]
    assert config.sources.arxiv.include_cross_list is True
    assert config.ranking.final_top_n == 10


def test_unknown_key_reports_its_location(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"ranking\.top_n"):
        load_config(write(tmp_path, "ranking:\n  top_n: 5\n"), env={})


def test_top_n_cannot_exceed_top_k(tmp_path: Path) -> None:
    text = "ranking:\n  coarse_top_k: 5\n  final_top_n: 10\n"
    with pytest.raises(ConfigError, match="final_top_n"):
        load_config(write(tmp_path, text), env={})


def test_invalid_timezone(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Mars/Base"):
        load_config(write(tmp_path, "timezone: Mars/Base\n"), env={})


def test_rejects_api_key_in_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        load_config(write(tmp_path, "llm:\n  api_key: sk-xxx\n"), env={})


def test_rejects_non_mapping(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="顶层"):
        load_config(write(tmp_path, "- a\n- b\n"), env={})


def test_rejects_invalid_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="YAML"):
        load_config(write(tmp_path, "llm: [unclosed\n"), env={})


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="不存在"):
        load_config(tmp_path / "nope.yaml", env={})


def test_env_provides_secrets(tmp_path: Path) -> None:
    env = {"LLM_API_KEY": "sk-test", "LLM_BASE_URL": "https://dashscope.example/v1"}
    config = load_config(write(tmp_path, ""), env=env)
    assert config.llm.api_key is not None
    assert config.llm.api_key.get_secret_value() == "sk-test"
    assert config.llm.base_url == "https://dashscope.example/v1"


def test_provider_specific_env_aliases(tmp_path: Path) -> None:
    config = load_config(
        write(tmp_path, ""),
        env={
            "DEEPSEEK_API_KEY": "sk-deepseek",
            "DEEPSEEK_BASE_URL": "https://deepseek.example/v1",
        },
    )
    assert config.llm.api_key is not None
    assert config.llm.api_key.get_secret_value() == "sk-deepseek"
    assert config.llm.base_url == "https://deepseek.example/v1"

    config = load_config(
        write(tmp_path, ""),
        env={
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-openai",
            "OPENAI_BASE_URL": "https://openai.example/v1",
            "OPENAI_FAST_MODEL": "gpt-test-fast",
            "OPENAI_DEEP_MODEL": "gpt-test-deep",
            "DEEPSEEK_API_KEY": "sk-deepseek",
        },
    )
    assert config.llm.api_key is not None
    assert config.llm.api_key.get_secret_value() == "sk-openai"
    assert config.llm.base_url == "https://openai.example/v1"
    assert config.llm.models.fast == "gpt-test-fast"
    assert config.llm.models.deep == "gpt-test-deep"


def test_empty_env_values_are_ignored(tmp_path: Path) -> None:
    # GitHub Actions 中未配置的 Secret 是空字符串
    config = load_config(write(tmp_path, ""), env={"LLM_API_KEY": "", "LLM_BASE_URL": " "})
    assert config.llm.api_key is None
    assert config.llm.base_url == "https://api.deepseek.com"


def test_api_key_is_not_serialized(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, ""), env={"LLM_API_KEY": "sk-test"})
    assert "sk-test" not in config.model_dump_json()
    assert "api_key" not in config.model_dump()["llm"]


def test_resolve_prefers_explicit_path(tmp_path: Path) -> None:
    assert resolve_config_path(tmp_path / "custom.yaml") == tmp_path / "custom.yaml"


def test_resolve_falls_back_to_example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.example.yaml").write_text("")
    assert resolve_config_path() == Path("config.example.yaml")

    (tmp_path / "config.yaml").write_text("")
    assert resolve_config_path() == Path("config.yaml")


def test_resolve_without_any_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="config.example.yaml"):
        resolve_config_path()


def test_embedding_secrets_from_env(tmp_path: Path) -> None:
    env = {"EMBEDDING_API_KEY": "sk-embed", "EMBEDDING_BASE_URL": "https://embed.example/v1"}
    config = load_config(write(tmp_path, "ranking:\n  embedding:\n    provider: api\n"), env=env)
    embedding = config.ranking.embedding
    assert embedding.api_key is not None
    assert embedding.api_key.get_secret_value() == "sk-embed"
    assert embedding.base_url == "https://embed.example/v1"


def test_rejects_embedding_api_key_in_file(tmp_path: Path) -> None:
    text = "ranking:\n  embedding:\n    api_key: sk-xxx\n"
    with pytest.raises(ConfigError, match="EMBEDDING_API_KEY"):
        load_config(write(tmp_path, text), env={})


# --- 推送渠道与 Zotero 的密钥 ---------------------------------------------------

CHANNEL_ENV = {
    "SMTP_HOST": "smtp.qq.com",
    "SMTP_PORT": "587",
    "SMTP_USER": "me@qq.com",
    "SMTP_PASSWORD": "auth-code-123",
    "EMAIL_TO": "a@example.com, b@example.com；c@example.com",
    "FEISHU_WEBHOOK": "https://open.feishu.cn/open-apis/bot/v2/hook/token-abc",
    "FEISHU_SECRET": "sign-secret",
    "WECOM_WEBHOOK": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=wecom-key",
    "ZOTERO_ID": "1234567",
    "ZOTERO_KEY": "zotero-key-xyz",
}


def test_channel_and_zotero_secrets_from_env(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, ""), env=CHANNEL_ENV)

    email = config.notify.email
    assert (email.smtp_host, email.smtp_port) == ("smtp.qq.com", 587)
    assert email.smtp_user is not None and email.smtp_user.get_secret_value() == "me@qq.com"
    assert email.smtp_password is not None
    assert email.smtp_password.get_secret_value() == "auth-code-123"
    assert email.to == ["a@example.com", "b@example.com", "c@example.com"]
    feishu, wecom = config.notify.feishu, config.notify.wecom
    assert feishu.webhook is not None and feishu.webhook.get_secret_value().endswith("token-abc")
    assert feishu.secret is not None and feishu.secret.get_secret_value() == "sign-secret"
    assert wecom.webhook is not None and wecom.webhook.get_secret_value().endswith("key=wecom-key")
    zotero = config.profile.zotero
    assert zotero.library_id is not None and zotero.library_id.get_secret_value() == "1234567"
    assert zotero.api_key is not None and zotero.api_key.get_secret_value() == "zotero-key-xyz"


def test_channel_secrets_are_not_serialized(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, ""), env=CHANNEL_ENV)
    dumped = config.model_dump_json()
    for value in CHANNEL_ENV.values():
        if value not in {"smtp.qq.com", "587"}:
            assert value.split(",")[0] not in dumped
    assert "token-abc" not in repr(config)
    assert "auth-code-123" not in repr(config)


@pytest.mark.parametrize(
    ("text", "env_name"),
    [
        ("notify:\n  email:\n    smtp_password: x\n", "SMTP_PASSWORD"),
        ("notify:\n  email:\n    password: x\n", "SMTP_PASSWORD"),
        ("notify:\n  email:\n    to: [a@b.c]\n", "EMAIL_TO"),
        ("notify:\n  email:\n    smtp_host: smtp.qq.com\n", "SMTP_HOST"),
        ("notify:\n  feishu:\n    webhook: https://x\n", "FEISHU_WEBHOOK"),
        ("notify:\n  feishu:\n    secret: x\n", "FEISHU_SECRET"),
        ("notify:\n  wecom:\n    webhook: https://x\n", "WECOM_WEBHOOK"),
        ("profile:\n  zotero:\n    api_key: x\n", "ZOTERO_KEY"),
        ("profile:\n  zotero:\n    key: x\n", "ZOTERO_KEY"),
        ("profile:\n  zotero:\n    library_id: '1'\n", "ZOTERO_ID"),
    ],
)
def test_rejects_channel_secrets_in_file(tmp_path: Path, text: str, env_name: str) -> None:
    with pytest.raises(ConfigError, match=env_name) as info:
        load_config(write(tmp_path, text), env={})
    assert "https://x" not in str(info.value)


def test_invalid_smtp_port(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="SMTP_PORT"):
        load_config(write(tmp_path, ""), env={"SMTP_PORT": "smtp"})


def test_invalid_email_to_does_not_echo_addresses(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="EMAIL_TO") as info:
        load_config(write(tmp_path, ""), env={"EMAIL_TO": "a@example.com, not-an-address"})
    assert "not-an-address" not in str(info.value)


def test_channel_and_zotero_options(tmp_path: Path) -> None:
    text = (
        "notify:\n"
        "  email: {enabled: true, subject_prefix: '[论文]'}\n"
        "  wecom: {enabled: true, max_papers: 3}\n"
        "profile:\n"
        "  zotero:\n"
        "    enabled: true\n"
        "    library_type: group\n"
        "    collections: [Detection, ABCD2345]\n"
        "    include_subcollections: false\n"
        "    max_items: 50\n"
    )
    config = load_config(write(tmp_path, text), env={})
    assert config.notify.email.subject_prefix == "[论文]"
    assert config.notify.wecom.max_papers == 3
    zotero = config.profile.zotero
    assert (zotero.library_type, zotero.collections) == ("group", ["Detection", "ABCD2345"])
    assert (zotero.include_subcollections, zotero.max_items) == (False, 50)
    with pytest.raises(ConfigError, match="library_type"):
        load_config(write(tmp_path, "profile:\n  zotero:\n    library_type: team\n"), env={})
