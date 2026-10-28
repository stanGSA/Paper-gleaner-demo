"""Claude Code Skill：frontmatter 合法，里面引用的命令都存在。"""

import re
import tomllib
from pathlib import Path

import typer.main
import yaml

from gleaner.cli import app

REPO_ROOT = Path(__file__).parent.parent
SKILL = REPO_ROOT / ".claude" / "skills" / "paper-gleaner" / "SKILL.md"


def frontmatter(text: str) -> tuple[dict, str]:
    match = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", text, re.DOTALL)
    assert match is not None, "SKILL.md 必须以 YAML frontmatter 开头"
    return yaml.safe_load(match.group(1)), match.group(2)


def test_frontmatter() -> None:
    meta, body = frontmatter(SKILL.read_text(encoding="utf-8"))

    assert set(meta) == {"name", "description"}
    assert meta["name"] == SKILL.parent.name == "paper-gleaner"
    assert re.fullmatch(r"[a-z0-9-]{1,64}", meta["name"])
    description = meta["description"]
    assert 50 < len(description) <= 1024
    assert "今天推了什么" in description  # 说明什么时候用这个 Skill
    assert body.lstrip().startswith("# Paper Gleaner")


def test_referenced_commands_exist() -> None:
    text = SKILL.read_text(encoding="utf-8")
    commands = set(typer.main.get_command(app).commands)  # type: ignore[attr-defined]

    referenced = set(re.findall(r"\bgleaner ([a-z][a-z-]*)", text))

    assert referenced >= {"digest", "deep", "compare", "ask", "feedback"}
    assert referenced <= commands, referenced - commands
    for option in re.findall(r"gleaner digest (--[a-z]+)", text):
        assert option in {"--day", "--all"}
    scripts = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if "gleaner-mcp" in text:
        assert "gleaner-mcp" in scripts["project"]["scripts"]
