from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import BaseModel

from gleaner.store import DataBranch, DataBranchError, DataStore
from tests.conftest import git


class Item(BaseModel):
    name: str
    score: int


def branch_for(repo: Path) -> DataBranch:
    return DataBranch(repo, repo / ".gleaner-data", backoff_seconds=0)


def remote_file(remote: Path, path: str) -> str:
    return git("show", f"gleaner-data:{path}", cwd=remote)


# --- DataStore ---------------------------------------------------------------


def test_store_missing_files(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    assert store.read_text("a.txt") is None
    assert store.read_json("a.json") is None
    assert store.read_model("a.json", Item) is None
    assert store.read_jsonl("a.jsonl") == []


def test_store_round_trips(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    store.write_json("nested/dir/data.json", {"标题": "中文", "n": 1})
    store.write_model("item.json", Item(name="x", score=3))

    assert store.read_json("nested/dir/data.json") == {"标题": "中文", "n": 1}
    assert "中文" in store.read_text("nested/dir/data.json")  # 不转义为 \uXXXX
    assert store.read_model("item.json", Item) == Item(name="x", score=3)
    assert b"\r\n" not in (tmp_path / "item.json").read_bytes()


def test_store_jsonl(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    store.append_jsonl("log.jsonl", [{"a": 1}, {"a": 2}])
    store.append_jsonl("log.jsonl", [])
    store.append_jsonl("log.jsonl", [{"a": 3}])
    with (tmp_path / "log.jsonl").open("a", encoding="utf-8") as f:
        f.write("not json\n\n")

    assert store.read_jsonl("log.jsonl") == [{"a": 1}, {"a": 2}, {"a": 3}]


# --- DataBranch --------------------------------------------------------------


def test_checkout_creates_orphan_branch(clone: Callable[[str], Path], git_remote: Path) -> None:
    repo = clone("repo")
    branch = branch_for(repo)

    store = branch.checkout()
    assert sorted(p.name for p in store.root.iterdir() if p.name != ".git") == [".gitattributes"]

    store.write_json("history/a.json", {"x": 1})
    assert branch.commit_and_push("data: first") is True

    assert remote_file(git_remote, "history/a.json") == '{\n  "x": 1\n}'
    assert "merge=union" in remote_file(git_remote, ".gitattributes")
    # 孤儿分支：只有一个提交，且不包含 main 的文件
    assert git("rev-list", "--count", "gleaner-data", cwd=git_remote) == "1"
    files = git("ls-tree", "-r", "--name-only", "gleaner-data", cwd=git_remote).split()
    assert "README.md" not in files
    # main 的工作区不受影响
    assert (repo / "README.md").is_file()
    assert git("branch", "--show-current", cwd=repo) == "main"


def test_commit_without_changes(clone: Callable[[str], Path]) -> None:
    repo = clone("repo")
    branch = branch_for(repo)
    branch.checkout().write_text("a.txt", "a")
    assert branch.commit_and_push("first") is True
    assert branch.commit_and_push("again") is False


def test_another_clone_sees_pushed_data(clone: Callable[[str], Path]) -> None:
    first = branch_for(clone("first"))
    first.checkout().write_text("note.txt", "hello\n")
    first.commit_and_push("data")

    store = branch_for(clone("second")).checkout()

    assert store.read_text("note.txt") == "hello\n"


def test_concurrent_appends_are_merged(clone: Callable[[str], Path], git_remote: Path) -> None:
    first = branch_for(clone("first"))
    first.checkout().append_jsonl("feedback.jsonl", [{"n": 0}])
    first.commit_and_push("first")

    a, b = branch_for(clone("a")), branch_for(clone("b"))
    store_a, store_b = a.checkout(), b.checkout()
    store_a.append_jsonl("feedback.jsonl", [{"n": 1}])
    store_b.append_jsonl("feedback.jsonl", [{"n": 2}])

    a.commit_and_push("a")
    b.commit_and_push("b")  # 第一次推送被拒，rebase 后成功

    lines = remote_file(git_remote, "feedback.jsonl").splitlines()
    assert sorted(lines) == ['{"n": 0}', '{"n": 1}', '{"n": 2}']


def test_rerun_discards_uncommitted_and_pulls_updates(clone: Callable[[str], Path]) -> None:
    repo = clone("repo")
    branch = branch_for(repo)
    store = branch.checkout()
    store.write_text("kept.txt", "v1")
    branch.commit_and_push("v1")

    other = branch_for(clone("other"))
    other.checkout().write_text("kept.txt", "v2")
    other.commit_and_push("v2")

    store.write_text("dry-run-cache.json", "{}")
    store = branch.checkout()

    assert store.read_text("kept.txt") == "v2"
    assert store.read_text("dry-run-cache.json") is None


def test_recreates_deleted_worktree(clone: Callable[[str], Path]) -> None:
    repo = clone("repo")
    branch = branch_for(repo)
    branch.checkout().write_text("a.txt", "a")
    branch.commit_and_push("a")

    git("worktree", "remove", "--force", str(branch.worktree), cwd=repo)
    assert branch.checkout().read_text("a.txt") == "a"


def test_local_repo_without_remote(tmp_path: Path) -> None:
    repo = tmp_path / "local"
    repo.mkdir()
    git("init", "--quiet", "--initial-branch=main", cwd=repo)
    git("config", "user.name", "Test", cwd=repo)
    git("config", "user.email", "test@example.com", cwd=repo)
    git("config", "commit.gpgsign", "false", cwd=repo)
    (repo / "README.md").write_text("x")
    git("add", ".", cwd=repo)
    git("commit", "--quiet", "-m", "init", cwd=repo)

    branch = branch_for(repo)
    branch.checkout().write_text("a.txt", "a")

    assert branch.commit_and_push("local only") is True
    assert git("show", "gleaner-data:a.txt", cwd=repo) == "a"
    # 再次检出时保留本地已提交的数据
    assert branch.checkout().read_text("a.txt") == "a"


def test_unreachable_remote(clone: Callable[[str], Path], tmp_path: Path) -> None:
    repo = clone("repo")
    git("remote", "set-url", "origin", str(tmp_path / "missing.git"), cwd=repo)
    with pytest.raises(DataBranchError, match="无法访问远端"):
        branch_for(repo).checkout()
