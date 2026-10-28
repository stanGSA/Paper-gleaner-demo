"""gleaner-data 分支：机器写入的状态（历史、缓存、统计）。

- ``DataStore`` 只负责目录内的文件读写，不涉及 git；
- ``DataBranch`` 把 gleaner-data 分支检出为一个独立的 git worktree，并在运行结束后提交、推送。

状态单独放在孤儿分支上，main 的提交历史不会被每天的自动提交淹没，
用户从上游同步代码时也不会冲突（见 docs/architecture.md 第 4 节）。
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

DATA_BRANCH = "gleaner-data"
# 多个任务同时追加同一个 jsonl 时，rebase 会把双方的行都保留下来
GITATTRIBUTES = "*.jsonl merge=union\n"

logger = logging.getLogger(__name__)
M = TypeVar("M", bound=BaseModel)


class DataBranchError(Exception):
    """检出、提交或推送 gleaner-data 分支失败。"""


class DataStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, relative: str) -> Path:
        return self.root / relative

    def read_text(self, relative: str) -> str | None:
        path = self.path(relative)
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def write_text(self, relative: str, text: str) -> None:
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 统一使用 LF，避免 Windows 上产生无意义的换行符差异
        path.write_text(text, encoding="utf-8", newline="\n")

    def read_json(self, relative: str) -> Any | None:
        text = self.read_text(relative)
        return None if text is None else json.loads(text)

    def write_json(self, relative: str, data: Any) -> None:
        self.write_text(relative, json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    def read_model(self, relative: str, model: type[M]) -> M | None:
        text = self.read_text(relative)
        return None if text is None else model.model_validate_json(text)

    def write_model(self, relative: str, value: BaseModel) -> None:
        self.write_text(relative, value.model_dump_json(indent=2) + "\n")

    def read_jsonl(self, relative: str) -> list[dict[str, Any]]:
        text = self.read_text(relative)
        if text is None:
            return []
        records = []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("跳过 %s 第 %d 行：不是合法的 JSON", relative, number)
        return records

    def append_jsonl(self, relative: str, records: Iterable[dict[str, Any]]) -> None:
        lines = "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in records)
        if not lines:
            return
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(lines)


class DataBranch:
    def __init__(
        self,
        repo: Path,
        worktree: Path,
        *,
        branch: str = DATA_BRANCH,
        remote: str = "origin",
        push_attempts: int = 4,
        backoff_seconds: float = 2.0,
    ) -> None:
        self.repo = repo
        self.worktree = worktree
        self.branch = branch
        self.remote = remote
        self.push_attempts = push_attempts
        self.backoff_seconds = backoff_seconds

    @property
    def _remote_ref(self) -> str:
        return f"refs/remotes/{self.remote}/{self.branch}"

    @property
    def _refspec(self) -> str:
        return f"+refs/heads/{self.branch}:{self._remote_ref}"

    def checkout(self) -> DataStore:
        """检出分支（远端还没有时创建孤儿分支），返回其目录的 DataStore。"""
        remote_exists = self._fetch()
        self._git("worktree", "prune")

        if (self.worktree / ".git").exists():
            # 本地重复运行：丢弃上次未提交的改动（如 dry-run 写下的缓存），与远端对齐
            target = [self._remote_ref] if remote_exists else []
            if remote_exists or self._has_commits(self.worktree):
                self._git("reset", "--hard", "--quiet", *target, cwd=self.worktree)
            self._git("clean", "-fd", "--quiet", cwd=self.worktree)
        elif remote_exists:
            self._git(
                "worktree",
                "add",
                "--quiet",
                "-B",
                self.branch,
                str(self.worktree),
                self._remote_ref,
            )
        elif self._local_branch_exists():
            self._git("worktree", "add", "--quiet", str(self.worktree), self.branch)
        else:
            self._create_orphan()

        store = DataStore(self.worktree)
        if store.read_text(".gitattributes") is None:
            store.write_text(".gitattributes", GITATTRIBUTES)
        return store

    def commit_and_push(self, message: str) -> bool:
        """提交全部改动并推送；没有改动时返回 False。推送被拒时 rebase 后重试。"""
        self._git("add", "--all", cwd=self.worktree)
        if self._git("diff", "--cached", "--quiet", cwd=self.worktree, check=False).returncode == 0:
            return False
        self._git("commit", "--quiet", "-m", message, cwd=self.worktree)
        if not self._has_remote():
            return True

        for attempt in range(1, self.push_attempts + 1):
            target = f"HEAD:refs/heads/{self.branch}"
            push = self._git("push", "--quiet", self.remote, target, cwd=self.worktree, check=False)
            if push.returncode == 0:
                return True
            if attempt == self.push_attempts:
                raise DataBranchError(f"推送 {self.branch} 失败：{push.stderr.strip()}")
            logger.info("推送被拒（第 %d 次），rebase 远端后重试", attempt)
            time.sleep(self.backoff_seconds * 2 ** (attempt - 1))
            self._git("fetch", "--quiet", self.remote, self._refspec, cwd=self.worktree)
            rebase = self._git(
                "rebase", "--quiet", self._remote_ref, cwd=self.worktree, check=False
            )
            if rebase.returncode != 0:
                self._git("rebase", "--abort", cwd=self.worktree, check=False)
                raise DataBranchError(f"rebase {self.branch} 时发生冲突：{rebase.stderr.strip()}")
        return True  # pragma: no cover

    def _fetch(self) -> bool:
        """拉取远端分支，返回远端是否存在该分支。"""
        if not self._has_remote():
            return False
        probe = self._git(
            "ls-remote", "--exit-code", "--heads", self.remote, self.branch, check=False
        )
        if probe.returncode == 2:  # --exit-code：没有匹配的分支
            return False
        if probe.returncode != 0:
            raise DataBranchError(f"无法访问远端 {self.remote}：{probe.stderr.strip()}")
        self._git("fetch", "--quiet", self.remote, self._refspec)
        return True

    def _create_orphan(self) -> None:
        self._git("worktree", "add", "--quiet", "--detach", str(self.worktree))
        self._git("checkout", "--quiet", "--orphan", self.branch, cwd=self.worktree)
        self._git("rm", "-r", "-f", "--quiet", "--ignore-unmatch", ".", cwd=self.worktree)

    def _has_remote(self) -> bool:
        return self._git("remote", "get-url", self.remote, check=False).returncode == 0

    def _local_branch_exists(self) -> bool:
        ref = f"refs/heads/{self.branch}"
        return self._git("rev-parse", "--verify", "--quiet", ref, check=False).returncode == 0

    def _has_commits(self, cwd: Path) -> bool:
        return (
            self._git("rev-parse", "--verify", "--quiet", "HEAD", cwd=cwd, check=False).returncode
            == 0
        )

    def _git(
        self, *args: str, cwd: Path | None = None, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd or self.repo,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if check and result.returncode != 0:
            raise DataBranchError(f"git {' '.join(args)} 失败：{result.stderr.strip()}")
        return result
