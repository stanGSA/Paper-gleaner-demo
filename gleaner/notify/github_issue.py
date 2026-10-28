"""把日报发布成 GitHub Issue：正文是总览，每篇论文一条评论（方便逐篇点 👍/👎）。"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

import httpx

from gleaner.config import GithubIssueNotifyConfig
from gleaner.models import DeepReport, Digest, ScoredPaper
from gleaner.render import issue_body, issue_title, issue_title_prefix, paper_section

LABELS = {
    "gleaner": ("1f6feb", "Paper Gleaner 自动生成"),
    "daily": ("0e8a16", "每日论文推送"),
}
# 与 render.paper_section 写在每条论文评论末尾的标记对应
_PAPER_MARKER = re.compile(r"<!-- gleaner:paper id=(\S+) index=\d+ -->")
_SERVER_ERRORS = {500, 502, 503, 504}

logger = logging.getLogger(__name__)


class GitHubError(Exception):
    """调用 GitHub API 失败。status 是 HTTP 状态码。"""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class IssueRef:
    number: int
    url: str


class GitHubClient:
    """GitHub REST API 的最小封装，只包含日报、指令交互和表情同步需要的几个接口。"""

    def __init__(
        self,
        token: str,
        repository: str,
        *,
        api_url: str = "https://api.github.com",
        client: httpx.Client | None = None,
        max_attempts: int = 3,
        backoff_seconds: float = 2.0,
    ) -> None:
        self.repository = repository
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self._client = client or httpx.Client(
            base_url=api_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
            transport=httpx.HTTPTransport(retries=3),
        )

    def ensure_label(self, name: str, color: str, description: str) -> None:
        response = self._request(
            "POST",
            f"/repos/{self.repository}/labels",
            json={"name": name, "color": color, "description": description},
        )
        if response.status_code == 422:  # 标签已存在
            return
        self._check(response, "创建标签")

    def list_issues(self, labels: list[str], *, state: str = "all") -> list[dict[str, Any]]:
        items = self._paginate(
            f"/repos/{self.repository}/issues",
            "查询 Issue",
            params={
                "labels": ",".join(labels),
                "state": state,
                "sort": "created",
                "direction": "desc",
            },
        )
        # 这个接口也会返回 Pull Request，需要排除
        return [item for item in items if "pull_request" not in item]

    def create_issue(self, title: str, body: str, labels: list[str]) -> IssueRef:
        response = self._request(
            "POST",
            f"/repos/{self.repository}/issues",
            json={"title": title, "body": body, "labels": labels},
        )
        data = self._check(response, "创建 Issue").json()
        return IssueRef(number=data["number"], url=data["html_url"])

    def list_comments(self, issue_number: int) -> list[dict[str, Any]]:
        return self._paginate(
            f"/repos/{self.repository}/issues/{issue_number}/comments", "查询评论"
        )

    def list_reactions(self, comment_id: int) -> list[dict[str, Any]]:
        """Issue 评论上的全部表情，每项包含 content（+1、-1、eyes 等）和 user.login。"""
        return self._paginate(
            f"/repos/{self.repository}/issues/comments/{comment_id}/reactions", "查询表情"
        )

    def create_comment(self, issue_number: int, body: str) -> int:
        response = self._request(
            "POST", f"/repos/{self.repository}/issues/{issue_number}/comments", json={"body": body}
        )
        return self._check(response, "发表评论").json()["id"]

    def add_reaction(self, comment_id: int, content: str) -> None:
        """给 Issue 评论加表情（eyes、rocket、confused 等）；加过的再加一次不会重复。"""
        response = self._request(
            "POST",
            f"/repos/{self.repository}/issues/comments/{comment_id}/reactions",
            json={"content": content},
        )
        self._check(response, "添加表情")

    def close_issue(self, issue_number: int) -> None:
        response = self._request(
            "PATCH",
            f"/repos/{self.repository}/issues/{issue_number}",
            json={"state": "closed", "state_reason": "completed"},
        )
        self._check(response, "关闭 Issue")

    def update_issue(
        self, issue_number: int, *, body: str, state: Literal["open", "closed"]
    ) -> None:
        response = self._request(
            "PATCH",
            f"/repos/{self.repository}/issues/{issue_number}",
            json={
                "body": body,
                "state": state,
                "state_reason": "completed" if state == "closed" else "reopened",
            },
        )
        self._check(response, "更新引导 Issue")

    def _paginate(
        self, url: str, action: str, *, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """按页读取列表接口，直到某一页不满 100 条。"""
        items: list[dict[str, Any]] = []
        page = 1
        while True:
            response = self._request(
                "GET", url, params={**(params or {}), "per_page": 100, "page": page}
            )
            batch = self._check(response, action).json()
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """发送请求；遇到限流或（幂等请求的）服务端错误时等待后重试。"""
        for attempt in range(self.max_attempts):
            response = self._client.request(method, url, **kwargs)
            delay = self._retry_delay(method, response, attempt)
            if delay is None or attempt == self.max_attempts - 1:
                return response
            logger.info("%s %s 返回 %d，%.0f 秒后重试", method, url, response.status_code, delay)
            time.sleep(delay)
        raise AssertionError("unreachable")

    def _retry_delay(self, method: str, response: httpx.Response, attempt: int) -> float | None:
        status = response.status_code
        rate_limited = status == 429 or (
            status == 403
            and (
                response.headers.get("x-ratelimit-remaining") == "0"
                or "rate limit" in response.text.lower()
            )
        )
        # 限流时请求没有被处理，任何请求都可以重试；服务端错误时 POST 可能已经生效，
        # 盲目重试会产生重复的 Issue 或评论，所以只重试 GET / PATCH
        if not rate_limited and not (status in _SERVER_ERRORS and method in {"GET", "PATCH"}):
            return None
        retry_after = response.headers.get("retry-after", "")
        if retry_after.isdigit():
            return min(float(retry_after), 60.0)
        return self.backoff_seconds * 2**attempt

    @staticmethod
    def _check(response: httpx.Response, action: str) -> httpx.Response:
        if response.is_success:
            return response
        try:
            message = response.json().get("message", "")
        except ValueError:
            message = response.text[:200]
        raise GitHubError(
            f"{action}失败（HTTP {response.status_code}）：{message}", response.status_code
        )


class IssuePublisher:
    def __init__(
        self,
        github: GitHubClient,
        config: GithubIssueNotifyConfig,
        *,
        pause_seconds: float = 1.0,
    ) -> None:
        self.github = github
        self.config = config
        # GitHub 建议连续创建内容时每次间隔约 1 秒，避免触发次级限流
        self.pause_seconds = pause_seconds

    def find_existing(self, day: date) -> IssueRef | None:
        prefix = issue_title_prefix(day)
        for issue in self.github.list_issues(list(LABELS)):
            if issue["title"].startswith(prefix):
                return IssueRef(number=issue["number"], url=issue["html_url"])
        return None

    def publish(self, digest: Digest, reports: Mapping[str, DeepReport] | None = None) -> Digest:
        """创建当日 Issue 和每篇论文的评论，返回填好 Issue 信息的 digest。

        reports 是自动深读的报告（论文 ID → 报告），附在对应论文的评论里。
        单条评论失败不中断发布；发完后再补发一轮，仍失败的留给下一次运行（见 repair）。
        """
        for name, (color, description) in LABELS.items():
            self.github.ensure_label(name, color, description)

        body = issue_body(digest, per_paper_comment=self.config.per_paper_comment)
        issue = self.github.create_issue(issue_title(digest), body, list(LABELS))
        digest = digest.model_copy(update={"issue_number": issue.number, "issue_url": issue.url})
        if not self.config.per_paper_comment:
            return digest

        comment_ids: dict[str, int] = {}
        papers = list(enumerate(digest.papers, start=1))
        self._post_comments(issue.number, papers, comment_ids, reports or {})
        return self.repair(digest.model_copy(update={"comment_ids": comment_ids}), reports)

    def repair(self, digest: Digest, reports: Mapping[str, DeepReport] | None = None) -> Digest:
        """补发 digest 中缺失的论文评论，返回更新了 comment_ids 的 digest。

        先读取 Issue 上已有的评论：上次的请求可能已经生效、只是没收到响应，
        按评论末尾的标记识别，避免重复发布。
        """
        if not self.config.per_paper_comment or digest.issue_number is None:
            return digest
        missing = [
            (index, paper)
            for index, paper in enumerate(digest.papers, start=1)
            if paper.id not in digest.comment_ids
        ]
        if not missing:
            return digest

        comment_ids = dict(digest.comment_ids)
        try:
            for comment in self.github.list_comments(digest.issue_number):
                if match := _PAPER_MARKER.search(comment.get("body") or ""):
                    comment_ids.setdefault(match.group(1), comment["id"])
        except (GitHubError, httpx.HTTPError) as e:
            logger.warning("读取 Issue 评论失败，暂不补发：%s", e)
            return digest
        self._post_comments(
            digest.issue_number,
            [(index, paper) for index, paper in missing if paper.id not in comment_ids],
            comment_ids,
            reports or {},
        )
        return digest.model_copy(update={"comment_ids": comment_ids})

    def _post_comments(
        self,
        issue_number: int,
        papers: list[tuple[int, ScoredPaper]],
        comment_ids: dict[str, int],
        reports: Mapping[str, DeepReport],
    ) -> None:
        for index, paper in papers:
            time.sleep(self.pause_seconds)
            try:
                comment_ids[paper.id] = self.github.create_comment(
                    issue_number, paper_section(index, paper, reports.get(paper.id))
                )
            except (GitHubError, httpx.HTTPError) as e:
                logger.warning("第 %d 篇论文的评论发布失败：%s", index, e)

    def close_stale(self, now: datetime | None = None) -> list[int]:
        """关闭超过 close_after_days 天的旧日报，返回关闭的 Issue 编号。关闭后仍可评论。"""
        if self.config.close_after_days is None:
            return []
        cutoff = (now or datetime.now(UTC)) - timedelta(days=self.config.close_after_days)
        closed: list[int] = []
        try:
            for issue in self.github.list_issues(list(LABELS), state="open"):
                created = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
                if created < cutoff:
                    self.github.close_issue(issue["number"])
                    closed.append(issue["number"])
        except (GitHubError, httpx.HTTPError) as e:
            # 关闭旧日报只是整理，失败不影响当天的推送
            logger.warning("关闭旧日报失败：%s", e)
        return closed
