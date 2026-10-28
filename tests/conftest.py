import json
import re
import subprocess
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pymupdf
import pytest

from gleaner.agent import Services
from gleaner.config import ArxivConfig, LLMConfig
from gleaner.llm import LLMClient
from gleaner.models import DeepReport, Digest, Paper, ScoredPaper
from gleaner.pipelines.daily import digest_path
from gleaner.profile import command_feedback, record_feedback
from gleaner.ranking.embedder import normalize
from gleaner.reader import FulltextFetcher, save_report
from gleaner.sources import ArxivSource, SemanticScholar
from gleaner.sources.arxiv import ARXIV_RATE_LIMITER, RateLimiter

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def no_arxiv_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    # arXiv 要求请求间隔 3 秒，测试中不等待
    monkeypatch.setattr(ARXIV_RATE_LIMITER, "interval", 0.0)


@pytest.fixture
def rss_bytes() -> bytes:
    return (FIXTURES / "arxiv_rss.xml").read_bytes()


@pytest.fixture
def empty_rss_bytes() -> bytes:
    return (FIXTURES / "arxiv_rss_empty.xml").read_bytes()


@pytest.fixture
def api_bytes() -> bytes:
    return (FIXTURES / "arxiv_api.xml").read_bytes()


@pytest.fixture
def html_text() -> str:
    return (FIXTURES / "arxiv_html.html").read_text(encoding="utf-8")


def make_pdf(pages: list[list[str]], links: tuple[str, ...] = ()) -> bytes:
    """用 PyMuPDF 现场生成 PDF：每个字符串是一个文本块，links 作为第一页的链接注释。"""
    document = pymupdf.open()
    for blocks in pages:
        page = document.new_page()
        y = 40.0
        for text in blocks:
            height = 13.0 * (text.count("\n") + len(text) // 80 + 2)
            page.insert_textbox(pymupdf.Rect(50, y, 560, y + height), text, fontsize=10)
            y += height + 6
    for number, uri in enumerate(links):
        rect = pymupdf.Rect(50, 700 + number * 12, 300, 710 + number * 12)
        document[0].insert_link({"kind": pymupdf.LINK_URI, "from": rect, "uri": uri})
    return document.tobytes()


def git(*args: str, cwd: Path) -> str:
    # git 输出 UTF-8；不指定编码时 Windows 会按系统默认编码（如 GBK）解码而出错
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout.strip()


@pytest.fixture
def git_remote(tmp_path: Path) -> Path:
    """一个裸仓库作为远端，main 分支上有一个提交。"""
    remote = tmp_path / "remote.git"
    git("init", "--bare", "--quiet", "--initial-branch=main", str(remote), cwd=tmp_path)
    seed = _clone(remote, tmp_path / "seed")
    (seed / "README.md").write_text("main\n")
    git("add", ".", cwd=seed)
    git("commit", "--quiet", "-m", "init", cwd=seed)
    git("push", "--quiet", "origin", "HEAD:main", cwd=seed)
    return remote


@pytest.fixture
def clone(tmp_path: Path, git_remote: Path) -> Callable[[str], Path]:
    """克隆远端到 tmp_path/<name>，并配置好提交身份。"""
    return lambda name: _clone(git_remote, tmp_path / name)


def _clone(remote: Path, path: Path) -> Path:
    git("clone", "--quiet", str(remote), str(path), cwd=path.parent)
    git("config", "user.name", "Test", cwd=path)
    git("config", "user.email", "test@example.com", cwd=path)
    git("config", "commit.gpgsign", "false", cwd=path)
    return path


class FakeOpenAI:
    """模拟 openai.OpenAI：handler 根据请求参数返回回复文本，也可以抛出异常。"""

    def __init__(self, handler: Callable[[dict[str, Any]], str]) -> None:
        self.handler = handler
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        reply = self.handler(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )


def fake_llm(
    handler: Callable[[dict[str, Any]], str], **config: Any
) -> tuple[LLMClient, FakeOpenAI]:
    fake = FakeOpenAI(handler)
    return LLMClient(LLMConfig(**config), client=fake), fake


def user_prompt(request: dict[str, Any]) -> str:
    """取出请求中最后一条用户消息的内容。"""
    return next(m["content"] for m in reversed(request["messages"]) if m["role"] == "user")


class KeywordEmbedder:
    """按关键词出现次数生成向量的假嵌入模型，便于构造可预期的相似度。"""

    def __init__(self, vocabulary: list[str]) -> None:
        self.vocabulary = vocabulary
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        # 最后一维是常数，避免出现零向量
        rows = [[text.lower().count(word) for word in self.vocabulary] + [0.1] for text in texts]
        return normalize(np.asarray(rows, dtype=np.float32))


# 与 tests/fixtures/arxiv_rss.xml 中的论文配套的假 LLM 和假嵌入
PIPELINE_ANCHORS = {
    "anchors": [
        {"name": "长上下文推理", "query": "KV cache compression for long-context inference"},
        {"name": "检索增强生成", "query": "Retrieval augmented generation"},
        {"name": "医学影像", "query": "Medical imaging", "negative": True},
    ]
}
PIPELINE_VOCABULARY = ["kv cache", "retrieval", "attention", "medical"]


def pipeline_llm(scores: dict[str, int]) -> Callable[[dict[str, Any]], str]:
    """按 prompt 类型分别模拟兴趣拆解、精排和概览三种调用。"""

    def handler(request: dict[str, Any]) -> str:
        system = request["messages"][0]["content"]
        prompt = user_prompt(request)
        if "拆解成若干个具体的子方向" in system:
            return json.dumps(PIPELINE_ANCHORS, ensure_ascii=False)
        if "<paper id=" in prompt:
            ids = re.findall(r'<paper id="([^"]+)">', prompt)
            results = [
                {
                    "id": paper_id,
                    "score": scores[paper_id],
                    "matched_interests": ["长上下文推理"],
                    "reason": f"{paper_id} 与长上下文推理相关",
                    "tldr": f"{paper_id} 的一句话摘要",
                    "tags": ["llm"],
                }
                for paper_id in ids
                if paper_id in scores
            ]
            return json.dumps({"results": results}, ensure_ascii=False)
        if "概括今天的整体情况" in prompt:
            return "今天以长上下文推理为主，建议先读 [1]。"
        raise AssertionError(f"意外的 prompt：{prompt[:80]}")

    return handler


# --- 指令交互 -------------------------------------------------------------------

DIGEST_DAY = date(2026, 10, 25)
DEEP_REPORT = "### 一句话总结\n分层压缩 KV cache。\n\n### 实验\n| 方法 | PPL |\n| --- | --- |"
COMPARE_BODY = "### 一句话结论\n[1] 更省显存。"
RELATED_BODY = "### 定位\n延续了 R1。"


def interact_digest(**fields: Any) -> Digest:
    """2026-10-25 的日报：推荐 2 篇（序号 1–2），另有 3 篇候选（序号 3–5）。"""
    papers = [
        ScoredPaper(
            id="2610.00001",
            title="LongKV: Hierarchical KV Cache Compression",
            abstract="We compress the KV cache.",
            authors=["Alice Zhang", "Bob Li"],
            categories=["cs.CL"],
            llm_score=9,
        ),
        ScoredPaper(id="2610.00002", title="RAG for Science", abstract="A RAG benchmark."),
    ]
    others = [
        ScoredPaper(
            id=f"2610.0001{i}",
            title=f"Candidate {i}",
            abstract=f"Candidate abstract {i}.",
            llm_score=5,
            reason=f"候选 {i} 的理由",
            tldr=f"候选 {i} 的一句话摘要",
        )
        for i in range(3)
    ]
    values = {
        "day": DIGEST_DAY,
        "generated_at": datetime(2026, 10, 25, 1, 30, tzinfo=UTC),
        "papers": papers,
        "others": others,
        "issue_number": 1,
        "issue_url": "https://github.com/o/r/issues/1",
    } | fields
    return Digest(**values)


def interact_llm(request: dict[str, Any]) -> str:
    """按 system prompt 区分深读、多篇对比和相关工作对比。"""
    system = request["messages"][0]["content"]
    if "深读报告" in system:
        return DEEP_REPORT
    if "横向对比" in system:
        return COMPARE_BODY
    if "放到相关工作中比较" in system:
        return RELATED_BODY
    raise AssertionError(f"意外的 prompt：{system[:80]}")


def interact_services(
    handler: Callable[[dict[str, Any]], str] = interact_llm,
) -> tuple[Services, FakeOpenAI]:
    """真实的 arXiv / 全文 / Semantic Scholar 客户端（请求由 respx 模拟）加假 LLM。"""
    fake = FakeOpenAI(handler)
    services = Services(
        arxiv=ArxivSource(ArxivConfig()),
        fetcher=FulltextFetcher(limiter=RateLimiter(0)),
        s2=SemanticScholar(backoff_seconds=0),
        llm_factory=lambda: LLMClient(LLMConfig(), client=fake),
    )
    return services, fake


class FakeNotifier:
    """代替邮件、飞书、企业微信渠道：记录发送的 digest；error 不为空时每次都抛出。"""

    def __init__(self, name: str, error: Exception | None = None) -> None:
        self.name = name
        self.error = error
        self.sent: list[tuple[Digest, bool]] = []

    def send(self, digest: Digest, *, test: bool = False) -> None:
        self.sent.append((digest, test))
        if self.error is not None:
            raise self.error


# --- Agent 工具、/ask、MCP -----------------------------------------------------

OLDER_DAY = date(2026, 10, 20)
CACHED_REPORT = "### 一句话总结\nRAG 的基准测试。\n\n### 实验\n在 5 个数据集上评测。"


def seed_agent_data(store) -> None:
    """两份日报、几条反馈、一个屏蔽主题、学到的偏好和一份深读报告缓存。"""
    store.write_model(digest_path(DIGEST_DAY), interact_digest())
    older = Digest(
        day=OLDER_DAY,
        generated_at=datetime(2026, 10, 20, 1, 30, tzinfo=UTC),
        papers=[
            ScoredPaper(
                id="2610.00100",
                title="Domain Generalization for Object Detectors",
                abstract="Single-source domain generalization with style augmentation.",
                llm_score=8,
                reason="单源域泛化检测，和你的方向直接相关。",
            )
        ],
        issue_number=0,
    )
    store.write_model(digest_path(OLDER_DAY), older)
    now = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
    liked = Paper(id="2610.00100", title=older.papers[0].title, abstract=older.papers[0].abstract)
    other = Paper(id="2609.00900", title="Style Randomization", abstract="Random style transfer.")
    record_feedback(
        store,
        [
            command_feedback(liked, "like", now=now, note="有开源代码"),
            command_feedback(other, "dislike", now=now),
        ],
        timezone="Asia/Shanghai",
    )
    store.append_jsonl("profile/muted.jsonl", [{"action": "mute", "topic": "医学影像"}])
    store.write_text("profile/learned_preferences.md", "### 偏好\n- 偏好有开源代码的工作")
    save_report(
        store,
        DeepReport(
            paper_id="2610.00002",
            title="RAG for Science",
            model="deepseek-reasoner",
            source="html",
            generated_at=datetime(2026, 10, 25, 3, 0, tzinfo=UTC),
            body=CACHED_REPORT,
        ),
    )
