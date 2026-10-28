"""核心数据模型。"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Paper(BaseModel):
    id: str  # 不带版本号的 arXiv ID，如 "2410.12345"
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    announce_type: str | None = None  # new / cross / replace / replace-cross
    announced: datetime | None = None

    @property
    def url(self) -> str:
        return f"https://arxiv.org/abs/{self.id}"

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.id}"


class ScoredPaper(Paper):
    """经过粗筛、精排后的论文。"""

    coarse_score: float | None = None
    coarse_match: str | None = None  # 向量最接近的兴趣子方向
    # 粗筛分数的分项（docs 3.1.1）：coarse_score = 兴趣 + 正反馈 + Zotero - 负反馈，都已乘过权重。
    # 没有正反馈锚点时 positive_score 为 None，没有负向锚点时 negative_score 为 None，
    # 没有使用 Zotero 时 zotero_score 为 None
    interest_score: float | None = None
    positive_score: float | None = None
    negative_score: float | None = None
    zotero_score: float | None = None
    forced: bool = False  # 命中 include_keywords，跳过了粗筛
    llm_score: int | None = None  # 0–10；None 表示精排失败或未精排
    matched_interests: list[str] = Field(default_factory=list)
    reason: str = ""
    tldr: str = ""
    tags: list[str] = Field(default_factory=list)


class FeedbackStats(BaseModel):
    """日报运行时的反馈概况，显示在 Issue 的「本次运行」中，用来观察反馈是否生效。

    前几项是按论文计的当前状态（同一篇论文的重复反馈只算一次），见 profile/memory.py 的聚合规则。
    """

    up: int = 0  # 当前为 👍 的论文
    down: int = 0  # 当前为 👎 的论文
    like: int = 0  # 最新一条显式反馈为 /like 的论文
    dislike: int = 0
    implicit: int = 0  # /deep、/compare 过的论文
    positive: int = 0  # 参与粗筛的正反馈锚点（A2）
    negative: int = 0  # 参与粗筛的负反馈锚点（A3 中的论文）
    muted: int = 0  # 当前屏蔽的主题
    synced: int = 0  # 本次表情同步新写入的记录
    sync_error: str | None = None  # 表情同步失败的原因

    @property
    def empty(self) -> bool:
        return self == FeedbackStats()


class ZoteroStats(BaseModel):
    """Zotero 锚点（A4）的使用情况，显示在「本次运行」中。只记数量，不记文献标题。"""

    used: int = 0  # 参与粗筛的篇数
    total: int = 0  # 读取范围内的有效文献数（去掉附件、笔记、回收站条目和重复项）
    # api：重新拉取；cache：文库没有变化（304），用了缓存；stale：读取失败，用了旧缓存；
    # none：读取失败又没有缓存，这次没有使用
    source: Literal["api", "cache", "stale", "none"] = "api"
    error: str | None = None


class RunStats(BaseModel):
    fetched: int = 0
    duplicates: int = 0
    excluded: int = 0
    candidates: int = 0  # 进入精排的篇数
    selected: int = 0
    failed_batches: int = 0
    degraded: bool = False
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_seconds: float = 0.0
    feedback: FeedbackStats = Field(default_factory=FeedbackStats)
    zotero: ZoteroStats | None = None  # 没有启用 Zotero 时为 None
    deep_read: int = 0  # 自动深读的篇数（报告附在论文评论里）
    deep_read_notes: list[str] = Field(default_factory=list)  # 没有自动深读的原因，如超出预算


class ChannelResult(BaseModel):
    """一个推送渠道（邮件、飞书、企业微信）的发送结果，记在 digest 里。

    兜底运行据此只补发上次失败的渠道，不重复发送。error 已经脱敏（gleaner-data 是公开的）。
    """

    status: Literal["sent", "failed"]
    error: str | None = None
    at: datetime


class Digest(BaseModel):
    """一天的推送结果，保存在 gleaner-data 分支的 history/digests/<日期>.json。"""

    day: date
    generated_at: datetime
    overview: str | None = None
    papers: list[ScoredPaper] = Field(default_factory=list)  # 推送的论文，序号从 1 开始
    others: list[ScoredPaper] = Field(default_factory=list)  # 精排过但未推送的候选
    stats: RunStats = Field(default_factory=RunStats)
    issue_number: int | None = None
    issue_url: str | None = None
    comment_ids: dict[str, int] = Field(default_factory=dict)  # 论文 ID → 评论 ID
    notifications: dict[str, ChannelResult] = Field(default_factory=dict)  # 渠道名 → 发送结果
    # 自动深读过的论文：报告缓存在 reports/<arxiv_id>.md，补发评论时从缓存读取，不重新生成
    deep_reads: list[str] = Field(default_factory=list)


class DeepReport(BaseModel):
    """/deep 生成的深读报告，缓存在 gleaner-data 分支的 reports/<arxiv_id>.md。"""

    paper_id: str
    title: str
    model: str
    source: Literal["html", "pdf", "abstract"]  # 全文来源；abstract 表示只用了摘要
    generated_at: datetime
    body: str  # 模型输出的 Markdown，未转义；发到 GitHub 前再转义


class Feedback(BaseModel):
    """一条反馈，按月追加到 gleaner-data 分支的 feedback/<YYYY-MM>.jsonl（只追加，不修改）。

    表情记录的 weight 是同步时的状态：+1（👍）、-1（👎）、0（撤销，signal 为被撤销的那个）。
    聚合规则见 profile/memory.py。
    """

    paper_id: str
    title: str
    abstract: str
    signal: Literal["up", "down", "like", "dislike", "deep", "compare"]
    weight: float
    note: str | None = None  # /like 或 /dislike 附带的原因
    source: Literal["reaction", "command"]
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def _assume_utc(cls, value: datetime) -> datetime:
        # 手改的记录可能不带时区；不统一的话与其它记录比较先后时会出错
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
