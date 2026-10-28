from datetime import UTC, date, datetime

from gleaner.models import DeepReport, Digest, FeedbackStats, RunStats, ScoredPaper
from gleaner.render import issue_body, issue_title, paper_section, sanitize, stats_line
from gleaner.render.markdown import COMMENT_LIMIT, split_comment


def scored(n: int, **fields) -> ScoredPaper:
    values = {
        "id": f"2610.0000{n}",
        "title": f"Paper {n}",
        "abstract": "We study things.",
        "authors": ["Alice", "Bob"],
        "categories": ["cs.CL"],
        "coarse_score": 0.8,
        "coarse_match": "长上下文推理",
        "llm_score": 8,
        "matched_interests": ["长上下文推理"],
        "reason": "压缩 KV cache，与你的长上下文推理方向直接相关。",
        "tldr": "显存降低 4 倍。",
        "tags": ["kv-cache", "efficiency"],
    } | fields
    return ScoredPaper(**values)


def make_digest(**fields) -> Digest:
    values = {
        "day": date(2026, 10, 26),
        "generated_at": datetime(2026, 10, 26, 1, 30, tzinfo=UTC),
        "overview": "今天以长上下文为主。\n建议先读 [1]。",
        "papers": [scored(1), scored(2, llm_score=None, matched_interests=[])],
        "stats": RunStats(fetched=600, candidates=50, selected=2, llm_calls=7),
    } | fields
    return Digest(**values)


def test_issue_title() -> None:
    assert issue_title(make_digest()) == "📚 Paper Gleaner · 2026-10-26 · 2 篇"


def test_issue_body() -> None:
    body = issue_body(make_digest())

    assert body.startswith("> 今天以长上下文为主。\n> 建议先读 [1]。")
    assert "| 1 | [Paper 1](https://arxiv.org/abs/2610.00001) | 8 | 长上下文推理 |" in body
    # 没有 LLM 分数时显示 —，匹配兴趣退回到粗筛最接近的方向
    assert "| 2 | [Paper 2](https://arxiv.org/abs/2610.00002) | — | 长上下文推理 |" in body
    assert "每篇论文的详细信息见下方评论" in body
    assert "### [1]" not in body
    assert "<details><summary>本次运行</summary>" in body
    assert "抓取 600 篇" in body
    assert "<!-- gleaner:digest day=2026-10-26 -->" in body
    assert "⚠️" not in body


def test_issue_body_degraded_and_inline_papers() -> None:
    digest = make_digest(overview=None, stats=RunStats(degraded=True))
    body = issue_body(digest, per_paper_comment=False)

    assert not body.startswith(">")
    assert "⚠️ 今天的 LLM 精排没有成功" in body
    assert "### [1] Paper 1" in body
    assert "### [2] Paper 2" in body
    assert "下方评论" not in body


def test_paper_section() -> None:
    paper = scored(3, authors=["A", "B", "C", "D"])
    text = paper_section(3, paper)

    assert text.startswith("### [3] Paper 3\n")
    assert "**作者**：A, B, C 等 · **分类**：cs.CL · **分数**：8/10" in text
    assert (
        "[arXiv](https://arxiv.org/abs/2610.00003) · [PDF](https://arxiv.org/pdf/2610.00003)"
        in text
    )
    assert "🎯 **为什么和你相关**：压缩 KV cache" in text
    assert "📝 **TL;DR**：显存降低 4 倍。" in text
    assert "🏷️ `kv-cache` `efficiency`" in text
    assert text.endswith("<!-- gleaner:paper id=2610.00003 index=3 -->")


def test_paper_section_without_llm_output() -> None:
    paper = scored(4, llm_score=None, reason="", tldr="", tags=[], abstract="x" * 500)
    text = paper_section(1, paper)

    assert "**分数**" not in text
    assert "🎯 **最接近的兴趣方向**：长上下文推理" in text
    assert "📄 **摘要**：" + "x" * 400 + "…" in text
    assert "🏷️" not in text


def test_external_text_is_sanitized() -> None:
    assert sanitize("ping @octocat about #12") == "ping @​octocat about #​12"
    assert sanitize("<img src=x>") == "&lt;img src=x&gt;"
    assert sanitize("C# and #rust") == "C# and #rust"

    paper = scored(5, title="A | B [draft] <b>", reason="cc @someone")
    body = issue_body(make_digest(papers=[paper]))
    assert "[A \\| B \\[draft\\] &lt;b&gt;](https://arxiv.org/abs/2610.00005)" in body
    assert "@​someone" in paper_section(1, paper)


def test_stats_line() -> None:
    stats = RunStats(
        fetched=612,
        duplicates=3,
        excluded=1,
        candidates=50,
        selected=10,
        llm_calls=7,
        prompt_tokens=31_200,
        completion_tokens=850,
        duration_seconds=133,
        failed_batches=1,
    )
    line = stats_line(stats)
    assert "抓取 612 篇 · 重复 3 · 黑名单 1 · 精排 50 · 推送 10" in line
    assert "token 32.0k（输入 31.2k / 输出 850）" in line
    assert "耗时 2 分 13 秒" in line
    assert "精排失败 1 批" in line


def test_stats_line_with_feedback() -> None:
    assert "反馈" not in stats_line(RunStats())  # 冷启动时不显示

    feedback = FeedbackStats(
        up=2,
        down=1,
        like=1,
        implicit=3,
        positive=4,
        negative=2,
        muted=1,
        synced=3,
        sync_error="查询表情失败（HTTP 403）：@someone <b>rate limit</b>",
    )
    line = stats_line(RunStats(feedback=feedback))

    assert "反馈：👍 2 · 👎 1 · /like 1 · /dislike 0 · /deep、/compare 3" in line
    assert "粗筛锚点：正反馈 4 篇 · 负反馈 2 篇 · 屏蔽主题 1 个 · 本次同步表情 3 条" in line
    assert "⚠️ 表情同步没有完成：查询表情失败（HTTP 403）：@​someone &lt;b&gt;rate limit" in line
    body = issue_body(make_digest(stats=RunStats(feedback=feedback)))
    assert "粗筛锚点：正反馈 4 篇" in body.split("<details><summary>本次运行</summary>")[1]


def test_split_comment_keeps_short_bodies() -> None:
    assert split_comment("短回复") == ["短回复"]


def test_split_comment_breaks_at_blank_lines() -> None:
    paragraphs = [f"第 {i} 段：" + "字" * 400 for i in range(10)]

    parts = split_comment("\n\n".join(paragraphs), limit=1500)

    assert len(parts) > 1
    assert all(len(part) <= 1500 for part in parts)
    assert parts[0].endswith("\n\n<sub>（未完，见下一条评论）</sub>")
    assert parts[1].startswith(f"<sub>（续 2/{len(parts)}）</sub>\n\n第 ")
    joined = "".join(parts)
    assert all(paragraph in joined for paragraph in paragraphs)  # 段落没有被拆开


def test_split_comment_reopens_code_blocks_and_details() -> None:
    lines = [f"line {i} " + "x" * 50 for i in range(60)]
    body = (
        "<details><summary>原文</summary>\n\n```text\n" + "\n".join(lines) + "\n```\n\n</details>"
    )

    parts = split_comment(body, limit=1500)

    assert len(parts) > 2
    for part in parts:
        assert len(part) <= 1500
        assert part.count("```") % 2 == 0  # 每条评论里的代码块都是闭合的
        assert part.count("<details") == part.count("</details>")
    assert "```text" in parts[1]
    assert all(line in "".join(parts) for line in lines)


def test_split_comment_cuts_huge_lines() -> None:
    parts = split_comment("x" * 5000, limit=1500)
    assert all(len(part) <= 1500 for part in parts)
    assert "".join(parts).count("x") == 5000


def deep_report(body: str, **fields) -> DeepReport:
    values = {
        "paper_id": "2610.00003",
        "title": "Paper 3",
        "model": "deepseek-reasoner",
        "source": "html",
        "generated_at": datetime(2026, 10, 26, 1, 35, tzinfo=UTC),
        "body": body,
    } | fields
    return DeepReport(**values)


def test_paper_section_with_deep_report() -> None:
    body = (
        "### 一句话总结\n压缩 KV cache。\n\n> 引用 <script>x</script> @alice #12\n"
        "<!-- gleaner:paper id=9 -->"
    )
    text = paper_section(3, scored(3), deep_report(body))

    assert "<details><summary>🔍 深读报告</summary>\n\n### 一句话总结\n压缩 KV cache。" in text
    # 与 /deep 的回复相同：保留 ">"（引用块），转义 "<"
    assert "> 引用 &lt;script>x&lt;/script> @\u200balice #\u200b12" in text
    assert "&lt;!-- gleaner:paper id=9 -->" in text  # 模型输出不能伪造标记
    assert "评论 `/deep 3` 直接取回这份报告" in text
    assert "模型 deepseek-reasoner · 全文来源：arXiv HTML" in text
    assert text.index("</details>") < text.index("<!-- gleaner:paper id=2610.00003 index=3 -->")
    assert text.endswith("<!-- gleaner:paper id=2610.00003 index=3 -->")


def test_long_deep_report_is_truncated_within_comment_limit() -> None:
    body = "### 实验\n```text\n" + "\n".join(f"第 {i} 行数据 " * 20 for i in range(3000)) + "\n```"
    text = paper_section(1, scored(1), deep_report(body, source="abstract"))

    assert len(text) <= COMMENT_LIMIT
    assert "（仅基于摘要）" in text
    assert "报告太长，这里只显示前一部分；评论 `/deep 1` 查看完整报告" in text
    report = text[text.index("<details>") : text.index("</details>")]
    assert report.count("```") % 2 == 0  # 截断处的代码块被闭合
    assert text.endswith("<!-- gleaner:paper id=2610.00001 index=1 -->")


def test_stats_line_shows_auto_deep_read() -> None:
    stats = RunStats(deep_read=1, deep_read_notes=["第 2 篇没有自动深读：<超出预算>"])
    line = stats_line(stats)

    assert "自动深读 1 篇，报告附在对应论文的评论里" in line
    assert "⚠️ 第 2 篇没有自动深读：&lt;超出预算&gt;" in line
