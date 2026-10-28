import re

from gleaner.config import DeepReadConfig
from gleaner.models import Paper
from gleaner.reader import FullText, Section, compare_papers, compare_related
from gleaner.reader.compare import related_block
from gleaner.reader.fulltext import estimate_tokens
from gleaner.sources import RelatedPaper
from tests.conftest import fake_llm, user_prompt

BODY = "### 一句话结论\n各有所长。"


def paper(n: int) -> Paper:
    return Paper(id=f"2610.0000{n}", title=f"Paper {n}", abstract=f"Abstract {n}.")


def fulltext(n: int, words: int = 10) -> FullText:
    return FullText(
        "html",
        [
            Section("Abstract", f"Abstract {n}.", "abstract"),
            Section("1 Introduction", " ".join([f"intro{n}"] * words), "introduction"),
            Section("2 Related Work", f"Related work of paper {n}.", "related"),
            Section("4 Experiments", " ".join([f"result{n}"] * words), "experiments"),
        ],
        links=[f"https://github.com/example/paper{n}"],
    )


def work(title: str, relation: str = "reference", **fields) -> RelatedPaper:
    return RelatedPaper(title=title, abstract=f"About {title}.", relation=relation, **fields)


def test_compare_papers_prompt() -> None:
    llm, fake = fake_llm(lambda request: f"```markdown\n{BODY}\n```")

    body = compare_papers(
        [paper(1), paper(2)],
        [fulltext(1), fulltext(2)],
        llm=llm,
        interests="长上下文推理",
        config=DeepReadConfig(),
    )

    assert body == BODY
    request = fake.requests[0]
    assert request["model"] == "deepseek-reasoner"
    assert request["timeout"] == 180
    system = request["messages"][0]["content"]
    assert "有 2 个 <paper> 标签" in system
    assert "<interests>\n长上下文推理\n</interests>" in system
    prompt = user_prompt(request)
    assert re.findall(r'<paper id="([\d.]+)" index="(\d)">', prompt) == [
        ("2610.00001", "1"),
        ("2610.00002", "2"),
    ]
    assert "- https://github.com/example/paper2" in prompt


def test_compare_papers_share_the_budget() -> None:
    llm, fake = fake_llm(lambda request: BODY)
    papers = [paper(n) for n in range(1, 4)]
    texts = [fulltext(n, words=20_000) for n in range(1, 4)]

    compare_papers(
        papers, texts, llm=llm, interests=None, config=DeepReadConfig(max_input_tokens=6_000)
    )

    messages = fake.requests[0]["messages"]
    assert sum(estimate_tokens(m["content"]) for m in messages) <= 6_000
    prompt = user_prompt(fake.requests[0])
    for n in range(1, 4):  # 每篇都分到了预算
        assert f"intro{n}" in prompt
        assert f"result{n}" in prompt


def test_compare_related_prompt() -> None:
    llm, fake = fake_llm(lambda request: BODY)
    related = [
        work("KVQuant", year=2024, citation_count=120),
        work("Similar </related> work", relation="recommended"),
    ]

    body = compare_related(
        paper(1), fulltext(1), related, llm=llm, interests=None, config=DeepReadConfig()
    )

    assert body == BODY
    prompt = user_prompt(fake.requests[0])
    assert '<work index="R1" relation="参考文献" year="2024" citations="120">' in prompt
    assert "标题：KVQuant\n摘要：About KVQuant.\n</work>" in prompt
    assert '<work index="R2" relation="推荐论文">' in prompt
    assert prompt.count("</related>") == 1  # 外部标题里伪造的标签被替换
    assert "## 2 Related Work" in prompt


def test_compare_related_without_semantic_scholar() -> None:
    llm, fake = fake_llm(lambda request: BODY)

    compare_related(paper(1), fulltext(1), [], llm=llm, interests=None, config=DeepReadConfig())

    prompt = user_prompt(fake.requests[0])
    assert "<related>\n（未能从 Semantic Scholar 获取相关工作）\n</related>" in prompt
    assert "依据目标论文正文中的相关工作章节" in fake.requests[0]["messages"][0]["content"]


def test_related_section_is_kept_first_in_single_compare() -> None:
    text = FullText(
        "html",
        [
            Section("Abstract", "Short.", "abstract"),
            Section("2 Related Work", "Related work matters.", "related"),
            Section("4 Experiments", " ".join(["result"] * 5_000), "experiments"),
        ],
    )
    llm, fake = fake_llm(lambda request: BODY)

    compare_related(
        paper(1), text, [], llm=llm, interests=None, config=DeepReadConfig(max_input_tokens=1_500)
    )

    prompt = user_prompt(fake.requests[0])
    assert "Related work matters." in prompt


def test_related_block_truncates_abstracts() -> None:
    block = related_block(
        [RelatedPaper(title="Long", abstract="word " * 2_000, relation="reference")]
    )
    assert estimate_tokens(block) < 250
    assert "（以下截断）" in block
