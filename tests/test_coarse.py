from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from gleaner.config import RankingWeights
from gleaner.models import Paper, ScoredPaper
from gleaner.profile import Anchor, AnchorSet, Memory, MutedTopic, PaperFeedback
from gleaner.ranking import coarse_rank
from gleaner.ranking.coarse import coarse_scores, paper_text
from gleaner.ranking.embedder import normalize
from tests.conftest import KeywordEmbedder

VOCABULARY = ["kv cache", "retrieval", "medical", "speech"]


def make_anchors(*, with_negative: bool = True) -> AnchorSet:
    anchors = [
        Anchor(name="长上下文", query="KV cache compression"),
        Anchor(name="RAG", query="retrieval augmented generation"),
    ]
    if with_negative:
        anchors.append(Anchor(name="医学影像", query="medical imaging", negative=True))
    return AnchorSet(source_hash="h", model="m", generated_at=datetime.now(UTC), anchors=anchors)


def paper(id: str, title: str, abstract: str = "") -> Paper:
    return Paper(id=id, title=title, abstract=abstract)


def rank(anchors: AnchorSet, top_k: int) -> list[str]:
    result = coarse_rank(
        PAPERS, anchors, KeywordEmbedder(VOCABULARY), top_k=top_k, weights=RankingWeights()
    )
    return [p.id for p in result]


PAPERS = [
    paper("speech", "Speech recognition"),
    paper("kv", "KV cache tricks", "We compress the KV cache."),
    paper("rag", "Retrieval for QA"),
    paper("weak", "Retrieval of speech segments"),
    paper("medical-rag", "Retrieval for medical reports", "Retrieval."),
]


def test_ranks_by_anchor_similarity() -> None:
    embedder = KeywordEmbedder(VOCABULARY)
    anchors = make_anchors(with_negative=False)
    ranked = coarse_rank(PAPERS, anchors, embedder, top_k=5, weights=RankingWeights())

    assert {p.id for p in ranked[:2]} == {"kv", "rag"}
    assert ranked[-1].id == "speech"
    assert ranked == sorted(ranked, key=lambda p: p.coarse_score, reverse=True)
    by_id = {p.id: p for p in ranked}
    assert by_id["kv"].coarse_match == "长上下文"
    assert by_id["rag"].coarse_match == "RAG"
    assert all(p.coarse_score is not None for p in ranked)


def test_negative_anchor_penalizes() -> None:
    without = rank(make_anchors(with_negative=False), top_k=5)
    with_negative = rank(make_anchors(), top_k=5)

    # 不加负向锚点时，medical-rag 与 RAG 更相关，排在 weak 前面；加上后被压到后面
    assert without.index("medical-rag") < without.index("weak")
    assert with_negative.index("medical-rag") > with_negative.index("weak")


def test_top_k_keeps_forced_papers() -> None:
    ranked = coarse_rank(
        PAPERS,
        make_anchors(),
        KeywordEmbedder(VOCABULARY),
        top_k=2,
        weights=RankingWeights(),
        forced_ids={"speech"},
    )

    assert {p.id for p in ranked[:2]} == {"kv", "rag"}
    assert ranked[2].id == "speech"
    assert [p.forced for p in ranked] == [False, False, True]


def test_embeds_anchors_and_papers_in_one_call() -> None:
    embedder = KeywordEmbedder(VOCABULARY)
    coarse_rank(PAPERS, make_anchors(), embedder, top_k=2, weights=RankingWeights())

    assert len(embedder.calls) == 1
    texts = embedder.calls[0]
    assert texts[:3] == [
        "KV cache compression",
        "retrieval augmented generation",
        "medical imaging",
    ]
    assert texts[4] == "KV cache tricks. We compress the KV cache."


def test_empty_papers() -> None:
    embedder = KeywordEmbedder(VOCABULARY)
    assert coarse_rank([], make_anchors(), embedder, top_k=5, weights=RankingWeights()) == []
    assert embedder.calls == []


# --- 反馈锚点（A2 / A3） ---------------------------------------------------------

NOW = datetime(2026, 10, 26, 2, 0, tzinfo=UTC)


def liked(n: int, *, days_ago: float = 0, signal: str = "like") -> list[PaperFeedback]:
    """n 篇讲语音的正反馈论文。"""
    return [
        PaperFeedback(
            paper_id=f"2609.0000{i}",
            title=f"Speech model {i}",
            abstract="Speech recognition with speech encoders.",
            command=2.0 if signal == "like" else 0.0,
            reaction=1.0 if signal == "up" else 0.0,
            updated_at=NOW - timedelta(days=days_ago),
        )
        for i in range(n)
    ]


def disliked_medical() -> PaperFeedback:
    return PaperFeedback(
        paper_id="2609.00100",
        title="Medical imaging",
        abstract="Medical segmentation.",
        command=-2.0,
        updated_at=NOW,
    )


def ranked(memory: Memory | None, *, anchors: AnchorSet | None = None, embedder=None):
    result = coarse_rank(
        PAPERS,
        anchors or make_anchors(),
        embedder or KeywordEmbedder(VOCABULARY),
        top_k=5,
        weights=RankingWeights(),
        memory=memory,
        half_life_days=90,
        now=NOW,
    )
    return {p.id: p for p in result}, [p.id for p in result]


def test_cold_start_memory_changes_nothing() -> None:
    without, order = ranked(None)
    empty, empty_order = ranked(Memory())
    no_query, _ = ranked(Memory(muted=[MutedTopic("医学影像", None, NOW)]))  # 还没有英文描述

    assert empty_order == order
    for result in (empty, no_query):
        assert {i: p.coarse_score for i, p in result.items()} == {
            i: p.coarse_score for i, p in without.items()
        }
    kv = without["kv"]
    assert kv.positive_score is None
    assert kv.coarse_score == pytest.approx(kv.interest_score - kv.negative_score, abs=1e-4)


def test_liked_direction_rises() -> None:
    before, before_order = ranked(Memory())
    after, after_order = ranked(Memory(feedback=liked(3)))

    # 喜欢了三篇语音论文：同时涉及检索和语音的 weak 升到第一，纯语音的 speech 也往前了
    assert before_order.index("weak") == 2
    assert after_order[0] == "weak"
    assert after_order.index("speech") < before_order.index("speech")
    assert after["weak"].positive_score > 0.4
    assert after["kv"].positive_score == pytest.approx(0.0, abs=0.01)
    assert after["kv"].interest_score == before["kv"].interest_score
    for paper in after.values():
        total = paper.interest_score + paper.positive_score - paper.negative_score
        assert paper.coarse_score == pytest.approx(total, abs=1e-3)


def test_positive_term_grows_with_more_feedback() -> None:
    one, _ = ranked(Memory(feedback=liked(1)))
    three, _ = ranked(Memory(feedback=liked(3)))
    thumbs, _ = ranked(Memory(feedback=liked(1, signal="up")))

    # mean_top3：不足 3 个按 0 计；一个 👍 的强度是 /like 的一半
    assert one["speech"].positive_score == pytest.approx(three["speech"].positive_score / 3, 1e-3)
    assert thumbs["speech"].positive_score == pytest.approx(one["speech"].positive_score / 2, 1e-3)


def test_dislike_and_mute_push_similar_papers_down() -> None:
    anchors = make_anchors(with_negative=False)
    _, before = ranked(Memory(), anchors=anchors)
    disliked, after_dislike = ranked(Memory(feedback=[disliked_medical()]), anchors=anchors)
    muted = Memory(muted=[MutedTopic("医学影像", "Medical imaging", NOW)])
    _, after_mute = ranked(muted, anchors=anchors)

    assert before.index("medical-rag") < before.index("weak")
    assert after_dislike.index("medical-rag") > after_dislike.index("weak")
    assert after_mute.index("medical-rag") > after_mute.index("weak")
    assert disliked["medical-rag"].negative_score > 0.3
    assert disliked["kv"].negative_score < 0.01


def test_feedback_decays_with_half_life() -> None:
    fresh, _ = ranked(Memory(feedback=liked(3)))
    old, _ = ranked(Memory(feedback=liked(3, days_ago=90)))
    older, _ = ranked(Memory(feedback=liked(3, days_ago=180)))

    score = fresh["speech"].positive_score
    assert old["speech"].positive_score == pytest.approx(score / 2, rel=1e-3)
    assert older["speech"].positive_score == pytest.approx(score / 4, rel=1e-3)


def test_feedback_texts_are_embedded_after_papers() -> None:
    embedder = KeywordEmbedder(VOCABULARY)
    bare = PaperFeedback(paper_id="2609.09999", title="2609.09999", command=2.0, updated_at=NOW)
    memory = Memory(
        feedback=[*liked(1), bare, disliked_medical()],
        muted=[MutedTopic("语音", "Speech processing.", NOW)],
    )

    ranked(memory, embedder=embedder)

    [texts] = embedder.calls
    assert texts[3] == "Speech recognition. "  # 论文的位置不变
    assert texts[-3:] == [
        "Speech model 0. Speech recognition with speech encoders.",
        "Medical imaging. Medical segmentation.",
        "Speech processing.",
    ]  # 只有 ID、没有标题和摘要的反馈不参与


# --- Zotero（A4） ---------------------------------------------------------------


class RandomEmbedder:
    """给每段文本分配一个固定的随机向量（同一实例中相同文本总是同一个向量）。"""

    def __init__(self, seed: int, dim: int = 24) -> None:
        self.rng = np.random.default_rng(seed)
        self.dim = dim
        self.vectors: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = [self.vectors.setdefault(t, self.rng.normal(size=self.dim)) for t in texts]
        return normalize(np.asarray(rows, dtype=np.float32))


def legacy_scores(papers, anchors, embedder, *, weights, memory=None, half_life_days=90, now=None):
    """Phase 3（接入 Zotero 之前，main 分支上）coarse_rank 的打分部分，原样照抄。"""
    now = now or datetime.now(UTC)
    positive, negative = anchors.positive, anchors.negative
    liked = memory.positive(now, half_life_days) if memory else []
    disliked = memory.negative(now, half_life_days) if memory else []
    muted = [topic.query for topic in memory.muted if topic.query] if memory else []

    groups = [
        [a.query for a in positive],
        [a.query for a in negative],
        [paper_text(p) for p in papers],
        [paper_text(item.paper) for item, _ in liked],
        [paper_text(item.paper) for item, _ in disliked],
        muted,
    ]
    vectors = embedder.embed([text for group in groups for text in group])
    bounds = np.cumsum([0, *(len(group) for group in groups)])
    (
        positive_vectors,
        negative_vectors,
        paper_vectors,
        liked_vectors,
        disliked_vectors,
        muted_vectors,
    ) = (vectors[start:end] for start, end in zip(bounds[:-1], bounds[1:], strict=True))

    def weights_of(pairs):
        return np.asarray([weight for _, weight in pairs], dtype=np.float32)

    similarity = paper_vectors @ positive_vectors.T
    interest = weights.interest * similarity.max(axis=1)
    scores = interest
    positive_part = None
    if liked:
        contributions = (paper_vectors @ liked_vectors.T) * weights_of(liked)
        top = np.sort(contributions, axis=1)[:, -3:]
        positive_part = weights.positive * top.sum(axis=1) / 3
        scores = scores + positive_part
    negative_part = None
    penalties = []
    if negative:
        penalties.append(paper_vectors @ negative_vectors.T)
    if muted:
        penalties.append(paper_vectors @ muted_vectors.T)
    if disliked:
        penalties.append((paper_vectors @ disliked_vectors.T) * weights_of(disliked))
    if penalties:
        negative_part = weights.negative * np.concatenate(penalties, axis=1).max(axis=1)
        scores = scores - negative_part
    return scores, interest, positive_part, negative_part, similarity.argmax(axis=1)


def random_papers(count: int) -> list[Paper]:
    return [paper(f"2610.{i:05d}", f"Random paper {i}", f"Abstract {i}.") for i in range(count)]


MEMORIES = {
    "none": None,
    "empty": Memory(),
    "feedback": Memory(
        feedback=[*liked(4), disliked_medical()],
        muted=[MutedTopic("语音", "Speech processing.", NOW), MutedTopic("x", None, NOW)],
    ),
}


@pytest.mark.parametrize("memory", MEMORIES.values(), ids=MEMORIES.keys())
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_without_zotero_scores_are_bit_identical(seed: int, memory: Memory | None) -> None:
    papers = random_papers(40)
    embedder = RandomEmbedder(seed)
    kwargs = {"weights": RankingWeights(), "memory": memory, "half_life_days": 90, "now": NOW}

    old = legacy_scores(papers, make_anchors(), embedder, **kwargs)
    new = coarse_scores(papers, make_anchors(), embedder, **kwargs)

    assert new.total.tobytes() == old[0].tobytes()  # 逐位相同，不只是近似相等
    assert new.interest.tobytes() == old[1].tobytes()
    for new_part, old_part in ((new.positive, old[2]), (new.negative, old[3])):
        assert (new_part is None) == (old_part is None)
        if new_part is not None:
            assert new_part.tobytes() == old_part.tobytes()
    assert new.best_match.tobytes() == old[4].tobytes()
    assert new.zotero is None
    ranked = coarse_rank(papers, make_anchors(), embedder, top_k=10, **kwargs)
    assert all(p.zotero_score is None for p in ranked)


def with_zotero(count: int) -> dict[str, ScoredPaper]:
    """count 篇讲语音的 Zotero 文献作为锚点。"""
    zotero = [
        paper(f"zotero:{i}", f"Speech model {i}", "Speech recognition with speech encoders.")
        for i in range(count)
    ]
    result = coarse_rank(
        PAPERS,
        make_anchors(),
        KeywordEmbedder(VOCABULARY),
        top_k=5,
        weights=RankingWeights(),
        memory=Memory(),
        now=NOW,
        zotero=zotero,
    )
    return {p.id: p for p in result}


def test_zotero_anchors_lift_similar_papers() -> None:
    before, before_order = ranked(Memory())
    after = with_zotero(3)
    after_order = sorted(after, key=lambda i: after[i].coarse_score, reverse=True)

    # 文库里有三篇语音论文：同时涉及检索和语音的 weak 升到第一，纯语音的 speech 分数明显提高
    assert before_order.index("weak") == 2 and after_order[0] == "weak"
    assert after["speech"].coarse_score > before["speech"].coarse_score + 0.4
    assert after["kv"].zotero_score == pytest.approx(0.0, abs=0.01)
    for p in after.values():
        assert p.positive_score is None
        total = p.interest_score + p.zotero_score - p.negative_score
        assert p.coarse_score == pytest.approx(total, abs=1e-3)
        assert p.interest_score == before[p.id].interest_score


def test_zotero_term_is_mean_of_top3() -> None:
    # 不足 3 篇时按 0 计：两篇锚点的分项是三篇时的 2/3
    two, three = with_zotero(2), with_zotero(3)
    assert two["speech"].zotero_score == pytest.approx(three["speech"].zotero_score * 2 / 3, 1e-3)


def test_zotero_texts_are_embedded_last() -> None:
    embedder = KeywordEmbedder(VOCABULARY)
    memory = Memory(muted=[MutedTopic("语音", "Speech processing.", NOW)])
    coarse_rank(
        PAPERS,
        make_anchors(),
        embedder,
        top_k=5,
        weights=RankingWeights(),
        memory=memory,
        now=NOW,
        zotero=[paper("zotero:A", "Zotero paper", "Its abstract.")],
    )

    [texts] = embedder.calls
    assert texts[-2:] == ["Speech processing.", "Zotero paper. Its abstract."]
