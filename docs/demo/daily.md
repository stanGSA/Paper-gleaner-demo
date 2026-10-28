> **历史论文演示**：候选为人工选定的 5 篇公开论文，不代表今天的 arXiv 公告。排序、推荐理由和深读由实际模型调用生成；该样本不用于推荐质量评测。

> 今天推荐的论文主要集中在 LLM 高效服务与长上下文建模、检索增强生成及自反思、以及长上下文信息利用评估几个方向。值得注意的趋势是，研究既关注 KV cache 与流式注意力带来的效率提升，也开始重视 RAG 的自适应检索与自我评判，以及长上下文中的位置偏差问题。建议先读 [1]，它相关度最高，且对 LLM 服务系统的吞吐与显存管理有直接工程价值。

| # | 论文 | 分数 | 匹配兴趣 |
| --: | --- | :-: | --- |
| 1 | [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | 10 | KV cache 管理与显存优化、高效推理与 RAG 的可复现系统实现 |
| 2 | [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) | 10 | KV cache 管理与显存优化、长上下文与流式推理 |
| 3 | [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) | 10 | 自适应检索与检索必要性判断、RAG 事实性与无依据回答减少 |
| 4 | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | 8 | RAG 事实性与无依据回答减少、高效推理与 RAG 的可复现系统实现 |
| 5 | [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) | 7 | 长上下文与流式推理、检索结果重排与上下文压缩 |

每篇论文的详细信息见下方评论。

<details><summary>本次运行</summary>

抓取 5 篇 · 重复 0 · 黑名单 0 · 精排 5 · 推送 5<br>
LLM 调用 3 次 · token 6.2k（输入 3.2k / 输出 3.0k） · 耗时 0 分 38 秒

</details>

<!-- gleaner:digest day=2026-10-28 -->
