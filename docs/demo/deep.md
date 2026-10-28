## 🔍 深读 · [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

**作者**：Woosuk Kwon, Zhuohan Li, Siyuan Zhuang 等 · **分类**：cs.LG, cs.DC
[arXiv](https://arxiv.org/abs/2309.06180) · [PDF](https://arxiv.org/pdf/2309.06180) · [HTML](https://arxiv.org/html/2309.06180)

### 一句话总结
论文提出受操作系统虚拟内存与分页启发的 PagedAttention，把 KV cache 分成非连续物理块管理，并构建 vLLM 服务系统，在相同延迟下相对 FasterTransformer、Orca 等系统提升 2-4× 吞吐。

### 问题与动机
LLM 自回归生成时，KV cache 随请求动态增长，且长度事先未知；服务吞吐受 GPU 显存容量限制，核心是 KV cache 管理效率。现有系统把每个请求的 KV cache 存在连续内存中，并按最大可能长度预分配，导致三类浪费：为未来 token 预留空间、内部碎片和外部碎片；论文测得实际用于存储 token 状态的 KV cache 只有 20.4% - 38.2%。此外，连续预分配使 parallel sampling、beam search、shared prefix 等场景难以共享 KV cache，进一步限制 batch size。已有 fine-grained batching 降低了计算和 padding 浪费，但 batch size 仍受 KV cache 显存约束。

### 方法要点
- 将每个序列的 KV cache 划分为固定大小的 KV blocks，PagedAttention 按 block 读取非连续的 key/value，并做 block-wise attention 计算；逻辑上连续、物理上可不连续。
- KV cache manager 维护 logical KV block 到 physical KV block 的 block table 和填充位置记录，按需分配物理块；请求的浪费被限制在最后一个 block 内，从而接近零浪费并消除外部碎片。
- 用引用计数和 block 级 copy-on-write 支持共享：parallel sampling 共享 prompt 的 KV cache；beam search 在不同候选间动态共享和释放 block；shared prefix 可预置共享前缀的物理块。
- 调度采用 first-come-first-serve，并以 sequence group 为单位 gang-schedule；显存不足时按 all-or-nothing 策略驱逐整个序列的块，恢复方式包括 swapping 到 CPU 内存和 recomputation。
- 分布式执行支持 Megatron-LM 风格 tensor model parallelism，中心 scheduler 共享 block table，GPU worker 只保存对应 attention heads 的 KV cache。
- 实现上提供 fused reshape and block write、fusing block read and attention、fused block copy 等 GPU kernel；用 fork、append、free 三个操作支持多种 decoding 算法。默认 block size 为 16。

### 实验
实验使用 OPT-13B、OPT-66B、OPT-175B 和 LLaMA-13B，硬件为 NVIDIA A100；工作负载由 ShareGPT 和 Alpaca 的输入/输出长度合成，请求到达时间用 Poisson 分布生成；shared prefix 实验使用 WMT16 English-to-German 翻译数据。基线为 FasterTransformer 和作者实现的 Orca，包括 Orca (Oracle)、Orca (Pow2)、Orca (Max)。主要指标是 normalized latency，即每个请求端到端延迟除以其输出长度。

| 场景 | 对比基线 | 关键结果 |
| --- | --- | --- |
| Basic sampling，ShareGPT | Orca、FasterTransformer | vLLM 可承受的请求率是 Orca (Oracle) 的 1.7×–2.7×、Orca (Max) 的 2.7×–8×，相对 FasterTransformer 至多 22×；OPT-13B 同时批处理的请求数为 Orca (Oracle) 的 2.2×、Orca (Max) 的 4.3× |
| Basic sampling，Alpaca | Orca、FasterTransformer | 趋势类似；OPT-175B 上 vLLM 相对 Orca (Oracle)/Orca (Pow2) 的优势较弱，原因是该配置 KV cache 空间较大且 Alpaca 序列短，系统转为 compute-bound |
| Parallel sampling / beam search，OPT-13B，Alpaca | Orca (Oracle) | 相对 Orca (Oracle) 的提升从 basic sampling 的 1.3× 增至 beam search width 6 的 2.3× |
| KV 共享节省 | — | Alpaca：parallel sampling 6.1% - 9.8%，beam search 37.6% - 55.2%；ShareGPT：parallel sampling 16.2% - 30.5%，beam search 44.3% - 66.3% |
| Shared prefix，LLaMA-13B，WMT16 En-De | Orca (Oracle) | 共享 1 个示例、80 tokens 时吞吐为 1.67×；共享 5 个示例、341 tokens 时为 3.58× |
| Chatbot，ShareGPT | 三个 Orca 基线 | vLLM 可承受 2× 更高请求率 |
| Attention kernel microbenchmark | FasterTransformer | PagedAttention attention kernel 延迟高 20–26% |

消融实验结论：block size 过小不能充分利用 GPU 并行度，过大则增加内部碎片并降低共享概率；ShareGPT 上 16 到 128 的 block size 表现最好，Alpaca 上 16 和 32 较好，更大 block size 明显退化，因此默认取 16。Recomputation 与 swapping 的权衡是：小 block size 下 swapping 因大量小数据传输开销过大，recomputation 开销在不同 block size 下基本恒定；recomputation 开销不超过 swapping 延迟的 20%；中等 block size 16 到 64 时两者端到端性能接近。

### 代码与资源
官方代码：https://github.com/vllm-project/vllm ，论文摘要和正文均说明 vLLM 源码公开在此仓库。&lt;links> 中未提供数据集或模型链接；正文使用了 ShareGPT、Alpaca、WMT16 以及 OPT、LLaMA 模型，但对应下载或模型仓库链接文中未给出。

### 与你研究的关系
这篇论文直接对应“KV cache 管理、长上下文与流式推理、显存和吞吐改进”的兴趣点。可借鉴之处很具体：用 block table 把逻辑 KV 块映射到非连续物理块，按需分配并把浪费限制在一个 block 内；用引用计数和 copy-on-write 做块级共享；把 prefill 阶段的共享前缀缓存成物理块，减少重复计算；在显存压力下用 swapping 或 recomputation 做抢占恢复。这些设计适合复现为长上下文/流式服务的 KV cache 管理基线，也适合分析 block size、抢占策略和共享粒度对吞吐与尾延迟的影响。

对 RAG 兴趣点，这篇论文不是关于“何时需要检索”或“减少无依据回答”，不评估检索必要性和 groundedness；但它对 RAG 的长 system prompt、few-shot 示例、重复文档前缀有间接价值：shared prefix 的 KV block 缓存可以减少相同前缀的重复 prefill 开销。若要把检索到的文档块做成可复用前缀，仍需自行设计缓存失效、淘汰和跨请求一致性策略，论文未说明。限制也很明确：它不压缩 KV cache 本身，主要解决 memory-bound serving；在 compute-bound 或短序列场景下收益会减弱。

### 局限与可追问点
- 正文主要报告吞吐、延迟和内存节省，没有给出精度评测表；摘要声称“不影响模型准确率”，但共享块、copy-on-write 和调度策略对生成质量的独立验证文中未说明。
- Orca 并非公开可用，作者实现了自己的 Orca 版本并假定 buddy allocation；Orca (Oracle) 还假设知道真实输出长度，实际不可实现，因此对比结果依赖复现实现和该上界设定。
- 工作负载由 ShareGPT、Alpaca 的 token 长度合成，并用 Poisson 分布生成到达时间，不是带真实时间戳的线上 trace；多数实验用 1 小时 trace，OPT-175B 因成本限制只用 15 分钟 trace。
- Shared prefix 的缓存生命周期、淘汰策略、前缀更新后的失效机制未展开；用于 RAG 动态检索文档时，如何管理不断变化的共享块值得追问。
- 抢占时 vLLM 会停止接受新请求直到所有被抢占序列完成；这对 burst 流量下的尾延迟和 SLO 影响未充分评估。中心 scheduler 和全局 block table 在大规模分布式、多租户场景下的扩展性也未评估。
- Block size 默认 16，但最优值依赖序列长度分布；小 block 增加 swapping 开销，大 block 增加内部碎片并降低共享概率。PagedAttention attention kernel 本身比 FasterTransformer 高 20–26% 延迟，虽然端到端仍占优。
- 可追问：能否把块式管理与 KV cache 量化/压缩、稀疏注意力结合？能否自适应选择 block size？在长上下文 RAG 中，检索文档变化时共享前缀缓存如何失效和重建？在真实高并发线上 trace 下，抢占策略对 P99 延迟的影响如何？

---
<sub>全文来源：arXiv HTML · 模型 deepseek-v4.1-flash-1017 · 2026-10-28 生成</sub>