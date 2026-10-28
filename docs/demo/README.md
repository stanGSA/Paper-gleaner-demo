# 公开 Demo 的来源与复现

公开仓库：[stanGSA/Paper-gleaner-demo](https://github.com/stanGSA/Paper-gleaner-demo)。

兴趣描述使用本目录的 `interests.md`，与个人配置隔离。候选人工选择以下 5 篇历史论文：PagedAttention（2309.06180）、StreamingLLM（2309.17453）、Self-RAG（2310.11511）、RAG（2005.11401）与 Lost in the Middle（2307.03172）。实际生成的样本及计时见 `run.json`。

这些候选用于展示推荐理由和深读界面，不用于 precision@10 评测，也不冒充当日公告。排序、解释和报告调用真实模型生成。生成器复用生产流水线和渲染函数，不写入个人 `gleaner-data`，不发布 GitHub Issue。

在仓库根目录安装依赖、配置本地 `.env` 后：

```bash
uv run python scripts/build_demo.py --output .pytest_cache/phase5-demo-live
```

生成的 `digest.json`、`daily.md`、`paper-*.md`、`deep.md` 和 `run.json` 包含公开结果及计时。需要换用服务商提供的模型别名时，可为此命令设置 `LLM_FAST_MODEL` 和 `LLM_DEEP_MODEL`；不要修改或提交密钥。

Demo 仓库维护可浏览的固定示例；手动 Actions 演示需要仓库所有者配置自己的 Secrets。演示仓库关闭定时触发，避免无人维护时持续产生费用。访客可以浏览；交互命令受 OWNER / MEMBER / COLLABORATOR 权限限制。
