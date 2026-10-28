"""Generate real model output from public, explicitly curated historical papers.

Run from the repository root: python scripts/build_demo.py
No GitHub writes. Artifacts contain public paper data and this repo's demo profile.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

# Also works when invoked directly rather than as an installed entry point.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gleaner.config import load_config  # noqa: E402
from gleaner.llm import LLMClient  # noqa: E402
from gleaner.models import Paper  # noqa: E402
from gleaner.pipelines.daily import run_daily  # noqa: E402
from gleaner.ranking import build_embedder  # noqa: E402
from gleaner.reader import FulltextFetcher, deep_read  # noqa: E402
from gleaner.render import issue_body, paper_section  # noqa: E402
from gleaner.render.replies import deep_reply  # noqa: E402
from gleaner.sources import ArxivSource  # noqa: E402
from gleaner.store import DataStore  # noqa: E402

PAPER_IDS = ["2307.03172", "2309.06180", "2309.17453", "2310.11511", "2005.11401"]
NOTICE = (
    "> **历史论文演示**：候选为人工选定的 5 篇公开论文，不代表今天的 arXiv 公告。"
    "排序、推荐理由和深读由实际模型调用生成；该样本不用于推荐质量评测。\n\n"
)


@dataclass
class HistoricalSource:
    papers: list[Paper]

    def fetch_latest(self) -> list[Paper]:
        return self.papers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".pytest_cache/phase5-demo"))
    parser.add_argument("--skip-deep", action="store_true")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    # The config lives under docs/demo, but local credentials live at the repo root.
    load_dotenv(Path(".env"), override=False)
    config = load_config(Path("docs/demo/config.yaml"))
    if config.llm.api_key is None:
        raise SystemExit("Missing local LLM credentials; configure .env first.")
    started = time.monotonic()
    print("Fetching public paper metadata...", flush=True)
    papers = ArxivSource(config.sources.arxiv).fetch_by_ids(PAPER_IDS)
    if {paper.id for paper in papers} != set(PAPER_IDS):
        raise SystemExit("Incomplete arXiv response; demo was not generated.")
    llm = LLMClient(config.llm)
    print("Running the production ranking pipeline...", flush=True)
    result = run_daily(
        config,
        store=DataStore(output / "state" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")),
        llm=llm,
        embedder=build_embedder(config),
        today=datetime.now(UTC).date(),
        source=HistoricalSource(papers),
    )
    if result.digest is None or not result.digest.papers or result.digest.stats.degraded:
        raise SystemExit("No successful LLM-ranked digest; demo was not generated.")
    digest = result.digest
    (output / "digest.json").write_text(digest.model_dump_json(indent=2), encoding="utf-8")
    (output / "daily.md").write_text(NOTICE + issue_body(digest), encoding="utf-8")
    for index, paper in enumerate(digest.papers, 1):
        (output / f"paper-{index}.md").write_text(paper_section(index, paper), encoding="utf-8")
    ranking_seconds = time.monotonic() - started
    if not args.skip_deep:
        print("Generating a real deep-read example...", flush=True)
        paper = digest.papers[0]
        fulltext = FulltextFetcher().fetch(paper)
        report = deep_read(
            paper,
            fulltext,
            llm=llm,
            interests=Path("docs/demo/interests.md").read_text(encoding="utf-8"),
            config=config.deep_read,
            language=config.language,
        )
        (output / "deep.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        (output / "deep.md").write_text(deep_reply(paper, report, cached=False), encoding="utf-8")
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "paper_ids": PAPER_IDS,
        "profile": "docs/demo/interests.md",
        "historical_sample": True,
        "ranking_seconds": round(ranking_seconds, 2),
        "total_seconds": round(time.monotonic() - started, 2),
        "fast_model": llm.model_for("fast"),
        "deep_model": llm.model_for("deep"),
        "stats": digest.stats.model_dump(mode="json"),
    }
    (output / "run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "generated",
                "selected": len(digest.papers),
                "ranking_seconds": metadata["ranking_seconds"],
                "total_seconds": metadata["total_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
