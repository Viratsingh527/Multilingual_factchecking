import os
import json
import asyncio
import pandas as pd
from typing import List, Dict
from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CacheMode,
    DefaultMarkdownGenerator,
    PruningContentFilter,
)
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from crawl4ai import RateLimiter
from tqdm import tqdm
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from multilingual_factchecking.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, EXTERNAL_DATA_DIR,INTERIM_DATA_DIR,DATA_DIR

async def process_claim_with_links(
    claim: str, label: str, language: str, links: List[str]
) -> Dict:
    claim_data = {"claim": claim, "label": label, "language": language, "sources": []}

    prune_filter = PruningContentFilter(
        threshold=0.45, threshold_type="dynamic", min_word_threshold=5
    )

    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)

    run_config = CrawlerRunConfig(
        markdown_generator=md_generator, cache_mode=CacheMode.BYPASS
    )

    valid_urls = [
        url for url in links if isinstance(url, str) and url.startswith("http")
    ]

    if not valid_urls:
        return claim_data

    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=90.0,
        check_interval=1.0,
        max_session_permit=5,
        rate_limiter=RateLimiter(base_delay=(1.0, 2.0), max_delay=30.0, max_retries=1),
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun_many(
            urls=valid_urls, config=run_config, dispatcher=dispatcher
        )

        for result in results:
            if result.success and result.markdown.fit_markdown:
                claim_data["sources"].append(
                    {"source": result.url, "content": result.markdown.fit_markdown}
                )
            else:
                print(f"Failed to crawl {result.url}: {result.error_message}")

    return claim_data


async def process_dataset(file_path: str, output_path: str, limit: int = None):
    df = pd.read_csv(file_path, sep="\t", quoting=3, on_bad_lines="skip")

    if limit:
        df = df.head(limit)

    # ✅ ADDED: Load already-processed claims from existing output
    already_processed = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    already_processed.add(data["claim"])
                except:
                    continue

    # ✅ CHANGED: Open output in append mode and save incrementally
    with open(output_path, "a", encoding="utf-8") as f:
        for idx, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Processing claims with multi-URL crawling",
        ):
            claim = row["claim"]
            if claim in already_processed:
                continue  # ✅ ADDED: skip already processed claims

            label = row.get("label", "NOT_SPECIFIED")
            language = row.get("language", "unknown")
            links = [row.get(f"link_{i}") for i in range(1, 6)]

            print(f"Processing claim {idx + 1}/{len(df)}")
            try:
                claim_entry = await process_claim_with_links(
                    claim, label, language, links
                )
                json.dump(claim_entry, f, ensure_ascii=False)
                f.write("\n")
                f.flush()  # ✅ OPTIONAL: ensure data is actually written to disk immediately
            except Exception as e:
                print(f"Error processing claim {idx}: {e}")


if __name__ == "__main__":
    input_file = RAW_DATA_DIR/"ood.tsv"
    output_file = INTERIM_DATA_DIR/"webdata_no_chunks_ood.jsonl"
    asyncio.run(process_dataset(input_file, output_file))
