import os
import json
import asyncio
import argparse
import pandas as pd
from typing import List, Dict, Union
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

from multilingual_factchecking.config import RAW_DATA_DIR, INTERIM_DATA_DIR


async def process_claim_with_links(
    claim: str, label: str, language: str, links: List[str], semaphore: asyncio.Semaphore
) -> Dict:
    """Crawl evidence links and return structured claim data."""
    async with semaphore:  # global concurrency limiter
        claim_data = {"claim": claim, "label": label, "language": language, "sources": []}

        prune_filter = PruningContentFilter(
            threshold=0.45, threshold_type="dynamic", min_word_threshold=5
        )
        md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)
        run_config = CrawlerRunConfig(
            markdown_generator=md_generator, cache_mode=CacheMode.BYPASS
        )

        valid_urls = [url for url in links if isinstance(url, str) and url.startswith("http")]
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
                    logger.warning(f"Failed to crawl {result.url}: {result.error_message}")

        return claim_data


def normalize_row(row: pd.Series, dataset_type: str) -> Dict[str, Union[str, List[str]]]:
    """Normalize row into standard schema regardless of dataset type."""
    if dataset_type == "xfact":
        claim = row.get("claim", "")
        label = row.get("label", "NOT_SPECIFIED")
        language = row.get("language", "unknown")
        links = [row.get(f"link_{i}") for i in range(1, 6)]
    elif dataset_type == "ru22fact":
        claim = row.get("claim", "")
        label = row.get("label", "NOT_SPECIFIED")
        language = row.get("language", "unknown")
        links = [row.get("url")] if pd.notna(row.get("url")) else []
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return {"claim": claim, "label": label, "language": language, "links": links}


async def process_dataset(
    file_path: Path,
    output_path: Path,
    dataset_type: str,
    limit: int = None,
    batch_size: int = 10,
    max_concurrency: int = 50,
):
    """Main pipeline: load dataset, normalize, crawl, and save results in parallel with global concurrency cap."""

    # Load correct file format
    if dataset_type == "xfact":
        df = pd.read_csv(file_path, sep="\t", quoting=3, on_bad_lines="skip")
    elif dataset_type == "ru22fact":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    if limit:
        df = df.head(limit)

    # Load already-processed claims
    already_processed = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    already_processed.add(data["claim"])
                except:
                    continue

    rows = [normalize_row(row, dataset_type) for _, row in df.iterrows()]
    rows = [r for r in rows if r["claim"] and r["claim"] not in already_processed]

    semaphore = asyncio.Semaphore(max_concurrency)

    with open(output_path, "a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(rows), batch_size), desc=f"Processing {dataset_type} in batches"):
            batch = rows[i : i + batch_size]

            tasks = [
                process_claim_with_links(r["claim"], r["label"], r["language"], r["links"], semaphore)
                for r in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for claim_entry in results:
                if isinstance(claim_entry, Exception):
                    logger.error(f"Batch error: {claim_entry}")
                    continue
                json.dump(claim_entry, f, ensure_ascii=False)
                f.write("\n")
                f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web crawl claims for fact-check datasets")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset type (xfact or ru22fact)")
    parser.add_argument("--split", type=str, required=True,
                        help="Which split to process")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional: limit number of rows for debugging")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of claims to process in parallel")
    parser.add_argument("--max_concurrency", type=int, default=50,
                        help="Global concurrency cap for all crawling tasks")

    args = parser.parse_args()

    if args.dataset == "xfact":
        input_file = RAW_DATA_DIR / f"{args.dataset}" / f"{args.split}.tsv"
    elif args.dataset == "ru22fact":
        input_file = RAW_DATA_DIR / f"{args.dataset}" / f"{args.split}.csv"
    else:
        raise ValueError("Unknown dataset")

    output_file = INTERIM_DATA_DIR / f"{args.dataset}" / f"{args.dataset}_{args.split}_with_webdata.jsonl"

    asyncio.run(process_dataset(input_file, output_file, args.dataset, args.limit, args.batch_size, args.max_concurrency))
