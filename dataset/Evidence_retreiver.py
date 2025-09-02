#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
# --- Optional-but-helpful: quiets TF/Torch logs if installed ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# 3rd-party deps
import sys
from pathlib import Path

# /data2/Gaurav/Babu/Multilingual_factchecking/dataset/Evidence_retreiver.py
# -> repo root is parents[1]
repo_root = Path(__file__).resolve().parents[1]
if repo_root.as_posix() not in sys.path:
    sys.path.insert(0, repo_root.as_posix())

import nltk
from transformers import AutoTokenizer
from langchain.docstore.document import Document
from multilingual_factchecking.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, EXTERNAL_DATA_DIR,INTERIM_DATA_DIR,DATA_DIR

# Some users have a newer NLTK requiring "punkt" and "punkt_tab" for sentencepiece-like tokenizers.
def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        # Not present on older NLTK; ignore failures
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

_ensure_nltk()

# =========================
# Utilities
# =========================
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

def clean_markdown_links(text: str) -> str:
    """Strip markdown links to plain text."""
    return MARKDOWN_LINK_RE.sub(r"\1", text or "")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

# =========================
# Chunkers
# =========================
def build_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)

def sentence_chunker(
    text: str,
    tokenizer: AutoTokenizer,
    max_tokens: int = 512,
) -> List[str]:
    """Chunk text by sentences, grouping until token budget is reached."""
    if not text:
        return []
    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk, current_len = [], [], 0

    for sentence in sentences:
        tok_count = len(tokenizer.tokenize(sentence))
        if current_len + tok_count > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())
            current_chunk = [sentence]
            current_len = tok_count
        else:
            current_chunk.append(sentence)
            current_len += tok_count

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    # Fallback: if tokenization somehow yields nothing, return the raw text
    return [c for c in chunks if c] or [text.strip()]

def build_semantic_splitter(
    model_name: str,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: int = 95,
    min_chunk_size: int = 2,
):
    """Create a SemanticChunker instance with a reusable embedding model."""
    # Imported here to avoid import cost for sentence-only mode
    from langchain_experimental.text_splitter import SemanticChunker

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
        min_chunk_size=min_chunk_size,
    )
    return splitter, embeddings

# =========================
# Core pipeline
# =========================
def retrieve_evidence_for_dp(
    dp: Dict[str, Any],
    chunk_mode: str,
    tokenizer: AutoTokenizer,
    embeddings: HuggingFaceEmbeddings,
    splitter,  # SemanticChunker or None
    instruction: str,
    k: int = 1,
    sent_max_tokens: int = 128,
) -> Dict[str, Any]:
    claim = dp.get("claim", "")
    sources = dp.get("sources", [])
    language = dp.get("language")
    formatted_claim = f"{instruction}\nQuery: {claim}".strip()

    evidence_chunks = []

    for s_idx, source in enumerate(sources or []):
        content = clean_markdown_links(source.get("content", ""))
        if not content.strip():
            continue

        # ---- choose chunker ----
        if chunk_mode == "semantic":
            if splitter is None:
                raise RuntimeError("Semantic mode selected but splitter is None.")
            chunks = splitter.split_text(content)
        else:  # "sentence"
            chunks = sentence_chunker(content, tokenizer=tokenizer, max_tokens=sent_max_tokens)

        docs = [Document(page_content=c.strip()) for c in chunks if c and c.strip()]
        if not docs:
            continue

        # Reuse one vectorstore per source (keeps memory manageable across dataset)
        vectorstore = FAISS.from_documents(docs, embeddings)
        result = vectorstore.similarity_search(formatted_claim, k=k)

        if result:
            evidence_chunks.append(
                {
                    "source_index": s_idx,
                    "source_url": source.get("source"),
                    "evidence": result[0].page_content,
                }
            )

    return {
        "claim": claim,
        "label": dp.get("label"),
        "evidences": evidence_chunks,
        "language": language,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve web evidence per claim with selectable chunking."
    )
    # I/O
    parser.add_argument(
        "--input",
        default="train",
        help="Name of input JSONL file that needs to be retrieved.",
    )

    # Chunking mode
    parser.add_argument(
        "--chunker",
        choices=["sentence", "semantic"],
        default="semantic",
        help="Choose sentence-level or semantic chunking.",
    )

    # Sentence chunker params
    parser.add_argument(
        "--sent-tokenizer-model",
        default="xlm-roberta-large",
        help="HF tokenizer for sentence token counts.",
    )
    parser.add_argument(
        "--sent-max-tokens",
        type=int,
        default=128,
        help="Max tokens per sentence chunk group.",
    )

    # Semantic chunker params
    parser.add_argument(
        "--semantic-embed-model",
        default="intfloat/multilingual-e5-large-instruct",
        help="Embedding model for semantic chunking + retrieval.",
    )
    parser.add_argument(
        "--breakpoint-threshold-type",
        default="percentile",
        choices=["percentile", "standard_deviation"],
        help="SemanticChunker thresholding type.",
    )
    parser.add_argument(
        "--breakpoint-threshold-amount",
        type=int,
        default=95,
        help="Threshold amount (meaning depends on type).",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=2,
        help="Minimum number of sentences per semantic chunk.",
    )

    # Retrieval
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Top-k documents to retrieve from FAISS per source.",
    )
    parser.add_argument(
        "--instruction",
        default="Instruct: Given a claim, retrieve relevant evidence from web documents that support or refute the claim",
        help="Instruction prefix for the query.",
    )

    args = parser.parse_args()

    # ---- Load data ----
    input_file = INTERIM_DATA_DIR/f"xfact_{args.input}_with_webdata.jsonl"
    datapoints = load_jsonl(input_file)

    # ---- Build models once ----
    tokenizer = build_tokenizer(args.sent_tokenizer_model)

    splitter = None
    embeddings = None

    if args.chunker == "semantic":
        splitter, embeddings = build_semantic_splitter(
            model_name=args.semantic_embed_model,
            breakpoint_threshold_type=args.breakpoint_threshold_type,
            breakpoint_threshold_amount=args.breakpoint_threshold_amount,
            min_chunk_size=args.min_chunk_size,
        )
    else:
        # Even in sentence mode, we still need embeddings for FAISS
        embeddings = HuggingFaceEmbeddings(model_name=args.semantic_embed_model)

    # ---- Process ----
    all_results = []
    for dp in tqdm(datapoints, desc="Processing datapoints"):
        result = retrieve_evidence_for_dp(
            dp=dp,
            chunk_mode=args.chunker,
            tokenizer=tokenizer,
            embeddings=embeddings,
            splitter=splitter,
            instruction=args.instruction,
            k=args.k,
            sent_max_tokens=args.sent_max_tokens,
        )
        all_results.append(result)

    # ---- Save ----
    output_file = PROCESSED_DATA_DIR/f"xfact_{args.input}_with_retrieved_evidence.jsonl"
    save_jsonl(output_file, all_results)

    print(f"Done. Wrote {len(all_results)} rows to {output_file}")

if __name__ == "__main__":
    main()
