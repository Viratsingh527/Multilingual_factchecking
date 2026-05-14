#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import csv
import sys
import gc
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

import nltk
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- Optional-but-helpful: quiets TF/Torch logs if installed ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# /data2/Gaurav/Babu/Multilingual_factchecking/dataset/Evidence_retreiver.py
# -> repo root is parents[1]
repo_root = Path(__file__).resolve().parents[1]
if repo_root.as_posix() not in sys.path:
    sys.path.insert(0, repo_root.as_posix())

from multilingual_factchecking.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
)

LANGUAGE_MAP = {
    "tr": "Turkish",
    "ka": "Georgian",
    "pt": "Portuguese",
    "id": "Indonesian",
    "sr": "Serbian",
    "it": "Italian",
    "de": "German",
    "ro": "Romanian",
    "ta": "Tamil",
    "pl": "Polish",
    "hi": "Hindi",
    "ar": "Arabic",
    "es": "Spanish",
    "bn": "Bengali",
    "fa": "Persian",
    "gu": "Gujarati",
    "mr": "Marathi",
    "pa": "Punjabi",
    "no": "Norwegian",
    "si": "Sinhala",
    "sq": "Albanian",
    "ru": "Russian",
    "az": "Azerbaijani",
    "nl": "Dutch",
    "fr": "French",
}


# =========================================================
# NLTK setup
# =========================================================
def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass


_ensure_nltk()


# =========================================================
# Utilities
# =========================================================
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def clean_markdown_links(text: str) -> str:
    return MARKDOWN_LINK_RE.sub(r"\1", text or "")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False)
        f.write("\n")


def load_processed_ids(path: str) -> Set[str]:
    path = Path(path)
    if not path.exists():
        return set()

    processed_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                row_id = row.get("id")
                if row_id is not None:
                    processed_ids.add(str(row_id))
            except Exception:
                continue
    return processed_ids


def make_dp_key(dp: Dict[str, Any]) -> str:
    if dp.get("id") is not None:
        return str(dp["id"])
    raw = f"{dp.get('claim', '')} || {dp.get('language', '')} || {dp.get('label', '')}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def maybe_clear_cuda(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def translated_xfact_to_internal(dp: Dict[str, Any]) -> Dict[str, Any]:
    evidences = dp.get("translated_evidences", []) or []

    sources = []
    for i, ev in enumerate(evidences):
        if isinstance(ev, str) and ev.strip():
            sources.append({
                "source": f"evidence_{i}",
                "content": ev
            })

    return {
        "id": dp.get("id"),
        "claim": dp.get("translated_claim", ""),
        "label": dp.get("label"),
        "language": dp.get("language", "en"),
        "sources": sources
    }


def convert_csv_to_jsonl(path: str):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for _, row in enumerate(reader):
            results.append({
                "claim": row.get("claim", ""),
                "label": row.get("label", ""),
                "language": row.get("language", ""),
                "evidences": [
                    {
                        "source_index": 0,
                        "source_url": row.get("url", ""),
                        "evidence": row.get("evidence", "")
                    }
                ] if row.get("evidence") else []
            })
    return results


def convert_tsv_to_jsonl(path: str):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for _, row in enumerate(reader):
            evidences = []
            for j in range(1, 6):
                ev_text = row.get(f"evidence_{j}", "")
                ev_url = row.get(f"link_{j}", "")
                if ev_text.strip():
                    evidences.append({
                        "source_index": j - 1,
                        "source_url": ev_url,
                        "evidence": ev_text
                    })
            results.append({
                "claim": row.get("claim", ""),
                "label": row.get("label", ""),
                "language": LANGUAGE_MAP.get(row.get("language", ""), "Unknown"),
                "evidences": evidences
            })
    return results


# =========================================================
# Sentence tokenization helpers
# =========================================================
def split_text_into_sentence_dicts(text: str) -> List[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return []

    sentences = nltk.sent_tokenize(text)
    return [{"sent": s.strip(), "is_evidence": None} for s in sentences if s.strip()]


def build_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


def sentence_chunker(
    text: str,
    tokenizer: AutoTokenizer,
    max_tokens: int = 512,
) -> List[str]:
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

    return [c for c in chunks if c] or [text.strip()]


def fixed_size_chunker(
    text: str,
    tokenizer: AutoTokenizer,
    chunk_size: int = 256,
    chunk_overlap: int = 32,
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    tokens = tokenizer.tokenize(text)
    if not tokens:
        return []

    chunks = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(tokens), step):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]

        if not chunk_tokens:
            continue

        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens).strip()
        if chunk_text:
            chunks.append(chunk_text)

        if end >= len(tokens):
            break

    return chunks


# =========================================================
# Custom semantic chunking
# =========================================================
@torch.no_grad()
def mean_pool_embeddings(
    model,
    tokenizer,
    texts,
    device="cpu",
    max_length=256,
    batch_size=16,
):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        out = model(**enc)
        token_emb = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)

        pooled = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = F.normalize(pooled, p=2, dim=1)

        all_embeddings.append(pooled.cpu())

        del enc, out, token_emb, mask, pooled
        maybe_clear_cuda(device)

    return torch.cat(all_embeddings, dim=0)


def semantic_chunk_from_tokenized(
    tokenized: List[Dict[str, Any]],
    tokenizer,
    model,
    label_key: str = "is_evidence",
    min_chunk_sentences: int = 2,
    alpha: float = 1.0,
    device: str = "cpu",
    max_length: int = 256,
    batch_size: int = 16,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    sentences = []
    labels = []

    for item in tokenized or []:
        s = (item.get("sent") or "").strip()
        if not s:
            continue
        sentences.append(s)
        labels.append(item.get(label_key, None))

    n = len(sentences)
    if n == 0:
        return [], []
    if n == 1:
        chunk = {sentences[0]: labels[0]}
        return [chunk], [sentences[0]]

    emb = mean_pool_embeddings(
        model=model,
        tokenizer=tokenizer,
        texts=sentences,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )

    sims = (emb[:-1] * emb[1:]).sum(dim=1)
    distances = 1.0 - sims
    d = distances.detach().cpu()

    mean = d.mean().item()
    std = d.std(unbiased=False).item()
    threshold = mean + alpha * std

    breakpoints = [i for i, val in enumerate(d.tolist()) if val > threshold]

    chunks_dicts = []
    chunk_texts = []

    start = 0
    for b in breakpoints:
        end = b + 1
        if end - start >= min_chunk_sentences:
            chunk_dict = {sentences[i]: labels[i] for i in range(start, end)}
            chunks_dicts.append(chunk_dict)
            chunk_texts.append(" ".join(sentences[start:end]))
            start = end

    if start < n:
        chunk_dict = {sentences[i]: labels[i] for i in range(start, n)}
        chunks_dicts.append(chunk_dict)
        chunk_texts.append(" ".join(sentences[start:n]))

    if len(chunks_dicts) >= 2 and len(chunks_dicts[-1]) < min_chunk_sentences:
        prev = chunks_dicts[-2]
        prev.update(chunks_dicts[-1])
        chunk_texts[-2] = chunk_texts[-2] + " " + chunk_texts[-1]
        chunks_dicts.pop()
        chunk_texts.pop()

    del emb, sims, distances, d
    gc.collect()
    maybe_clear_cuda(device)

    return chunks_dicts, chunk_texts


def build_custom_semantic_models(model_name: str, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


# =========================================================
# Core pipeline
# =========================================================
def retrieve_evidence_for_dp(
    dp: Dict[str, Any],
    chunk_mode: str,
    sent_tokenizer: AutoTokenizer,
    retrieval_embeddings: HuggingFaceEmbeddings,
    semantic_tokenizer=None,
    semantic_model=None,
    semantic_alpha: float = 0.5,
    semantic_min_chunk_sentences: int = 2,
    semantic_max_length: int = 256,
    semantic_batch_size: int = 16,
    semantic_device: str = "cpu",
    instruction: str = "",
    k: int = 1,
    sent_max_tokens: int = 128,
    fixed_chunk_size: int = 256,
    fixed_chunk_overlap: int = 32,
) -> Dict[str, Any]:
    claim = dp.get("claim", "")
    sources = dp.get("sources", [])
    language = dp.get("language", "en")
    formatted_claim = f"{instruction}\nQuery: {claim}".strip()

    evidence_chunks = []

    for s_idx, source in enumerate(sources or []):
        content = clean_markdown_links(source.get("content", ""))
        if not content.strip():
            continue

        if chunk_mode == "semantic":
            tokenized_sentences = split_text_into_sentence_dicts(content)
            _, chunks = semantic_chunk_from_tokenized(
                tokenized=tokenized_sentences,
                tokenizer=semantic_tokenizer,
                model=semantic_model,
                label_key="is_evidence",
                min_chunk_sentences=semantic_min_chunk_sentences,
                alpha=semantic_alpha,
                device=semantic_device,
                max_length=semantic_max_length,
                batch_size=semantic_batch_size,
            )
            del tokenized_sentences
            maybe_clear_cuda(semantic_device)

        elif chunk_mode == "sentence":
            chunks = sentence_chunker(
                content,
                tokenizer=sent_tokenizer,
                max_tokens=sent_max_tokens,
            )

        elif chunk_mode == "fixed":
            chunks = fixed_size_chunker(
                content,
                tokenizer=sent_tokenizer,
                chunk_size=fixed_chunk_size,
                chunk_overlap=fixed_chunk_overlap,
            )

        else:
            raise ValueError(f"Unsupported chunk_mode: {chunk_mode}")

        docs = [Document(page_content=c.strip()) for c in chunks if c and c.strip()]
        if not docs:
            continue

        vectorstore = FAISS.from_documents(docs, retrieval_embeddings)
        results = vectorstore.similarity_search(formatted_claim, k=k)

        for res in results:
            evidence_chunks.append({
                "source_index": s_idx,
                "source_url": source.get("source"),
                "evidence": res.page_content,
            })

        del docs, vectorstore, results, chunks
        gc.collect()
        maybe_clear_cuda(semantic_device)

    return {
        "id": dp.get("id"),
        "claim": claim,
        "label": dp.get("label"),
        "evidences": evidence_chunks,
        "language": language,
    }


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Retrieve evidence with selectable sentence/custom-semantic/fixed-size chunking."
    )

    parser.add_argument("--dataset", default="xfact", help="Name of dataset.")
    parser.add_argument("--input", default="train", help="Input split/file stem.")

    parser.add_argument(
        "--chunker",
        choices=["sentence", "semantic", "fixed"],
        default="semantic",
        help="Choose sentence-level, custom semantic, or fixed-size chunking."
    )

    parser.add_argument(
        "--sent-tokenizer-model",
        default="xlm-roberta-large",
        help="HF tokenizer for sentence token counts and fixed-size chunking."
    )
    parser.add_argument(
        "--sent-max-tokens",
        type=int,
        default=256,
        help="Max tokens per sentence chunk group."
    )
    parser.add_argument(
        "--fixed-chunk-size",
        type=int,
        default=256,
        help="Number of tokens per fixed chunk."
    )
    parser.add_argument(
        "--fixed-chunk-overlap",
        type=int,
        default=32,
        help="Token overlap between consecutive fixed chunks."
    )

    parser.add_argument(
        "--retrieval-embed-model",
        default="intfloat/multilingual-e5-large-instruct",
        help="Embedding model used for FAISS retrieval."
    )

    parser.add_argument(
        "--semantic-model",
        # default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        default="intfloat/multilingual-e5-large-instruct",
        help="HF model for custom semantic chunking sentence embeddings."
    )
    parser.add_argument(
        "--semantic-alpha",
        type=float,
        default=1.0,
        help="Adaptive threshold factor for custom semantic chunking."
    )
    parser.add_argument(
        "--semantic-min-chunk-sentences",
        type=int,
        default=2,
        help="Minimum number of sentences per semantic chunk."
    )
    parser.add_argument(
        "--semantic-max-length",
        type=int,
        default=256,
        help="Max token length per sentence for semantic embedding."
    )
    parser.add_argument(
        "--semantic-batch-size",
        type=int,
        default=16,
        help="Batch size for semantic sentence embedding."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for custom semantic chunking model, e.g. cpu or cuda."
    )

    parser.add_argument("--k", type=int, default=1, help="Top-k chunks per source.")
    parser.add_argument(
        "--instruction",
        default="Instruct: Given a claim, retrieve relevant evidence from web documents that support or refute the claim",
        help="Instruction prefix for the query."
    )

    parser.add_argument(
        "--retriever",
        dest="retriever",
        action="store_true",
        help="Enable evidence retrieval (default: True)."
    )
    parser.add_argument(
        "--no-retriever",
        dest="retriever",
        action="store_false",
        help="Disable evidence retrieval: only convert CSV/TSV into JSONL."
    )
    parser.set_defaults(retriever=True)

    args = parser.parse_args()

    if args.fixed_chunk_overlap >= args.fixed_chunk_size:
        raise ValueError("--fixed-chunk-overlap must be smaller than --fixed-chunk-size")

    type_of_evidence = "default"

    if args.retriever:
        if args.dataset == "translated_xfact":
            input_file = INTERIM_DATA_DIR / "translated_xfact" / f"{args.input}.jsonl"
        else:
            input_file = INTERIM_DATA_DIR / "X-FACT" / f"{args.input}.jsonl"
            # input_file = INTERIM_DATA_DIR / args.dataset / f"{args.dataset}_{args.input}_with_webdata.jsonl"
    else:
        if args.dataset == "xfact":
            input_file = RAW_DATA_DIR / args.dataset / f"{args.input}.tsv"
            type_of_evidence = "search_snippet"
        elif args.dataset == "ru22fact":
            input_file = RAW_DATA_DIR / args.dataset / f"{args.input}.csv"
            type_of_evidence = "llm_generated"
        else:
            raise ValueError("Unsupported dataset for --no-retriever mode.")

        input_ext = Path(input_file).suffix.lower()
        if input_ext == ".csv":
            all_results = convert_csv_to_jsonl(input_file)
        elif input_ext == ".tsv":
            all_results = convert_tsv_to_jsonl(input_file)
        else:
            raise ValueError("When --no-retriever is set, input must be CSV or TSV.")

        output_file = PROCESSED_DATA_DIR / args.dataset / f"{args.dataset}_{args.input}_with_{type_of_evidence}_evidences.jsonl"
        save_jsonl(output_file, all_results)
        print(f"Done. Wrote {len(all_results)} rows to {output_file}")
        return

    datapoints = load_jsonl(input_file)

    if args.chunker == "sentence":
        output_file = (
            PROCESSED_DATA_DIR / f"{args.dataset}" /
            f"{args.dataset}_{args.input}_with_sentence_level_chunked_retrieved_evidence.jsonl"
        )
    elif args.chunker == "semantic":
        output_file = (
            PROCESSED_DATA_DIR / f"{args.dataset}_alpha={args.semantic_alpha}" /
            f"{args.dataset}_{args.input}_with_custom_semantic_chunked_retrieved_evidence.jsonl"
        )
    elif args.chunker == "fixed":
        output_file = (
            PROCESSED_DATA_DIR / f"{args.dataset}_fixed_{args.fixed_chunk_size}_{args.fixed_chunk_overlap}" /
            f"{args.dataset}_{args.input}_with_fixed_size_chunked_retrieved_evidence.jsonl"
        )
    else:
        raise ValueError(f"Unsupported chunker: {args.chunker}")

    processed_ids = load_processed_ids(output_file)
    print(f"Found {len(processed_ids)} already processed datapoints in {output_file}")

    sent_tokenizer = build_tokenizer(args.sent_tokenizer_model)

    retrieval_embeddings = HuggingFaceEmbeddings(
        model_name=args.retrieval_embed_model
    )

    semantic_tokenizer = None
    semantic_model = None

    if args.chunker == "semantic":
        semantic_tokenizer, semantic_model = build_custom_semantic_models(
            model_name=args.semantic_model,
            device=args.device,
        )

    skipped = 0
    processed_now = 0

    for dp in tqdm(datapoints, desc="Processing datapoints"):
        if args.dataset == "translated_xfact":
            dp = translated_xfact_to_internal(dp)

        dp_key = make_dp_key(dp)

        if dp_key in processed_ids:
            skipped += 1
            continue

        result = retrieve_evidence_for_dp(
            dp=dp,
            chunk_mode=args.chunker,
            sent_tokenizer=sent_tokenizer,
            retrieval_embeddings=retrieval_embeddings,
            semantic_tokenizer=semantic_tokenizer,
            semantic_model=semantic_model,
            semantic_alpha=args.semantic_alpha,
            semantic_min_chunk_sentences=args.semantic_min_chunk_sentences,
            semantic_max_length=args.semantic_max_length,
            semantic_batch_size=args.semantic_batch_size,
            semantic_device=args.device,
            instruction=args.instruction,
            k=args.k,
            sent_max_tokens=args.sent_max_tokens,
            fixed_chunk_size=args.fixed_chunk_size,
            fixed_chunk_overlap=args.fixed_chunk_overlap,
        )

        if result.get("id") is None:
            result["id"] = dp_key

        append_jsonl(output_file, result)
        processed_ids.add(str(result["id"]))
        processed_now += 1

        del result
        gc.collect()
        maybe_clear_cuda(args.device)

    print(f"Done. Newly processed: {processed_now}, skipped already done: {skipped}")
    print(f"Output saved incrementally to {output_file}")


if __name__ == "__main__":
    main()