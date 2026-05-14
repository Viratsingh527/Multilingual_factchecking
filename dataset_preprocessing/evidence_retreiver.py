# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import argparse
# import json
# import os
# import re
# import csv
# import sys
# import gc
# import hashlib
# from pathlib import Path
# from typing import List, Dict, Any, Tuple, Set, Optional
# from dataclasses import dataclass

# import numpy as np
# import nltk
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModel
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document

# # -----------------------------
# # Optional: quiet tokenizer logs
# # -----------------------------
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# # /data2/Gaurav/Babu/Multilingual_factchecking/dataset/Evidence_retreiver.py
# # -> repo root is parents[1]
# repo_root = Path(__file__).resolve().parents[1]
# if repo_root.as_posix() not in sys.path:
#     sys.path.insert(0, repo_root.as_posix())

# from multilingual_factchecking.config import (
#     PROCESSED_DATA_DIR,
#     RAW_DATA_DIR,
#     INTERIM_DATA_DIR,
# )

# LANGUAGE_MAP = {
#     "tr": "Turkish",
#     "ka": "Georgian",
#     "pt": "Portuguese",
#     "id": "Indonesian",
#     "sr": "Serbian",
#     "it": "Italian",
#     "de": "German",
#     "ro": "Romanian",
#     "ta": "Tamil",
#     "pl": "Polish",
#     "hi": "Hindi",
#     "ar": "Arabic",
#     "es": "Spanish",
#     "bn": "Bengali",
#     "fa": "Persian",
#     "gu": "Gujarati",
#     "mr": "Marathi",
#     "pa": "Punjabi",
#     "no": "Norwegian",
#     "si": "Sinhala",
#     "sq": "Albanian",
#     "ru": "Russian",
#     "az": "Azerbaijani",
#     "nl": "Dutch",
#     "fr": "French",
# }


# # =========================================================
# # NLTK setup
# # =========================================================
# def _ensure_nltk():
#     try:
#         nltk.data.find("tokenizers/punkt")
#     except LookupError:
#         nltk.download("punkt", quiet=True)

#     try:
#         nltk.data.find("tokenizers/punkt_tab")
#     except LookupError:
#         try:
#             nltk.download("punkt_tab", quiet=True)
#         except Exception:
#             pass


# _ensure_nltk()


# # =========================================================
# # Utilities
# # =========================================================
# MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


# def clean_markdown_links(text: str) -> str:
#     return MARKDOWN_LINK_RE.sub(r"\1", text or "")


# def load_jsonl(path: str) -> List[Dict[str, Any]]:
#     with open(path, "r", encoding="utf-8") as f:
#         return [json.loads(line) for line in f if line.strip()]


# def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with open(path, "w", encoding="utf-8") as f:
#         for row in rows:
#             json.dump(row, f, ensure_ascii=False)
#             f.write("\n")


# def append_jsonl(path: str, row: Dict[str, Any]) -> None:
#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with open(path, "a", encoding="utf-8") as f:
#         json.dump(row, f, ensure_ascii=False)
#         f.write("\n")


# def load_processed_ids(path: str) -> Set[str]:
#     path = Path(path)
#     if not path.exists():
#         return set()

#     processed_ids = set()
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 row = json.loads(line)
#                 row_id = row.get("id")
#                 if row_id is not None:
#                     processed_ids.add(str(row_id))
#             except Exception:
#                 continue
#     return processed_ids


# def make_dp_key(dp: Dict[str, Any]) -> str:
#     if dp.get("id") is not None:
#         return str(dp["id"])
#     raw = f"{dp.get('claim', '')} || {dp.get('language', '')} || {dp.get('label', '')}"
#     return hashlib.md5(raw.encode("utf-8")).hexdigest()


# def maybe_clear_cuda(device: str) -> None:
#     if str(device).startswith("cuda") and torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()


# def translated_xfact_to_internal(dp: Dict[str, Any]) -> Dict[str, Any]:
#     evidences = dp.get("translated_evidences", []) or []

#     sources = []
#     for i, ev in enumerate(evidences):
#         if isinstance(ev, str) and ev.strip():
#             sources.append({
#                 "source": f"evidence_{i}",
#                 "content": ev
#             })

#     return {
#         "id": dp.get("id"),
#         "claim": dp.get("translated_claim", ""),
#         "label": dp.get("label"),
#         "language": dp.get("language", "en"),
#         "sources": sources
#     }


# def convert_csv_to_jsonl(path: str):
#     results = []
#     with open(path, "r", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         for _, row in enumerate(reader):
#             results.append({
#                 "claim": row.get("claim", ""),
#                 "label": row.get("label", ""),
#                 "language": row.get("language", ""),
#                 "evidences": [
#                     {
#                         "source_index": 0,
#                         "source_url": row.get("url", ""),
#                         "evidence": row.get("evidence", "")
#                     }
#                 ] if row.get("evidence") else []
#             })
#     return results


# def convert_tsv_to_jsonl(path: str):
#     results = []
#     with open(path, "r", encoding="utf-8") as f:
#         reader = csv.DictReader(f, delimiter="\t")
#         for _, row in enumerate(reader):
#             evidences = []
#             for j in range(1, 6):
#                 ev_text = row.get(f"evidence_{j}", "")
#                 ev_url = row.get(f"link_{j}", "")
#                 if ev_text.strip():
#                     evidences.append({
#                         "source_index": j - 1,
#                         "source_url": ev_url,
#                         "evidence": ev_text
#                     })
#             results.append({
#                 "claim": row.get("claim", ""),
#                 "label": row.get("label", ""),
#                 "language": LANGUAGE_MAP.get(row.get("language", ""), "Unknown"),
#                 "evidences": evidences
#             })
#     return results


# # =========================================================
# # Sentence tokenization helpers
# # =========================================================
# def split_text_into_sentence_dicts(text: str) -> List[Dict[str, Any]]:
#     text = (text or "").strip()
#     if not text:
#         return []

#     sentences = nltk.sent_tokenize(text)
#     return [{"sent": s.strip(), "is_evidence": None} for s in sentences if s.strip()]


# def build_tokenizer(model_name: str) -> AutoTokenizer:
#     return AutoTokenizer.from_pretrained(model_name)


# def sentence_chunker(
#     text: str,
#     tokenizer: AutoTokenizer,
#     max_tokens: int = 512,
# ) -> List[str]:
#     if not text:
#         return []

#     sentences = nltk.sent_tokenize(text)
#     chunks, current_chunk, current_len = [], [], 0

#     for sentence in sentences:
#         tok_count = len(tokenizer.tokenize(sentence))

#         if current_len + tok_count > max_tokens:
#             if current_chunk:
#                 chunks.append(" ".join(current_chunk).strip())
#             current_chunk = [sentence]
#             current_len = tok_count
#         else:
#             current_chunk.append(sentence)
#             current_len += tok_count

#     if current_chunk:
#         chunks.append(" ".join(current_chunk).strip())

#     return [c for c in chunks if c] or [text.strip()]


# def fixed_size_chunker(
#     text: str,
#     tokenizer: AutoTokenizer,
#     chunk_size: int = 256,
#     chunk_overlap: int = 32,
# ) -> List[str]:
#     text = (text or "").strip()
#     if not text:
#         return []

#     if chunk_overlap >= chunk_size:
#         raise ValueError("chunk_overlap must be smaller than chunk_size")

#     tokens = tokenizer.tokenize(text)
#     if not tokens:
#         return []

#     chunks = []
#     step = chunk_size - chunk_overlap

#     for start in range(0, len(tokens), step):
#         end = start + chunk_size
#         chunk_tokens = tokens[start:end]

#         if not chunk_tokens:
#             continue

#         chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens).strip()
#         if chunk_text:
#             chunks.append(chunk_text)

#         if end >= len(tokens):
#             break

#     return chunks


# # =========================================================
# # Embedding helpers
# # =========================================================
# @torch.no_grad()
# def mean_pool_embeddings(
#     model,
#     tokenizer,
#     texts,
#     device="cpu",
#     max_length=256,
#     batch_size=16,
# ):
#     all_embeddings = []

#     for i in range(0, len(texts), batch_size):
#         batch_texts = texts[i:i + batch_size]

#         enc = tokenizer(
#             batch_texts,
#             padding=True,
#             truncation=True,
#             max_length=max_length,
#             return_tensors="pt"
#         ).to(device)

#         out = model(**enc)
#         token_emb = out.last_hidden_state
#         mask = enc["attention_mask"].unsqueeze(-1)

#         pooled = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
#         pooled = F.normalize(pooled, p=2, dim=1)

#         all_embeddings.append(pooled.cpu())

#         del enc, out, token_emb, mask, pooled
#         maybe_clear_cuda(device)

#     return torch.cat(all_embeddings, dim=0)


# @torch.no_grad()
# def embed_texts_mean_pool(
#     model,
#     tokenizer,
#     texts: List[str],
#     device: str = "cpu",
#     max_length: int = 256,
#     batch_size: int = 16,
# ) -> torch.Tensor:
#     return mean_pool_embeddings(
#         model=model,
#         tokenizer=tokenizer,
#         texts=texts,
#         device=device,
#         max_length=max_length,
#         batch_size=batch_size,
#     )


# # =========================================================
# # Original custom semantic chunking
# # =========================================================
# def semantic_chunk_from_tokenized(
#     tokenized: List[Dict[str, Any]],
#     tokenizer,
#     model,
#     label_key: str = "is_evidence",
#     min_chunk_sentences: int = 2,
#     alpha: float = 1.0,
#     device: str = "cpu",
#     max_length: int = 256,
#     batch_size: int = 16,
# ) -> Tuple[List[Dict[str, Any]], List[str]]:
#     sentences = []
#     labels = []

#     for item in tokenized or []:
#         s = (item.get("sent") or "").strip()
#         if not s:
#             continue
#         sentences.append(s)
#         labels.append(item.get(label_key, None))

#     n = len(sentences)
#     if n == 0:
#         return [], []
#     if n == 1:
#         chunk = {sentences[0]: labels[0]}
#         return [chunk], [sentences[0]]

#     emb = mean_pool_embeddings(
#         model=model,
#         tokenizer=tokenizer,
#         texts=sentences,
#         device=device,
#         max_length=max_length,
#         batch_size=batch_size,
#     )

#     sims = (emb[:-1] * emb[1:]).sum(dim=1)
#     distances = 1.0 - sims
#     d = distances.detach().cpu()

#     mean = d.mean().item()
#     std = d.std(unbiased=False).item()
#     threshold = mean + alpha * std

#     breakpoints = [i for i, val in enumerate(d.tolist()) if val > threshold]

#     chunks_dicts = []
#     chunk_texts = []

#     start = 0
#     for b in breakpoints:
#         end = b + 1
#         if end - start >= min_chunk_sentences:
#             chunk_dict = {sentences[i]: labels[i] for i in range(start, end)}
#             chunks_dicts.append(chunk_dict)
#             chunk_texts.append(" ".join(sentences[start:end]))
#             start = end

#     if start < n:
#         chunk_dict = {sentences[i]: labels[i] for i in range(start, n)}
#         chunks_dicts.append(chunk_dict)
#         chunk_texts.append(" ".join(sentences[start:n]))

#     if len(chunks_dicts) >= 2 and len(chunks_dicts[-1]) < min_chunk_sentences:
#         prev = chunks_dicts[-2]
#         prev.update(chunks_dicts[-1])
#         chunk_texts[-2] = chunk_texts[-2] + " " + chunk_texts[-1]
#         chunks_dicts.pop()
#         chunk_texts.pop()

#     del emb, sims, distances, d
#     gc.collect()
#     maybe_clear_cuda(device)

#     return chunks_dicts, chunk_texts


# def build_custom_semantic_models(model_name: str, device: str = "cpu"):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name).to(device)
#     model.eval()
#     return tokenizer, model


# # =========================================================
# # Improved semantic chunking
# # =========================================================
# @dataclass
# class ChunkRecord:
#     source_index: int
#     source_url: str
#     chunk_text: str
#     chunk_id: str


# def smooth_array(values: List[float], window: int = 3) -> List[float]:
#     if not values:
#         return values
#     if window <= 1:
#         return values

#     arr = np.array(values, dtype=np.float32)
#     pad = window // 2
#     padded = np.pad(arr, (pad, pad), mode="edge")

#     smoothed = []
#     for i in range(len(arr)):
#         smoothed.append(float(np.mean(padded[i:i + window])))
#     return smoothed


# def detect_semantic_boundaries_windowed(
#     sentences: List[str],
#     tokenizer,
#     model,
#     device: str = "cpu",
#     max_length: int = 256,
#     batch_size: int = 16,
#     left_window: int = 2,
#     right_window: int = 2,
#     smoothing_window: int = 3,
#     percentile: float = 85.0,
# ) -> List[int]:
#     """
#     boundary = 4 means split between sentences[3] and sentences[4]
#     """
#     n = len(sentences)
#     if n <= 1:
#         return []

#     sent_emb = embed_texts_mean_pool(
#         model=model,
#         tokenizer=tokenizer,
#         texts=sentences,
#         device=device,
#         max_length=max_length,
#         batch_size=batch_size,
#     )

#     shift_scores = []
#     candidate_positions = []

#     for split_pos in range(1, n):
#         l_start = max(0, split_pos - left_window)
#         l_end = split_pos
#         r_start = split_pos
#         r_end = min(n, split_pos + right_window)

#         if l_start >= l_end or r_start >= r_end:
#             continue

#         left_vec = sent_emb[l_start:l_end].mean(dim=0)
#         right_vec = sent_emb[r_start:r_end].mean(dim=0)

#         left_vec = F.normalize(left_vec, p=2, dim=0)
#         right_vec = F.normalize(right_vec, p=2, dim=0)

#         sim = torch.dot(left_vec, right_vec).item()
#         dist = 1.0 - sim

#         shift_scores.append(dist)
#         candidate_positions.append(split_pos)

#     if not shift_scores:
#         return []

#     smoothed_scores = smooth_array(shift_scores, window=smoothing_window)
#     threshold = float(np.percentile(smoothed_scores, percentile))

#     boundaries = [
#         pos for pos, score in zip(candidate_positions, smoothed_scores)
#         if score >= threshold
#     ]

#     del sent_emb
#     gc.collect()
#     maybe_clear_cuda(device)

#     return boundaries


# def semantic_token_capped_segments(
#     sentences: List[str],
#     boundaries: List[int],
#     tokenizer,
#     max_chunk_tokens: int = 220,
#     min_chunk_sentences: int = 2,
# ) -> List[List[str]]:
#     n = len(sentences)
#     if n == 0:
#         return []

#     boundary_set = set(boundaries)
#     segments = []
#     current = []
#     current_tokens = 0

#     for i, sent in enumerate(sentences):
#         sent_tokens = len(tokenizer.tokenize(sent))

#         force_semantic_split = (i in boundary_set and len(current) >= min_chunk_sentences)
#         force_token_split = (current_tokens + sent_tokens > max_chunk_tokens and len(current) > 0)

#         if force_semantic_split or force_token_split:
#             segments.append(current)
#             current = [sent]
#             current_tokens = sent_tokens
#         else:
#             current.append(sent)
#             current_tokens += sent_tokens

#     if current:
#         segments.append(current)

#     merged = []
#     for seg in segments:
#         if merged and len(seg) < min_chunk_sentences:
#             merged[-1].extend(seg)
#         else:
#             merged.append(seg)

#     return merged


# def sliding_chunks_from_segments(
#     segments: List[List[str]],
#     window_size: int = 2,
#     stride: int = 1,
# ) -> List[str]:
#     if not segments:
#         return []

#     segment_texts = [" ".join(seg).strip() for seg in segments if seg]
#     if not segment_texts:
#         return []

#     if len(segment_texts) <= window_size:
#         return [" ".join(segment_texts).strip()]

#     chunks = []
#     for start in range(0, len(segment_texts), stride):
#         end = start + window_size
#         window = segment_texts[start:end]
#         if not window:
#             continue
#         chunk_text = " ".join(window).strip()
#         if chunk_text:
#             chunks.append(chunk_text)
#         if end >= len(segment_texts):
#             break

#     return chunks


# def improved_semantic_sliding_chunker(
#     text: str,
#     tokenizer,
#     model,
#     device: str = "cpu",
#     embed_max_length: int = 256,
#     embed_batch_size: int = 16,
#     left_window: int = 2,
#     right_window: int = 2,
#     smoothing_window: int = 3,
#     percentile: float = 85.0,
#     max_chunk_tokens: int = 220,
#     min_chunk_sentences: int = 2,
#     sliding_window_size: int = 2,
#     sliding_stride: int = 1,
# ) -> List[str]:
#     text = (text or "").strip()
#     if not text:
#         return []

#     sentences = [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
#     if not sentences:
#         return []

#     if len(sentences) <= min_chunk_sentences:
#         return [text]

#     boundaries = detect_semantic_boundaries_windowed(
#         sentences=sentences,
#         tokenizer=tokenizer,
#         model=model,
#         device=device,
#         max_length=embed_max_length,
#         batch_size=embed_batch_size,
#         left_window=left_window,
#         right_window=right_window,
#         smoothing_window=smoothing_window,
#         percentile=percentile,
#     )

#     segments = semantic_token_capped_segments(
#         sentences=sentences,
#         boundaries=boundaries,
#         tokenizer=tokenizer,
#         max_chunk_tokens=max_chunk_tokens,
#         min_chunk_sentences=min_chunk_sentences,
#     )

#     chunks = sliding_chunks_from_segments(
#         segments=segments,
#         window_size=sliding_window_size,
#         stride=sliding_stride,
#     )

#     return [c for c in chunks if c.strip()]


# # =========================================================
# # Query-aware chunking
# # =========================================================
# def merge_overlapping_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
#     if not spans:
#         return []

#     spans = sorted(spans, key=lambda x: (x[0], x[1]))
#     merged = [spans[0]]

#     for s, e in spans[1:]:
#         last_s, last_e = merged[-1]
#         if s <= last_e + 1:
#             merged[-1] = (last_s, max(last_e, e))
#         else:
#             merged.append((s, e))

#     return merged


# def split_long_span_by_tokens(
#     sentences: List[str],
#     tokenizer,
#     max_chunk_tokens: int = 220,
# ) -> List[str]:
#     if not sentences:
#         return []

#     chunks = []
#     current = []
#     current_tokens = 0

#     for sent in sentences:
#         sent_tokens = len(tokenizer.tokenize(sent))
#         if current and current_tokens + sent_tokens > max_chunk_tokens:
#             chunks.append(" ".join(current).strip())
#             current = [sent]
#             current_tokens = sent_tokens
#         else:
#             current.append(sent)
#             current_tokens += sent_tokens

#     if current:
#         chunks.append(" ".join(current).strip())

#     return [c for c in chunks if c.strip()]


# def select_query_anchor_indices(
#     claim: str,
#     sentences: List[str],
#     tokenizer,
#     model,
#     device: str = "cpu",
#     max_length: int = 256,
#     batch_size: int = 16,
#     top_m: int = 3,
#     score_percentile: Optional[float] = None,
#     min_anchors: int = 1,
# ) -> List[int]:
#     if not claim.strip() or not sentences:
#         return []

#     claim_emb = embed_texts_mean_pool(
#         model=model,
#         tokenizer=tokenizer,
#         texts=[claim],
#         device=device,
#         max_length=max_length,
#         batch_size=1,
#     )[0]

#     sent_embs = embed_texts_mean_pool(
#         model=model,
#         tokenizer=tokenizer,
#         texts=sentences,
#         device=device,
#         max_length=max_length,
#         batch_size=batch_size,
#     )

#     scores = torch.matmul(sent_embs, claim_emb.unsqueeze(-1)).squeeze(-1).cpu().tolist()

#     # primary selection: top_m
#     ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
#     selected = ranked[:min(top_m, len(ranked))]

#     # optional threshold-based expansion
#     if score_percentile is not None and len(scores) > 0:
#         threshold = float(np.percentile(scores, score_percentile))
#         extra = [i for i, s in enumerate(scores) if s >= threshold]
#         selected = sorted(set(selected) | set(extra))

#     if not selected and len(scores) > 0:
#         selected = ranked[:min_anchors]

#     del claim_emb, sent_embs
#     gc.collect()
#     maybe_clear_cuda(device)

#     return selected


# def query_aware_chunker(
#     claim: str,
#     text: str,
#     tokenizer,
#     model,
#     device: str = "cpu",
#     embed_max_length: int = 256,
#     embed_batch_size: int = 16,
#     top_m_sentences: int = 3,
#     score_percentile: Optional[float] = None,
#     left_context: int = 1,
#     right_context: int = 1,
#     max_chunk_tokens: int = 220,
# ) -> List[str]:
#     text = (text or "").strip()
#     claim = (claim or "").strip()

#     if not text:
#         return []

#     sentences = [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
#     if not sentences:
#         return []
#     if len(sentences) == 1:
#         return [sentences[0]]

#     anchor_indices = select_query_anchor_indices(
#         claim=claim,
#         sentences=sentences,
#         tokenizer=tokenizer,
#         model=model,
#         device=device,
#         max_length=embed_max_length,
#         batch_size=embed_batch_size,
#         top_m=top_m_sentences,
#         score_percentile=score_percentile,
#         min_anchors=1,
#     )

#     if not anchor_indices:
#         return split_long_span_by_tokens(
#             sentences=sentences,
#             tokenizer=tokenizer,
#             max_chunk_tokens=max_chunk_tokens,
#         )

#     spans = []
#     n = len(sentences)
#     for idx in anchor_indices:
#         start = max(0, idx - left_context)
#         end = min(n - 1, idx + right_context)
#         spans.append((start, end))

#     merged_spans = merge_overlapping_spans(spans)

#     chunks = []
#     for start, end in merged_spans:
#         span_sentences = sentences[start:end + 1]
#         span_chunks = split_long_span_by_tokens(
#             sentences=span_sentences,
#             tokenizer=tokenizer,
#             max_chunk_tokens=max_chunk_tokens,
#         )
#         chunks.extend(span_chunks)

#     # Fallback if something odd happens
#     if not chunks:
#         chunks = split_long_span_by_tokens(
#             sentences=sentences,
#             tokenizer=tokenizer,
#             max_chunk_tokens=max_chunk_tokens,
#         )

#     return [c for c in chunks if c.strip()]


# @torch.no_grad()
# def rerank_chunks_with_biencoder(
#     query: str,
#     chunks: List[ChunkRecord],
#     tokenizer,
#     model,
#     device: str = "cpu",
#     max_length: int = 256,
#     batch_size: int = 16,
#     top_k: int = 5,
# ) -> List[ChunkRecord]:
#     if not chunks:
#         return []

#     query_emb = embed_texts_mean_pool(
#         model=model,
#         tokenizer=tokenizer,
#         texts=[query],
#         device=device,
#         max_length=max_length,
#         batch_size=1,
#     )[0]

#     doc_texts = [c.chunk_text for c in chunks]
#     doc_embs = embed_texts_mean_pool(
#         model=model,
#         tokenizer=tokenizer,
#         texts=doc_texts,
#         device=device,
#         max_length=max_length,
#         batch_size=batch_size,
#     )

#     scores = torch.matmul(doc_embs, query_emb.unsqueeze(-1)).squeeze(-1).cpu().tolist()
#     ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

#     return [item[0] for item in ranked[:top_k]]


# # =========================================================
# # Core retrieval pipelines
# # =========================================================
# def retrieve_evidence_for_dp(
#     dp: Dict[str, Any],
#     chunk_mode: str,
#     sent_tokenizer: AutoTokenizer,
#     retrieval_embeddings: HuggingFaceEmbeddings,
#     semantic_tokenizer=None,
#     semantic_model=None,
#     semantic_alpha: float = 0.5,
#     semantic_min_chunk_sentences: int = 2,
#     semantic_max_length: int = 256,
#     semantic_batch_size: int = 16,
#     semantic_device: str = "cpu",
#     instruction: str = "",
#     k: int = 1,
#     sent_max_tokens: int = 128,
#     fixed_chunk_size: int = 256,
#     fixed_chunk_overlap: int = 32,
# ) -> Dict[str, Any]:
#     """
#     Old per-source retrieval pipeline.
#     Used for: sentence, fixed, semantic_old
#     """
#     claim = dp.get("claim", "")
#     sources = dp.get("sources", [])
#     language = dp.get("language", "en")
#     formatted_claim = f"{instruction}\nQuery: {claim}".strip()

#     evidence_chunks = []

#     for s_idx, source in enumerate(sources or []):
#         content = clean_markdown_links(source.get("content", ""))
#         if not content.strip():
#             continue

#         if chunk_mode == "semantic_old":
#             tokenized_sentences = split_text_into_sentence_dicts(content)
#             _, chunks = semantic_chunk_from_tokenized(
#                 tokenized=tokenized_sentences,
#                 tokenizer=semantic_tokenizer,
#                 model=semantic_model,
#                 label_key="is_evidence",
#                 min_chunk_sentences=semantic_min_chunk_sentences,
#                 alpha=semantic_alpha,
#                 device=semantic_device,
#                 max_length=semantic_max_length,
#                 batch_size=semantic_batch_size,
#             )
#             del tokenized_sentences
#             maybe_clear_cuda(semantic_device)

#         elif chunk_mode == "sentence":
#             chunks = sentence_chunker(
#                 content,
#                 tokenizer=sent_tokenizer,
#                 max_tokens=sent_max_tokens,
#             )

#         elif chunk_mode == "fixed":
#             chunks = fixed_size_chunker(
#                 content,
#                 tokenizer=sent_tokenizer,
#                 chunk_size=fixed_chunk_size,
#                 chunk_overlap=fixed_chunk_overlap,
#             )

#         else:
#             raise ValueError(f"Unsupported chunk_mode: {chunk_mode}")

#         docs = [Document(page_content=c.strip()) for c in chunks if c and c.strip()]
#         if not docs:
#             continue

#         vectorstore = FAISS.from_documents(docs, retrieval_embeddings)
#         results = vectorstore.similarity_search(formatted_claim, k=k)

#         for res in results:
#             evidence_chunks.append({
#                 "source_index": s_idx,
#                 "source_url": source.get("source"),
#                 "evidence": res.page_content,
#             })

#         del docs, vectorstore, results, chunks
#         gc.collect()
#         maybe_clear_cuda(semantic_device)

#     return {
#         "id": dp.get("id"),
#         "claim": claim,
#         "label": dp.get("label"),
#         "evidences": evidence_chunks,
#         "language": language,
#     }


# def retrieve_evidence_for_dp_global_sliding_semantic(
#     dp: Dict[str, Any],
#     sent_tokenizer: AutoTokenizer,
#     retrieval_embeddings: HuggingFaceEmbeddings,
#     semantic_tokenizer,
#     semantic_model,
#     instruction: str = "",
#     k: int = 5,
#     rerank_top_n: int = 20,
#     semantic_device: str = "cpu",
#     semantic_max_length: int = 256,
#     semantic_batch_size: int = 16,
#     left_window: int = 2,
#     right_window: int = 2,
#     smoothing_window: int = 3,
#     percentile: float = 85.0,
#     semantic_max_chunk_tokens: int = 220,
#     semantic_min_chunk_sentences: int = 2,
#     sliding_window_size: int = 2,
#     sliding_stride: int = 1,
# ) -> Dict[str, Any]:
#     claim = dp.get("claim", "")
#     sources = dp.get("sources", [])
#     language = dp.get("language", "en")
#     formatted_claim = f"{instruction}\nQuery: {claim}".strip()

#     all_docs = []

#     for s_idx, source in enumerate(sources or []):
#         content = clean_markdown_links(source.get("content", ""))
#         if not content.strip():
#             continue

#         chunks = improved_semantic_sliding_chunker(
#             text=content,
#             tokenizer=semantic_tokenizer,
#             model=semantic_model,
#             device=semantic_device,
#             embed_max_length=semantic_max_length,
#             embed_batch_size=semantic_batch_size,
#             left_window=left_window,
#             right_window=right_window,
#             smoothing_window=smoothing_window,
#             percentile=percentile,
#             max_chunk_tokens=semantic_max_chunk_tokens,
#             min_chunk_sentences=semantic_min_chunk_sentences,
#             sliding_window_size=sliding_window_size,
#             sliding_stride=sliding_stride,
#         )

#         for c_idx, chunk_text in enumerate(chunks):
#             chunk_text = chunk_text.strip()
#             if not chunk_text:
#                 continue

#             chunk_id = f"{s_idx}_{c_idx}"
#             metadata = {
#                 "source_index": s_idx,
#                 "source_url": source.get("source"),
#                 "chunk_id": chunk_id,
#             }

#             all_docs.append(Document(page_content=chunk_text, metadata=metadata))

#     if not all_docs:
#         return {
#             "id": dp.get("id"),
#             "claim": claim,
#             "label": dp.get("label"),
#             "evidences": [],
#             "language": language,
#         }

#     vectorstore = FAISS.from_documents(all_docs, retrieval_embeddings)

#     initial_hits = vectorstore.similarity_search(
#         formatted_claim,
#         k=min(rerank_top_n, len(all_docs))
#     )

#     candidate_records = []
#     for hit in initial_hits:
#         meta = hit.metadata
#         candidate_records.append(
#             ChunkRecord(
#                 source_index=meta["source_index"],
#                 source_url=meta["source_url"],
#                 chunk_text=hit.page_content,
#                 chunk_id=meta["chunk_id"],
#             )
#         )

#     reranked = rerank_chunks_with_biencoder(
#         query=formatted_claim,
#         chunks=candidate_records,
#         tokenizer=semantic_tokenizer,
#         model=semantic_model,
#         device=semantic_device,
#         max_length=semantic_max_length,
#         batch_size=semantic_batch_size,
#         top_k=k,
#     )

#     evidence_chunks = []
#     for rec in reranked:
#         evidence_chunks.append({
#             "source_index": rec.source_index,
#             "source_url": rec.source_url,
#             "evidence": rec.chunk_text,
#         })

#     del all_docs, vectorstore, initial_hits, candidate_records, reranked
#     gc.collect()
#     maybe_clear_cuda(semantic_device)

#     return {
#         "id": dp.get("id"),
#         "claim": claim,
#         "label": dp.get("label"),
#         "evidences": evidence_chunks,
#         "language": language,
#     }


# def retrieve_evidence_for_dp_query_aware(
#     dp: Dict[str, Any],
#     retrieval_embeddings: HuggingFaceEmbeddings,
#     semantic_tokenizer,
#     semantic_model,
#     instruction: str = "",
#     k: int = 5,
#     rerank_top_n: int = 20,
#     semantic_device: str = "cpu",
#     semantic_max_length: int = 256,
#     semantic_batch_size: int = 16,
#     top_m_sentences: int = 3,
#     query_score_percentile: Optional[float] = None,
#     query_left_context: int = 1,
#     query_right_context: int = 1,
#     query_max_chunk_tokens: int = 220,
# ) -> Dict[str, Any]:
#     claim = dp.get("claim", "")
#     sources = dp.get("sources", [])
#     language = dp.get("language", "en")
#     formatted_claim = f"{instruction}\nQuery: {claim}".strip()

#     all_docs = []

#     for s_idx, source in enumerate(sources or []):
#         content = clean_markdown_links(source.get("content", ""))
#         if not content.strip():
#             continue

#         chunks = query_aware_chunker(
#             claim=formatted_claim,
#             text=content,
#             tokenizer=semantic_tokenizer,
#             model=semantic_model,
#             device=semantic_device,
#             embed_max_length=semantic_max_length,
#             embed_batch_size=semantic_batch_size,
#             top_m_sentences=top_m_sentences,
#             score_percentile=query_score_percentile,
#             left_context=query_left_context,
#             right_context=query_right_context,
#             max_chunk_tokens=query_max_chunk_tokens,
#         )

#         for c_idx, chunk_text in enumerate(chunks):
#             chunk_text = chunk_text.strip()
#             if not chunk_text:
#                 continue

#             chunk_id = f"{s_idx}_{c_idx}"
#             metadata = {
#                 "source_index": s_idx,
#                 "source_url": source.get("source"),
#                 "chunk_id": chunk_id,
#             }
#             all_docs.append(Document(page_content=chunk_text, metadata=metadata))

#     if not all_docs:
#         return {
#             "id": dp.get("id"),
#             "claim": claim,
#             "label": dp.get("label"),
#             "evidences": [],
#             "language": language,
#         }

#     vectorstore = FAISS.from_documents(all_docs, retrieval_embeddings)

#     initial_hits = vectorstore.similarity_search(
#         formatted_claim,
#         k=min(rerank_top_n, len(all_docs))
#     )

#     candidate_records = []
#     for hit in initial_hits:
#         meta = hit.metadata
#         candidate_records.append(
#             ChunkRecord(
#                 source_index=meta["source_index"],
#                 source_url=meta["source_url"],
#                 chunk_text=hit.page_content,
#                 chunk_id=meta["chunk_id"],
#             )
#         )

#     reranked = rerank_chunks_with_biencoder(
#         query=formatted_claim,
#         chunks=candidate_records,
#         tokenizer=semantic_tokenizer,
#         model=semantic_model,
#         device=semantic_device,
#         max_length=semantic_max_length,
#         batch_size=semantic_batch_size,
#         top_k=k,
#     )

#     evidence_chunks = []
#     for rec in reranked:
#         evidence_chunks.append({
#             "source_index": rec.source_index,
#             "source_url": rec.source_url,
#             "evidence": rec.chunk_text,
#         })

#     del all_docs, vectorstore, initial_hits, candidate_records, reranked
#     gc.collect()
#     maybe_clear_cuda(semantic_device)

#     return {
#         "id": dp.get("id"),
#         "claim": claim,
#         "label": dp.get("label"),
#         "evidences": evidence_chunks,
#         "language": language,
#     }


# # =========================================================
# # Main
# # =========================================================
# def main():
#     parser = argparse.ArgumentParser(
#         description="Retrieve evidence with sentence, fixed, semantic-old, semantic-improved, or query-aware chunking."
#     )

#     parser.add_argument("--dataset", default="xfact", help="Name of dataset.")
#     parser.add_argument("--input", default="train", help="Input split/file stem.")

#     parser.add_argument(
#         "--chunker",
#         choices=["sentence", "fixed", "semantic_old", "semantic_improved", "query_aware"],
#         default="semantic_improved",
#         help="Chunking strategy."
#     )

#     parser.add_argument(
#         "--sent-tokenizer-model",
#         default="xlm-roberta-large",
#         help="HF tokenizer for sentence token counts and fixed-size chunking."
#     )
#     parser.add_argument(
#         "--sent-max-tokens",
#         type=int,
#         default=256,
#         help="Max tokens per sentence chunk group."
#     )
#     parser.add_argument(
#         "--fixed-chunk-size",
#         type=int,
#         default=256,
#         help="Number of tokens per fixed chunk."
#     )
#     parser.add_argument(
#         "--fixed-chunk-overlap",
#         type=int,
#         default=32,
#         help="Token overlap between consecutive fixed chunks."
#     )

#     parser.add_argument(
#         "--retrieval-embed-model",
#         default="intfloat/multilingual-e5-large-instruct",
#         help="Embedding model used for FAISS retrieval."
#     )

#     parser.add_argument(
#         "--semantic-model",
#         default="intfloat/multilingual-e5-large-instruct",
#         help="HF model for semantic/query-aware sentence embeddings."
#     )
#     parser.add_argument(
#         "--semantic-alpha",
#         type=float,
#         default=1.0,
#         help="Adaptive threshold factor for original semantic chunking."
#     )
#     parser.add_argument(
#         "--semantic-min-chunk-sentences",
#         type=int,
#         default=2,
#         help="Minimum number of sentences per semantic chunk."
#     )
#     parser.add_argument(
#         "--semantic-max-length",
#         type=int,
#         default=256,
#         help="Max token length per sentence for semantic embedding."
#     )
#     parser.add_argument(
#         "--semantic-batch-size",
#         type=int,
#         default=16,
#         help="Batch size for semantic sentence embedding."
#     )
#     parser.add_argument(
#         "--device",
#         default="cpu",
#         help="Device for semantic/query-aware model, e.g. cpu or cuda."
#     )

#     parser.add_argument("--k", type=int, default=5, help="Top-k chunks to return.")
#     parser.add_argument(
#         "--instruction",
#         default="Instruct: Given a claim, retrieve relevant evidence from web documents that support or refute the claim",
#         help="Instruction prefix for the query."
#     )

#     parser.add_argument(
#         "--retriever",
#         dest="retriever",
#         action="store_true",
#         help="Enable evidence retrieval (default: True)."
#     )
#     parser.add_argument(
#         "--no-retriever",
#         dest="retriever",
#         action="store_false",
#         help="Disable evidence retrieval: only convert CSV/TSV into JSONL."
#     )
#     parser.set_defaults(retriever=True)

#     # ---------------- Improved semantic args ----------------
#     parser.add_argument(
#         "--rerank-top-n",
#         type=int,
#         default=20,
#         help="Number of FAISS candidates to rerank."
#     )
#     parser.add_argument(
#         "--left-window",
#         type=int,
#         default=2,
#         help="Left context window for semantic-improved boundary detection."
#     )
#     parser.add_argument(
#         "--right-window",
#         type=int,
#         default=2,
#         help="Right context window for semantic-improved boundary detection."
#     )
#     parser.add_argument(
#         "--smoothing-window",
#         type=int,
#         default=3,
#         help="Moving average window for semantic-improved shift smoothing."
#     )
#     parser.add_argument(
#         "--semantic-percentile",
#         type=float,
#         default=85.0,
#         help="Percentile threshold for semantic-improved split detection."
#     )
#     parser.add_argument(
#         "--semantic-max-chunk-tokens",
#         type=int,
#         default=220,
#         help="Max token cap for semantic-improved chunks."
#     )
#     parser.add_argument(
#         "--sliding-window-size",
#         type=int,
#         default=2,
#         help="How many semantic segments to join per final semantic-improved chunk."
#     )
#     parser.add_argument(
#         "--sliding-stride",
#         type=int,
#         default=1,
#         help="Stride for sliding semantic-improved chunks."
#     )

#     # ---------------- Query-aware args ----------------
#     parser.add_argument(
#         "--query-top-m-sentences",
#         type=int,
#         default=3,
#         help="Top-m most claim-relevant sentences to use as anchors."
#     )
#     parser.add_argument(
#         "--query-score-percentile",
#         type=float,
#         default=None,
#         help="Optional percentile threshold over claim-sentence scores for extra anchors."
#     )
#     parser.add_argument(
#         "--query-left-context",
#         type=int,
#         default=1,
#         help="Number of sentences to include before each query anchor."
#     )
#     parser.add_argument(
#         "--query-right-context",
#         type=int,
#         default=1,
#         help="Number of sentences to include after each query anchor."
#     )
#     parser.add_argument(
#         "--query-max-chunk-tokens",
#         type=int,
#         default=220,
#         help="Max token cap for query-aware chunks."
#     )

#     args = parser.parse_args()

#     if args.fixed_chunk_overlap >= args.fixed_chunk_size:
#         raise ValueError("--fixed-chunk-overlap must be smaller than --fixed-chunk-size")

#     type_of_evidence = "default"

#     if args.retriever:
#         if args.dataset == "translated_xfact":
#             input_file = INTERIM_DATA_DIR / "translated_xfact" / f"{args.input}.jsonl"
#         else:
#             input_file = INTERIM_DATA_DIR / "X-FACT" / f"{args.input}.jsonl"
#     else:
#         if args.dataset == "xfact":
#             input_file = RAW_DATA_DIR / args.dataset / f"{args.input}.tsv"
#             type_of_evidence = "search_snippet"
#         elif args.dataset == "ru22fact":
#             input_file = RAW_DATA_DIR / args.dataset / f"{args.input}.csv"
#             type_of_evidence = "llm_generated"
#         else:
#             raise ValueError("Unsupported dataset for --no-retriever mode.")

#         input_ext = Path(input_file).suffix.lower()
#         if input_ext == ".csv":
#             all_results = convert_csv_to_jsonl(input_file)
#         elif input_ext == ".tsv":
#             all_results = convert_tsv_to_jsonl(input_file)
#         else:
#             raise ValueError("When --no-retriever is set, input must be CSV or TSV.")

#         output_file = PROCESSED_DATA_DIR / args.dataset / f"{args.dataset}_{args.input}_with_{type_of_evidence}_evidences.jsonl"
#         save_jsonl(output_file, all_results)
#         print(f"Done. Wrote {len(all_results)} rows to {output_file}")
#         return

#     datapoints = load_jsonl(input_file)

#     if args.chunker == "sentence":
#         output_file = (
#             PROCESSED_DATA_DIR / f"{args.dataset}" /
#             f"{args.dataset}_{args.input}_with_sentence_level_chunked_retrieved_evidence.jsonl"
#         )
#     elif args.chunker == "fixed":
#         output_file = (
#             PROCESSED_DATA_DIR / f"{args.dataset}_fixed_{args.fixed_chunk_size}_{args.fixed_chunk_overlap}" /
#             f"{args.dataset}_{args.input}_with_fixed_size_chunked_retrieved_evidence.jsonl"
#         )
#     elif args.chunker == "semantic_old":
#         output_file = (
#             PROCESSED_DATA_DIR / f"{args.dataset}_alpha={args.semantic_alpha}" /
#             f"{args.dataset}_{args.input}_with_custom_semantic_chunked_retrieved_evidence.jsonl"
#         )
#     elif args.chunker == "semantic_improved":
#         output_file = (
#             PROCESSED_DATA_DIR / f"{args.dataset}_global_semantic_p{args.semantic_percentile}_tok{args.semantic_max_chunk_tokens}" /
#             f"{args.dataset}_{args.input}_with_global_sliding_semantic_retrieved_evidence.jsonl"
#         )
#     elif args.chunker == "query_aware":
#         output_file = (
#             PROCESSED_DATA_DIR / f"{args.dataset}_queryaware_m{args.query_top_m_sentences}_l{args.query_left_context}_r{args.query_right_context}_tok{args.query_max_chunk_tokens}" /
#             f"{args.dataset}_{args.input}_with_query_aware_retrieved_evidence.jsonl"
#         )
#     else:
#         raise ValueError(f"Unsupported chunker: {args.chunker}")

#     processed_ids = load_processed_ids(output_file)
#     print(f"Found {len(processed_ids)} already processed datapoints in {output_file}")

#     sent_tokenizer = build_tokenizer(args.sent_tokenizer_model)

#     retrieval_embeddings = HuggingFaceEmbeddings(
#         model_name=args.retrieval_embed_model
#     )

#     semantic_tokenizer = None
#     semantic_model = None

#     if args.chunker in {"semantic_old", "semantic_improved", "query_aware"}:
#         semantic_tokenizer, semantic_model = build_custom_semantic_models(
#             model_name=args.semantic_model,
#             device=args.device,
#         )

#     skipped = 0
#     processed_now = 0

#     for dp in tqdm(datapoints, desc="Processing datapoints"):
#         if args.dataset == "translated_xfact":
#             dp = translated_xfact_to_internal(dp)

#         dp_key = make_dp_key(dp)

#         if dp_key in processed_ids:
#             skipped += 1
#             continue

#         if args.chunker == "semantic_improved":
#             result = retrieve_evidence_for_dp_global_sliding_semantic(
#                 dp=dp,
#                 sent_tokenizer=sent_tokenizer,
#                 retrieval_embeddings=retrieval_embeddings,
#                 semantic_tokenizer=semantic_tokenizer,
#                 semantic_model=semantic_model,
#                 instruction=args.instruction,
#                 k=args.k,
#                 rerank_top_n=args.rerank_top_n,
#                 semantic_device=args.device,
#                 semantic_max_length=args.semantic_max_length,
#                 semantic_batch_size=args.semantic_batch_size,
#                 left_window=args.left_window,
#                 right_window=args.right_window,
#                 smoothing_window=args.smoothing_window,
#                 percentile=args.semantic_percentile,
#                 semantic_max_chunk_tokens=args.semantic_max_chunk_tokens,
#                 semantic_min_chunk_sentences=args.semantic_min_chunk_sentences,
#                 sliding_window_size=args.sliding_window_size,
#                 sliding_stride=args.sliding_stride,
#             )
#         elif args.chunker == "query_aware":
#             result = retrieve_evidence_for_dp_query_aware(
#                 dp=dp,
#                 retrieval_embeddings=retrieval_embeddings,
#                 semantic_tokenizer=semantic_tokenizer,
#                 semantic_model=semantic_model,
#                 instruction=args.instruction,
#                 k=args.k,
#                 rerank_top_n=args.rerank_top_n,
#                 semantic_device=args.device,
#                 semantic_max_length=args.semantic_max_length,
#                 semantic_batch_size=args.semantic_batch_size,
#                 top_m_sentences=args.query_top_m_sentences,
#                 query_score_percentile=args.query_score_percentile,
#                 query_left_context=args.query_left_context,
#                 query_right_context=args.query_right_context,
#                 query_max_chunk_tokens=args.query_max_chunk_tokens,
#             )
#         else:
#             result = retrieve_evidence_for_dp(
#                 dp=dp,
#                 chunk_mode=args.chunker,
#                 sent_tokenizer=sent_tokenizer,
#                 retrieval_embeddings=retrieval_embeddings,
#                 semantic_tokenizer=semantic_tokenizer,
#                 semantic_model=semantic_model,
#                 semantic_alpha=args.semantic_alpha,
#                 semantic_min_chunk_sentences=args.semantic_min_chunk_sentences,
#                 semantic_max_length=args.semantic_max_length,
#                 semantic_batch_size=args.semantic_batch_size,
#                 semantic_device=args.device,
#                 instruction=args.instruction,
#                 k=args.k,
#                 sent_max_tokens=args.sent_max_tokens,
#                 fixed_chunk_size=args.fixed_chunk_size,
#                 fixed_chunk_overlap=args.fixed_chunk_overlap,
#             )

#         if result.get("id") is None:
#             result["id"] = dp_key

#         append_jsonl(output_file, result)
#         processed_ids.add(str(result["id"]))
#         processed_now += 1

#         del result
#         gc.collect()
#         maybe_clear_cuda(args.device)

#     print(f"Done. Newly processed: {processed_now}, skipped already done: {skipped}")
#     print(f"Output saved incrementally to {output_file}")


# if __name__ == "__main__":
#     main()


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
from typing import List, Dict, Any, Tuple, Set, Optional
from dataclasses import dataclass

import numpy as np
import nltk
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ------------------------------
# HuggingFace cache
# ------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/data2/Gaurav/Babu/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data2/Gaurav/Babu/hf_cache/transformers")
os.environ.setdefault("HF_DATASETS_CACHE", "/data2/Gaurav/Babu/hf_cache/datasets")

# ------------------------------
# Repo root
# ------------------------------
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
# Tokenization / chunk helpers
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
# Embedding helpers
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


@torch.no_grad()
def embed_texts_mean_pool(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cpu",
    max_length: int = 256,
    batch_size: int = 16,
) -> torch.Tensor:
    return mean_pool_embeddings(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )


def build_custom_semantic_models(model_name: str, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


# =========================================================
# OLD semantic chunking
# =========================================================
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


# =========================================================
# IMPROVED semantic chunking
# =========================================================
@dataclass
class ChunkRecord:
    source_index: int
    source_url: str
    chunk_text: str
    chunk_id: str


def smooth_array(values: List[float], window: int = 3) -> List[float]:
    if not values:
        return values
    if window <= 1:
        return values

    arr = np.array(values, dtype=np.float32)
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    smoothed = []
    for i in range(len(arr)):
        smoothed.append(float(np.mean(padded[i:i + window])))
    return smoothed


def detect_semantic_boundaries_windowed(
    sentences: List[str],
    tokenizer,
    model,
    device: str = "cpu",
    max_length: int = 256,
    batch_size: int = 16,
    left_window: int = 2,
    right_window: int = 2,
    smoothing_window: int = 3,
    percentile: float = 85.0,
) -> List[int]:
    n = len(sentences)
    if n <= 1:
        return []

    sent_emb = embed_texts_mean_pool(
        model=model,
        tokenizer=tokenizer,
        texts=sentences,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )

    shift_scores = []
    candidate_positions = []

    for split_pos in range(1, n):
        l_start = max(0, split_pos - left_window)
        l_end = split_pos
        r_start = split_pos
        r_end = min(n, split_pos + right_window)

        if l_start >= l_end or r_start >= r_end:
            continue

        left_vec = sent_emb[l_start:l_end].mean(dim=0)
        right_vec = sent_emb[r_start:r_end].mean(dim=0)

        left_vec = F.normalize(left_vec, p=2, dim=0)
        right_vec = F.normalize(right_vec, p=2, dim=0)

        sim = torch.dot(left_vec, right_vec).item()
        dist = 1.0 - sim

        shift_scores.append(dist)
        candidate_positions.append(split_pos)

    if not shift_scores:
        return []

    smoothed_scores = smooth_array(shift_scores, window=smoothing_window)
    threshold = float(np.percentile(smoothed_scores, percentile))

    boundaries = [
        pos for pos, score in zip(candidate_positions, smoothed_scores)
        if score >= threshold
    ]

    del sent_emb
    gc.collect()
    maybe_clear_cuda(device)

    return boundaries


def semantic_token_capped_segments(
    sentences: List[str],
    boundaries: List[int],
    tokenizer,
    max_chunk_tokens: int = 220,
    min_chunk_sentences: int = 2,
) -> List[List[str]]:
    if not sentences:
        return []

    boundary_set = set(boundaries)
    segments = []

    current = []
    current_tokens = 0

    for i, sent in enumerate(sentences):
        sent_tokens = len(tokenizer.tokenize(sent))

        force_semantic_split = (i in boundary_set and len(current) >= min_chunk_sentences)
        force_token_split = (current_tokens + sent_tokens > max_chunk_tokens and len(current) > 0)

        if force_semantic_split or force_token_split:
            segments.append(current)
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens

    if current:
        segments.append(current)

    merged = []
    for seg in segments:
        if merged and len(seg) < min_chunk_sentences:
            merged[-1].extend(seg)
        else:
            merged.append(seg)

    return merged


def sliding_chunks_from_segments(
    segments: List[List[str]],
    window_size: int = 2,
    stride: int = 1,
) -> List[str]:
    if not segments:
        return []

    segment_texts = [" ".join(seg).strip() for seg in segments if seg]
    if not segment_texts:
        return []

    if len(segment_texts) <= window_size:
        return [" ".join(segment_texts).strip()]

    chunks = []
    for start in range(0, len(segment_texts), stride):
        end = start + window_size
        window = segment_texts[start:end]
        if not window:
            continue
        chunk_text = " ".join(window).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(segment_texts):
            break

    return chunks


def improved_semantic_sliding_chunker(
    text: str,
    tokenizer,
    model,
    device: str = "cpu",
    embed_max_length: int = 256,
    embed_batch_size: int = 16,
    left_window: int = 2,
    right_window: int = 2,
    smoothing_window: int = 3,
    percentile: float = 85.0,
    max_chunk_tokens: int = 220,
    min_chunk_sentences: int = 2,
    sliding_window_size: int = 2,
    sliding_stride: int = 1,
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    sentences = [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    if not sentences:
        return []

    if len(sentences) <= min_chunk_sentences:
        return [text]

    boundaries = detect_semantic_boundaries_windowed(
        sentences=sentences,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=embed_max_length,
        batch_size=embed_batch_size,
        left_window=left_window,
        right_window=right_window,
        smoothing_window=smoothing_window,
        percentile=percentile,
    )

    segments = semantic_token_capped_segments(
        sentences=sentences,
        boundaries=boundaries,
        tokenizer=tokenizer,
        max_chunk_tokens=max_chunk_tokens,
        min_chunk_sentences=min_chunk_sentences,
    )

    chunks = sliding_chunks_from_segments(
        segments=segments,
        window_size=sliding_window_size,
        stride=sliding_stride,
    )

    return [c for c in chunks if c.strip()]


# =========================================================
# QUERY-AWARE helpers
# =========================================================
def merge_consecutive_indices(indices: List[int]) -> List[List[int]]:
    if not indices:
        return []

    indices = sorted(set(indices))
    groups = [[indices[0]]]

    for idx in indices[1:]:
        if idx == groups[-1][-1] + 1:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    return groups


@torch.no_grad()
def compute_query_sentence_similarities(
    claim: str,
    sentences: List[str],
    tokenizer,
    model,
    device: str = "cpu",
    max_length: int = 256,
    batch_size: int = 16,
) -> List[float]:
    if not claim.strip() or not sentences:
        return []

    claim_emb = embed_texts_mean_pool(
        model=model,
        tokenizer=tokenizer,
        texts=[claim],
        device=device,
        max_length=max_length,
        batch_size=1,
    )[0]

    sent_embs = embed_texts_mean_pool(
        model=model,
        tokenizer=tokenizer,
        texts=sentences,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )

    sims = torch.matmul(sent_embs, claim_emb.unsqueeze(-1)).squeeze(-1)
    scores = sims.detach().cpu().tolist()

    del claim_emb, sent_embs, sims
    gc.collect()
    maybe_clear_cuda(device)

    return scores


def select_query_relevant_sentence_indices(
    scores: List[float],
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    min_keep: int = 1,
) -> List[int]:
    if not scores:
        return []

    indexed_scores = list(enumerate(scores))
    selected = set()

    if threshold is not None:
        for i, sc in indexed_scores:
            if sc >= threshold:
                selected.add(i)

    if top_k is not None and top_k > 0:
        ranked = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        for i, _ in ranked[:top_k]:
            selected.add(i)

    if len(selected) < min_keep:
        ranked = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        for i, _ in ranked[:min_keep]:
            selected.add(i)

    return sorted(selected)


def expand_indices_with_context(
    indices: List[int],
    total_sentences: int,
    left_context: int = 1,
    right_context: int = 1,
) -> List[int]:
    if not indices:
        return []

    expanded = set()
    for idx in indices:
        for j in range(idx - left_context, idx + right_context + 1):
            if 0 <= j < total_sentences:
                expanded.add(j)

    return sorted(expanded)


def semantic_query_aware_chunker(
    claim: str,
    text: str,
    tokenizer,
    model,
    device: str = "cpu",
    embed_max_length: int = 256,
    embed_batch_size: int = 16,
    left_window: int = 2,
    right_window: int = 2,
    smoothing_window: int = 3,
    percentile: float = 85.0,
    max_chunk_tokens: int = 220,
    min_chunk_sentences: int = 2,
    qa_top_k_per_segment: int = 2,
    qa_threshold: Optional[float] = None,
    qa_left_context: int = 1,
    qa_right_context: int = 1,
    final_max_chunk_tokens: int = 260,
) -> List[str]:
    text = (text or "").strip()
    claim = (claim or "").strip()

    if not text:
        return []

    sentences = [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    if not sentences:
        return []

    if len(sentences) <= min_chunk_sentences:
        return [text]

    boundaries = detect_semantic_boundaries_windowed(
        sentences=sentences,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=embed_max_length,
        batch_size=embed_batch_size,
        left_window=left_window,
        right_window=right_window,
        smoothing_window=smoothing_window,
        percentile=percentile,
    )

    segments = semantic_token_capped_segments(
        sentences=sentences,
        boundaries=boundaries,
        tokenizer=tokenizer,
        max_chunk_tokens=max_chunk_tokens,
        min_chunk_sentences=min_chunk_sentences,
    )

    if not segments:
        return []

    final_chunks = []

    for seg in segments:
        seg_sentences = [s.strip() for s in seg if s.strip()]
        if not seg_sentences:
            continue

        if not claim:
            seg_text = " ".join(seg_sentences).strip()
            if seg_text:
                final_chunks.append(seg_text)
            continue

        scores = compute_query_sentence_similarities(
            claim=claim,
            sentences=seg_sentences,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=embed_max_length,
            batch_size=embed_batch_size,
        )

        selected = select_query_relevant_sentence_indices(
            scores=scores,
            top_k=min(qa_top_k_per_segment, len(seg_sentences)) if qa_top_k_per_segment is not None else None,
            threshold=qa_threshold,
            min_keep=1,
        )

        expanded = expand_indices_with_context(
            indices=selected,
            total_sentences=len(seg_sentences),
            left_context=qa_left_context,
            right_context=qa_right_context,
        )

        groups = merge_consecutive_indices(expanded)

        for grp in groups:
            chunk_sents = [seg_sentences[i] for i in grp]
            if not chunk_sents:
                continue

            current = []
            current_tokens = 0

            for sent in chunk_sents:
                sent_tokens = len(tokenizer.tokenize(sent))
                if current and current_tokens + sent_tokens > final_max_chunk_tokens:
                    chunk_text = " ".join(current).strip()
                    if chunk_text:
                        final_chunks.append(chunk_text)
                    current = [sent]
                    current_tokens = sent_tokens
                else:
                    current.append(sent)
                    current_tokens += sent_tokens

            if current:
                chunk_text = " ".join(current).strip()
                if chunk_text:
                    final_chunks.append(chunk_text)

    deduped = []
    seen = set()
    for ch in final_chunks:
        key = ch.strip()
        if key and key not in seen:
            deduped.append(key)
            seen.add(key)

    return deduped


# =========================================================
# ADAPTIVE semantic-stopping query-aware helpers
# =========================================================
def mean_pool_tensor_rows(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("Expected 2D tensor [n, d]")
    v = x.mean(dim=0)
    v = F.normalize(v, p=2, dim=0)
    return v


def cosine_sim_vec(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a, p=2, dim=0)
    b = F.normalize(b, p=2, dim=0)
    return float(torch.dot(a, b).item())


def embed_sentences_once(
    sentences: List[str],
    tokenizer,
    model,
    device: str = "cpu",
    max_length: int = 256,
    batch_size: int = 16,
) -> torch.Tensor:
    if not sentences:
        return torch.empty(0, 1)
    return embed_texts_mean_pool(
        model=model,
        tokenizer=tokenizer,
        texts=sentences,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )


def select_top_anchor_indices(
    claim_emb: torch.Tensor,
    sent_embs: torch.Tensor,
    top_k: int = 2,
    min_query_sim: Optional[float] = None,
) -> List[int]:
    if sent_embs.numel() == 0:
        return []

    sims = torch.matmul(sent_embs, claim_emb.unsqueeze(-1)).squeeze(-1)
    pairs = list(enumerate(sims.detach().cpu().tolist()))
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

    selected = []
    for idx, score in pairs:
        if min_query_sim is not None and score < min_query_sim:
            continue
        selected.append(idx)
        if len(selected) >= top_k:
            break

    if not selected and len(pairs) > 0:
        selected.append(pairs[0][0])

    return sorted(selected)


def adaptive_semantic_expand_from_anchor(
    anchor_idx: int,
    sent_embs: torch.Tensor,
    claim_emb: torch.Tensor,
    semantic_expand_threshold: float = 0.65,
    query_min_threshold: Optional[float] = None,
    max_expand_sentences: Optional[int] = None,
) -> Tuple[int, int]:
    n = sent_embs.shape[0]
    if n == 0:
        return (0, -1)

    start = anchor_idx
    end = anchor_idx
    current_region = sent_embs[anchor_idx].clone()

    while True:
        expanded = False

        left_idx = start - 1
        if left_idx >= 0:
            left_vec = sent_embs[left_idx]
            sem_sim = cosine_sim_vec(left_vec, current_region)

            if query_min_threshold is None:
                query_ok = True
            else:
                qsim = cosine_sim_vec(left_vec, claim_emb)
                query_ok = qsim >= query_min_threshold

            length_ok = True
            if max_expand_sentences is not None:
                length_ok = (end - start + 1) < max_expand_sentences

            if sem_sim >= semantic_expand_threshold and query_ok and length_ok:
                start = left_idx
                current_region = mean_pool_tensor_rows(sent_embs[start:end + 1])
                expanded = True

        right_idx = end + 1
        if right_idx < n:
            right_vec = sent_embs[right_idx]
            sem_sim = cosine_sim_vec(right_vec, current_region)

            if query_min_threshold is None:
                query_ok = True
            else:
                qsim = cosine_sim_vec(right_vec, claim_emb)
                query_ok = qsim >= query_min_threshold

            length_ok = True
            if max_expand_sentences is not None:
                length_ok = (end - start + 1) < max_expand_sentences

            if sem_sim >= semantic_expand_threshold and query_ok and length_ok:
                end = right_idx
                current_region = mean_pool_tensor_rows(sent_embs[start:end + 1])
                expanded = True

        if not expanded:
            break

    return start, end


def merge_overlapping_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []

    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    merged = [spans[0]]

    for s, e in spans[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + 1:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))

    return merged


def build_chunks_from_spans_with_token_cap(
    sentences: List[str],
    spans: List[Tuple[int, int]],
    tokenizer,
    final_max_chunk_tokens: int = 260,
) -> List[str]:
    final_chunks = []

    for s, e in spans:
        chunk_sents = sentences[s:e + 1]
        if not chunk_sents:
            continue

        current = []
        current_tokens = 0

        for sent in chunk_sents:
            sent_tokens = len(tokenizer.tokenize(sent))
            if current and current_tokens + sent_tokens > final_max_chunk_tokens:
                chunk_text = " ".join(current).strip()
                if chunk_text:
                    final_chunks.append(chunk_text)
                current = [sent]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                final_chunks.append(chunk_text)

    deduped = []
    seen = set()
    for ch in final_chunks:
        key = ch.strip()
        if key and key not in seen:
            deduped.append(key)
            seen.add(key)

    return deduped


def semantic_query_aware_adaptive_chunker(
    claim: str,
    text: str,
    tokenizer,
    model,
    device: str = "cpu",
    embed_max_length: int = 256,
    embed_batch_size: int = 16,
    left_window: int = 2,
    right_window: int = 2,
    smoothing_window: int = 3,
    percentile: float = 85.0,
    max_chunk_tokens: int = 220,
    min_chunk_sentences: int = 2,
    qa_top_k_anchors_per_segment: int = 2,
    qa_anchor_min_query_sim: Optional[float] = None,
    qa_semantic_expand_threshold: float = 0.65,
    qa_query_min_threshold: Optional[float] = None,
    qa_max_expand_sentences: Optional[int] = None,
    final_max_chunk_tokens: int = 260,
) -> List[str]:
    text = (text or "").strip()
    claim = (claim or "").strip()

    if not text:
        return []

    sentences = [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    if not sentences:
        return []

    if len(sentences) <= min_chunk_sentences:
        return [text]

    boundaries = detect_semantic_boundaries_windowed(
        sentences=sentences,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=embed_max_length,
        batch_size=embed_batch_size,
        left_window=left_window,
        right_window=right_window,
        smoothing_window=smoothing_window,
        percentile=percentile,
    )

    segments = semantic_token_capped_segments(
        sentences=sentences,
        boundaries=boundaries,
        tokenizer=tokenizer,
        max_chunk_tokens=max_chunk_tokens,
        min_chunk_sentences=min_chunk_sentences,
    )

    if not segments:
        return []

    claim_emb = embed_texts_mean_pool(
        model=model,
        tokenizer=tokenizer,
        texts=[claim],
        device=device,
        max_length=embed_max_length,
        batch_size=1,
    )[0]

    final_chunks = []

    for seg in segments:
        seg_sentences = [s.strip() for s in seg if s.strip()]
        if not seg_sentences:
            continue

        if not claim:
            seg_text = " ".join(seg_sentences).strip()
            if seg_text:
                final_chunks.append(seg_text)
            continue

        seg_embs = embed_sentences_once(
            sentences=seg_sentences,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=embed_max_length,
            batch_size=embed_batch_size,
        )

        anchor_indices = select_top_anchor_indices(
            claim_emb=claim_emb,
            sent_embs=seg_embs,
            top_k=min(qa_top_k_anchors_per_segment, len(seg_sentences)),
            min_query_sim=qa_anchor_min_query_sim,
        )

        spans = []
        for anchor_idx in anchor_indices:
            span = adaptive_semantic_expand_from_anchor(
                anchor_idx=anchor_idx,
                sent_embs=seg_embs,
                claim_emb=claim_emb,
                semantic_expand_threshold=qa_semantic_expand_threshold,
                query_min_threshold=qa_query_min_threshold,
                max_expand_sentences=qa_max_expand_sentences,
            )
            spans.append(span)

        spans = merge_overlapping_spans(spans)

        seg_chunks = build_chunks_from_spans_with_token_cap(
            sentences=seg_sentences,
            spans=spans,
            tokenizer=tokenizer,
            final_max_chunk_tokens=final_max_chunk_tokens,
        )

        final_chunks.extend(seg_chunks)

        del seg_embs
        gc.collect()
        maybe_clear_cuda(device)

    del claim_emb
    gc.collect()
    maybe_clear_cuda(device)

    deduped = []
    seen = set()
    for ch in final_chunks:
        key = ch.strip()
        if key and key not in seen:
            deduped.append(key)
            seen.add(key)

    return deduped


# =========================================================
# Reranker
# =========================================================
@torch.no_grad()
def rerank_chunks_with_biencoder(
    query: str,
    chunks: List[ChunkRecord],
    tokenizer,
    model,
    device: str = "cpu",
    max_length: int = 256,
    batch_size: int = 16,
    top_k: int = 5,
) -> List[ChunkRecord]:
    if not chunks:
        return []

    query_emb = embed_texts_mean_pool(
        model=model,
        tokenizer=tokenizer,
        texts=[query],
        device=device,
        max_length=max_length,
        batch_size=1,
    )[0]

    doc_texts = [c.chunk_text for c in chunks]
    doc_embs = embed_texts_mean_pool(
        model=model,
        tokenizer=tokenizer,
        texts=doc_texts,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )

    scores = torch.matmul(doc_embs, query_emb.unsqueeze(-1)).squeeze(-1).cpu().tolist()
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    return [item[0] for item in ranked[:top_k]]


# =========================================================
# Retrieval pipelines
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


def retrieve_evidence_for_dp_global_sliding_semantic(
    dp: Dict[str, Any],
    retrieval_embeddings: HuggingFaceEmbeddings,
    semantic_tokenizer,
    semantic_model,
    instruction: str = "",
    k: int = 5,
    rerank_top_n: int = 20,
    semantic_device: str = "cpu",
    semantic_max_length: int = 256,
    semantic_batch_size: int = 16,
    left_window: int = 2,
    right_window: int = 2,
    smoothing_window: int = 3,
    percentile: float = 85.0,
    semantic_max_chunk_tokens: int = 220,
    semantic_min_chunk_sentences: int = 2,
    sliding_window_size: int = 2,
    sliding_stride: int = 1,
) -> Dict[str, Any]:
    claim = dp.get("claim", "")
    sources = dp.get("sources", [])
    language = dp.get("language", "en")
    formatted_claim = f"{instruction}\nQuery: {claim}".strip()

    all_docs = []

    for s_idx, source in enumerate(sources or []):
        content = clean_markdown_links(source.get("content", ""))
        if not content.strip():
            continue

        chunks = improved_semantic_sliding_chunker(
            text=content,
            tokenizer=semantic_tokenizer,
            model=semantic_model,
            device=semantic_device,
            embed_max_length=semantic_max_length,
            embed_batch_size=semantic_batch_size,
            left_window=left_window,
            right_window=right_window,
            smoothing_window=smoothing_window,
            percentile=percentile,
            max_chunk_tokens=semantic_max_chunk_tokens,
            min_chunk_sentences=semantic_min_chunk_sentences,
            sliding_window_size=sliding_window_size,
            sliding_stride=sliding_stride,
        )

        for c_idx, chunk_text in enumerate(chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunk_id = f"{s_idx}_{c_idx}"
            metadata = {
                "source_index": s_idx,
                "source_url": source.get("source"),
                "chunk_id": chunk_id,
            }

            all_docs.append(Document(page_content=chunk_text, metadata=metadata))

    if not all_docs:
        return {
            "id": dp.get("id"),
            "claim": claim,
            "label": dp.get("label"),
            "evidences": [],
            "language": language,
        }

    vectorstore = FAISS.from_documents(all_docs, retrieval_embeddings)

    initial_hits = vectorstore.similarity_search(
        formatted_claim,
        k=min(rerank_top_n, len(all_docs))
    )

    candidate_records = []
    for hit in initial_hits:
        meta = hit.metadata
        candidate_records.append(
            ChunkRecord(
                source_index=meta["source_index"],
                source_url=meta["source_url"],
                chunk_text=hit.page_content,
                chunk_id=meta["chunk_id"],
            )
        )

    reranked = rerank_chunks_with_biencoder(
        query=formatted_claim,
        chunks=candidate_records,
        tokenizer=semantic_tokenizer,
        model=semantic_model,
        device=semantic_device,
        max_length=semantic_max_length,
        batch_size=semantic_batch_size,
        top_k=k,
    )

    evidence_chunks = []
    for rec in reranked:
        evidence_chunks.append({
            "source_index": rec.source_index,
            "source_url": rec.source_url,
            "evidence": rec.chunk_text,
        })

    del all_docs, vectorstore, initial_hits, candidate_records, reranked
    gc.collect()
    maybe_clear_cuda(semantic_device)

    return {
        "id": dp.get("id"),
        "claim": claim,
        "label": dp.get("label"),
        "evidences": evidence_chunks,
        "language": language,
    }


def retrieve_evidence_for_dp_global_semantic_query_aware(
    dp: Dict[str, Any],
    retrieval_embeddings: HuggingFaceEmbeddings,
    semantic_tokenizer,
    semantic_model,
    instruction: str = "",
    k: int = 5,
    rerank_top_n: int = 20,
    semantic_device: str = "cpu",
    semantic_max_length: int = 256,
    semantic_batch_size: int = 16,
    left_window: int = 2,
    right_window: int = 2,
    smoothing_window: int = 3,
    percentile: float = 85.0,
    semantic_max_chunk_tokens: int = 220,
    semantic_min_chunk_sentences: int = 2,
    qa_top_k_per_segment: int = 2,
    qa_threshold: Optional[float] = None,
    qa_left_context: int = 1,
    qa_right_context: int = 1,
    qa_final_max_chunk_tokens: int = 260,
) -> Dict[str, Any]:
    claim = dp.get("claim", "")
    sources = dp.get("sources", [])
    language = dp.get("language", "en")
    formatted_claim = f"{instruction}\nQuery: {claim}".strip()

    all_docs = []

    for s_idx, source in enumerate(sources or []):
        content = clean_markdown_links(source.get("content", ""))
        if not content.strip():
            continue

        chunks = semantic_query_aware_chunker(
            claim=claim,
            text=content,
            tokenizer=semantic_tokenizer,
            model=semantic_model,
            device=semantic_device,
            embed_max_length=semantic_max_length,
            embed_batch_size=semantic_batch_size,
            left_window=left_window,
            right_window=right_window,
            smoothing_window=smoothing_window,
            percentile=percentile,
            max_chunk_tokens=semantic_max_chunk_tokens,
            min_chunk_sentences=semantic_min_chunk_sentences,
            qa_top_k_per_segment=qa_top_k_per_segment,
            qa_threshold=qa_threshold,
            qa_left_context=qa_left_context,
            qa_right_context=qa_right_context,
            final_max_chunk_tokens=qa_final_max_chunk_tokens,
        )

        for c_idx, chunk_text in enumerate(chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunk_id = f"{s_idx}_{c_idx}"
            metadata = {
                "source_index": s_idx,
                "source_url": source.get("source"),
                "chunk_id": chunk_id,
            }

            all_docs.append(
                Document(
                    page_content=chunk_text,
                    metadata=metadata
                )
            )

    if not all_docs:
        return {
            "id": dp.get("id"),
            "claim": claim,
            "label": dp.get("label"),
            "evidences": [],
            "language": language,
        }

    vectorstore = FAISS.from_documents(all_docs, retrieval_embeddings)

    initial_hits = vectorstore.similarity_search(
        formatted_claim,
        k=min(rerank_top_n, len(all_docs))
    )

    candidate_records = []
    for hit in initial_hits:
        meta = hit.metadata
        candidate_records.append(
            ChunkRecord(
                source_index=meta["source_index"],
                source_url=meta["source_url"],
                chunk_text=hit.page_content,
                chunk_id=meta["chunk_id"],
            )
        )

    reranked = rerank_chunks_with_biencoder(
        query=formatted_claim,
        chunks=candidate_records,
        tokenizer=semantic_tokenizer,
        model=semantic_model,
        device=semantic_device,
        max_length=semantic_max_length,
        batch_size=semantic_batch_size,
        top_k=k,
    )

    evidence_chunks = []
    for rec in reranked:
        evidence_chunks.append({
            "source_index": rec.source_index,
            "source_url": rec.source_url,
            "evidence": rec.chunk_text,
        })

    del all_docs, vectorstore, initial_hits, candidate_records, reranked
    gc.collect()
    maybe_clear_cuda(semantic_device)

    return {
        "id": dp.get("id"),
        "claim": claim,
        "label": dp.get("label"),
        "evidences": evidence_chunks,
        "language": language,
    }


def retrieve_evidence_for_dp_global_semantic_query_aware_adaptive(
    dp: Dict[str, Any],
    retrieval_embeddings: HuggingFaceEmbeddings,
    semantic_tokenizer,
    semantic_model,
    instruction: str = "",
    k: int = 5,
    rerank_top_n: int = 20,
    semantic_device: str = "cpu",
    semantic_max_length: int = 256,
    semantic_batch_size: int = 16,
    left_window: int = 2,
    right_window: int = 2,
    smoothing_window: int = 3,
    percentile: float = 85.0,
    semantic_max_chunk_tokens: int = 220,
    semantic_min_chunk_sentences: int = 2,
    qa_top_k_anchors_per_segment: int = 2,
    qa_anchor_min_query_sim: Optional[float] = None,
    qa_semantic_expand_threshold: float = 0.65,
    qa_query_min_threshold: Optional[float] = None,
    qa_max_expand_sentences: Optional[int] = None,
    qa_final_max_chunk_tokens: int = 260,
) -> Dict[str, Any]:
    claim = dp.get("claim", "")
    sources = dp.get("sources", [])
    language = dp.get("language", "en")
    formatted_claim = f"{instruction}\nQuery: {claim}".strip()

    all_docs = []

    for s_idx, source in enumerate(sources or []):
        content = clean_markdown_links(source.get("content", ""))
        if not content.strip():
            continue

        chunks = semantic_query_aware_adaptive_chunker(
            claim=claim,
            text=content,
            tokenizer=semantic_tokenizer,
            model=semantic_model,
            device=semantic_device,
            embed_max_length=semantic_max_length,
            embed_batch_size=semantic_batch_size,
            left_window=left_window,
            right_window=right_window,
            smoothing_window=smoothing_window,
            percentile=percentile,
            max_chunk_tokens=semantic_max_chunk_tokens,
            min_chunk_sentences=semantic_min_chunk_sentences,
            qa_top_k_anchors_per_segment=qa_top_k_anchors_per_segment,
            qa_anchor_min_query_sim=qa_anchor_min_query_sim,
            qa_semantic_expand_threshold=qa_semantic_expand_threshold,
            qa_query_min_threshold=qa_query_min_threshold,
            qa_max_expand_sentences=qa_max_expand_sentences,
            final_max_chunk_tokens=qa_final_max_chunk_tokens,
        )

        for c_idx, chunk_text in enumerate(chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunk_id = f"{s_idx}_{c_idx}"
            metadata = {
                "source_index": s_idx,
                "source_url": source.get("source"),
                "chunk_id": chunk_id,
            }

            all_docs.append(
                Document(
                    page_content=chunk_text,
                    metadata=metadata
                )
            )

    if not all_docs:
        return {
            "id": dp.get("id"),
            "claim": claim,
            "label": dp.get("label"),
            "evidences": [],
            "language": language,
        }

    vectorstore = FAISS.from_documents(all_docs, retrieval_embeddings)

    initial_hits = vectorstore.similarity_search(
        formatted_claim,
        k=min(rerank_top_n, len(all_docs))
    )

    candidate_records = []
    for hit in initial_hits:
        meta = hit.metadata
        candidate_records.append(
            ChunkRecord(
                source_index=meta["source_index"],
                source_url=meta["source_url"],
                chunk_text=hit.page_content,
                chunk_id=meta["chunk_id"],
            )
        )

    reranked = rerank_chunks_with_biencoder(
        query=formatted_claim,
        chunks=candidate_records,
        tokenizer=semantic_tokenizer,
        model=semantic_model,
        device=semantic_device,
        max_length=semantic_max_length,
        batch_size=semantic_batch_size,
        top_k=k,
    )

    evidence_chunks = []
    for rec in reranked:
        evidence_chunks.append({
            "source_index": rec.source_index,
            "source_url": rec.source_url,
            "evidence": rec.chunk_text,
        })

    del all_docs, vectorstore, initial_hits, candidate_records, reranked
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
        description="Evidence retrieval with sentence / fixed / semantic_old / semantic_improved / semantic_query_aware / semantic_query_aware_adaptive chunking."
    )

    parser.add_argument("--dataset", default="xfact", help="Name of dataset.")
    parser.add_argument("--input", default="train", help="Input split/file stem.")

    parser.add_argument(
        "--chunker",
        choices=[
            "sentence",
            "fixed",
            "semantic_old",
            "semantic_improved",
            "semantic_query_aware",
            "semantic_query_aware_adaptive",
        ],
        default="semantic_query_aware_adaptive",
        help="Choose chunking strategy."
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
        default="intfloat/multilingual-e5-large-instruct",
        help="HF model for semantic chunking sentence embeddings."
    )
    parser.add_argument(
        "--semantic-alpha",
        type=float,
        default=1.0,
        help="Adaptive threshold factor for old semantic chunking."
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
        help="Device for semantic model, e.g. cpu or cuda."
    )

    parser.add_argument("--left-window", type=int, default=2)
    parser.add_argument("--right-window", type=int, default=2)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument("--semantic-percentile", type=float, default=85.0)
    parser.add_argument("--semantic-max-chunk-tokens", type=int, default=220)
    parser.add_argument("--sliding-window-size", type=int, default=2)
    parser.add_argument("--sliding-stride", type=int, default=1)

    parser.add_argument("--qa-top-k-per-segment", type=int, default=2)
    parser.add_argument("--qa-threshold", type=float, default=None)
    parser.add_argument("--qa-left-context", type=int, default=1)
    parser.add_argument("--qa-right-context", type=int, default=1)
    parser.add_argument("--qa-final-max-chunk-tokens", type=int, default=260)

    parser.add_argument("--qa-top-k-anchors-per-segment", type=int, default=2)
    parser.add_argument("--qa-anchor-min-query-sim", type=float, default=None)
    parser.add_argument("--qa-semantic-expand-threshold", type=float, default=0.65)
    parser.add_argument("--qa-query-min-threshold", type=float, default=None)
    parser.add_argument("--qa-max-expand-sentences", type=int, default=None)

    parser.add_argument("--k", type=int, default=5, help="Top-k final chunks to return.")
    parser.add_argument("--rerank-top-n", type=int, default=20, help="Number of FAISS hits to rerank.")
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
        elif args.dataset == "xfact":
            input_file = INTERIM_DATA_DIR / "X-FACT" / f"{args.input}.jsonl"
        elif args.dataset == "translated_ru22fact":
            input_file = INTERIM_DATA_DIR / "translated_ru22fact" / f"{args.input}.jsonl"
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

    elif args.chunker == "semantic_old":
        output_file = (
            PROCESSED_DATA_DIR / f"{args.dataset}_alpha={args.semantic_alpha}" /
            f"{args.dataset}_{args.input}_with_custom_semantic_chunked_retrieved_evidence.jsonl"
        )

    elif args.chunker == "semantic_improved":
        output_file = (
            PROCESSED_DATA_DIR / f"{args.dataset}_global_semantic_p{args.semantic_percentile}_tok{args.semantic_max_chunk_tokens}" /
            f"{args.dataset}_{args.input}_with_global_sliding_semantic_retrieved_evidence.jsonl"
        )

    elif args.chunker == "semantic_query_aware":
        output_file = (
            PROCESSED_DATA_DIR / f"{args.dataset}_semantic_queryaware_p{args.semantic_percentile}_tok{args.semantic_max_chunk_tokens}" /
            f"{args.dataset}_{args.input}_with_semantic_query_aware_retrieved_evidence.jsonl"
        )

    elif args.chunker == "semantic_query_aware_adaptive":
        output_file = (
            PROCESSED_DATA_DIR /
            f"{args.dataset}_semantic_queryaware_adaptive_p{args.semantic_percentile}_tok{args.semantic_max_chunk_tokens}" /
            f"{args.dataset}_{args.input}_with_semantic_query_aware_adaptive_retrieved_evidence.jsonl"
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

    if args.chunker in [
        "semantic_old",
        "semantic_improved",
        "semantic_query_aware",
        "semantic_query_aware_adaptive",
    ]:
        semantic_tokenizer, semantic_model = build_custom_semantic_models(
            model_name=args.semantic_model,
            device=args.device,
        )

    skipped = 0
    processed_now = 0

    for dp in tqdm(datapoints, desc="Processing datapoints"):
        if args.dataset == "translated_xfact" or args.dataset == "translated_ru22fact":
            dp = translated_xfact_to_internal(dp)

        dp_key = make_dp_key(dp)

        if dp_key in processed_ids:
            skipped += 1
            continue

        if args.chunker == "semantic_query_aware_adaptive":
            result = retrieve_evidence_for_dp_global_semantic_query_aware_adaptive(
                dp=dp,
                retrieval_embeddings=retrieval_embeddings,
                semantic_tokenizer=semantic_tokenizer,
                semantic_model=semantic_model,
                instruction=args.instruction,
                k=args.k,
                rerank_top_n=args.rerank_top_n,
                semantic_device=args.device,
                semantic_max_length=args.semantic_max_length,
                semantic_batch_size=args.semantic_batch_size,
                left_window=args.left_window,
                right_window=args.right_window,
                smoothing_window=args.smoothing_window,
                percentile=args.semantic_percentile,
                semantic_max_chunk_tokens=args.semantic_max_chunk_tokens,
                semantic_min_chunk_sentences=args.semantic_min_chunk_sentences,
                qa_top_k_anchors_per_segment=args.qa_top_k_anchors_per_segment,
                qa_anchor_min_query_sim=args.qa_anchor_min_query_sim,
                qa_semantic_expand_threshold=args.qa_semantic_expand_threshold,
                qa_query_min_threshold=args.qa_query_min_threshold,
                qa_max_expand_sentences=args.qa_max_expand_sentences,
                qa_final_max_chunk_tokens=args.qa_final-max-chunk-tokens if False else args.qa_final_max_chunk_tokens,  # keep parser name
            )

        elif args.chunker == "semantic_query_aware":
            result = retrieve_evidence_for_dp_global_semantic_query_aware(
                dp=dp,
                retrieval_embeddings=retrieval_embeddings,
                semantic_tokenizer=semantic_tokenizer,
                semantic_model=semantic_model,
                instruction=args.instruction,
                k=args.k,
                rerank_top_n=args.rerank_top_n,
                semantic_device=args.device,
                semantic_max_length=args.semantic_max_length,
                semantic_batch_size=args.semantic_batch_size,
                left_window=args.left_window,
                right_window=args.right_window,
                smoothing_window=args.smoothing_window,
                percentile=args.semantic_percentile,
                semantic_max_chunk_tokens=args.semantic_max_chunk_tokens,
                semantic_min_chunk_sentences=args.semantic_min_chunk_sentences,
                qa_top_k_per_segment=args.qa_top_k_per_segment,
                qa_threshold=args.qa_threshold,
                qa_left_context=args.qa_left_context,
                qa_right_context=args.qa_right_context,
                qa_final_max_chunk_tokens=args.qa_final_max_chunk_tokens,
            )

        elif args.chunker == "semantic_improved":
            result = retrieve_evidence_for_dp_global_sliding_semantic(
                dp=dp,
                retrieval_embeddings=retrieval_embeddings,
                semantic_tokenizer=semantic_tokenizer,
                semantic_model=semantic_model,
                instruction=args.instruction,
                k=args.k,
                rerank_top_n=args.rerank_top_n,
                semantic_device=args.device,
                semantic_max_length=args.semantic_max_length,
                semantic_batch_size=args.semantic_batch_size,
                left_window=args.left_window,
                right_window=args.right_window,
                smoothing_window=args.smoothing_window,
                percentile=args.semantic_percentile,
                semantic_max_chunk_tokens=args.semantic_max_chunk_tokens,
                semantic_min_chunk_sentences=args.semantic_min_chunk_sentences,
                sliding_window_size=args.sliding_window_size,
                sliding_stride=args.sliding_stride,
            )

        elif args.chunker == "semantic_old":
            result = retrieve_evidence_for_dp(
                dp=dp,
                chunk_mode="semantic",
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

        else:
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