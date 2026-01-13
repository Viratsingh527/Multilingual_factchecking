# semantic_chunker.py
import torch
import torch.nn.functional as F

# semantic_chunker_langchain.py
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings


def semantic_chunk_langchain_from_tokenized(
    tokenized: list,
    embedding_model_name: str = "intfloat/multilingual-e5-base",
    min_chunk_sentences: int = 2,
):
    """
    tokenized: list of dicts with keys: "sent", "is_evidence", ...

    Returns:
      chunk_dicts: List[Dict[sentence -> label]]
      chunk_texts: List[str]
    """

    # Extract sentences + labels
    sentences = []
    labels = {}

    for item in tokenized:
        sent = (item.get("sent") or "").strip()
        if not sent:
            continue
        sentences.append(sent)
        labels[sent] = item.get("is_evidence")

    if not sentences:
        return [], []

    full_text = " ".join(sentences)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=0.5,   # LangChain-like default
        min_chunk_size=min_chunk_sentences,
    )

    raw_chunks = splitter.split_text(full_text)

    # ---- Align chunks back to sentences ----
    chunk_dicts = []
    chunk_texts = []

    for chunk in raw_chunks:
        chunk_dict = {}
        for sent in sentences:
            if sent in chunk:
                chunk_dict[sent] = labels.get(sent)
        if chunk_dict:
            chunk_dicts.append(chunk_dict)
            chunk_texts.append(" ".join(chunk_dict.keys()))

    return chunk_dicts, chunk_texts

@torch.no_grad()
def mean_pool_embeddings(model, tokenizer, texts, device="cpu", max_length=256):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    out = model(**enc)
    token_emb = out.last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1)

    pooled = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return F.normalize(pooled, p=2, dim=1)


def semantic_chunk_from_tokenized(
    tokenized: list,
    tokenizer,
    model,
    label_key: str = "is_evidence",
    min_chunk_sentences: int = 2,
    alpha: float = 0.5,
    device: str = "cpu",
):
    """
    tokenized: list of dicts like:
      {"sent": "...", "is_evidence": 0/1, ...}

    Returns:
      chunks_dicts: List[Dict[sentence_text -> label_value]]
      chunk_texts:  List[str]   (joined sentence text per chunk, used for retrieval)
    """
    # Extract ordered sentences and labels
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

    # Embed each sentence
    emb = mean_pool_embeddings(model, tokenizer, sentences, device=device)

    # Adjacent similarity & distance
    sims = (emb[:-1] * emb[1:]).sum(dim=1)
    distances = 1.0 - sims
    d = distances.detach().cpu()

    # Adaptive threshold like LangChain style
    mean = d.mean().item()
    std = d.std(unbiased=False).item()
    threshold = mean + alpha * std

    # Breakpoints: split after i if distance is big
    breakpoints = [i for i, val in enumerate(d.tolist()) if val > threshold]

    # Build chunks by sentence indices
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

    # tail
    if start < n:
        chunk_dict = {sentences[i]: labels[i] for i in range(start, n)}
        chunks_dicts.append(chunk_dict)
        chunk_texts.append(" ".join(sentences[start:n]))

    # Merge tiny tail chunk
    if len(chunks_dicts) >= 2 and len(chunks_dicts[-1]) < min_chunk_sentences:
        # merge into previous
        prev = chunks_dicts[-2]
        prev.update(chunks_dicts[-1])
        chunk_texts[-2] = chunk_texts[-2] + " " + chunk_texts[-1]
        chunks_dicts.pop()
        chunk_texts.pop()

    return chunks_dicts, chunk_texts
