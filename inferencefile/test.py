# test_fixed.py
"""A cleaned‑up evaluation script for X‑FACT multilingual fact‑checking.

Key fixes
---------
1. **Canonical label handling** – All possible spelling / hyphen variations are
   mapped to a single canonical label so both gold labels and predictions are
   evaluated consistently.
2. **Consistent label list** – `LABEL_ORDER` is used everywhere (prompt
   cleaning, metrics) so we never accidentally drop a class in the report.
3. **Utility functions re‑organised** – Helpers for prompt construction and
   label normalisation live at the top of the file for clarity.
4. **Evaluation over the full dev/test set** – The artificial 50‑example slice
   has been removed; evaluate on the whole file unless the caller passes a
   smaller `--max_examples` flag.

Usage
-----
```
python test_fixed.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --adapter saves/llama3-8b_llama_webreteived_evidences/lora/sft \
  --data test_xfact.jsonl \
  --out_dir outputs/xfact_predictions/llama3_lora
```
"""
from __future__ import annotations
import unicodedata
import argparse
import json
import os
import re
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List, Optional
import csv
import torch
from peft import PeftModel
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

###############################################################################
# Canonical labels & normalisation helpers
###############################################################################

# Canonical order in which we want metrics to be reported
LABEL_ORDER: List[str] = [
    "true",
    "mostly true",
    "partly true/misleading",
    "false",
    "mostly false",
    "complicated/hard-to-categorise",
    "other",
]

# All acceptable surface forms → canonical form
LABEL_CANONICAL: Dict[str, str] = {
    "true": "true",
    "mostly true": "mostly true",
    "mostly-true": "mostly true",
    "partly true/misleading": "partly true/misleading",
    "partly-true/misleading": "partly true/misleading",
    "false": "false",
    "mostly false": "mostly false",
    "mostly-false": "mostly false",
    "complicated/hard-to-categorise": "complicated/hard-to-categorise",
    "complicated/hard to categorise": "complicated/hard-to-categorise",
    "complicated/hard to categorize": "complicated/hard-to-categorise",
    "other": "other",
}

# For fuzzy‑matching predictions that are slightly off (e.g. missing spaces)

LABEL_SET_LOWER = set(LABEL_CANONICAL.keys())  # for fuzzy matching

_DASH_CHARS = re.compile(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D\u002D]")


def _standardise_dashes(text: str) -> str:
    """Replace *any* Unicode dash with simple ASCII hyphen (-)."""
    return _DASH_CHARS.sub("-", text)


def normalise_label(text: str) -> str:
    """Lower‑cases, ASCII‑hyphenises, then maps to canonical form."""
    if not text:
        return "other"

    # 1) Unicode normalisation + dash folding
    cleaned = _standardise_dashes(unicodedata.normalize("NFKC", text))
    cleaned = re.sub(r"\s+", " ", cleaned)  # collapse whitespace
    cleaned = cleaned.strip().lower()

    if cleaned in LABEL_CANONICAL:
        return LABEL_CANONICAL[cleaned]

    # fuzzy backup
    match = get_close_matches(cleaned, LABEL_SET_LOWER, n=1, cutoff=0.7)
    return LABEL_CANONICAL[match[0]] if match else "other"


###############################################################################
# Prompt construction
###############################################################################

# SYSTEM_PROMPT = (
#     "You are a multilingual fact‑checking expert. You are given a news claim "
#     "along with raw evidence related to the claim. Do not consider any other "
#     "evidence. Your task is to evaluate the veracity of the claim based solely "
#     "on the provided raw evidence.\n\n"
#     "Output your answer in exactly the following format:\n"
#     "Claim Veracity: [label]\n\n"
#     "Answer with one word: TRUE, MOSTLY‑TRUE, PARTLY‑TRUE/MISLEADING, "
#     "FALSE, MOSTLY‑FALSE, COMPLICATED/HARD‑TO‑CATEGORISE, OTHER."
#     "Think step by step"
# )

LANGUAGE_MAP  = {
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
    "fr": "French"
}

def get_model_shortname(model_path: str) -> str:
    """
    Given a model path like 'meta-llama/Llama-3.1-8B-Instruct',
    return a short identifier like 'llama'.
    """
    # Take last part of path (e.g. 'Llama-3.1-8B-Instruct')
    name = Path(model_path).name.lower()
    # Split on dash and take first part
    return name.split("-")[0]

def extract_chunking_or_retrieval(path: str):
    """
    Extracts the chunking/retrieval technique and random seed from an adapter path.
    Handles both long names (model_dataset_technique_evidences) and shorter names.
    """

    # Normalize path to avoid empty basename from trailing slash
    path = path.rstrip("/")

    # Try to get a random seed from the last directory name
    last_dir = os.path.basename(path)
    if "_" in last_dir and last_dir.split("_")[-1].isdigit():
        random_seed = last_dir.split("_")[-1]
    else:
        random_seed = "default"

    # Adapter folder (two levels up: .../<adapter_name>/lora/sft_xxx)
    adapter_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    parts = adapter_name.split("_")

    # If the name follows the expected long format
    if len(parts) >= 4:
        technique = "_".join(parts[3:])
    elif len(parts) >= 2:
        # Fallback: take the last part(s)
        technique = "_".join(parts[1:])
    else:
        technique = adapter_name

    # Remove trailing "_evidences" if present
    if technique.endswith("_evidences"):
        technique = technique.rsplit("_evidences", 1)[0]

    return technique, random_seed


def extract_dataset_and_testset(name: str):
    """
    Extracts dataset name and testset type from a test data string.
    
    Example:
    >>> extract_dataset_and_testset("xfact_ood_with_sentence_level_chunked_retrieved_evidence")
    ('xfact', 'ood')
    """
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError("Name format not valid. Expected at least 2 parts: dataset and testset")
    
    dataset = parts[0]
    testset = parts[1]
    
    return dataset, testset

def build_prompt(claim: str, evidences: List[Dict[str, str]], lang_code: str) -> str:
    """Assemble the full prompt string."""

    language = LANGUAGE_MAP.get(lang_code)
    instruction = (
        f"Classify the given {language} claim into one of the seven categories "
        "(TRUE, MOSTLY‑TRUE, PARTLY‑TRUE/MISLEADING, FALSE, MOSTLY‑FALSE, "
        "COMPLICATED/HARD‑TO‑CATEGORISE, OTHER) based on the provided evidence."
        "\n\n1. TRUE: Fully supported by the evidence."
        "\n2. MOSTLY‑TRUE: Mostly supported but with minor inaccuracies."
        "\n3. PARTLY‑TRUE/MISLEADING: Partially supported, but includes significant "
        "omissions or could mislead."
        "\n4. FALSE: Clearly contradicted or unsupported by the evidence."
        "\n5. MOSTLY‑FALSE: Largely incorrect with only a small element of truth."
        "\n6. COMPLICATED/HARD‑TO‑CATEGORISE: Too complex or nuanced to assign a "
        "straightforward label."
        "\n7. OTHER: Does not fit into any of the above categories."
        "\n\nProvide exactly one label."
    )

    evidence_blocks = "\n\n".join(
        [f"Evidence_{1+i}: {ev['evidence']}" for i, ev in enumerate(evidences)]
    )

    return (
        f"##Instruction: {instruction}\n\nClaim: {claim}\n\n"
        f"Evidences:\n{evidence_blocks} \n\n so, the Claim Veracity is: "
    )


###############################################################################
# Response cleaning
###############################################################################

# Regex caches (compiled once)
STRUCTURED_RE = re.compile(
    r"(claim veracity|verdict|label|prediction)\s*:\s*([^\n]+)",
    re.I,
)
# Build a regex with *any* dash variant folded to ASCII ‘-’ beforehand.
LABEL_PATTERN_RE = re.compile(
    r"\b("
    + "|".join(map(re.escape, [_standardise_dashes(l) for l in LABEL_SET_LOWER]))
    + r")\b",
    re.I,
)


def extract_label(response: str) -> str:
    if not response:
        return "other"

    resp_std = _standardise_dashes(response)  # fold dashes once

    # 1) "Claim Veracity: …" pattern
    m = STRUCTURED_RE.search(resp_std)
    if m:
        return normalise_label(m.group(2))

    # 2) First token matching any label surface form
    m2 = LABEL_PATTERN_RE.search(resp_std)
    if m2:
        return normalise_label(m2.group(0))

    # 3) fallback fuzzy
    return normalise_label(resp_std)


###############################################################################
# Evaluation logic
###############################################################################

def load_data(file_path, lang="all"):
    ext = os.path.splitext(file_path)[-1].lower()

    # ---- Load data depending on format ----
    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                data = [data]

    elif ext == ".jsonl":
        data = []
        with open(file_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    data.append(json.loads(line))

    elif ext == ".csv":
        data = []
        with open(file_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                data.append(row)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # ---- Language filtering ----
    if lang.lower() != "all":
        # Accept both language codes ("hi") and names ("Hindi")
        lang_code = None
        # normalize: if user gives "Hindi", map back to "hi"
        for code, name in LANGUAGE_MAP.items():
            if lang.lower() in (code.lower(), name.lower()):
                lang_code = code
                break

        if not lang_code:
            raise ValueError(f"Unsupported language: {lang}")

        filtered_data = []
        for item in data:
            item_lang = item.get("language") or item.get("lang")
            if item_lang and item_lang.lower() == lang_code:
                filtered_data.append(item)
        data = filtered_data

    return data

def evaluate(
    lang: str,
    base_model_path: str,
    adapter_path: str,
    test_data_path: str,
    output_dir: str,
    max_examples: Optional[int] = None,
    max_length: int = 10000,
):
    """Run inference + classification report on dataset with 'instruction', 'input', and 'output' fields."""

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model …")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading LoRA adapter …")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print("Reading examples …")
    data = load_data(test_data_path,lang)  # works for .json, .jsonl, or .csv
    print(f"Loaded {len(data)} examples")

    if max_examples is not None:
        data = data[:max_examples]

    print(f"→ Evaluating on {len(data)} examples\n")

    predictions, gold = [], []
    results_dump = []

    for ex in tqdm(data):
        try:
            # Use fields from new format
            claim: str = ex["claim"]
            gold_label: str = normalise_label(ex["label"])
            lang_code = ex["language"]
            evidences = ex["evidences"]
            language = LANGUAGE_MAP.get(lang_code)
            # prompt = f"##Instruction: {ex['instruction']}\n\n{full_input} \n\nso, the Claim Veracity is: "
            prompt = build_prompt(claim,evidences,lang_code)

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(model.device)

            out = model.generate(
                **enc,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

            decoded = tokenizer.decode(
                out[0][enc.input_ids.shape[1] :], skip_special_tokens=True
            )
            pred_label = extract_label(decoded)

            predictions.append(pred_label)
            gold.append(gold_label)
            results_dump.append(
                {
                    "input": prompt,
                    "true_label": gold_label,
                    "predicted_label": pred_label,
                    "model_response": decoded,
                    "language": language,
                }
            )

        except Exception as exc:
            print(f"[!] Error on example → {exc}")
            predictions.append("other")
            gold.append("other")

    from collections import Counter

    print("\nClassification report (canonical labels):")
    gold_counts = Counter(gold)
    filtered_labels = [label for label in LABEL_ORDER if gold_counts[label] > 0]

    print("Label counts:")
    for label in LABEL_ORDER:
        print(f"{label:35s}  true: {gold_counts.get(label, 0):4d}")

    report = classification_report(
        gold, predictions, labels=filtered_labels, zero_division=0,digits=4
    )
    print(report)
    model_shortname = get_model_shortname(base_model_path)
    dataset, testset = extract_dataset_and_testset(Path(test_data_path).stem)
    technique, random_seed = extract_chunking_or_retrieval(adapter_path)
    if testset == "test":
        testset = "Indomain"
    output_dir = Path(output_dir) / f"{dataset}/{testset}/{model_shortname}/{technique}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(
        Path(output_dir) / f"predictions_{random_seed}.json", "w", encoding="utf-8"
    ) as f_pred:
        json.dump(results_dump, f_pred, indent=2, ensure_ascii=False)

    with open(
        Path(output_dir) / f"metrics__{random_seed}.txt", "w", encoding="utf-8"
    ) as f_met:
        f_met.write(report)

    print(f"\nSaved predictions & report to → {output_dir}")


###############################################################################
# CLI entry‑point
###############################################################################


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate X‑FACT model with LoRA adapter")
    p.add_argument(
        "--base_model", required=True, help="HF path / local dir of the base model"
    )
    p.add_argument(
        "--adapter", required=True, help="Path to the LoRA adapter checkpoint"
    )
    p.add_argument("--data", required=True, help="Path to test file")
    p.add_argument(
        "--out_dir", default="outputs", help="Where to store predictions + metrics"
    )
    p.add_argument("--language",default="all",help="Specify langauge of test data, e.g. for all langauge give input all for hindi write only hindi etc.")
    p.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optionally evaluate only the first N examples",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=10000,
        help="length of input sequence to the model (default: 10000)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        lang = args.language,
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        test_data_path=args.data,
        output_dir=args.out_dir,
        max_examples=args.max_examples,
        max_length=args.max_length,
    )
