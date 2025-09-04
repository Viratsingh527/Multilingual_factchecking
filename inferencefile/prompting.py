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


import re

def make_prompt_from_input(full_input: str, instruction_text: str, prompting: str = "vanilla") -> str:
    """
    Builds a prompt using the combined input and instruction text.
    Supports multiple prompting strategies.
    """
    match = re.match(r"Classify the given ([A-Za-z\-]+)", instruction_text)
    language = match.group(1) if match else "unknown"

    base_instruction = (
        f"Classify the given {language} claim into one of the seven categories "
        "(TRUE, MOSTLY‑TRUE, PARTLY‑TRUE/MISLEADING, FALSE, MOSTLY‑FALSE, "
        "COMPLICATED/HARD‑TO‑CATEGORISE, OTHER) based on the provided evidence."
        "\n\n1. TRUE: Fully supported by the evidence."
        "\n2. MOSTLY‑TRUE: Mostly supported but with minor inaccuracies."
        "\n3. PARTLY‑TRUE/MISLEADING: Partially supported, but includes significant omissions or could mislead."
        "\n4. FALSE: Clearly contradicted or unsupported by the evidence."
        "\n5. MOSTLY‑FALSE: Largely incorrect with only a small element of truth."
        "\n6. COMPLICATED/HARD‑TO‑CATEGORISE: Too complex or nuanced to assign a straightforward label."
        "\n7. OTHER: Does not fit into any of the above categories."
    )

    if prompting == "vanilla":
        prompt = f"##Instruction: {base_instruction}\n\n##input: {full_input} \n\n##output: "
    elif prompting == "cot":
        prompt = (
            f"##Instruction: {base_instruction}\n\n"
            f"##input: {full_input} \n\n"
            "Let's think step by step. So, the claim veracity is: "
        )
    elif prompting == "role-based":
        prompt = (
            f"You are an expert fact-checker in this {language} language. Based on the evidence, classify the following claim.\n"
            f"##input: {full_input}\nLabel: "
        )
    # elif prompting == "structured":
    #     prompt = (
    #         f"##Instruction: You will receive a claim and evidence. Your task is to verify the claim "
    #         f"based strictly on the evidence using the following label set: TRUE, MOSTLY‑TRUE, PARTLY‑TRUE/MISLEADING, "
    #         f"FALSE, MOSTLY‑FALSE, COMPLICATED/HARD‑TO‑CATEGORISE, OTHER.\n\n"
    #         f"##input: {full_input}\n\nProvide your answer in this format: Claim Veracity: <LABEL>\n\n"
    #     )
    # elif prompting == "zero-shot":
    #     prompt = (
    #         f"Classify the following claim as one of: TRUE, MOSTLY-TRUE, PARTLY-TRUE/MISLEADING, FALSE, MOSTLY-FALSE, COMPLICATED/HARD-TO-CATEGORISE, OTHER.\n"
    #         f"##input: {full_input}\nLabel: "
    #     )
    # elif prompting == "few-shot":
    #     # Hardcoded few-shot examples for illustration
    #     prompt = (
    #         "Classify the following claims:\n"
    #         "Example 1:\nClaim: The earth revolves around the sun.\nLabel: true\n\n"
    #         "Example 2:\nClaim: Water boils at 50 degrees Celsius.\nLabel: false\n\n"
    #         f"Now classify this claim:\n##input: {full_input}\nLabel: "
    #     )
    # elif prompting == "rationale":
    #     prompt = (
    #         f"Given the claim and evidence, explain your reasoning and then classify the claim.\n"
    #         f"##input: {full_input}\nRationale:\nLabel: "
    #     )
    # elif prompting == "structured-json":
    #     prompt = (
    #         f"Classify the claim and provide your answer in the following JSON format:\n"
    #         "{\n  \"rationale\": \"<your reasoning>\",\n  \"label\": \"<one of the categories>\"\n}\n"
    #         f"##input: {full_input}\n"
    #     )
    # elif prompting == "self-consistency":
    #     prompt = (
    #         f"Classify the claim. Repeat your answer three times, thinking step by step each time.\n"
    #         f"##input: {full_input}\nAnswer 1:\nAnswer 2:\nAnswer 3:\nFinal Label (most common): "
    #     )

    else:
        raise ValueError(f"Unsupported prompting type: {prompting}")

    return prompt

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

def extract_dataset_type(data_path: str) -> str:
    """
    Extracts the dataset type from file name. Assumes filename starts with 'xfact_'.
    Example: 'xfact_zeroshot_concrete.json' → 'zeroshot_concrete'
    """
    base = Path(data_path).stem  # e.g. xfact_zeroshot_concrete
    if base.startswith("xfact_"):
        return base[len("xfact_"):]  # strip prefix
    return base

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

# Removed imports
# from peft import PeftModel

# ...

def evaluate(
    base_model_path: str,
    test_data_path: str,
    output_dir: str,
    prompting: str,
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
    model.eval()

    print("Reading examples …")
    with open(test_data_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if max_examples is not None:
        data = data[:max_examples]

    print(f"→ Evaluating on {len(data)} examples\n")

    predictions, gold = [], []
    results_dump = []

    for ex in tqdm(data):
        try:
            full_input: str = ex["input"]
            gold_label: str = normalise_label(ex["output"])
            match = re.match(r"Classify the given ([A-Za-z\-]+)", ex["instruction"])
            language = match.group(1) if match else "unknown"

            prompt = make_prompt_from_input(full_input, ex["instruction"], prompting=prompting)

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(model.device)

            out = model.generate(
                **enc,
                max_new_tokens=1000,
                temperature=0.0,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

            decoded = tokenizer.decode(
                out[0][enc.input_ids.shape[1]:], skip_special_tokens=True
            )
            pred_label = extract_label(decoded)

            predictions.append(pred_label)
            gold.append(gold_label)
            results_dump.append({
                "instruction": ex["instruction"],
                "input": ex["input"],
                "true_label": gold_label,
                "predicted_label": pred_label,
                "model_response": decoded,
                "language": language,
            })

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
        gold, predictions, labels=filtered_labels, zero_division=0, digits=4
    )
    print(report)
    model_shortname = get_model_shortname(base_model_path)
    dataset_type = extract_dataset_type(test_data_path)
    file_prefix = f"{model_shortname}_{prompting}_{dataset_type}"


    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / f"{file_prefix}_predictions.json", "w", encoding="utf-8") as f_pred:
        json.dump(results_dump, f_pred, indent=2, ensure_ascii=False)

    with open(Path(output_dir) / f"{file_prefix}_metrics.txt", "w", encoding="utf-8") as f_met:
        f_met.write(report)

    print(f"\nSaved predictions & report to → {output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate X‑FACT base model")
    p.add_argument("--base_model", required=True, help="HF path / local dir of the base model")
    p.add_argument("--data", required=True, help="Path to .jsonl test file")
    p.add_argument("--out_dir", default="outputs", help="Where to store predictions + metrics")
    p.add_argument("--max_examples", type=int, default=None, help="Optionally evaluate only the first N examples")
    p.add_argument("--max_length", type=int, default=10000, help="length of input sequence to the model (default: 10000)")
    p.add_argument("--prompting", type=str, default="vanilla", help="Prompting strategy: vanilla | cot | structured | ...")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate(
        base_model_path=args.base_model,
        test_data_path=args.data,
        output_dir=args.out_dir,
        prompting=args.prompting,
        max_examples=args.max_examples,
        max_length=args.max_length,
    )