# test_fixed.py

"""A cleaned-up evaluation script for X-FACT multilingual fact-checking.

Key fixes
---------
1. Canonical label handling – All possible spelling/hyphen variations are
   mapped to a single canonical label so both gold labels and predictions are
   evaluated consistently.
2. Consistent label list – LABEL_ORDER is used everywhere (prompt
   cleaning, metrics) so we never accidentally drop a class in the report.
3. Utility functions re-organised – Helpers for prompt construction and
   label normalisation live at the top of the file for clarity.
4. Evaluation over the full dev/test set – Evaluate on the whole file
   unless the caller passes a smaller --max_examples flag.

Usage
-----
Example:

    python test_fixed.py \
      --base_model meta-llama/Llama-3.1-8B-Instruct \
      --adapter saves/llama3-8b_llama_webreteived_evidences/lora/sft \
      --data test_xfact.json \
      --out_dir outputs/xfact_predictions/llama3_lora \
      --language hindi
"""

from __future__ import annotations
import unicodedata
import argparse
import json
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

LABEL_ORDER: List[str] = [
    "true",
    "mostly true",
    "partly true/misleading",
    "false",
    "mostly false",
    "complicated/hard-to-categorise",
    "other",
]

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

LABEL_SET_LOWER = set(LABEL_CANONICAL.keys())

_DASH_CHARS = re.compile(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D\u002D]")


def _standardise_dashes(text: str) -> str:
    """Replace *any* Unicode dash with simple ASCII hyphen (-)."""
    return _DASH_CHARS.sub("-", text)


def normalise_label(text: str) -> str:
    """Lower-cases, ASCII-hyphenises, then maps to canonical form."""
    if not text:
        return "other"

    cleaned = _standardise_dashes(unicodedata.normalize("NFKC", text))
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip().lower()

    if cleaned in LABEL_CANONICAL:
        return LABEL_CANONICAL[cleaned]

    # fuzzy backup
    match = get_close_matches(cleaned, LABEL_SET_LOWER, n=1, cutoff=0.7)
    return LABEL_CANONICAL[match[0]] if match else "other"


###############################################################################
# Response cleaning
###############################################################################

STRUCTURED_RE = re.compile(
    r"(claim veracity|verdict|label|prediction)\s*:\s*([^\n]+)",
    re.I,
)

LABEL_PATTERN_RE = re.compile(
    r"\b("
    + "|".join(map(re.escape, [_standardise_dashes(l) for l in LABEL_SET_LOWER]))
    + r")\b",
    re.I,
)


def extract_label(response: str) -> str:
    if not response:
        return "other"

    resp_std = _standardise_dashes(response)

    # 1) Look for "Claim Veracity: …" pattern
    m = STRUCTURED_RE.search(resp_std)
    if m:
        return normalise_label(m.group(2))

    # 2) Look for any known label word in text
    m2 = LABEL_PATTERN_RE.search(resp_std)
    if m2:
        return normalise_label(m2.group(0))

    # 3) Fallback to fuzzy matching entire response
    return normalise_label(resp_std)


###############################################################################
# Evaluation logic
###############################################################################
def get_model_shortname(model_path: str) -> str:
    """
    Given a model path like 'meta-llama/Llama-3.1-8B-Instruct',
    return a short identifier like 'llama'.
    """
    # Take last part of path (e.g. 'Llama-3.1-8B-Instruct')
    name = Path(model_path).name.lower()
    # Split on dash and take first part
    return name.split("-")[0]

def evaluate(
    base_model_path: str,
    adapter_path: str,
    test_data_path: str,
    output_dir: str,
    max_examples: Optional[int] = None,
    language: str = "hindi",
):
    """Run inference + classification report."""

    lang_cap = language.capitalize()

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model…")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading LoRA adapter…")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print("Reading examples…")
    with open(test_data_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Filter only examples for this language
    data = [
        ex
        for ex in data
        if ex.get("instruction", "").startswith(f"Classify the given {lang_cap}")
    ]

    if max_examples is not None:
        data = data[:max_examples]

    print(f"→ Evaluating on {len(data)} examples\n")

    predictions = []
    gold = []
    results_dump = []

    for ex in tqdm(data):
        try:
            full_input = ex["input"]
            gold_label = normalise_label(ex["output"])

            prompt = (
                f"##Instruction:You are an expert fact-checker in {lang_cap}."
                f"{ex['instruction']}\n\n##input: {full_input}\n\n##output: "
            )

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=10000,
            ).to(model.device)

            out = model.generate(
                **enc,
                max_new_tokens=10,
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
                "language": language.lower(),
            })

        except Exception as e:
            print(f"[!] Error on example → {e}")
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
        gold,
        predictions,
        labels=filtered_labels,
        zero_division=0,
        digits=4,
    )
    print(report)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_shortname = get_model_shortname(base_model_path)
    with open(
        Path(output_dir) / f"{model_shortname}_predictions_{language.lower()}.json",
        "w",
        encoding="utf-8",
    ) as f_pred:
        json.dump(results_dump, f_pred, indent=2, ensure_ascii=False)

    with open(
        Path(output_dir) / f"{model_shortname}_metrics_{language.lower()}.txt",
        "w",
        encoding="utf-8",
    ) as f_met:
        f_met.write(report)

    print(f"\nSaved predictions & report to → {output_dir}")


###############################################################################
# CLI entry-point
###############################################################################


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate X-FACT model with LoRA adapter"
    )
    parser.add_argument(
        "--base_model",
        required=True,
        help="Hugging Face path or local dir of the base model",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to the LoRA adapter checkpoint",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to .json test file",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs",
        help="Where to store predictions + metrics",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optionally evaluate only the first N examples",
    )
    parser.add_argument(
        "--language",
        required=True,
        help="Language to evaluate (e.g. hindi, arabic, spanish, etc.)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        test_data_path=args.data,
        output_dir=args.out_dir,
        max_examples=args.max_examples,
        language=args.language,
    )
