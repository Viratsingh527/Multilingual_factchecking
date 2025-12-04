# test_fixed.py
"""A cleaned-up evaluation script for X-FACT multilingual fact-checking.

Key fixes
---------
1. Canonical label handling
2. Consistent label list
3. Utility functions cleaned
4. Full dataset evaluation
5. Cross-lingual evidence evaluation when lang != 'all'

Usage
-----
python test_fixed.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --adapter saves/llama3-8b_llama_webreteived_evidences/lora/sft \
  --data test_xfact.jsonl \
  --out_dir outputs/xfact_predictions/llama3_lora
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
# Argument parsing
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Finetuned model with LoRA adapter")
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--language", default="all")
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--max_length", type=int, default=10000)
    return p.parse_args()

args = parse_args()

def extract_dataset_and_testset(path: str):
    p = Path(path)

    dataset = p.parent.name       # last folder name → dataset
    testset = p.stem              # file name without extension → testset

    if not dataset or not testset:
        raise ValueError("Could not extract dataset or testset from path")

    return dataset, testset

###############################################################################
# Canonical labels
###############################################################################

LABEL_SETS = {
    "xfact": {
        "order": [
            "true", "mostly true", "partly true/misleading", "false",
            "mostly false", "complicated/hard-to-categorise", "other",
        ],
        "canonical": {
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
        },
    },
    "ru22fact": {
        "order": ["supported", "refuted", "nei"],
        "canonical": {
            "supported": "supported",
            "support": "supported",
            "refuted": "refuted",
            "refute": "refuted",
            "nei": "nei",
            "not enough info": "nei",
        },
    },
}

# dataset_name, _ = extract_dataset_and_testset(Path(args.data).stem)
dataset_name, _ = extract_dataset_and_testset(args.data)
label_config = LABEL_SETS[dataset_name]
LABEL_ORDER = label_config["order"]
LABEL_CANONICAL = label_config["canonical"]

LABEL_SET_LOWER = set(LABEL_CANONICAL.keys())
_DASH_CHARS = re.compile(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D\u002D]")

def _standardise_dashes(text: str) -> str:
    return _DASH_CHARS.sub("-", text)

def normalise_label(text: str) -> str:
    if not text:
        return "other"
    cleaned = _standardise_dashes(unicodedata.normalize("NFKC", text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    if cleaned in LABEL_CANONICAL:
        return LABEL_CANONICAL[cleaned]
    match = get_close_matches(cleaned, LABEL_SET_LOWER, n=1, cutoff=0.7)
    return LABEL_CANONICAL[match[0]] if match else "other"

###############################################################################
# Prompt construction
###############################################################################

LANGUAGE_MAP = {
    "tr": "Turkish", "ka": "Georgian", "pt": "Portuguese", "id": "Indonesian",
    "sr": "Serbian", "it": "Italian", "de": "German", "ro": "Romanian",
    "ta": "Tamil", "pl": "Polish", "hi": "Hindi", "ar": "Arabic", "es": "Spanish",
    "bn": "Bengali", "fa": "Persian", "gu": "Gujarati", "mr": "Marathi",
    "pa": "Punjabi", "no": "Norwegian", "si": "Sinhala", "sq": "Albanian",
    "ru": "Russian", "az": "Azerbaijani", "nl": "Dutch", "fr": "French"
}

FIXED_EVIDENCE_LANGS = [
    "tr","ka","pt","id","sr","it","de","ro","ta","pl","hi","ar","es"
]

def get_model_shortname(model_path: str) -> str:
    name = Path(model_path).name.lower()
    return name.split("-")[0]

def extract_chunking_or_retrieval(path: str):
    path = path.rstrip("/")
    last_dir = os.path.basename(path)

    if "_" in last_dir and last_dir.split("_")[-1].isdigit():
        random_seed = last_dir.split("_")[-1]
    else:
        random_seed = "default"

    adapter_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    parts = adapter_name.split("_")

    if len(parts) >= 4:
        technique = "_".join(parts[3:])
    elif len(parts) >= 2:
        technique = "_".join(parts[1:])
    else:
        technique = adapter_name

    if technique.endswith("_evidences"):
        technique = technique.rsplit("_evidences", 1)[0]
    return technique, random_seed

SYSTEM_PROMPT = {
    "xfact":
        "You are a multilingual fact-checking expert. You are given a news claim along with raw evidence. "
        "Your task is to classify the veracity strictly based on provided evidence.\n\n"
        "Output format:\nClaim Veracity: [label]\n\n"
        "Labels: TRUE, MOSTLY-TRUE, PARTLY-TRUE/MISLEADING, FALSE, MOSTLY-FALSE, "
        "COMPLICATED/HARD-TO-CATEGORISE, OTHER.",
    "ru22fact":
        "You are a multilingual fact-checking expert. Classify the claim into: SUPPORTED, REFUTED, NEI.\n\n"
        "Output format:\nClaim Veracity: [label]"
}

def build_prompt(claim: str, evidences: List[Dict[str, str]], language: str, dataset: str) -> str:
    if dataset == "xfact":
        instruction = (
            f"Classify the given {language} claim into seven categories: "
            "TRUE, MOSTLY-TRUE, PARTLY-TRUE/MISLEADING, FALSE, MOSTLY-FALSE, "
            "COMPLICATED/HARD-TO-CATEGORISE, OTHER.\nProvide exactly one label."
        )
    else:
        instruction = (
            f"Classify the given {language} claim into: SUPPORTED, REFUTED, NEI.\nProvide exactly one label."
        )

    evidence_blocks = "\n\n".join(
        [f"Evidence_{1+i}: {ev['evidence']}" for i, ev in enumerate(evidences)]
    )

    return (
        f"##Instruction: {instruction}\n\n"
        f"##input: Claim: {claim}\n\n"
        f"Evidences:\n{evidence_blocks}\n\n"
        "##output: "
    )

def build_chat_prompt(tokenizer, claim, evidences, language, dataset):
    user_prompt = build_prompt(claim, evidences, language, dataset)
    system_prompt = SYSTEM_PROMPT[dataset]

    if not hasattr(tokenizer, "apply_chat_template"):
        return system_prompt + "\n\n" + user_prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except:
        messages = [{"role": "user", "content": system_prompt + "\n\n" + user_prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

###############################################################################
# Response cleaning
###############################################################################

STRUCTURED_RE = re.compile(
    r"(claim veracity|verdict|label|prediction)\s*:\s*([^\n]+)", re.I
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
    m = STRUCTURED_RE.search(resp_std)
    if m:
        return normalise_label(m.group(2))
    m2 = LABEL_PATTERN_RE.search(resp_std)
    if m2:
        return normalise_label(m2.group(0))
    return normalise_label(resp_std)

###############################################################################
# Data loading
###############################################################################

import os
import json
import csv

def load_data(file_path, lang="all", dataset="xfact"):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]

    elif ext == ".jsonl":
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    elif ext == ".csv":
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, on_bad_lines='skip')
            for r in reader:
                data.append(r)

    elif ext == ".tsv":
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", on_bad_lines='skip')
            for r in reader:
                data.append(r)

    else:
        raise ValueError(f"Unsupported format: {ext}")

    # ---- Filter claims only if lang != all ----
    if lang.lower() != "all":
        filtered = []
        for ex in data:
            ex_lang = ex.get("language")
            if ex_lang and ex_lang.lower() == lang.lower():
                filtered.append(ex)
        data = filtered

    return data



###############################################################################
# Evaluation logic (UPDATED)
###############################################################################

def evaluate(
    lang: str,
    base_model_path: str,
    adapter_path: str,
    test_data_path: str,
    output_dir: str,
    max_examples: Optional[int] = None,
    max_length: int = 10000,
):

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
    data = load_data(test_data_path, lang)
    print(f"Loaded {len(data)} examples")

    if max_examples is not None:
        data = data[:max_examples]

    print(f"→ Evaluating on {len(data)} examples\n")

    model_shortname = get_model_shortname(base_model_path)
    dataset, testset = extract_dataset_and_testset(test_data_path)
    # if testset == "test":
    #     testset = "Indomain"

    technique, random_seed = extract_chunking_or_retrieval(adapter_path)

    #######################################################################
    # MODE 1 — Normal evaluation when lang == all
    #######################################################################
    if lang.lower() == "all":
        print("[Mode] Normal evaluation (lang == all). No cross-evidence looping.")

        predictions, gold, results_dump = [], [], []
        for ex in tqdm(data):

            try:
                evidences = []
                claim = ex["claim"]
                gold_label = normalise_label(ex["label"])
                lang_code = ex["language"]
                if technique == "semantic_chunking":
                    evidences = ex["evidences"]
                elif technique == "sentence_chunking":
                    evidences = ex["evidences"]
                elif technique == "search_snippet":
                    for i in range(1,6):
                        evidences.append({"evidence": ex[f"evidence_{i}"]})
                elif technique == "llm":
                    evidences = ex["evidences"]
                elif technique == "concrete":
                    evidences = ex["evidences"]

                language = LANGUAGE_MAP.get(lang_code, lang_code)
                prompt = build_chat_prompt(tokenizer, claim, evidences, language, dataset)

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
                    out[0][enc.input_ids.shape[1]:],
                    skip_special_tokens=True
                )

                pred_label = extract_label(decoded)

                predictions.append(pred_label)
                gold.append(gold_label)
                results_dump.append({
                    "input": prompt,
                    "true_label": gold_label,
                    "predicted_label": pred_label,
                    "model_response": decoded,
                    "language": language,
                })

            except Exception as exc:
                print(f"[!] Error → {exc}")
                predictions.append("other")
                gold.append("other")

        # Normal directory (no language folder)
        out_dir_full = Path(output_dir) / f"{dataset}/{testset}/{model_shortname}/{technique}"
        out_dir_full.mkdir(parents=True, exist_ok=True)

        # Save predictions
        with open(out_dir_full / f"predictions_{random_seed}.json", "w", encoding="utf-8") as f:
            json.dump(results_dump, f, indent=2, ensure_ascii=False)

        # Save metrics
        from collections import Counter
        gold_counts = Counter(gold)
        filtered_labels = [l for l in LABEL_ORDER if gold_counts[l] > 0]

        report = classification_report(
            gold, predictions,
            labels=filtered_labels,
            zero_division=0,
            digits=4
        )

        with open(out_dir_full / f"metrics_{random_seed}.txt", "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\nSaved → {out_dir_full}")
        return  # END MODE 1

    #######################################################################
    # MODE 2 — Cross-lingual evidence evaluation when lang != all
    #######################################################################
    print("[Mode] Cross-lingual evaluation (lang != all).")
    claim_lang = lang.lower()

    # Output directory includes claim language folder
    base_out_dir = Path(output_dir) / f"{dataset}/{testset}/{model_shortname}/{technique}/{claim_lang}"
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # Sequential evaluation over 13 evidence languages
    for evidence_lang in FIXED_EVIDENCE_LANGS:

        print(f"\n=== Evaluating CLAIM={claim_lang} | EVIDENCE={evidence_lang} ===")
        if evidence_lang == claim_lang:
            evidence_lang = "original"
        gold, predictions, results_dump = [], [], []

        for ex in tqdm(data, desc=f"Eval {claim_lang}-{evidence_lang}"):

            try:
                claim = ex["claim"]
                gold_label = normalise_label(ex["label"])

                # Extract translated evidences for this evidence language
                translated_list = []
                for ev in ex["evidences"]:
                    translated_text = ev["translated_evidences"][evidence_lang]
                    translated_list.append({"evidence": translated_text})

                # Build correct prompt
                claim_language_name = LANGUAGE_MAP.get(claim_lang, claim_lang)
                evidence_language_name = LANGUAGE_MAP.get(evidence_lang, evidence_lang)

                prompt = build_chat_prompt(
                    tokenizer,
                    claim,
                    translated_list,
                    claim_language_name,
                    dataset
                )

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
                    out[0][enc.input_ids.shape[1]:],
                    skip_special_tokens=True
                )

                pred_label = extract_label(decoded)

                predictions.append(pred_label)
                gold.append(gold_label)

                results_dump.append({
                    "claim_language": claim_language_name,
                    "evidence_language": evidence_language_name,
                    "prompt": prompt,
                    "true_label": gold_label,
                    "predicted_label": pred_label,
                    "model_response": decoded,
                })

            except Exception as exc:
                print(f"[!] Error → {exc}")
                predictions.append("other")
                gold.append("other")

        # ---- SAVE RESULTS FOR THIS LANGUAGE PAIR ----

        out_dir_full = base_out_dir
        out_dir_full.mkdir(parents=True, exist_ok=True)

        preds_file = out_dir_full / f"predictions_{claim_lang}_{evidence_lang}.jsonl"
        metrics_file = out_dir_full / f"metrics_{claim_lang}_{evidence_lang}.txt"

        # Save predictions JSONL
        with open(preds_file, "w", encoding="utf-8") as fp:
            for row in results_dump:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Save classification report
        from collections import Counter
        gold_counts = Counter(gold)
        filtered_labels = [l for l in LABEL_ORDER if gold_counts[l] > 0]

        report = classification_report(
            gold, predictions,
            labels=filtered_labels,
            zero_division=0,
            digits=4,
        )

        with open(metrics_file, "w", encoding="utf-8") as fm:
            fm.write(report)

        print(f"Saved results → {preds_file}")
        print(f"Saved metrics → {metrics_file}")
###############################################################################
# CLI entry-point
###############################################################################

if __name__ == "__main__":
    evaluate(
        lang=args.language,
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        test_data_path=args.data,
        output_dir=args.out_dir,
        max_examples=args.max_examples,
        max_length=args.max_length,
    )

