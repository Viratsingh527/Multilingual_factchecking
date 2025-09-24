import json
import argparse
from tqdm import tqdm
from multilingual_factchecking.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

import json
import csv
import os

def load_dataset(path):
    """Load a dataset from jsonl, json, csv, or tsv into a list of dictionaries."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):  # sometimes json can be dict with 'data' key
                return data.get("data", [])
            return data

    elif ext in [".csv", ".tsv"]:
        delimiter = "," if ext == ".csv" else "\t"
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            return [row for row in reader]

    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# Language map for xfact
language_map = {
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

def get_instruction_and_label(item, dataset):
    """Return instruction text and standardized label depending on dataset."""
    label = item["label"].strip()

    if dataset == "xfact":
        language = language_map.get(item.get("language"), item.get("language"))
        instruction = (
            f"Classify the given {language} claim into one of the seven categories "
            "(TRUE, MOSTLY-TRUE, PARTLY-TRUE/MISLEADING, FALSE, MOSTLY-FALSE, "
            "COMPLICATED/HARD-TO-CATEGORISE, OTHER) based on the provided evidence.\n\n"
            "1. TRUE: Fully supported by the evidence.\n"
            "2. MOSTLY-TRUE: Mostly supported but with minor inaccuracies.\n"
            "3. PARTLY-TRUE/MISLEADING: Partially supported, but includes significant omissions.\n"
            "4. FALSE: Clearly contradicted or unsupported by the evidence.\n"
            "5. MOSTLY-FALSE: Largely incorrect with only a small element of truth.\n"
            "6. COMPLICATED/HARD-TO-CATEGORISE: Too complex to assign a straightforward label.\n"
            "7. OTHER: Does not fit into any of the above categories.\n\n"
            "Provide exactly one label."
        )
        output_label = label.lower()

    elif dataset == "ru22fact":
        language = item.get("language", "Unknown")
        instruction = (
            f"Classify the given {language} claim into one of three categories "
            "(SUPPORTED, REFUTED, NEI) based on the provided evidence.\n\n"
            "1. SUPPORTED: The evidence supports the claim.\n"
            "2. REFUTED: The evidence contradicts the claim.\n"
            "3. NEI: Not enough information to verify.\n\n"
            "Provide exactly one label."
        )
        output_label = label.lower()  # supported, refuted, nei

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return instruction, output_label


def convert_to_alpaca_format(input_path, output_path, dataset, data_type=None):
    """Convert fact-checking dataset (xfact or ru22fact) in various formats to Alpaca format."""
    data = load_dataset(input_path)
    alpaca_data = []

    for item in tqdm(data, desc=f"Converting {dataset} data..."):

        # --- Normalize claim & evidence depending on dataset/type ---
        if dataset == "ru22fact" and data_type == "llm":
            # CSV structure: id,claim,evidence,referenced_explanation,label,url,date,claimant,language
            claim = item["claim"]
            evidences = [{"evidence": item.get("evidence","")}]
            label = item["label"]
            language = item.get("language", "Unknown")

        elif dataset == "xfact" and data_type == "search_snippet":
            # TSV structure: language site evidence_1 ... evidence_5 link_1 ... link_5 claimDate reviewDate claimant claim label
            claim = item["claim"]
            evidences = []
            for i in range(1, 6):  # up to 5 evidences
                ev_text = item[f"evidence_{i}"].strip()
                if ev_text:
                    evidences.append({"evidence": ev_text})
            label = item["label"]
            language = item.get("language", "Unknown")

        else:
            # Default JSON/JSONL style (xfact, ru22fact semantic/sentence/concrete)
            claim = item["claim"]
            evidences = item.get("evidences", [])
            label = item["label"]
            language = item.get("language", "Unknown")

        # --- Process evidence texts uniformly ---
        ev_texts = []
        for i, evidence in enumerate(evidences):
            ev_text = evidence.get("evidence", "").strip()
            if ev_text:
                ev_texts.append(f"Evidence_{i+1}: {ev_text}")

        combined_evidence = "\n\n".join(ev_texts)

        # Build instruction + label using existing helper
        instruction, output_label = get_instruction_and_label(
            {"claim": claim, "label": label, "language": language},
            dataset
        )

        alpaca_data.append(
            {
                "instruction": instruction,
                "input": f"Claim: {claim}\n\nEvidences:\n{combined_evidence}",
                "output": output_label,
            }
        )

    # --- Save as JSON ---
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="xfact")
    parser.add_argument("--input", type=str, default="train",
                        help="Name of input file (without extension).")
    parser.add_argument("--type", type=str,
                        choices=["semantic", "sentence", "search_snippet", "concrete","llm"],
                        default="semantic",
                        help="Type of chunking or retrieval (only used for xfact).")

    args = parser.parse_args()

    if args.dataset == "xfact":
        if args.type in ["semantic", "sentence"]:
            input_file = PROCESSED_DATA_DIR/f"{args.dataset}"/f"xfact_{args.input}_with_{args.type}_level_chunked_retrieved_evidence.jsonl"
            output_file = f"LLaMA-Factory/data/xfact_{args.input}_data_with_{args.type}_level_chunked_retrieved_evidence.json"
        elif args.type == "search_snippet":
            input_file = RAW_DATA_DIR / f"{args.input}.tsv"
            output_file = f"LLaMA-Factory/data/xfact_{args.input}_data_with_{args.type}.json"
        elif args.type == "concrete":
            input_file = RAW_DATA_DIR / f"{args.input}.jsonl"
            output_file = f"LLaMA-Factory/data/xfact_{args.input}_data_with_{args.type}.json"
        else:
            raise ValueError(f"Invalid type: {args.type}")

    elif args.dataset == "ru22fact":
        if args.type in ["semantic", "sentence"]:
            input_file = PROCESSED_DATA_DIR /f"{args.dataset}"/ f"ru22fact_{args.input}_with_{args.type}_level_chunked_retrieved_evidence.jsonl"
            output_file = f"LLaMA-Factory/data/ru22fact_{args.input}_data_with_{args.type}_level_chunked_retrieved_evidence.json"
        elif args.type == "llm":
            input_file = RAW_DATA_DIR /f"{args.dataset}"/f"{args.input}.csv"
            output_file = f"LLaMA-Factory/data/ru22fact_{args.input}_data_with_{args.type}_retrieved_evidence.json"            
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    convert_to_alpaca_format(input_file, output_file, args.dataset,args.type)
    print(f"Done. Wrote {output_file}")
