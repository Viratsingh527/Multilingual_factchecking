import json
import argparse
from tqdm import tqdm
from multilingual_factchecking.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, EXTERNAL_DATA_DIR,INTERIM_DATA_DIR,DATA_DIR


def load_jsonl(path):
    """Load a jsonl file into a list of dictionaries."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def convert_to_alpaca_format(input_path, output_path):
    """Convert the fact-checking dataset to Alpaca format compatible with LLaMA-Factory."""
    data = load_jsonl(input_path)

    alpaca_data = []

    for item in tqdm(data, desc="Converting data to LLaMA-Factory format..."):
        claim = item["claim"]
        label = item["label"]
        language = language_map.get(
            item.get("language")
        )  # Default to Turkish if not specified
        evidences = item.get("evidences", [])

        # Process evidence texts
        ev_texts = []
        for i, evidence in enumerate(evidences):
            # source = evidence.get("source", f"Source {i+1}")
            ev_text = evidence.get("evidence", "").strip()
            if ev_text:  # Only add if evidence text exists
                ev_texts.append(f"Evidence_{1+i}: {ev_text}")

        combined_evidence = "\n\n".join(ev_texts)

        # Create instruction with language consideration
        instruction = f"Classify the given {language} claim into one of the seven categories (TRUE, MOSTLY-TRUE, PARTLY-TRUE/MISLEADING, FALSE, MOSTLY-FALSE, COMPLICATED/HARD-TO-CATEGORISE, OTHER) based on the provided evidence.\n\n1. TRUE: Fully supported by the evidence.\n2. MOSTLY-TRUE: Mostly supported but with minor inaccuracies.\n3. PARTLY-TRUE/MISLEADING: Partially supported, but includes significant omissions or could mislead.\n4. FALSE: Clearly contradicted or unsupported by the evidence.\n5. MOSTLY-FALSE: Largely incorrect with only a small element of truth.\n6. COMPLICATED/HARD-TO-CATEGORISE: Too complex or nuanced to assign a straightforward label.\n7. OTHER: Does not fit into any of the above categories.\n\nProvide exactly one label."

        alpaca_data.append(
            {
                "instruction": instruction,
                "input": f"Claim: {claim}\n\nEvidences:\n{combined_evidence}",
                "output": label.lower().strip(),  # Standardize to lowercase
            }
        )

    # Save as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="train" ,help=
                    "Name of input JSONL file that needs to be converted to Alpaca format.")
parser.add_argument("--type", type=str,choices=["semantic", "sentence","search_snippet","concrete"], default="semantic", help="Type of chunking or retrieval.")

args = parser.parse_args()
# Usage example
if args.type == "semantic" or args.type == "sentence":
    input_file = PROCESSED_DATA_DIR/f"xfact_{args.input}_with_{args.type}_level_chunked_retrieved_evidence.jsonl"
    output_file = f"LLaMA-Factory/data/xfact_{args.input}_with_{args.type}_level_chunked_retrieved_evidence.json"    
elif args.type == "search_snippet":
    input_file = RAW_DATA_DIR/f"{args.input}.tsv"
    output_file = f"LLaMA-Factory/data/xfact_{args.input}_with_{args.type}.json"
elif args.type == "concrete":
    input_file = RAW_DATA_DIR/f"{args.input}.jsonl"
    output_file = f"LLaMA-Factory/data/xfact_{args.input}_with_{args.type}.json"
else:
    raise ValueError(f"Invalid type: {args.type}")

convert_to_alpaca_format(
    input_file,
    output_file,
)
print(f"Done. Wrote {output_file}")