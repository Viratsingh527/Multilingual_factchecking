import json
import argparse
from tqdm import tqdm
from collections import defaultdict

# -------------------------------
# Language Map
# -------------------------------
LANG_MAP = {
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
}

# -------------------------------
# Load JSONL file
# -------------------------------
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[Warning] Skipping invalid JSON at line {i}: {e}")
    return data

# -------------------------------
# Find Missing & Duplicate Translations
# -------------------------------
def analyze_missing(raw_path, translated_path, missing_claims_path, partial_missing_path, duplicate_claims_path):
    print("[INFO] Loading raw dataset...")
    raw_data = load_jsonl(raw_path)
    print("[INFO] Loading translated dataset...")
    translated_data = load_jsonl(translated_path)

    # ✅ Count how many times each claim appears
    claim_counts = defaultdict(int)
    for item in translated_data:
        claim_counts[item["claim"]] += 1

    # ✅ Build index (latest occurrence wins, but we track counts separately)
    translated_index = {item["claim"]: item for item in translated_data}
    print(f"[INFO] Total unique claims in translated dataset: {len(translated_index)}")

    # ✅ Detect duplicates
    duplicate_claims = []
    for claim_text, count in claim_counts.items():
        if count > 1:
            duplicate_claims.append({
                "claim": claim_text,
                "count": count
            })

    total_claims = len(raw_data)
    missing_claims = []
    partial_missing_claims = []

    # -------------------------------
    # Check for missing translations
    # -------------------------------
    for raw_item in tqdm(raw_data, desc="Analyzing claims"):
        claim_text = raw_item["claim"]
        claim_lang = raw_item.get("language")
        expected_langs = [lang for lang in LANG_MAP.keys() if lang != claim_lang]

        translated_item = translated_index.get(claim_text)

        # 🟥 Case 1: Entire claim missing
        if translated_item is None:
            missing_claims.append({
                "claim": claim_text,
                "language": claim_lang,
                "reason": "entirely_missing",
                "missing_languages": expected_langs
            })
            continue

        # 🟧 Case 2: Claim exists but has missing translations
        missing_langs_per_claim = set()
        for ev in translated_item.get("evidences", []):
            translated_dict = ev.get("translated_evidences", {})
            for lang in expected_langs:
                if lang not in translated_dict:
                    missing_langs_per_claim.add(lang)

        if missing_langs_per_claim:
            partial_missing_claims.append({
                "claim": claim_text,
                "language": claim_lang,
                "reason": "partial_missing",
                "missing_languages": sorted(list(missing_langs_per_claim))
            })

    # -------------------------------
    # Save results
    # -------------------------------
    with open(missing_claims_path, "w", encoding="utf-8") as f:
        for item in missing_claims:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(partial_missing_path, "w", encoding="utf-8") as f:
        for item in partial_missing_claims:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(duplicate_claims_path, "w", encoding="utf-8") as f:
        for item in duplicate_claims:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # -------------------------------
    # Summary
    # -------------------------------
    print("\n✅ Analysis complete!")
    print(f"Total raw claims: {total_claims}")
    print(f"🟥 Entirely missing claims: {len(missing_claims)} → {missing_claims_path}")
    print(f"🟧 Partial missing claims: {len(partial_missing_claims)} → {partial_missing_path}")
    print(f"🔁 Duplicate claims: {len(duplicate_claims)} → {duplicate_claims_path}")

# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare raw and translated datasets for missing and duplicate translations.")
    parser.add_argument("--raw", "-r", type=str, required=True, help="Path to raw JSONL dataset (untranslated).")
    parser.add_argument("--translated", "-t", type=str, required=True, help="Path to translated JSONL dataset.")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Directory to save missing/duplicate reports.")

    args = parser.parse_args()

    missing_claims_path = f"{args.output_dir}/missing_claims.jsonl"
    partial_missing_path = f"{args.output_dir}/partial_missing_claims.jsonl"
    duplicate_claims_path = f"{args.output_dir}/duplicate_claims.jsonl"

    analyze_missing(args.raw, args.translated, missing_claims_path, partial_missing_path, duplicate_claims_path)
