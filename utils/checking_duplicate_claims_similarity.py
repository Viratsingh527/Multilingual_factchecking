# import json
# import argparse
# from collections import defaultdict

# # -------------------------------
# # Language Map
# # -------------------------------
# LANG_MAP = {
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
# }

# # -------------------------------
# # Load JSONL
# # -------------------------------
# def load_jsonl(path):
#     data = []
#     with open(path, "r", encoding="utf-8", errors="replace") as f:
#         for i, line in enumerate(f, start=1):
#             line = line.strip()
#             if not line:  # ✅ skip empty lines
#                 continue
#             try:
#                 obj = json.loads(line)
#                 data.append(obj)
#             except json.JSONDecodeError as e:
#                 print(f"[Warning] Skipping invalid JSON at line {i}: {e}")
#     return data



# # -------------------------------
# # Check translations for identical duplicates
# # -------------------------------
# def check_translated_languages(input_path, output_path):
#     print("[INFO] Loading dataset...")
#     data = load_jsonl(input_path)

#     # Group by claim text
#     claims_map = defaultdict(list)
#     for item in data:
#         claims_map[item["claim"]].append(item)

#     results = []

#     # Iterate through claims with duplicates
#     for claim, items in claims_map.items():
#         if len(items) < 2:
#             continue  # only care about duplicates

#         # ✅ Check if evidences are identical
#         evidence_sets = {tuple(ev["evidence"] for ev in item["evidences"]) for item in items}
#         if len(evidence_sets) != 1:
#             continue  # not identical duplicates

#         # ✅ All identical duplicates – now check translations
#         for idx, item in enumerate(items):
#             language = item["language"]

#             for ev_idx, ev in enumerate(item["evidences"]):
#                 translated = ev.get("translated_evidences", {})
#                 missing_langs = [lang for lang in LANG_MAP.keys() if lang not in translated and lang != language]
#                 # missing_langs.remove(language)
#                 if missing_langs:
#                     results.append({
#                         "claim": claim,
#                         "duplicate_index": idx,
#                         "evidence_index": ev_idx,
#                         "evidence_text": ev.get("evidence", ""),
#                         "missing_languages": missing_langs
#                     })

#     # Save report
#     with open(output_path, "w", encoding="utf-8") as f:
#         for r in results:
#             f.write(json.dumps(r, ensure_ascii=False) + "\n")

#     print(f"\n✅ Check complete!")
#     print(f"Total duplicate claims checked: {len(claims_map)}")
#     print(f"Claims with missing translations: {len(results)}")
#     print(f"Report saved to: {output_path}")


# # -------------------------------
# # CLI
# # -------------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Check translated evidences for identical duplicate claims.")
#     parser.add_argument("--input", "-i", type=str, required=True, help="Path to translated dataset JSONL.")
#     parser.add_argument("--output", "-o", type=str, required=True, help="Path to save missing translation report.")

#     args = parser.parse_args()
#     check_translated_languages(args.input, args.output)
# import json
# import argparse
# from tqdm import tqdm

# def load_jsonl(path):
#     """Load a JSONL file into a list of dicts."""
#     data = []
#     with open(path, "r", encoding="utf-8", errors="replace") as f:
#         for i, line in enumerate(f, start=1):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 data.append(json.loads(line))
#             except json.JSONDecodeError as e:
#                 print(f"[Warning] Skipping invalid JSON at line {i}: {e}")
#     return data

# def remove_duplicates(input_path, output_path):
#     """Remove duplicate claims and save cleaned dataset."""
#     print("[INFO] Loading translated dataset...")
#     data = load_jsonl(input_path)
#     print(f"[INFO] Loaded {len(data)} total datapoints.")

#     seen_claims = set()
#     cleaned_data = []
#     duplicate_count = 0

#     for item in tqdm(data, desc="Processing datapoints"):
#         claim_text = item.get("claim", "").strip()
#         if claim_text in seen_claims:
#             duplicate_count += 1
#             continue  # skip duplicates
#         seen_claims.add(claim_text)
#         cleaned_data.append(item)

#     # Save cleaned dataset
#     with open(output_path, "w", encoding="utf-8") as f:
#         for item in cleaned_data:
#             f.write(json.dumps(item, ensure_ascii=False) + "\n")

#     print("\n✅ Done!")
#     print(f"📊 Original datapoints: {len(data)}")
#     print(f"🗑️ Removed duplicates: {duplicate_count}")
#     print(f"📁 Cleaned dataset saved to: {output_path}")
#     print(f"✅ Final datapoints: {len(cleaned_data)}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Remove duplicate claims from a translated JSONL dataset.")
#     parser.add_argument("--input", "-i", type=str, required=True, help="Path to the translated JSONL dataset.")
#     parser.add_argument("--output", "-o", type=str, required=True, help="Path to save the cleaned JSONL file.")
#     args = parser.parse_args()

#     remove_duplicates(args.input, args.output)

import json
import argparse
from tqdm import tqdm

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[Warning] Skipping invalid JSON: {e}")
    return data

def extract_missing_datapoints(raw_path, missing_path, partial_path, output_path):
    print("[INFO] Loading raw dataset...")
    raw_data = load_jsonl(raw_path)

    print("[INFO] Loading missing claims...")
    missing_claims = load_jsonl(missing_path)
    print("[INFO] Loading partial missing claims...")
    partial_missing_claims = load_jsonl(partial_path)

    # Collect all missing claims text
    missing_texts = set()
    for item in missing_claims + partial_missing_claims:
        c = item.get("claim", "").strip()
        if c:
            missing_texts.add(c)

    print(f"[INFO] Total missing+partial claims collected: {len(missing_texts)}")

    # Filter raw data
    filtered_data = [d for d in tqdm(raw_data, desc="Filtering datapoints") if d.get("claim", "").strip() in missing_texts]

    # Save filtered datapoints
    with open(output_path, "w", encoding="utf-8") as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(filtered_data)} datapoints to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract only missing/partial-missing datapoints from raw dataset.")
    parser.add_argument("--raw", "-r", type=str, required=True, help="Path to raw JSONL dataset.")
    parser.add_argument("--missing", "-m", type=str, required=True, help="Path to missing claims JSONL file.")
    parser.add_argument("--partial", "-p", type=str, required=True, help="Path to partial missing claims JSONL file.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to save filtered JSONL dataset.")

    args = parser.parse_args()
    extract_missing_datapoints(args.raw, args.missing, args.partial, args.output)

