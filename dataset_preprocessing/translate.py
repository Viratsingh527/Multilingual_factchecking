import json
import os
import time
import argparse
from tqdm import tqdm
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# Translation Helper with Retry
# -------------------------------
def translate_text(text: str, src_lang: str, tgt_lang: str, retries: int = 3, delay: float = 5.0):
    """Translate text from src_lang to tgt_lang using GoogleTranslator with retry."""
    for attempt in range(1, retries + 1):
        try:
            translated = GoogleTranslator(source=src_lang, target=tgt_lang).translate(text)
            if translated:
                return translated
        except Exception as e:
            print(f"[Warning] Failed to translate from {src_lang} to {tgt_lang} (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(delay * attempt)
    return None


# -------------------------------
# Evidence Translation Worker
# -------------------------------
def translate_evidence_for_all_langs(evidence_text, claim_lang):
    """Translate one evidence into all target languages using threads."""
    translated_dict = {"original": evidence_text}

    with ThreadPoolExecutor(max_workers=8) as executor:  # up to 8 parallel translations
        futures = {}
        for lang_code in LANG_MAP.keys():
            if lang_code == claim_lang:
                continue
            futures[executor.submit(translate_text, evidence_text, claim_lang, lang_code)] = lang_code

        for future in as_completed(futures):
            lang_code = futures[future]
            translated = future.result()
            if translated:
                translated_dict[lang_code] = translated

    return translated_dict


# -------------------------------
# Main Dataset Processing
# -------------------------------
def process_dataset(input_path: str, output_dir: str):
    """Translate all evidences in JSONL dataset and save to same filename under output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)
    missing_path = os.path.join(output_dir, filename.replace(".jsonl", "_missing.jsonl"))

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile, \
         open(missing_path, "w", encoding="utf-8") as missing_file:

        for line in tqdm(infile, desc=f"Processing {filename}"):
            if not line.strip():
                continue

            data = json.loads(line)
            claim_lang = data.get("language")
            processed_evidences = []
            missing_any = False

            for ev in data.get("evidences", []):
                original_text = ev.get("evidence", "")
                translated_dict = translate_evidence_for_all_langs(original_text, claim_lang)

                # mark as missing if less than half languages succeeded
                if len(translated_dict) < len(LANG_MAP) / 2:
                    missing_any = True

                ev["translated_evidences"] = translated_dict
                processed_evidences.append(ev)

            data["evidences"] = processed_evidences
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

            if missing_any:
                missing_file.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"\n✅ Translation complete.\n  Output: {output_path}\n  Missing translations: {missing_path}")


# -------------------------------
# Command Line Interface
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multithreaded translation of evidences in multilingual fact-checking JSONL dataset."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to the input JSONL dataset file."
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Directory to save the translated dataset (filename will be same as input)."
    )

    args = parser.parse_args()
    process_dataset(args.input, args.output)

