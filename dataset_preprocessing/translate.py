import json
import os
import time
import argparse
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

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
# Load model and tokenizer
# -------------------------------
print("🔹 Loading mBART-50 model and tokenizer...")
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
print(f"✅ Model loaded on {device.upper()}.\n")

# -------------------------------
# Helper: language code mapping
# -------------------------------
def get_lang_code(lang_code):
    lang_map = {
        "ar": "ar_AR", "de": "de_DE", "en": "en_XX", "es": "es_XX",
        "fr": "fr_XX", "hi": "hi_IN", "it": "it_IT", "ja": "ja_XX",
        "ko": "ko_KR", "pl": "pl_PL", "pt": "pt_XX", "ru": "ru_RU",
        "tr": "tr_TR", "id": "id_ID", "fa": "fa_IR", "ro": "ro_RO",
        "ta": "ta_IN", "bn": "bn_IN", "no": "sv_SE", "nl": "nl_XX",
        "ka": "ka_GE", "sr": "hr_HR"
    }
    return lang_map.get(lang_code, None)

# -------------------------------
# Helper: chunk long texts
# -------------------------------
def chunk_text(text, max_words=900):
    """
    Split long evidence text into smaller chunks (~900 words)
    so that each fits within the model's 1024-token limit.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks

# -------------------------------
# Translation function (offline)
# -------------------------------
def translate_text(text, src_lang, tgt_lang, max_retries=3):
    """
    Translate text using local mBART model inference.
    Automatically chunks long texts and merges results.
    Retries on GPU/CPU runtime errors with exponential backoff.
    """
    src_lang_code = get_lang_code(src_lang)
    tgt_lang_code = get_lang_code(tgt_lang)
    if not src_lang_code or not tgt_lang_code:
        print(f"[Warning] Skipping unsupported language pair {src_lang}->{tgt_lang}")
        return None

    chunks = chunk_text(text, max_words=900)
    translated_chunks = []

    for idx, chunk in enumerate(chunks):
        for attempt in range(max_retries):
            try:
                tokenizer.src_lang = src_lang_code
                encoded = tokenizer(chunk, return_tensors="pt",
                                    truncation=True, max_length=1024).to(device)
                generated_tokens = model.generate(
                    **encoded,
                    forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code],
                    num_beams=4,
                    max_length=1024
                )
                translated = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )[0]
                translated_chunks.append(translated.strip())
                break  # success → exit retry loop
            except Exception as e:
                print(f"[Warning] Chunk {idx+1}/{len(chunks)} failed "
                    f"({attempt+1}/{max_retries}) {src_lang}->{tgt_lang}: {e}")
                if attempt < max_retries - 1:
                    print("Retrying immediately...")
                    torch.cuda.empty_cache()  # free GPU memory just in case
                    continue
                else:
                    print(f"[Error] Giving up chunk {idx+1} after {max_retries} retries.")
                    return None


    return " ".join(translated_chunks).strip()

# -------------------------------
# Main Dataset Processing
# -------------------------------
def process_dataset(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)
    missing_path = os.path.join(output_dir, "missing_translations.jsonl")

    print(f"🔹 Processing dataset: {filename}")
    print(f"🔹 Output file: {output_path}")
    print(f"🔹 Missing translations file: {missing_path}\n")

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile, \
         open(missing_path, "w", encoding="utf-8") as missingfile:

        for line in tqdm(infile, desc=f"Translating {filename}"):
            if not line.strip():
                continue

            data = json.loads(line)
            claim_lang = data.get("language")
            if not claim_lang:
                print("[Warning] Missing language field; skipping datapoint.")
                continue

            processed_evidences = []
            missing_entry = {
                "claim": data.get("claim", ""),
                "language": claim_lang,
                "label": data.get("label", ""),
                "missing_evidences": []
            }

            for ev in data.get("evidences", []):
                original_text = ev.get("evidence", "")
                translated_dict = {"original": original_text}
                failed_langs = []

                for lang_code in LANG_MAP.keys():
                    if lang_code == claim_lang:
                        continue

                    translated = translate_text(original_text,
                                                src_lang=claim_lang,
                                                tgt_lang=lang_code)
                    if translated:
                        translated_dict[lang_code] = translated
                    else:
                        failed_langs.append(lang_code)

                ev["translated_evidences"] = translated_dict
                processed_evidences.append(ev)

                if failed_langs:
                    missing_entry["missing_evidences"].append({
                        "source_index": ev.get("source_index"),
                        "source_url": ev.get("source_url"),
                        "evidence": original_text,
                        "failed_languages": failed_langs
                    })

            data["evidences"] = processed_evidences
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

            if missing_entry["missing_evidences"]:
                missingfile.write(json.dumps(missing_entry, ensure_ascii=False) + "\n")

    print(f"\n✅ Translation complete.")
    print(f"   Output saved to: {output_path}")
    print(f"   Missing translations saved to: {missing_path}")

# -------------------------------
# Command Line Interface
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate multilingual fact-checking dataset using offline mBART-50 with automatic chunking."
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to the input JSONL dataset file.")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Directory to save the translated dataset and missing_translations.jsonl.")
    args = parser.parse_args()
    process_dataset(args.input, args.output)



# import json
# import os
# import time
# import argparse
# from tqdm import tqdm
# from deep_translator import GoogleTranslator

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
# # Translation Helper (with Retry)
# # -------------------------------
# def translate_text(text: str, src_lang: str, tgt_lang: str, max_retries: int = 5):
#     """
#     Translate text from src_lang to tgt_lang using GoogleTranslator.
#     Retries with exponential backoff on failure.
#     """
#     for attempt in range(max_retries):
#         try:
#             translated = GoogleTranslator(source=src_lang, target=tgt_lang).translate(text)
#             return translated
#         except Exception as e:
#             wait_time = 2 ** attempt
#             print(f"[Warning] Failed ({attempt+1}/{max_retries}) {src_lang}->{tgt_lang}: {e}")
#             if attempt < max_retries - 1:
#                 print(f"Retrying in {wait_time}s...")
#                 time.sleep(wait_time)
#             else:
#                 print(f"[Error] Giving up {src_lang}->{tgt_lang} after {max_retries} retries.")
#     return None


# # -------------------------------
# # Main Dataset Processing
# # -------------------------------
# def process_dataset(input_path: str, output_dir: str):
#     """
#     Translate all evidences in JSONL dataset and save to same filename under output_dir.
#     Also saves a 'missing_translations.jsonl' file for failed translations.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     filename = os.path.basename(input_path)
#     output_path = os.path.join(output_dir, filename)
#     missing_path = os.path.join(output_dir, "missing_translations.jsonl")

#     print(f"🔹 Processing dataset: {filename}")
#     print(f"🔹 Output file: {output_path}")
#     print(f"🔹 Missing translations file: {missing_path}\n")

#     with open(input_path, "r", encoding="utf-8") as infile, \
#          open(output_path, "w", encoding="utf-8") as outfile, \
#          open(missing_path, "w", encoding="utf-8") as missingfile:

#         for line in tqdm(infile, desc=f"Translating {filename}"):
#             if not line.strip():
#                 continue

#             data = json.loads(line)
#             claim_lang = data.get("language")
#             if not claim_lang:
#                 print("[Warning] Missing language field; skipping datapoint.")
#                 continue

#             processed_evidences = []
#             missing_entry = {
#                 "claim": data.get("claim", ""),
#                 "language": claim_lang,
#                 "label": data.get("label", ""),
#                 "missing_evidences": []
#             }

#             for ev in data.get("evidences", []):
#                 original_text = ev.get("evidence", "")
#                 translated_dict = {"original": original_text}
#                 failed_langs = []

#                 for lang_code in LANG_MAP.keys():
#                     if lang_code == claim_lang:
#                         continue

#                     translated = translate_text(original_text, src_lang=claim_lang, tgt_lang=lang_code)
#                     if translated:
#                         translated_dict[lang_code] = translated
#                     else:
#                         failed_langs.append(lang_code)

#                     # optional small delay to prevent rate limit
#                     time.sleep(0.5)

#                 ev["translated_evidences"] = translated_dict
#                 processed_evidences.append(ev)

#                 if failed_langs:
#                     missing_entry["missing_evidences"].append({
#                         "source_index": ev.get("source_index"),
#                         "source_url": ev.get("source_url"),
#                         "evidence": original_text,
#                         "failed_languages": failed_langs
#                     })

#             data["evidences"] = processed_evidences
#             outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

#             if missing_entry["missing_evidences"]:
#                 missingfile.write(json.dumps(missing_entry, ensure_ascii=False) + "\n")

#     print(f"\n✅ Translation complete.")
#     print(f"   Output saved to: {output_path}")
#     print(f"   Missing translations saved to: {missing_path}")


# # -------------------------------
# # Command Line Interface
# # -------------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Translate multilingual fact-checking dataset with retry and missing-translation logging."
#     )
#     parser.add_argument(
#         "--input", "-i", type=str, required=True,
#         help="Path to the input JSONL dataset file."
#     )
#     parser.add_argument(
#         "--output", "-o", type=str, required=True,
#         help="Directory to save the translated dataset and missing_translations.jsonl."
#     )

#     args = parser.parse_args()
#     process_dataset(args.input, args.output)
