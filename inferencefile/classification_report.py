import json
from sklearn.metrics import classification_report

# Load data from prediction.json
with open('../outputs/different_model_result/mistral_predictions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# If the file contains a list of records
if isinstance(data, dict):  # Single object
    data = [data]

# Extract true and predicted labels
true_labels = [item['true_label'] for item in data]
predicted_labels = [item['predicted_label'] for item in data]

# Define your label set
label_set = [
    "true",
    "mostly true",
    "partly true/misleading",
    "false",
    "mostly false",
    "complicated/hard-to-categorise",
    "other"
]

# Generate classification report
report = classification_report(
    true_labels,
    predicted_labels,
    labels=label_set,
    zero_division=0,
    digits=4  # Optional: controls decimal places in output
)

print(report)

import os
# base_filename = os.path.splitext(os.path.basename(f.name))[0]
output_path = f"../outputs/different_model_result/{base_filename}_classification_report.txt"

with open(output_path, "w", encoding="utf-8") as out_file:
    out_file.write(report)

print(f"Classification report saved to {output_path}")