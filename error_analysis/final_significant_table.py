# collect_significance_results.py

import os
import pandas as pd

ROOT = "."   # current error_analysis directory
OUTPUT = "all_significance_results.csv"

rows = []

for dirpath, dirnames, filenames in os.walk(ROOT):
    for file in filenames:
        if file.endswith("_mcnemar.csv"):
            path = os.path.join(dirpath, file)

            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue

            if len(df) == 0:
                continue

            row = df.iloc[0].to_dict()

            parts = dirpath.split(os.sep)

            # Example:
            # ./ours_vs_semantic/xfact/Indomain/gemma
            comparison = parts[1] if len(parts) > 1 else "unknown"
            dataset = parts[2] if len(parts) > 2 else "unknown"
            split = parts[3] if len(parts) > 3 else "unknown"
            model = parts[4] if len(parts) > 4 else "unknown"

            b_gt_o = int(row["baseline_correct_ours_wrong"])
            o_gt_b = int(row["ours_correct_baseline_wrong"])
            net_gain = o_gt_b - b_gt_o

            p_value = float(row["p_value"])

            rows.append({
                "comparison": comparison,
                "dataset": dataset,
                "split": split,
                "model": model,
                "B>O": b_gt_o,
                "O>B": o_gt_b,
                "net_gain": net_gain,
                "chi_square": row["chi_square"],
                "p_value": p_value,
                "significant_p_0.05": "Yes" if p_value < 0.05 else "No"
            })

result = pd.DataFrame(rows)

# Sort neatly
result = result.sort_values(
    by=["comparison", "dataset", "split", "model"]
)

result.to_csv(OUTPUT, index=False)

print(result)
print(f"\nSaved to: {OUTPUT}")