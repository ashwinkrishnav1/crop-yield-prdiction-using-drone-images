from pathlib import Path
import pandas as pd

ROOT = Path(
    r"C:\Users\ashwi\Music\cropyeildererrrrrrr\drone_dataset"
)

REPORTS = ROOT / "verification_reports"

unmatched_path = REPORTS / "plot_keys_without_csv_match.csv"
csv_path = ROOT / "GroundTruth" / "HYBRID_HIPS_V3.5_ALLPLOTS.csv"

unmatched = pd.read_csv(unmatched_path)
gt = pd.read_csv(csv_path, low_memory=False)

gt.columns = gt.columns.str.strip()

print("\nUNMATCHED UAV PLOT EXAMPLES")
print("=" * 65)

cols = [
    "location_raw",
    "experiment_raw",
    "range_raw",
    "row_raw",
    "candidates_same_exp_range_row_ignoring_location",
]

cols = [c for c in cols if c in unmatched.columns]
print(unmatched[cols].head(30).to_string(index=False))

print("\nUNMATCHED PLOTS BY LOCATION")
print("=" * 65)

print(
    unmatched.groupby("location_raw")
    .size()
    .sort_values(ascending=False)
    .to_string()
)

print("\nGROUND-TRUTH COLUMN NAMES")
print("=" * 65)
print(gt.columns.tolist())

print("\nGROUND-TRUTH EXAMPLES FOR AFFECTED LOCATIONS")
print("=" * 65)

if "location" in gt.columns:
    for location in unmatched["location_raw"].dropna().unique():
        subset = gt[
            gt["location"].astype(str).str.strip().str.lower()
            == str(location).strip().lower()
        ]

        print(f"\n--- {location}: {len(subset)} CSV rows ---")

        view_cols = [
            c for c in [
                "location", "experiment", "range", "row",
                "plotNumber", "qrCode", "yieldPerAcre"
            ]
            if c in subset.columns
        ]

        print(subset[view_cols].head(10).to_string(index=False))

print("\nNo original dataset files were changed.")

