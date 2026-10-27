from pathlib import Path
import pandas as pd

report_dir = Path(__file__).resolve().parent / "drone_dataset" / "audit_reports"

# 1. Inspect unmatched image counts
unmatched = pd.read_csv(report_dir / "unmatched_images.csv")

keys = [
    "location_key",
    "experiment_key",
    "range_key",
    "row_key",
]

print("Unmatched images:", len(unmatched))
print("Unique unmatched plot IDs:",
      unmatched[keys].drop_duplicates().shape[0])

# Support either column name
loc_col = "location_raw" if "location_raw" in unmatched.columns else "location_from_filename"

print("\nUnmatched images by location and time point:")
print(
    unmatched.groupby(
        [loc_col, "timepoint"]
    ).size().unstack(fill_value=0)
)

# Check whether each unmatched plot has all 3 time points
per_plot = unmatched.groupby(keys).agg(
    image_count=("filename", "size"),
    timepoint_count=("timepoint", "nunique"),
    timepoints=("timepoint",
                lambda s: ",".join(sorted(s.unique())))
)

print("\nUnmatched plot image counts:")
print(per_plot["image_count"].value_counts().sort_index())

print("\nUnmatched plots that do not have exactly 3 images:")
print(per_plot[per_plot["image_count"] != 3].head(30))

# 2. See which missing-label rows have UAV images
missing = pd.read_csv(
    report_dir / "csv_rows_missing_or_nonnumeric_yield.csv"
)

print("\nMissing-yield records:")
print(missing["has_matching_uav_plot_key"].value_counts(dropna=False))