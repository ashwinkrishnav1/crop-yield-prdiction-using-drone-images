from pathlib import Path
import pandas as pd

ROOT = Path(
    r"C:\Users\ashwi\Music\cropyeildererrrrrrr\drone_dataset"
)

REPORTS = ROOT / "verification_reports"

unmatched = pd.read_csv(
    REPORTS / "plot_keys_without_csv_match.csv"
)

gt = pd.read_csv(
    ROOT / "GroundTruth" / "HYBRID_HIPS_V3.5_ALLPLOTS.csv",
    low_memory=False
)

def norm(series):
    return series.astype("string").str.strip().str.lower()

# Normalize the fields on both sides.
for col in ["location", "range", "row", "experiment"]:
    gt[col + "_key"] = norm(gt[col])

for col in ["location_raw", "range_raw", "row_raw", "experiment_raw"]:
    unmatched[col + "_key"] = norm(unmatched[col])

# Candidate search: same location + range + row.
# Experiment is deliberately ignored for diagnosis only.
group_cols = ["location_key", "range_key", "row_key"]

gt_groups = {
    key: group
    for key, group in gt.groupby(group_cols, dropna=False)
}

results = []

for _, img in unmatched.iterrows():
    key = tuple(img[c] for c in [
        "location_raw_key", "range_raw_key", "row_raw_key"
    ])

    candidate = gt_groups.get(key)

    results.append({
        "location": img["location_raw"],
        "experiment": img["experiment_raw"],
        "range": img["range_raw"],
        "row": img["row_raw"],
        "candidate_csv_rows_ignoring_experiment":
            0 if candidate is None else len(candidate),
        "candidate_rows_with_numeric_yield":
            0 if candidate is None else
            int(pd.to_numeric(
                candidate["yieldPerAcre"], errors="coerce"
            ).notna().sum())
    })

out = pd.DataFrame(results)

out.to_csv(
    REPORTS / "alternate_match_candidates.csv",
    index=False
)

print("\nPossible alternate matches by location:")
print(
    out.groupby("location").agg(
        unmatched_plots=("range", "size"),
        plots_with_candidates=(
            "candidate_csv_rows_ignoring_experiment",
            lambda s: int((s > 0).sum())
        ),
        plots_with_numeric_yield_candidates=(
            "candidate_rows_with_numeric_yield",
            lambda s: int((s > 0).sum())
        )
    )
)

print("\nSaved report:", REPORTS / "alternate_match_candidates.csv")
print("Diagnostic only: no matches were automatically accepted.")

