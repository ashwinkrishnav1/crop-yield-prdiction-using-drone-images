"""Leakage-safe plot-level split for the multi-temporal maize UAV dataset.

Policy:
  * Every valid plot from Crawfordsville is held out for testing.
  * The remaining locations are split plot-by-plot into train/validation.
  * TP1, TP2, and TP3 remain together because each manifest row represents a plot.
  * Original files are not moved or modified.

Edit ROOT and MANIFEST_PATH if your project is stored elsewhere.
Requires: pandas, scikit-learn
Install if needed: python -m pip install pandas scikit-learn
"""
from pathlib import Path
import re
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# ---- Configuration: change only if your paths or preferences differ ----
ROOT = Path(r"C:\Users\ashwi\Music\cropyeildererrrrrrr")
MANIFEST_PATH = ROOT / "drone_dataset" / "valid_plots.csv"
OUTPUT_DIR = ROOT / "datasplitting_script"
TEST_LOCATION = "Crawfordsville"
VALIDATION_FRACTION = 0.20       # 20% of plots outside Crawfordsville
RANDOM_STATE = 42                # fixed seed makes this split reproducible
# ----------------------------------------------------------------------

LOCATION_ALIASES = {
    "ames": "Ames",
    "crawfordsville": "Crawfordsville",
    "lincoln": "Lincoln",
    "movalley": "MOValley",
    "missourivalley": "MOValley",
    "scottsbluff": "Scottsbluff",
}

LOCATION_COLUMNS = [
    "location", "location_raw", "location_from_filename", "location_key",
    "trial_location", "site", "Location",
]
PLOT_ID_COLUMNS = [
    "plot_id", "plot_key", "unique_plot_key", "plot_identifier", "plot_id_key",
    "uav_plot_key", "canonical_plot_key", "full_plot_key", "plot_uid",
]
YIELD_COLUMNS = ["yieldPerAcre", "yield_per_acre", "yield", "target", "YieldPerAcre"]
TP1_PATH_COLUMNS = ["TP1_image_path", "tp1_image_path", "TP1_path", "tp1_path"]


def column_lookup(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column by case-insensitive name, ignoring surrounding spaces."""
    lookup = {str(col).strip().casefold(): col for col in frame.columns}
    for candidate in candidates:
        found = lookup.get(candidate.strip().casefold())
        if found is not None:
            return found
    return None


def normalize_location(value) -> str | None:
    if pd.isna(value):
        return None
    raw = str(value).strip()
    compact = re.sub(r"[^a-z]", "", raw.casefold())
    if compact in LOCATION_ALIASES:
        return LOCATION_ALIASES[compact]
    # Also recognize names embedded in an image path or other metadata string.
    for token, canonical in [
        ("crawfordsville", "Crawfordsville"),
        ("scottsbluff", "Scottsbluff"),
        ("ames", "Ames"),
        ("lincoln", "Lincoln"),
        ("movalley", "MOValley"),
        ("missourivalley", "MOValley"),
    ]:
        if token in compact:
            return canonical
    return None


def main() -> int:
    if not MANIFEST_PATH.exists():
        print("ERROR: Training manifest was not found:")
        print(f"  {MANIFEST_PATH}")
        print("Edit MANIFEST_PATH near the top of this script to point to your verified manifest CSV.")
        return 1

    # Read columns as strings so plot IDs (including leading zeroes) are preserved.
    df = pd.read_csv(MANIFEST_PATH, dtype=str, low_memory=False)
    if df.empty:
        print("ERROR: The manifest CSV is empty.")
        return 1

    location_col = column_lookup(df, LOCATION_COLUMNS)
    plot_id_col = column_lookup(df, PLOT_ID_COLUMNS)
    yield_col = column_lookup(df, YIELD_COLUMNS)
    tp1_col = column_lookup(df, TP1_PATH_COLUMNS)

    if plot_id_col is None:
        print("ERROR: Could not identify the plot-ID column.")
        print("Columns found:", list(df.columns))
        print("Add the exact plot-ID column name to PLOT_ID_COLUMNS in this script, then rerun.")
        return 1
    if yield_col is None:
        print("ERROR: Could not identify the yield-label column.")
        print("Columns found:", list(df.columns))
        print("Add the exact yield column name to YIELD_COLUMNS in this script, then rerun.")
        return 1

    # Normalize / infer location. Prefer a location field, but fill its missing values
    # from the TP1 path if that path column is available.
    loc_values = df[location_col].map(normalize_location) if location_col else pd.Series(None, index=df.index)
    if tp1_col:
        from_path = df[tp1_col].map(normalize_location)
        loc_values = loc_values.where(loc_values.notna(), from_path)
    if location_col is None and tp1_col is None:
        print("ERROR: No recognized location column or TP1 image-path column was found.")
        print("Columns found:", list(df.columns))
        return 1
    df["_split_location"] = loc_values

    missing_location = df["_split_location"].isna()
    if missing_location.any():
        print(f"ERROR: {int(missing_location.sum())} manifest rows have an unrecognized/missing location.")
        print("Example rows:")
        print(df.loc[missing_location].head(8).to_string(index=False))
        return 1

    df["_plot_id_clean"] = df[plot_id_col].fillna("").astype(str).str.strip()
    missing_plot_id = df["_plot_id_clean"].eq("")
    if missing_plot_id.any():
        print(f"ERROR: {int(missing_plot_id.sum())} rows have an empty plot ID.")
        return 1

    # Use (location, plot ID) as the canonical key. Plot identifiers can sometimes
    # repeat between trial locations, so the location is part of the identity.
    df["_leakage_key"] = df["_split_location"] + "::" + df["_plot_id_clean"]
    duplicate_key = df["_leakage_key"].duplicated(keep=False)
    if duplicate_key.any():
        print("ERROR: The manifest has more than one row for at least one location + plot ID.")
        print("Do not split until this is investigated; duplicates could cause leakage.")
        print(df.loc[duplicate_key, [location_col or "_split_location", plot_id_col]].head(20).to_string(index=False))
        print("If each row is an image rather than one row per plot, first rebuild a one-row-per-plot manifest with TP1/TP2/TP3 attached.")
        return 1

    # Refuse to silently include invalid target rows. This file is expected to be the
    # verified training manifest containing numeric yield labels.
    numeric_yield = pd.to_numeric(df[yield_col], errors="coerce")
    if numeric_yield.isna().any():
        print(f"ERROR: {int(numeric_yield.isna().sum())} rows have missing/non-numeric yield in the verified manifest.")
        print("Fix or regenerate the verified manifest first; this script will not invent or fill labels.")
        return 1
    df[yield_col] = numeric_yield

    test_location_normalized = normalize_location(TEST_LOCATION)
    test_mask = df["_split_location"].eq(test_location_normalized)
    if not test_mask.any():
        print(f"ERROR: No rows found for test location {TEST_LOCATION!r}.")
        print("Locations found:", sorted(df["_split_location"].unique()))
        return 1

    # Location-based holdout: every Crawfordsville plot is test-only.
    test_df = df.loc[test_mask].copy()
    remaining = df.loc[~test_mask].copy()

    # Split separately inside each remaining location so every location is represented
    # in train and validation. Each row is one plot, not one image or patch.
    train_parts, validation_parts = [], []
    for location, group in remaining.groupby("_split_location", sort=True):
        if len(group) < 2:
            print(f"ERROR: Location {location} has fewer than 2 plots; cannot create train and validation subsets.")
            return 1
        train_part, validation_part = train_test_split(
            group,
            test_size=VALIDATION_FRACTION,
            random_state=RANDOM_STATE,
            shuffle=True,
        )
        train_parts.append(train_part)
        validation_parts.append(validation_part)

    train_df = pd.concat(train_parts, ignore_index=False)
    validation_df = pd.concat(validation_parts, ignore_index=False)

    # Validate the split identities do not overlap.
    train_keys = set(train_df["_leakage_key"])
    validation_keys = set(validation_df["_leakage_key"])
    test_keys = set(test_df["_leakage_key"])
    assert train_keys.isdisjoint(validation_keys), "Leakage: train and validation plot IDs overlap."
    assert train_keys.isdisjoint(test_keys), "Leakage: train and test plot IDs overlap."
    assert validation_keys.isdisjoint(test_keys), "Leakage: validation and test plot IDs overlap."
    assert set(test_df["_split_location"].unique()) == {test_location_normalized}, "Test set contains another location."
    assert test_location_normalized not in set(train_df["_split_location"]), "Crawfordsville leaked into training."
    assert test_location_normalized not in set(validation_df["_split_location"]), "Crawfordsville leaked into validation."

    # Detect duplicate image paths across sets, if the manifest has explicit TP/image paths.
    image_path_cols = [
        col for col in df.columns
        if any(tok in str(col).casefold() for tok in ("tp1", "tp2", "tp3"))
        and any(tok in str(col).casefold() for tok in ("path", "image", "file"))
    ]
    def paths_in(frame: pd.DataFrame) -> set[str]:
        values = set()
        for col in image_path_cols:
            for value in frame[col].dropna().astype(str):
                value = value.strip()
                if value and value.casefold() not in {"nan", "none", "null"}:
                    values.add(value.casefold().replace("\\", "/"))
        return values

    train_paths, validation_paths, test_paths = paths_in(train_df), paths_in(validation_df), paths_in(test_df)
    assert train_paths.isdisjoint(validation_paths), "Leakage: an image path is present in train and validation."
    assert train_paths.isdisjoint(test_paths), "Leakage: an image path is present in train and test."
    assert validation_paths.isdisjoint(test_paths), "Leakage: an image path is present in validation and test."

    # Remove internal helper columns, retaining original manifest columns (plot IDs,
    # yield, and TP1/TP2/TP3 path fields) so each split can directly drive the pipeline.
    internal_cols = ["_split_location", "_plot_id_clean", "_leakage_key"]
    original_cols = [col for col in df.columns if col not in internal_cols]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_out = train_df[original_cols].sort_values(by=[location_col] if location_col else [plot_id_col]).copy()
    validation_out = validation_df[original_cols].sort_values(by=[location_col] if location_col else [plot_id_col]).copy()
    test_out = test_df[original_cols].sort_values(by=[location_col] if location_col else [plot_id_col]).copy()

    train_file = OUTPUT_DIR / "train_plot_ids.csv"
    validation_file = OUTPUT_DIR / "validation_plot_ids.csv"
    test_file = OUTPUT_DIR / "test_plot_ids.csv"
    assignments_file = OUTPUT_DIR / "all_plot_split_assignments.csv"
    summary_file = OUTPUT_DIR / "split_summary_by_location.csv"

    train_out.to_csv(train_file, index=False)
    validation_out.to_csv(validation_file, index=False)
    test_out.to_csv(test_file, index=False)

    # A compact assignment file makes the exact split easy to audit/reuse.
    assignment_parts = []
    for split_name, frame in [("train", train_df), ("validation", validation_df), ("test", test_df)]:
        assignment_parts.append(pd.DataFrame({
            "location": frame["_split_location"].values,
            "plot_id": frame["_plot_id_clean"].values,
            "split": split_name,
        }))
    assignments = pd.concat(assignment_parts, ignore_index=True)
    assignments.to_csv(assignments_file, index=False)

    summary = assignments.groupby(["location", "split"]).size().unstack(fill_value=0)
    for needed in ["train", "validation", "test"]:
        if needed not in summary.columns:
            summary[needed] = 0
    summary = summary[["train", "validation", "test"]]
    summary["total"] = summary.sum(axis=1)
    summary.loc["TOTAL"] = summary.sum(axis=0)
    summary.to_csv(summary_file, index_label="location")

    print("\nSPLIT COMPLETED SUCCESSFULLY")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Total unique plots: {len(df):,}")
    print(f"Train plots:        {len(train_df):,}")
    print(f"Validation plots:   {len(validation_df):,}")
    print(f"Test plots:         {len(test_df):,} (all {TEST_LOCATION})")
    print("\nCounts by location and split:")
    print(summary.to_string())
    print("\nLeakage checks passed:")
    print("  - No location + plot ID appears in more than one split.")
    print("  - No image path appears in more than one split (where paths are listed in the manifest).")
    print("  - Test contains Crawfordsville only; Crawfordsville is absent from train/validation.")
    print("\nFiles created:")
    print(f"  {train_file}")
    print(f"  {validation_file}")
    print(f"  {test_file}")
    print(f"  {assignments_file}")
    print(f"  {summary_file}")
    print("\nNOTE: Fit normalization/statistics and model/preprocessing parameters using TRAIN only.")
    print("Do not create image patches before splitting; assign all patches from a plot to its split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
