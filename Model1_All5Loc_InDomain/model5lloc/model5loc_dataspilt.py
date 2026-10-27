"""
Model 1 (Model1_All5Loc_InDomain) — Step 1: Leakage-Safe 5-Location Plot-Level Split
Policy:
  * All 5 locations (Ames, Crawfordsville, Lincoln, MOValley, Scottsbluff) are included.
  * Within each location, plots are split into 70% Train, 15% Validation, and 15% Test.
  * TP1, TP2, and TP3 remain together per plot.
"""
from pathlib import Path
import re
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

MODEL_NAME = "Model1_All5Loc_InDomain"
ROOT = Path(r"C:\Users\ashwi\Music\cropyeildererrrrrrr")
MANIFEST_PATH = ROOT / "drone_dataset" / "valid_plots.csv"
OUTPUT_DIR = ROOT / MODEL_NAME / "step1_datasplitting"

VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_STATE = 42

LOCATION_ALIASES = {
    "ames": "Ames",
    "crawfordsville": "Crawfordsville",
    "lincoln": "Lincoln",
    "movalley": "MOValley",
    "missourivalley": "MOValley",
    "scottsbluff": "Scottsbluff",
}

LOCATION_COLUMNS = ["location", "location_raw", "location_from_filename", "location_key", "trial_location", "site", "Location"]
PLOT_ID_COLUMNS = ["plot_id", "plot_key", "unique_plot_key", "plot_identifier", "plot_id_key", "uav_plot_key", "canonical_plot_key"]
YIELD_COLUMNS = ["yieldPerAcre", "yield_per_acre", "yield", "target", "YieldPerAcre"]
TP1_PATH_COLUMNS = ["TP1_image_path", "tp1_image_path", "TP1_path", "tp1_path"]


def column_lookup(frame: pd.DataFrame, candidates: list[str]) -> str | None:
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
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
        return 1

    df = pd.read_csv(MANIFEST_PATH, dtype=str, low_memory=False)
    location_col = column_lookup(df, LOCATION_COLUMNS)
    plot_id_col = column_lookup(df, PLOT_ID_COLUMNS)
    yield_col = column_lookup(df, YIELD_COLUMNS)
    tp1_col = column_lookup(df, TP1_PATH_COLUMNS)

    if plot_id_col is None or yield_col is None:
        print("ERROR: Missing plot_id or yieldPerAcre column.")
        return 1

    loc_values = df[location_col].map(normalize_location) if location_col else pd.Series(None, index=df.index)
    if tp1_col:
        from_path = df[tp1_col].map(normalize_location)
        loc_values = loc_values.where(loc_values.notna(), from_path)
    df["_split_location"] = loc_values
    df["_plot_id_clean"] = df[plot_id_col].fillna("").astype(str).str.strip()
    df["_leakage_key"] = df["_split_location"] + "::" + df["_plot_id_clean"]

    if df["_leakage_key"].duplicated(keep=False).any():
        print("ERROR: Duplicate (location, plot_id) rows detected.")
        return 1

    df[yield_col] = pd.to_numeric(df[yield_col], errors="coerce")
    df = df.dropna(subset=[yield_col, "_split_location"]).reset_index(drop=True)

    train_parts, val_parts, test_parts = [], [], []
    holdout_fraction = VAL_FRACTION + TEST_FRACTION
    relative_test_ratio = TEST_FRACTION / holdout_fraction

    for location, group in df.groupby("_split_location", sort=True):
        # Stratify by yield quartiles within each location so Train, Val, and Test have balanced yield distributions
        y_bins = pd.qcut(group[yield_col], q=4, labels=False, duplicates="drop")
        train_part, temp_part = train_test_split(
            group,
            test_size=holdout_fraction,
            random_state=RANDOM_STATE,
            shuffle=True,
            stratify=y_bins,
        )
        temp_bins = pd.qcut(temp_part[yield_col], q=2, labels=False, duplicates="drop")
        val_part, test_part = train_test_split(
            temp_part,
            test_size=relative_test_ratio,
            random_state=RANDOM_STATE,
            shuffle=True,
            stratify=temp_bins,
        )
        train_parts.append(train_part)
        val_parts.append(val_part)
        test_parts.append(test_part)

    train_df = pd.concat(train_parts, ignore_index=True)
    validation_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    # Strict Zero-Leakage Assertions
    train_keys = set(train_df["_leakage_key"])
    val_keys = set(validation_df["_leakage_key"])
    test_keys = set(test_df["_leakage_key"])
    assert train_keys.isdisjoint(val_keys), "Leakage: Train and Val overlap!"
    assert train_keys.isdisjoint(test_keys), "Leakage: Train and Test overlap!"
    assert val_keys.isdisjoint(test_keys), "Leakage: Val and Test overlap!"
    assert len(set(train_df["_split_location"])) == 5, "All 5 locations must be in Train!"
    assert len(set(validation_df["_split_location"])) == 5, "All 5 locations must be in Val!"
    assert len(set(test_df["_split_location"])) == 5, "All 5 locations must be in Test!"

    internal_cols = ["_split_location", "_plot_id_clean", "_leakage_key"]
    original_cols = [c for c in df.columns if c not in internal_cols]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df[original_cols].to_csv(OUTPUT_DIR / "train_plot_ids.csv", index=False)
    validation_df[original_cols].to_csv(OUTPUT_DIR / "validation_plot_ids.csv", index=False)
    test_df[original_cols].to_csv(OUTPUT_DIR / "test_plot_ids.csv", index=False)

    assignments = pd.concat([
        pd.DataFrame({"location": train_df["_split_location"], "plot_id": train_df["_plot_id_clean"], "split": "train"}),
        pd.DataFrame({"location": validation_df["_split_location"], "plot_id": validation_df["_plot_id_clean"], "split": "val"}),
        pd.DataFrame({"location": test_df["_split_location"], "plot_id": test_df["_plot_id_clean"], "split": "test"}),
    ], ignore_index=True)
    assignments.to_csv(OUTPUT_DIR / "all_plot_split_assignments.csv", index=False)

    summary = assignments.groupby(["location", "split"]).size().unstack(fill_value=0)[["train", "val", "test"]]
    summary["total"] = summary.sum(axis=1)
    summary.loc["TOTAL"] = summary.sum(axis=0)
    summary.to_csv(OUTPUT_DIR / "split_summary_by_location.csv", index_label="location")

    print("=" * 80)
    print(f"[{MODEL_NAME}] 5-LOCATION PLOT SPLIT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(summary.to_string())
    print(f"\nSaved split files to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())