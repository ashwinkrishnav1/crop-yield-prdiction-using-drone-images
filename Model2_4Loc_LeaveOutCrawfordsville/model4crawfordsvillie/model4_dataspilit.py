"""
Model 2 (Model2_4Loc_LeaveOutCrawfordsville) — Step 1: 4-Location Train/Val + Unseen Crawfordsville Test Split
Policy:
  * Every valid plot from Crawfordsville is held out for testing (488 plots).
  * The remaining 4 locations (Ames, Lincoln, MOValley, Scottsbluff) are split plot-by-plot into Train (80%) and Val (20%).
  * TP1, TP2, and TP3 remain together per plot.
"""
from pathlib import Path
import re
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

MODEL_NAME = "Model2_4Loc_LeaveOutCrawfordsville"
ROOT = Path(r"C:\Users\ashwi\Music\cropyeildererrrrrrr")
MANIFEST_PATH = ROOT / "drone_dataset" / "valid_plots.csv"
OUTPUT_DIR = ROOT / MODEL_NAME / "step1_datasplitting"
TEST_LOCATION = "Crawfordsville"
VALIDATION_FRACTION = 0.20
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

    loc_values = df[location_col].map(normalize_location) if location_col else pd.Series(None, index=df.index)
    if tp1_col:
        from_path = df[tp1_col].map(normalize_location)
        loc_values = loc_values.where(loc_values.notna(), from_path)
    df["_split_location"] = loc_values
    df["_plot_id_clean"] = df[plot_id_col].fillna("").astype(str).str.strip()
    df["_leakage_key"] = df["_split_location"] + "::" + df["_plot_id_clean"]

    df[yield_col] = pd.to_numeric(df[yield_col], errors="coerce")
    df = df.dropna(subset=[yield_col, "_split_location"]).reset_index(drop=True)

    test_loc_norm = normalize_location(TEST_LOCATION)
    test_mask = df["_split_location"].eq(test_loc_norm)
    test_df = df.loc[test_mask].copy()
    remaining = df.loc[~test_mask].copy()

    train_parts, val_parts = [], []
    for location, group in remaining.groupby("_split_location", sort=True):
        tr_p, va_p = train_test_split(group, test_size=VALIDATION_FRACTION, random_state=RANDOM_STATE, shuffle=True)
        train_parts.append(tr_p)
        val_parts.append(va_p)

    train_df = pd.concat(train_parts, ignore_index=True)
    validation_df = pd.concat(val_parts, ignore_index=True)

    assert set(test_df["_split_location"].unique()) == {test_loc_norm}
    assert test_loc_norm not in set(train_df["_split_location"])
    assert test_loc_norm not in set(validation_df["_split_location"])

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
    print(f"[{MODEL_NAME}] 4-LOC TRAIN/VAL + CRAWFORDSVILLE TEST SPLIT COMPLETED")
    print("=" * 80)
    print(summary.to_string())
    print(f"\nSaved split files to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())