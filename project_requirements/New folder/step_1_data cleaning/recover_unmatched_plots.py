"""
Recovery Investigation Script
Investigates all 328 unmatched plots and recovers matches where possible.
"""
import pandas as pd
from pathlib import Path
import re

ROOT = Path(r"C:\Users\ashwi\Music\cropyeildererrrrrrr\drone_dataset")
GT_PATH = ROOT / "GroundTruth" / "HYBRID_HIPS_V3.5_ALLPLOTS.csv"
UAV_ROOT = ROOT / "UAV"
REPORTS = ROOT / "verification_reports"

gt = pd.read_csv(GT_PATH, low_memory=False)
gt.columns = gt.columns.str.strip()

# ===================================================================
# STEP 1: Understand the data on both sides
# ===================================================================
print("=" * 70)
print("STEP 1: UNDERSTANDING THE MISMATCH")
print("=" * 70)

# --- Missouri Valley ---
mv_gt = gt[gt["location"] == "MOValley"].copy()
print(f"\n--- MISSOURI VALLEY ---")
print(f"CSV rows: {len(mv_gt)}")
print(f"CSV experiment values: {mv_gt['experiment'].unique().tolist()}")

mv_tp1 = sorted((UAV_ROOT / "MOValley" / "TP1").glob("*.PNG"))
mv_exps_in_files = set()
for f in mv_tp1:
    parts = f.stem.split("-")
    exp_part = parts[2].split("_")[0]
    mv_exps_in_files.add(exp_part)
print(f"Image experiment values: {mv_exps_in_files}")
print(f"Image count (TP1): {len(mv_tp1)}")

mv_yield_valid = pd.to_numeric(mv_gt["yieldPerAcre"], errors="coerce").notna().sum()
print(f"CSV rows with valid yield: {mv_yield_valid} / {len(mv_gt)}")

# --- Ames ---
ames_gt = gt[gt["location"] == "Ames"].copy()
print(f"\n--- AMES ---")
print(f"CSV rows: {len(ames_gt)}")
print(f"CSV experiment values: {sorted(ames_gt['experiment'].dropna().unique().tolist())}")

ames_tp1 = sorted((UAV_ROOT / "Ames" / "TP1").glob("*.PNG"))
ames_exps_in_files = set()
for f in ames_tp1:
    parts = f.stem.split("-")
    exp_part = parts[2].split("_")[0]
    ames_exps_in_files.add(exp_part)
print(f"Image experiment values: {ames_exps_in_files}")
print(f"Image count (TP1): {len(ames_tp1)}")

ames_yield_valid = pd.to_numeric(ames_gt["yieldPerAcre"], errors="coerce").notna().sum()
print(f"CSV rows with valid yield: {ames_yield_valid} / {len(ames_gt)}")

# --- Scottsbluff ---
sb_gt = gt[gt["location"] == "Scottsbluff"].copy()
print(f"\n--- SCOTTSBLUFF ---")
print(f"CSV rows: {len(sb_gt)}")
sb_exp_vals = sb_gt["experiment"].unique().tolist()
print(f"CSV experiment values: {sb_exp_vals}")

sb_tp1 = sorted((UAV_ROOT / "Scottsbluff" / "TP1").glob("*.PNG"))
sb_exps_in_files = set()
for f in sb_tp1:
    parts = f.stem.split("-")
    exp_part = parts[2].split("_")[0]
    sb_exps_in_files.add(exp_part)
print(f"Image experiment values: {sb_exps_in_files}")
print(f"Image count (TP1): {len(sb_tp1)}")

sb_yield_valid = pd.to_numeric(sb_gt["yieldPerAcre"], errors="coerce").notna().sum()
print(f"CSV rows with valid yield: {sb_yield_valid} / {len(sb_gt)}")

# ===================================================================
# STEP 2: Build filename parser
# ===================================================================
FILENAME_PATTERN = re.compile(
    r"^(?P<location>.+?)-(?P<timepoint>TP[123])-"
    r"(?P<experiment>[^_]+)_(?P<range>[^_]+)_(?P<row>[^_]+)$",
    flags=re.IGNORECASE,
)

def parse_all_images():
    """Parse all UAV images into a DataFrame."""
    records = []
    for png in UAV_ROOT.rglob("*.PNG"):
        match = FILENAME_PATTERN.match(png.stem)
        if match:
            d = match.groupdict()
            records.append({
                "path": str(png.relative_to(ROOT)),
                "filename": png.name,
                "loc_file": d["location"].strip(),
                "tp": d["timepoint"].upper(),
                "exp_file": d["experiment"].strip(),
                "range_file": d["range"].strip(),
                "row_file": d["row"].strip(),
            })
    return pd.DataFrame(records)

df_img = parse_all_images()
print(f"\nTotal parsed images: {len(df_img)}")

# ===================================================================
# STEP 3: Normalize both sides for matching
# ===================================================================
def norm_loc(val):
    if pd.isna(val):
        return ""
    s = str(val).strip().lower()
    if "valley" in s or s.startswith("mo"):
        return "movalley"
    return s

def norm_exp(val):
    if pd.isna(val):
        return ""
    s = str(val).strip().lower()
    # Fix the Hyrbrids typo
    if s in ("hyrbrids", "hybrids"):
        return "hybrids"
    # Numeric experiment IDs
    try:
        return str(int(float(s)))
    except (ValueError, OverflowError):
        pass
    return s

def norm_id(val):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    try:
        return str(int(float(s)))
    except (ValueError, OverflowError):
        pass
    return s.lower()

# Normalize images
df_img["loc_key"] = df_img["loc_file"].apply(norm_loc)
df_img["exp_key"] = df_img["exp_file"].apply(norm_exp)
df_img["range_key"] = df_img["range_file"].apply(norm_id)
df_img["row_key"] = df_img["row_file"].apply(norm_id)
df_img["plot_key"] = (
    df_img["loc_key"] + "|" + df_img["exp_key"] + "|" +
    df_img["range_key"] + "|" + df_img["row_key"]
)

# Normalize ground truth
gt["loc_key"] = gt["location"].apply(norm_loc)
gt["exp_key"] = gt["experiment"].apply(norm_exp)
gt["range_key"] = gt["range"].apply(norm_id)
gt["row_key"] = gt["row"].apply(norm_id)
gt["plot_key"] = (
    gt["loc_key"] + "|" + gt["exp_key"] + "|" +
    gt["range_key"] + "|" + gt["row_key"]
)

# ===================================================================
# STEP 4: Match images to ground truth
# ===================================================================
gt_keys = set(gt["plot_key"])
df_img["matched"] = df_img["plot_key"].isin(gt_keys)

matched = df_img[df_img["matched"]]
unmatched = df_img[~df_img["matched"]]

print("\n" + "=" * 70)
print("STEP 2: RECOVERY RESULTS (after fixing experiment name normalization)")
print("=" * 70)

print(f"\nTotal images: {len(df_img)}")
print(f"Matched images: {len(matched)}")
print(f"Still unmatched: {len(unmatched)}")

# Break down unmatched by location
print("\nUnmatched images by location:")
if len(unmatched) > 0:
    print(unmatched.groupby("loc_file").size().to_string())
else:
    print("  None! All images matched.")

# Break down by unique plots
unmatched_plots = unmatched.drop_duplicates(subset=["plot_key"])
print(f"\nUnmatched unique plots: {len(unmatched_plots)}")

# ===================================================================
# STEP 5: Check yield availability for matched plots
# ===================================================================
# For matched images, join with GT to check yield
gt_deduped = gt.drop_duplicates(subset=["plot_key"])
df_merged = matched.merge(
    gt_deduped[["plot_key", "genotype", "yieldPerAcre", "totalStandCount",
                "plantingDate", "poundsOfNitrogenPerAcre"]],
    on="plot_key",
    how="left",
)

df_merged["yield_numeric"] = pd.to_numeric(df_merged["yieldPerAcre"], errors="coerce")
has_yield = df_merged["yield_numeric"].notna()

# Unique plots with yield
plots_with_yield = df_merged[has_yield].drop_duplicates(subset=["plot_key"])
plots_without_yield = df_merged[~has_yield].drop_duplicates(subset=["plot_key"])

print(f"\n--- YIELD AVAILABILITY (among matched plots) ---")
print(f"Unique matched plots with valid yield: {len(plots_with_yield)}")
print(f"Unique matched plots WITHOUT yield:    {len(plots_without_yield)}")
print(f"Total matched images with yield:       {has_yield.sum()}")
print(f"Total matched images without yield:    {(~has_yield).sum()}")

# By location
print("\nMatched plots WITH yield, by location:")
print(plots_with_yield.groupby("loc_file").size().to_string())

print("\nMatched plots WITHOUT yield, by location:")
if len(plots_without_yield) > 0:
    print(plots_without_yield.groupby("loc_file").size().to_string())
else:
    print("  None!")

# ===================================================================
# STEP 6: Investigate remaining unmatched
# ===================================================================
if len(unmatched) > 0:
    print("\n" + "=" * 70)
    print("STEP 3: INVESTIGATING REMAINING UNMATCHED PLOTS")
    print("=" * 70)

    for loc in unmatched["loc_file"].unique():
        loc_unmatched = unmatched[unmatched["loc_file"] == loc]
        loc_plots = loc_unmatched.drop_duplicates(subset=["plot_key"])
        print(f"\n--- {loc}: {len(loc_plots)} unmatched plots ({len(loc_unmatched)} images) ---")

        # Show sample filenames
        print("  Sample filenames:")
        for _, row in loc_plots.head(5).iterrows():
            print(f"    {row['filename']}  ->  key: {row['plot_key']}")

        # Check if any of these plots exist in GT with different experiment
        loc_key = norm_loc(loc)
        gt_loc = gt[gt["loc_key"] == loc_key]
        print(f"  CSV rows for this location: {len(gt_loc)}")

        # Check for range/row overlap
        img_coords = set(
            zip(loc_plots["range_key"], loc_plots["row_key"])
        )
        gt_coords = set(
            zip(gt_loc["range_key"], gt_loc["row_key"])
        )
        overlap = img_coords & gt_coords
        only_in_images = img_coords - gt_coords
        print(f"  (range, row) pairs in images: {len(img_coords)}")
        print(f"  (range, row) pairs in CSV:    {len(gt_coords)}")
        print(f"  Overlapping coordinates:      {len(overlap)}")
        print(f"  Only in images (true border):  {len(only_in_images)}")

        if overlap:
            print(f"\n  These {len(overlap)} plots EXIST in the CSV at same (range, row).")
            print(f"  Experiment mismatch is the only issue.")
            # Show examples
            for rng, rw in sorted(overlap)[:5]:
                img_exp = loc_plots[
                    (loc_plots["range_key"] == rng) & (loc_plots["row_key"] == rw)
                ]["exp_key"].values[0]
                csv_exp = gt_loc[
                    (gt_loc["range_key"] == rng) & (gt_loc["row_key"] == rw)
                ]["exp_key"].values[0]
                csv_yield = gt_loc[
                    (gt_loc["range_key"] == rng) & (gt_loc["row_key"] == rw)
                ]["yieldPerAcre"].values[0]
                print(f"    range={rng}, row={rw}: "
                      f"img_exp='{img_exp}' vs csv_exp='{csv_exp}', "
                      f"yield={csv_yield}")

# ===================================================================
# STEP 7: RECOVER by relaxed matching (location + range + row only)
# ===================================================================
print("\n" + "=" * 70)
print("STEP 4: RECOVERY — MATCHING BY (location, range, row) ONLY")
print("=" * 70)

# For still-unmatched images, try matching on just location + range + row
gt["lrr_key"] = gt["loc_key"] + "|" + gt["range_key"] + "|" + gt["row_key"]
unmatched["lrr_key"] = unmatched["loc_key"] + "|" + unmatched["range_key"] + "|" + unmatched["row_key"]

gt_lrr_keys = set(gt["lrr_key"])
unmatched["lrr_matched"] = unmatched["lrr_key"].isin(gt_lrr_keys)

recovered = unmatched[unmatched["lrr_matched"]]
still_lost = unmatched[~unmatched["lrr_matched"]]

recovered_plots = recovered.drop_duplicates(subset=["lrr_key"])
still_lost_plots = still_lost.drop_duplicates(subset=["plot_key"])

# Check uniqueness — each (loc, range, row) should map to exactly 1 GT row
gt_lrr_counts = gt.groupby("lrr_key").size()
ambiguous_lrr = gt_lrr_counts[gt_lrr_counts > 1]

recovered_ambiguous = recovered[recovered["lrr_key"].isin(ambiguous_lrr.index)]
recovered_unique = recovered[~recovered["lrr_key"].isin(ambiguous_lrr.index)]

# Get yield for recovered plots
gt_lrr_deduped = gt.drop_duplicates(subset=["lrr_key"])
recovered_with_gt = recovered_unique.merge(
    gt_lrr_deduped[["lrr_key", "experiment", "genotype", "yieldPerAcre"]],
    on="lrr_key",
    how="left",
    suffixes=("", "_csv"),
)
recovered_with_gt["yield_numeric"] = pd.to_numeric(
    recovered_with_gt["yieldPerAcre"], errors="coerce"
)
recovered_has_yield = recovered_with_gt["yield_numeric"].notna()

print(f"\nRecovered images (unique match): {len(recovered_unique)}")
print(f"Recovered plots with yield:     {recovered_with_gt[recovered_has_yield].drop_duplicates(subset=['lrr_key']).shape[0]}")
print(f"Recovered plots without yield:  {recovered_with_gt[~recovered_has_yield].drop_duplicates(subset=['lrr_key']).shape[0]}")
print(f"Ambiguous matches (skipped):    {len(recovered_ambiguous)}")
print(f"Truly unrecoverable images:     {len(still_lost)}")
print(f"Truly unrecoverable plots:      {len(still_lost_plots)}")

if len(recovered_unique) > 0:
    print("\nRecovered by location:")
    print(recovered_unique.groupby("loc_file").size().to_string())

if len(still_lost) > 0:
    print("\nStill lost by location:")
    print(still_lost.groupby("loc_file").size().to_string())

# ===================================================================
# STEP 8: FINAL SUMMARY
# ===================================================================
total_images = len(df_img)
total_matched_strict = len(matched)
total_recovered = len(recovered_unique)
total_recovered_with_yield = recovered_has_yield.sum()
total_lost = len(still_lost)
total_usable = total_matched_strict + total_recovered_with_yield

# Unique plots
strict_match_plots = matched.drop_duplicates(subset=["plot_key"])
strict_with_yield = df_merged[has_yield].drop_duplicates(subset=["plot_key"])
recovered_yield_plots = recovered_with_gt[recovered_has_yield].drop_duplicates(subset=["lrr_key"])

print("\n" + "=" * 70)
print("FINAL RECOVERY SUMMARY")
print("=" * 70)
print(f"Total drone images:                     {total_images}")
print(f"Strict match (experiment-exact):         {total_matched_strict} images ({strict_match_plots.shape[0]} plots)")
print(f"Recovered (location+range+row match):    {total_recovered} images ({recovered_plots.shape[0]} plots)")
print(f"  - with valid yield:                    {int(total_recovered_with_yield)} images ({recovered_yield_plots.shape[0]} plots)")
print(f"  - without yield (border plots):        {int(len(recovered_unique) - total_recovered_with_yield)} images")
print(f"Truly unrecoverable (no CSV match):      {total_lost} images ({still_lost_plots.shape[0]} plots)")
print(f"")
print(f"TOTAL USABLE FOR TRAINING:")
print(f"  Plots with yield:  {strict_with_yield.shape[0] + recovered_yield_plots.shape[0]}")
print(f"  Images with yield: {int(has_yield.sum() + total_recovered_with_yield)}")

# ===================================================================
# STEP 9: Save recovered mapping
# ===================================================================
REPORTS.mkdir(parents=True, exist_ok=True)

# Combine strict matches (with yield) + recovered (with yield)
strict_usable = df_merged[has_yield].copy()
strict_usable["match_type"] = "strict"

recovered_usable = recovered_with_gt[recovered_has_yield].copy()
recovered_usable["match_type"] = "recovered_by_location_range_row"

# Align columns for concatenation
common_cols = ["path", "filename", "loc_file", "tp", "exp_file",
               "range_file", "row_file", "plot_key", "genotype",
               "yieldPerAcre", "yield_numeric", "match_type"]

strict_out = strict_usable[[c for c in common_cols if c in strict_usable.columns]].copy()
recovered_out = recovered_usable[[c for c in common_cols if c in recovered_usable.columns]].copy()

final = pd.concat([strict_out, recovered_out], ignore_index=True)
final.to_csv(REPORTS / "recovered_training_manifest.csv", index=False)

# Save still-lost for reference
if len(still_lost) > 0:
    still_lost.to_csv(REPORTS / "truly_unrecoverable_images.csv", index=False)

# Save recovery details
recovered_with_gt.to_csv(REPORTS / "recovery_details.csv", index=False)

print(f"\nFiles saved to: {REPORTS}")
print(f"  recovered_training_manifest.csv  ({len(final)} rows)")
print(f"  recovery_details.csv             ({len(recovered_with_gt)} rows)")
if len(still_lost) > 0:
    print(f"  truly_unrecoverable_images.csv   ({len(still_lost)} rows)")

print("\nNo original dataset files were modified.")
