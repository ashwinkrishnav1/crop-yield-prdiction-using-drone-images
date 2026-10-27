"""Audit UAV image-to-ground-truth mapping for the Multi-state Maize dataset.

Expected dataset layout:
    drone_dataset/
      UAV/...
      GroundTruth/HYBRID_HIPS_V3.5_ALLPLOTS.csv

The script only reads original data and writes audit reports to audit_reports/.
"""

from pathlib import Path
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Find the dataset automatically.
# Supports either:
#   project_folder/drone_dataset/UAV + project_folder/drone_dataset/GroundTruth
# or
#   project_folder/UAV + project_folder/GroundTruth
# It also accepts UAV_Dataset instead of UAV.
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_CANDIDATES = [SCRIPT_DIR / "drone_dataset", SCRIPT_DIR]
CSV_RELATIVE_PATH = Path("GroundTruth") / "HYBRID_HIPS_V3.5_ALLPLOTS.csv"

ROOT = None
UAV_ROOT = None
for candidate in DATASET_CANDIDATES:
    candidate_uav = candidate / "UAV"
    if not candidate_uav.is_dir():
        candidate_uav = candidate / "UAV_Dataset"
    candidate_csv = candidate / CSV_RELATIVE_PATH
    if candidate_uav.is_dir() and candidate_csv.is_file():
        ROOT = candidate
        UAV_ROOT = candidate_uav
        CSV_PATH = candidate_csv
        break

if ROOT is None:
    print("ERROR: Could not locate the dataset.")
    print("Expected either:")
    print(f"  {SCRIPT_DIR / 'drone_dataset' / 'UAV'}")
    print(f"  {SCRIPT_DIR / 'drone_dataset' / CSV_RELATIVE_PATH}")
    print("or UAV and GroundTruth directly beside this script.")
    print("Edit DATASET_CANDIDATES / CSV_RELATIVE_PATH near the top if needed.")
    sys.exit(1)

REPORT_DIR = ROOT / "audit_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Example filename: Crawfordsville-TP1-4351_3_15.PNG
FILENAME_PATTERN = re.compile(
    r"^(?P<location>.+?)-(?P<timepoint>TP[123])-"
    r"(?P<experiment>[^_]+)_(?P<range>[^_]+)_(?P<row>[^_]+)$",
    flags=re.IGNORECASE,
)

IMAGE_KEY_COLUMNS = ["location_key", "experiment_key", "range_key", "row_key"]
TIMEPOINTS = ["TP1", "TP2", "TP3"]


def normalize_location(value) -> str:
    """Normalize location names and common Missouri Valley variants."""
    if pd.isna(value):
        return ""
    text = re.sub(r"[^a-z0-9]", "", str(value).strip().lower())
    if not text:
        return ""
    # The image directory commonly uses MOValley while CSVs may spell it out.
    if text.startswith(("movalley", "missourivalley")):
        return "movalley"
    # Remove common state suffixes naturally, e.g. Ames, IA -> amesia.
    for canonical in ("crawfordsville", "scottsbluff", "ames", "lincoln"):
        if text.startswith(canonical):
            return canonical
    return text


def normalize_identifier(value) -> str:
    """Normalize numeric IDs such as 15, 15.0, and '15' to the same string."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    # Convert integer-like numeric values without damaging alphanumeric IDs.
    if re.fullmatch(r"[+-]?\d+(?:\.0+)?", text):
        try:
            return str(int(float(text)))
        except (ValueError, OverflowError):
            pass
    # Handle known dataset variations (e.g. 'Hyrbrids' typo in CSV vs 'Hybrids' in filenames)
    cleaned = text.lower()
    if cleaned in {"hybrids", "hyrbrids"}:
        return "hybrids"
    return cleaned


# -------------------------------------------------------------------
# 1. Scan and parse all UAV image files
# -------------------------------------------------------------------
print(f"Scanning UAV images in: {UAV_ROOT}")
image_files = sorted(
    p for p in UAV_ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() == ".png"
)

image_records = []
unparsed_images = []

for p in image_files:
    match = FILENAME_PATTERN.match(p.stem)
    rel_path = str(p.relative_to(ROOT))
    if match:
        data = match.groupdict()
        loc_key = normalize_location(data["location"])
        exp_key = normalize_identifier(data["experiment"])
        range_key = normalize_identifier(data["range"])
        row_key = normalize_identifier(data["row"])

        image_records.append({
            "relative_path": rel_path,
            "filename": p.name,
            "location_raw": data["location"],
            "timepoint": data["timepoint"].upper(),
            "experiment_raw": data["experiment"],
            "range_raw": data["range"],
            "row_raw": data["row"],
            "location_key": loc_key,
            "experiment_key": exp_key,
            "range_key": range_key,
            "row_key": row_key,
            "plot_key": f"{loc_key}|{exp_key}|{range_key}|{row_key}",
        })
    else:
        unparsed_images.append({
            "relative_path": rel_path,
            "filename": p.name,
        })

df_images = pd.DataFrame(image_records)
print(f"Found {len(image_files)} PNG images. Successfully parsed: {len(df_images)}, Unparsed: {len(unparsed_images)}")

# -------------------------------------------------------------------
# 2. Load and normalize Ground-Truth CSV
# -------------------------------------------------------------------
print(f"\nLoading Ground-Truth CSV from: {CSV_PATH}")
df_gt = pd.read_csv(CSV_PATH, low_memory=False)
df_gt.columns = df_gt.columns.str.strip()

df_gt["location_key"] = df_gt["location"].apply(normalize_location)
df_gt["experiment_key"] = df_gt["experiment"].apply(normalize_identifier)
df_gt["range_key"] = df_gt["range"].apply(normalize_identifier)
df_gt["row_key"] = df_gt["row"].apply(normalize_identifier)
df_gt["plot_key"] = (
    df_gt["location_key"] + "|" +
    df_gt["experiment_key"] + "|" +
    df_gt["range_key"] + "|" +
    df_gt["row_key"]
)

print(f"Loaded {len(df_gt)} ground-truth plot records.")

# -------------------------------------------------------------------
# 3. Match Images to Ground Truth Plots
# -------------------------------------------------------------------
gt_keys = set(df_gt["plot_key"])
df_images["matched_to_gt"] = df_images["plot_key"].isin(gt_keys)

matched_images = df_images[df_images["matched_to_gt"]]
unmatched_images = df_images[~df_images["matched_to_gt"]]

# Check Ground-Truth plots with / without images
image_plot_keys = set(df_images["plot_key"])
df_gt["has_uav_images"] = df_gt["plot_key"].isin(image_plot_keys)

plots_with_images = df_gt[df_gt["has_uav_images"]]
plots_without_images = df_gt[~df_gt["has_uav_images"]]

# -------------------------------------------------------------------
# 4. Generate Reports and Summaries
# -------------------------------------------------------------------
# Merge matched images with their ground-truth labels
df_merged = matched_images.merge(
    df_gt,
    on="plot_key",
    how="left",
    suffixes=("_img", "_gt"),
)

# Export audit files
df_merged.to_csv(REPORT_DIR / "matched_images_with_ground_truth.csv", index=False)
unmatched_images.to_csv(REPORT_DIR / "unmatched_images.csv", index=False)
plots_without_images.to_csv(REPORT_DIR / "plots_without_images.csv", index=False)

if unparsed_images:
    pd.DataFrame(unparsed_images).to_csv(REPORT_DIR / "unparsed_images.csv", index=False)

# Breakdown by Location and Timepoint
coverage = df_images.groupby(["location_raw", "timepoint"]).size().unstack(fill_value=0)

# Target Yield analysis on matched images
yield_col = next((c for c in df_gt.columns if c.lower() == "yieldperacre"), None)

print("\n" + "=" * 50)
print("             AUDIT SUMMARY REPORT")
print("=" * 50)
print(f"Total UAV images found        : {len(image_files)}")
print(f"Images successfully matched   : {len(matched_images)} ({len(matched_images)/len(image_files)*100:.1f}%)")
print(f"Images unmatched              : {len(unmatched_images)}")
print(f"Total Ground-Truth plots      : {len(df_gt)}")
print(f"Plots with UAV imagery        : {len(plots_with_images)} ({len(plots_with_images)/len(df_gt)*100:.1f}%)")
print(f"Plots without UAV imagery     : {len(plots_without_images)} (border/unharvested plots)")

if yield_col:
    yield_s = pd.to_numeric(plots_with_images[yield_col], errors="coerce")
    valid_yield = yield_s.notna().sum()
    print(f"Plots with valid yield label  : {valid_yield} ({valid_yield/len(plots_with_images)*100:.1f}%)")

print("\nUAV Image Counts by Location & Time Point:")
print(coverage.to_string())

print("\nAudit reports saved to:")
print(f"  {REPORT_DIR.resolve()}")
print("=" * 50)