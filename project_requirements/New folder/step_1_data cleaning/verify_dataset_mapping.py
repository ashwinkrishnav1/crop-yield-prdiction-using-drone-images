"""Independent verification audit for the Multi-state Maize UAV dataset.

This script DOES NOT modify or delete source files. It checks:
- parsed UAV images and unique plot keys
- TP1/TP2/TP3 completeness per plot
- exact joins against location + experiment + range + row
- duplicate/ambiguous ground-truth keys
- valid/missing yield labels
- CSV-only and image-only plot keys
- diagnostic partial-key candidates for unmatched plots

Expected layout (either is supported):
  <script folder>/drone_dataset/UAV/...
  <script folder>/drone_dataset/GroundTruth/HYBRID_HIPS_V3.5_ALLPLOTS.csv
OR UAV and GroundTruth directly in the script folder.
"""
from pathlib import Path
from collections import defaultdict
import re
import sys
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_CANDIDATES = [SCRIPT_DIR / "drone_dataset", SCRIPT_DIR]
CSV_REL = Path("GroundTruth") / "HYBRID_HIPS_V3.5_ALLPLOTS.csv"

ROOT = UAV_ROOT = CSV_PATH = None
for root in ROOT_CANDIDATES:
    uav = root / "UAV"
    if not uav.is_dir():
        uav = root / "UAV_Dataset"
    csv_path = root / CSV_REL
    if uav.is_dir() and csv_path.is_file():
        ROOT, UAV_ROOT, CSV_PATH = root, uav, csv_path
        break

if ROOT is None:
    print("ERROR: Dataset not found. Edit ROOT_CANDIDATES/CSV_REL in this script.")
    print("Expected drone_dataset\\UAV and drone_dataset\\GroundTruth\\HYBRID_HIPS_V3.5_ALLPLOTS.csv")
    sys.exit(1)

REPORT_DIR = ROOT / "verification_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TIMEPOINTS = ["TP1", "TP2", "TP3"]
KEYS = ["location_key", "experiment_key", "range_key", "row_key"]
FILENAME_PATTERN = re.compile(
    r"^(?P<location>.+?)-(?P<timepoint>TP[123])-"
    r"(?P<experiment>[^_]+)_(?P<range>[^_]+)_(?P<row>[^_]+)$",
    re.IGNORECASE,
)


def norm_location(value):
    if pd.isna(value):
        return ""
    s = re.sub(r"[^a-z0-9]", "", str(value).strip().lower())
    if not s:
        return ""
    if s.startswith(("movalley", "missourivalley")):
        return "movalley"
    for known in ("crawfordsville", "scottsbluff", "ames", "lincoln"):
        if s.startswith(known):
            return known
    return s


def norm_id(value):
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    # normalize numeric values like 15, 15.0, and "15"
    if re.fullmatch(r"[+-]?\d+(?:\.0+)?", s):
        try:
            return str(int(float(s)))
        except (ValueError, OverflowError):
            pass
    return re.sub(r"\s+", "", s).lower()


def key_from_row(row):
    return tuple(str(row[c]) for c in KEYS)


def csv_list(values):
    return ";".join(str(v) for v in values)


def banner(title):
    print("\n" + "=" * 22 + f" {title} " + "=" * 22)

# ----------------------- Load ground-truth CSV -----------------------
gt = pd.read_csv(CSV_PATH, low_memory=False)
gt.columns = [str(c).strip() for c in gt.columns]
col_lookup = {c.lower(): c for c in gt.columns}
required = ["location", "experiment", "range", "row", "yieldperacre"]
missing_cols = [c for c in required if c not in col_lookup]
if missing_cols:
    raise ValueError(f"Required columns missing: {missing_cols}. Available: {gt.columns.tolist()}")

C_LOC = col_lookup["location"]
C_EXP = col_lookup["experiment"]
C_RANGE = col_lookup["range"]
C_ROW = col_lookup["row"]
C_YIELD = col_lookup["yieldperacre"]
C_PLOT = col_lookup.get("plotnumber")
C_INDEX = col_lookup.get("index")

# Approximate physical CSV row number: header is line 1.
gt["csv_record_number"] = range(2, len(gt) + 2)
gt["yield_numeric"] = pd.to_numeric(gt[C_YIELD], errors="coerce")
gt["location_key"] = gt[C_LOC].map(norm_location)
gt["experiment_key"] = gt[C_EXP].map(norm_id)
gt["range_key"] = gt[C_RANGE].map(norm_id)
gt["row_key"] = gt[C_ROW].map(norm_id)
gt["valid_join_key"] = gt[KEYS].ne("").all(axis=1)

valid_gt = gt.loc[gt["valid_join_key"]].copy()
lookup = defaultdict(list)
for idx, r in valid_gt.iterrows():
    lookup[key_from_row(r)].append(idx)

dup_mask = valid_gt.duplicated(KEYS, keep=False)
dup_gt = valid_gt.loc[dup_mask].copy()
dup_gt.to_csv(REPORT_DIR / "duplicate_csv_join_keys.csv", index=False)

# Diagnostic lookups for partial keys: useful only as leads, not proof of a match.
lookup_ewr = defaultdict(list)  # experiment + range + row, ignoring location
lookup_ler = defaultdict(list)  # location + experiment + row, ignoring range
lookup_lxr = defaultdict(list)  # location + range + row, ignoring experiment
lookup_lex = defaultdict(list)  # location + experiment + range, ignoring row
for idx, r in valid_gt.iterrows():
    lookup_ewr[(r["experiment_key"], r["range_key"], r["row_key"])].append(idx)
    lookup_ler[(r["location_key"], r["experiment_key"], r["row_key"])].append(idx)
    lookup_lxr[(r["location_key"], r["range_key"], r["row_key"])].append(idx)
    lookup_lex[(r["location_key"], r["experiment_key"], r["range_key"])].append(idx)

# --------------------------- Scan images ---------------------------
image_paths = sorted(p for p in UAV_ROOT.rglob("*") if p.is_file() and p.suffix.lower() == ".png")
image_records = []
for p in image_paths:
    m = FILENAME_PATTERN.match(p.stem)
    r = {
        "image_path": str(p.relative_to(ROOT)),
        "filename": p.name,
        "filename_parsed": bool(m),
        "location_raw": "", "timepoint": "", "experiment_raw": "", "range_raw": "", "row_raw": "",
        "location_key": "", "experiment_key": "", "range_key": "", "row_key": "",
    }
    if m:
        d = m.groupdict()
        r.update({
            "location_raw": d["location"],
            "timepoint": d["timepoint"].upper(),
            "experiment_raw": d["experiment"],
            "range_raw": d["range"],
            "row_raw": d["row"],
            "location_key": norm_location(d["location"]),
            "experiment_key": norm_id(d["experiment"]),
            "range_key": norm_id(d["range"]),
            "row_key": norm_id(d["row"]),
        })
    image_records.append(r)
images = pd.DataFrame(image_records)
parsed = images.loc[images["filename_parsed"]].copy()

image_rows = []
for _, im in images.iterrows():
    out = im.to_dict()
    out.update({"csv_match_count": 0, "csv_record_number": pd.NA,
                "csv_index_value": pd.NA, "yieldPerAcre": pd.NA,
                "mapping_status": "unparsed_filename"})
    if bool(im["filename_parsed"]):
        k = tuple(im[c] for c in KEYS)
        candidates = lookup.get(k, [])
        out["csv_match_count"] = len(candidates)
        if len(candidates) == 0:
            out["mapping_status"] = "no_csv_match"
        elif len(candidates) > 1:
            out["mapping_status"] = "ambiguous_csv_match"
        else:
            gr = gt.loc[candidates[0]]
            out["csv_record_number"] = int(gr["csv_record_number"])
            if C_INDEX:
                out["csv_index_value"] = gr[C_INDEX]
            if pd.isna(gr["yield_numeric"]):
                out["mapping_status"] = "matched_missing_yield"
            else:
                out["mapping_status"] = "matched_valid_yield"
                out["yieldPerAcre"] = float(gr["yield_numeric"])
    image_rows.append(out)
image_map = pd.DataFrame(image_rows)
image_map.to_csv(REPORT_DIR / "full_image_mapping_audit.csv", index=False)

# Unique image plot keys and TP completeness.
plot_records = []
if not parsed.empty:
    for key, group in parsed.groupby(KEYS, dropna=False, sort=True):
        # Grouping yields a tuple for four keys.
        key = tuple(key) if isinstance(key, tuple) else (key,)
        kmap = dict(zip(KEYS, key))
        counts = group["timepoint"].value_counts().to_dict()
        candidates = lookup.get(key, [])
        match_count = len(candidates)
        target = pd.NA
        csv_record = pd.NA
        csv_index = pd.NA
        mapping_status = "no_csv_match" if match_count == 0 else "ambiguous_csv_match"
        if match_count == 1:
            gr = gt.loc[candidates[0]]
            csv_record = int(gr["csv_record_number"])
            if C_INDEX:
                csv_index = gr[C_INDEX]
            if pd.isna(gr["yield_numeric"]):
                mapping_status = "matched_missing_yield"
            else:
                mapping_status = "matched_valid_yield"
                target = float(gr["yield_numeric"])
        exact_tp = all(counts.get(tp, 0) == 1 for tp in TIMEPOINTS) and set(counts).issubset(set(TIMEPOINTS))
        row = {
            **kmap,
            "location_raw": group["location_raw"].iloc[0],
            "experiment_raw": group["experiment_raw"].iloc[0],
            "range_raw": group["range_raw"].iloc[0],
            "row_raw": group["row_raw"].iloc[0],
            "image_count": len(group),
            "TP1_image_count": counts.get("TP1", 0),
            "TP2_image_count": counts.get("TP2", 0),
            "TP3_image_count": counts.get("TP3", 0),
            "exactly_one_image_each_tp": exact_tp,
            "csv_match_count": match_count,
            "csv_record_number": csv_record,
            "csv_index_value": csv_index,
            "mapping_status": mapping_status,
            "yieldPerAcre": target,
            "eligible_for_multitemporal_supervised_training": bool(exact_tp and match_count == 1 and pd.notna(target)),
        }
        # Partial-key diagnostics for unmatched keys.
        ewr_candidates = lookup_ewr.get((kmap["experiment_key"], kmap["range_key"], kmap["row_key"]), [])
        ler_candidates = lookup_ler.get((kmap["location_key"], kmap["experiment_key"], kmap["row_key"]), [])
        lxr_candidates = lookup_lxr.get((kmap["location_key"], kmap["range_key"], kmap["row_key"]), [])
        lex_candidates = lookup_lex.get((kmap["location_key"], kmap["experiment_key"], kmap["range_key"]), [])
        row["candidates_same_exp_range_row_ignoring_location"] = len(ewr_candidates)
        row["candidate_locations_ignoring_location"] = csv_list(sorted({str(gt.loc[i, C_LOC]) for i in ewr_candidates}))
        row["candidates_same_location_exp_row_ignoring_range"] = len(ler_candidates)
        row["candidates_same_location_range_row_ignoring_experiment"] = len(lxr_candidates)
        row["candidates_same_location_exp_range_ignoring_row"] = len(lex_candidates)
        row["candidate_csv_record_numbers_ignoring_location"] = csv_list([int(gt.loc[i, "csv_record_number"]) for i in ewr_candidates])
        plot_records.append(row)

plots = pd.DataFrame(plot_records)
plots.to_csv(REPORT_DIR / "plot_level_verification.csv", index=False)

# Precise image-level lists.
image_map.loc[image_map["mapping_status"].eq("no_csv_match")].to_csv(REPORT_DIR / "image_files_without_csv_match.csv", index=False)
image_map.loc[image_map["mapping_status"].eq("ambiguous_csv_match")].to_csv(REPORT_DIR / "images_with_ambiguous_csv_match.csv", index=False)
plots.loc[plots["mapping_status"].eq("no_csv_match")].to_csv(REPORT_DIR / "plot_keys_without_csv_match.csv", index=False)
plots.loc[plots["mapping_status"].eq("matched_missing_yield")].to_csv(REPORT_DIR / "matched_plots_with_missing_yield.csv", index=False)
plots.loc[~plots["exactly_one_image_each_tp"]].to_csv(REPORT_DIR / "plots_with_incomplete_or_duplicate_timepoints.csv", index=False)
plots.loc[plots["mapping_status"].eq("no_csv_match") & (plots["candidates_same_exp_range_row_ignoring_location"] > 0)].to_csv(REPORT_DIR / "unmatched_plots_with_possible_location_mismatch.csv", index=False)

# CSV rows whose normalized key does not exist among any parsed UAV filenames.
image_key_set = set(plots[KEYS].itertuples(index=False, name=None)) if not plots.empty else set()
gt["has_exact_uav_plot_key"] = gt.apply(lambda r: tuple(r[c] for c in KEYS) in image_key_set and bool(tuple(r[c] for c in KEYS)), axis=1)
gt.to_csv(REPORT_DIR / "all_ground_truth_rows_with_image_presence.csv", index=False)
gt.loc[~gt["has_exact_uav_plot_key"]].to_csv(REPORT_DIR / "csv_rows_without_exact_uav_match.csv", index=False)
missing_yield = gt.loc[gt["yield_numeric"].isna()].copy()
missing_yield.to_csv(REPORT_DIR / "all_csv_rows_missing_or_invalid_yield.csv", index=False)
missing_yield.loc[missing_yield["has_exact_uav_plot_key"]].to_csv(REPORT_DIR / "missing_yield_rows_with_uav_images.csv", index=False)
missing_yield.loc[~missing_yield["has_exact_uav_plot_key"]].to_csv(REPORT_DIR / "missing_yield_rows_without_uav_images.csv", index=False)

# Eligible manifest with explicit paths per time point.
eligible = plots.loc[plots["eligible_for_multitemporal_supervised_training"]].copy()
manifest_rows = []
for _, pr in eligible.iterrows():
    mask = True
    for col in KEYS:
        mask &= parsed[col].eq(pr[col])
    pg = parsed.loc[mask]
    paths = {}
    for tp in TIMEPOINTS:
        match_paths = pg.loc[pg["timepoint"].eq(tp), "image_path"].tolist()
        paths[f"{tp}_image_path"] = match_paths[0] if len(match_paths) == 1 else ""
    manifest_rows.append({**{c: pr[c] for c in KEYS}, "location_raw": pr["location_raw"],
                          **paths, "csv_record_number": pr["csv_record_number"],
                          "csv_index_value": pr["csv_index_value"], "yieldPerAcre": pr["yieldPerAcre"]})
pd.DataFrame(manifest_rows).to_csv(REPORT_DIR / "verified_multitemporal_training_manifest.csv", index=False)

# Counts per location and time point.
by_location_tp = parsed.groupby(["location_raw", "timepoint"]).size().unstack(fill_value=0).reindex(columns=TIMEPOINTS, fill_value=0)
by_location_tp.to_csv(REPORT_DIR / "image_counts_by_location_timepoint.csv")

# ------------------------------- Summary -------------------------------
banner("DATASET LOCATIONS")
print(f"Dataset root : {ROOT}")
print(f"UAV folder   : {UAV_ROOT}")
print(f"Ground-truth : {CSV_PATH}")
print(f"Reports      : {REPORT_DIR}")

banner("IMAGE AND PLOT COUNTS")
print(f"PNG images discovered                         : {len(images)}")
print(f"Filenames successfully parsed                : {int(images['filename_parsed'].sum())}")
print(f"Unique UAV plot keys (location+exp+range+row) : {len(plots)}")
print(f"Plots with exactly one TP1, TP2 and TP3       : {int(plots['exactly_one_image_each_tp'].sum())}")
print(f"Plots with incomplete/duplicate timepoints    : {int((~plots['exactly_one_image_each_tp']).sum())}")

banner("EXACT IMAGE-TO-CSV MATCH")
print(f"Ground-truth CSV rows                         : {len(gt)}")
print(f"Rows with valid join keys                     : {int(gt['valid_join_key'].sum())}")
print(f"Rows with invalid/incomplete join keys        : {int((~gt['valid_join_key']).sum())}")
print(f"CSV rows involved in duplicated normalized keys: {int(dup_gt['csv_record_number'].nunique())}")
print(f"Duplicate normalized key groups               : {int(dup_gt[KEYS].drop_duplicates().shape[0])}")
print(f"Image plot keys with unique CSV match         : {int(plots['csv_match_count'].eq(1).sum())}")
print(f"Image plot keys with no CSV match             : {int(plots['csv_match_count'].eq(0).sum())}")
print(f"Image plot keys with ambiguous CSV match      : {int(plots['csv_match_count'].gt(1).sum())}")
print(f"CSV rows with no exact UAV plot key            : {int((~gt['has_exact_uav_plot_key']).sum())}")

banner("YIELD LABEL COUNTS")
print(f"CSV rows with numeric yield                   : {int(gt['yield_numeric'].notna().sum())}")
print(f"CSV rows with missing/non-numeric yield       : {int(gt['yield_numeric'].isna().sum())}")
print(f"Missing-yield rows whose key occurs in UAV    : {int(missing_yield['has_exact_uav_plot_key'].sum())}")
print(f"Missing-yield rows whose key does not occur   : {int((~missing_yield['has_exact_uav_plot_key']).sum())}")

banner("FINAL MULTI-TEMPORAL ELIGIBILITY")
print(f"Plots with exactly 3 images + unique CSV match + numeric yield: {len(eligible)}")
print(f"Eligible image files expected (plots x 3)                    : {len(eligible) * 3}")
print("By location:")
if not eligible.empty:
    print(eligible.groupby("location_raw").size().to_string())
else:
    print("No eligible plots found")

banner("UNMATCHED KEY DIAGNOSTIC")
no_match = plots.loc[plots["mapping_status"].eq("no_csv_match")]
print(f"Unmatched image plot keys with exp/range/row present somewhere in CSV (location ignored): {int((no_match['candidates_same_exp_range_row_ignoring_location'] > 0).sum())}")
print(f"Unmatched image plot keys with no exp/range/row candidate anywhere in CSV: {int((no_match['candidates_same_exp_range_row_ignoring_location'] == 0).sum())}")
print("See unmatched_plots_with_possible_location_mismatch.csv for the partial-key candidates.")

banner("RECONCILIATION CHECKS")
print(f"Unique UAV plot keys = unique CSV-matched keys + unmatched + ambiguous? : {len(plots)} = {int(plots['csv_match_count'].eq(1).sum())} + {int(plots['csv_match_count'].eq(0).sum())} + {int(plots['csv_match_count'].gt(1).sum())}")
print(f"Images = {len(images)}; unique plot keys x 3 would equal {len(plots) * 3} only if every plot has exactly 3 images.")
print("NOTE: A partial-key candidate is diagnostic only; it is not accepted as a valid match automatically.")
print("Audit complete. Original images and CSV were not modified.")
