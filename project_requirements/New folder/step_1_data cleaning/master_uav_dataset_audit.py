"""Master audit for a UAV-based maize yield prediction dataset.

What this script does
---------------------
1. Finds the dataset folder automatically.
2. Scans all PNGs, parses filenames, checks image readability and dimensions.
3. Counts plots and verifies one TP1, TP2, TP3 image per plot.
4. Audits ground-truth yield labels and duplicate join keys.
5. Performs conservative full-key mapping using location + experiment + range + row.
6. Applies ONE documented spelling alias for Missouri Valley: Hyrbrids <-> Hybrids.
7. Reports possible alternate matches (e.g., same location/range/row but different or missing
   experiment) WITHOUT accepting those matches automatically.
8. Creates a supervised multi-temporal manifest only from plots that have three images,
   a unique full-key match, a numeric yield label, and no CSV row reused by another plot.
9. Writes detailed CSV reports and a plain-text summary. It never changes original data.

Expected layout:
  project_folder/\n    master_uav_dataset_audit.py\n    drone_dataset/\n      UAV/ ... (or UAV_Dataset/)\n      GroundTruth/HYBRID_HIPS_V3.5_ALLPLOTS.csv

Requires: pandas, numpy, Pillow
Install if needed: python -m pip install pandas numpy pillow
"""

from __future__ import annotations

from collections import defaultdict, Counter
from pathlib import Path
import hashlib
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

# ----------------------------- Configuration -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_CANDIDATES = [SCRIPT_DIR / "drone_dataset", SCRIPT_DIR]
CSV_RELATIVE_PATH = Path("GroundTruth") / "HYBRID_HIPS_V3.5_ALLPLOTS.csv"
REPORT_FOLDER_NAME = "master_audit_reports"
TIMEPOINTS = ("TP1", "TP2", "TP3")

# This alias is narrowly scoped to the reported Missouri Valley typo.
# It does NOT fuzzy-match other experiment IDs.
MO_VALLEY_EXPERIMENT_ALIASES = {"hyrbrids": "hybrids", "hybrids": "hybrids"}

# Checks full image bytes for exact duplicate files. Set False to make the scan faster.
CHECK_SHA256_DUPLICATES = True

FILENAME_PATTERN = re.compile(
    r"^(?P<location>.+?)-(?P<timepoint>TP[123])-"
    r"(?P<experiment>[^_]+)_(?P<range>[^_]+)_(?P<row>[^_]+)$",
    re.IGNORECASE,
)

KEY_COLS = ["location_key", "experiment_key", "range_key", "row_key"]
BASE_KEY_COLS = ["location_key", "experiment_base_key", "range_key", "row_key"]


def locate_dataset() -> tuple[Path, Path, Path]:
    for root in DATASET_CANDIDATES:
        uav = root / "UAV"
        if not uav.is_dir():
            uav = root / "UAV_Dataset"
        csv_path = root / CSV_RELATIVE_PATH
        if uav.is_dir() and csv_path.is_file():
            return root, uav, csv_path
    expected = (
        f"{SCRIPT_DIR / 'drone_dataset' / 'UAV'} (or UAV_Dataset) and "
        f"{SCRIPT_DIR / CSV_RELATIVE_PATH}"
    )
    raise FileNotFoundError(
        "Could not find dataset. Expected " + expected +
        ". Edit DATASET_CANDIDATES or CSV_RELATIVE_PATH near the top of the script."
    )


def missingish(value: Any) -> bool:
    if pd.isna(value):
        return True
    s = str(value).strip().lower()
    return s in {"", "nan", "none", "null", "<na>"}


def norm_location(value: Any) -> str:
    if missingish(value):
        return ""
    s = re.sub(r"[^a-z0-9]", "", str(value).strip().lower())
    if not s:
        return ""
    if s.startswith(("movalley", "missourivalley", "missourivally")):
        return "movalley"
    for known in ("crawfordsville", "scottsbluff", "ames", "lincoln"):
        if s.startswith(known):
            return known
    return s


def norm_id(value: Any) -> str:
    if missingish(value):
        return ""
    s = str(value).strip()
    # CSV numeric columns may load integers as floats (e.g. 4231.0).
    if re.fullmatch(r"[+-]?\d+(?:\.0+)?", s):
        try:
            return str(int(float(s)))
        except (ValueError, OverflowError):
            pass
    return re.sub(r"\s+", "", s).lower()


def norm_experiment_base(value: Any) -> str:
    """Normal experiment normalization, with no typo correction."""
    return norm_id(value)


def norm_experiment_alias(value: Any, location_key: str) -> str:
    """Apply only the documented Missouri Valley Hyrbrids/Hybrids alias."""
    base = norm_experiment_base(value)
    if location_key == "movalley" and base in MO_VALLEY_EXPERIMENT_ALIASES:
        return MO_VALLEY_EXPERIMENT_ALIASES[base]
    return base


def norm_key_tuple(row: pd.Series, columns: list[str]) -> tuple[str, ...]:
    return tuple(str(row[c]) for c in columns)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def counts_table(series: pd.Series, name: str = "count") -> pd.DataFrame:
    return series.value_counts(dropna=False).rename_axis("value").reset_index(name=name)


def print_header(text: str) -> None:
    print("\n" + "=" * 18 + f" {text} " + "=" * 18)


def main() -> int:
    try:
        ROOT, UAV_ROOT, CSV_PATH = locate_dataset()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    REPORT_DIR = ROOT / REPORT_FOLDER_NAME
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------- Load CSV -----------------------------
    gt = pd.read_csv(CSV_PATH, low_memory=False)
    gt.columns = [str(c).strip() for c in gt.columns]
    lookup_cols = {c.lower(): c for c in gt.columns}
    required = ["location", "experiment", "range", "row", "yieldperacre"]
    absent = [c for c in required if c not in lookup_cols]
    if absent:
        print(f"ERROR: Ground-truth CSV is missing columns: {absent}")
        print("Available columns:", gt.columns.tolist())
        return 1

    c_loc = lookup_cols["location"]
    c_exp = lookup_cols["experiment"]
    c_range = lookup_cols["range"]
    c_row = lookup_cols["row"]
    c_yield = lookup_cols["yieldperacre"]
    c_plot = lookup_cols.get("plotnumber")
    c_qr = lookup_cols.get("qrcode")
    c_index = lookup_cols.get("index")

    gt = gt.copy()
    # Spreadsheet row number (CSV header is line 1).
    gt["csv_record_number"] = np.arange(2, len(gt) + 2)
    gt["yield_numeric"] = pd.to_numeric(gt[c_yield], errors="coerce")
    gt["location_key"] = gt[c_loc].map(norm_location)
    gt["experiment_base_key"] = gt[c_exp].map(norm_experiment_base)
    gt["experiment_key"] = [
        norm_experiment_alias(exp, loc)
        for exp, loc in zip(gt[c_exp], gt["location_key"])
    ]
    gt["range_key"] = gt[c_range].map(norm_id)
    gt["row_key"] = gt[c_row].map(norm_id)
    gt["valid_full_key_base"] = gt[BASE_KEY_COLS].ne("").all(axis=1)
    gt["valid_full_key_alias"] = gt[KEY_COLS].ne("").all(axis=1)

    valid_gt_base = gt.loc[gt["valid_full_key_base"]].copy()
    valid_gt_alias = gt.loc[gt["valid_full_key_alias"]].copy()

    # Lookups return lists so ambiguous keys are never silently chosen.
    lookup_base: dict[tuple[str, ...], list[int]] = defaultdict(list)
    lookup_alias: dict[tuple[str, ...], list[int]] = defaultdict(list)
    lookup_loc_range_row: dict[tuple[str, ...], list[int]] = defaultdict(list)
    lookup_exp_range_row: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx, r in valid_gt_base.iterrows():
        lookup_base[norm_key_tuple(r, BASE_KEY_COLS)].append(idx)
    for idx, r in valid_gt_alias.iterrows():
        lookup_alias[norm_key_tuple(r, KEY_COLS)].append(idx)
        lookup_exp_range_row[(r["experiment_base_key"], r["range_key"], r["row_key"])].append(idx)
    # Diagnostic candidate lookup deliberately allows a missing experiment value,
    # so records such as the reported Scottsbluff rows can still be surfaced for review.
    for idx, r in gt.iterrows():
        if r["location_key"] and r["range_key"] and r["row_key"]:
            lookup_loc_range_row[(r["location_key"], r["range_key"], r["row_key"])].append(idx)

    duplicate_base = valid_gt_base.loc[valid_gt_base.duplicated(BASE_KEY_COLS, keep=False)].copy()
    duplicate_alias = valid_gt_alias.loc[valid_gt_alias.duplicated(KEY_COLS, keep=False)].copy()
    save_csv(duplicate_base, REPORT_DIR / "duplicate_csv_keys_before_alias.csv")
    save_csv(duplicate_alias, REPORT_DIR / "duplicate_csv_keys_after_alias.csv")

    # ----------------------------- Scan images -----------------------------
    paths = sorted(p for p in UAV_ROOT.rglob("*") if p.is_file() and p.suffix.lower() == ".png")
    image_rows: list[dict[str, Any]] = []
    for i, path in enumerate(paths, start=1):
        m = FILENAME_PATTERN.match(path.stem)
        rec: dict[str, Any] = {
            "image_path": str(path.relative_to(ROOT)),
            "filename": path.name,
            "filename_parsed": bool(m),
            "image_readable": False,
            "image_width": pd.NA,
            "image_height": pd.NA,
            "image_mode": "",
            "all_black": pd.NA,
            "black_pixel_percent": pd.NA,
            "sha256": "",
            "image_error": "",
            "location_raw": "",
            "timepoint": "",
            "experiment_raw": "",
            "range_raw": "",
            "row_raw": "",
            "location_key": "",
            "experiment_base_key": "",
            "experiment_key": "",
            "range_key": "",
            "row_key": "",
        }
        if m:
            d = m.groupdict()
            loc_key = norm_location(d["location"])
            rec.update({
                "location_raw": d["location"],
                "timepoint": d["timepoint"].upper(),
                "experiment_raw": d["experiment"],
                "range_raw": d["range"],
                "row_raw": d["row"],
                "location_key": loc_key,
                "experiment_base_key": norm_experiment_base(d["experiment"]),
                "experiment_key": norm_experiment_alias(d["experiment"], loc_key),
                "range_key": norm_id(d["range"]),
                "row_key": norm_id(d["row"]),
            })
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                rec["image_width"], rec["image_height"] = im.size
                rec["image_mode"] = im.mode
                rgb = np.asarray(im.convert("RGB"))
            rec["image_readable"] = bool(rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.shape[0] > 0 and rgb.shape[1] > 0)
            black_mask = np.all(rgb == 0, axis=2)
            rec["all_black"] = bool(black_mask.all())
            rec["black_pixel_percent"] = round(float(black_mask.mean() * 100), 2)
        except Exception as exc:
            rec["image_error"] = f"{type(exc).__name__}: {exc}"
        if CHECK_SHA256_DUPLICATES:
            try:
                rec["sha256"] = sha256_file(path)
            except Exception as exc:
                rec["image_error"] = (rec["image_error"] + " | " if rec["image_error"] else "") + f"SHA256: {exc}"
        image_rows.append(rec)
        if i % 500 == 0:
            print(f"Scanned {i}/{len(paths)} images")

    images = pd.DataFrame(image_rows)
    parsed = images.loc[images["filename_parsed"]].copy()
    save_csv(images, REPORT_DIR / "image_integrity_and_filename_audit.csv")
    save_csv(images.loc[~images["image_readable"]], REPORT_DIR / "unreadable_images.csv")
    save_csv(images.loc[~images["filename_parsed"]], REPORT_DIR / "unparsed_filenames.csv")
    save_csv(images.loc[images["all_black"].eq(True)], REPORT_DIR / "completely_black_images.csv")
    if CHECK_SHA256_DUPLICATES:
        valid_hash = images.loc[images["sha256"].ne("")]
        dup_hashes = valid_hash.loc[valid_hash.duplicated("sha256", keep=False)].copy()
        save_csv(dup_hashes, REPORT_DIR / "possible_exact_duplicate_images.csv")
    else:
        dup_hashes = pd.DataFrame()

    # ----------------------------- Plot keys and matches -----------------------------
    plot_rows: list[dict[str, Any]] = []
    for key, group in parsed.groupby(KEY_COLS, dropna=False, sort=True):
        key = tuple(key) if isinstance(key, tuple) else (key,)
        kmap = dict(zip(KEY_COLS, key))
        counts = group["timepoint"].value_counts().to_dict()
        tp_complete = all(counts.get(tp, 0) == 1 for tp in TIMEPOINTS) and set(counts).issubset(set(TIMEPOINTS))

        # Find canonical alias-aware full-key match. Preserve base status too.
        base_key = (
            kmap["location_key"],
            group["experiment_base_key"].iloc[0],
            kmap["range_key"],
            kmap["row_key"],
        )
        base_candidates = lookup_base.get(base_key, [])
        candidates = lookup_alias.get(tuple(kmap[c] for c in KEY_COLS), [])
        matched_idx = candidates[0] if len(candidates) == 1 else None
        gtrow = gt.loc[matched_idx] if matched_idx is not None else None
        base_match_count = len(base_candidates)
        alias_match_count = len(candidates)
        used_alias = bool(alias_match_count == 1 and base_match_count == 0)
        yield_value = float(gtrow["yield_numeric"]) if gtrow is not None and pd.notna(gtrow["yield_numeric"]) else np.nan
        if alias_match_count == 0:
            status = "no_full_key_match"
        elif alias_match_count > 1:
            status = "ambiguous_full_key_match"
        elif pd.isna(yield_value):
            status = "matched_but_yield_missing"
        elif not tp_complete:
            status = "matched_yield_but_timepoints_incomplete"
        else:
            status = "eligible_full_key_match"

        # Diagnostic alternative: same location + range + row, ignoring experiment.
        alt_key = (kmap["location_key"], kmap["range_key"], kmap["row_key"])
        alt = lookup_loc_range_row.get(alt_key, [])
        # Only needed to review un/ambiguous full-key cases; never auto-accepted.
        alt_count = len(alt)
        alt_candidate_details = []
        for ai in alt:
            ar = gt.loc[ai]
            alt_candidate_details.append(
                f"record={int(ar['csv_record_number'])};experiment={ar[c_exp]!s};yield={ar['yield_numeric']!s};"
                f"base_key_already_matches_image={len(lookup_base.get(norm_key_tuple(ar, BASE_KEY_COLS), [])) > 0}"
            )
        alt_yield_count = sum(pd.notna(gt.loc[ai, "yield_numeric"]) for ai in alt)
        if gtrow is not None:
            csv_record_number = int(gtrow["csv_record_number"])
            csv_index_value = gtrow[c_index] if c_index else pd.NA
            plot_number = gtrow[c_plot] if c_plot else pd.NA
            qr_code = gtrow[c_qr] if c_qr else pd.NA
        else:
            csv_record_number = pd.NA
            csv_index_value = pd.NA
            plot_number = pd.NA
            qr_code = pd.NA
        plot_rows.append({
            **kmap,
            "location_raw": group["location_raw"].iloc[0],
            "experiment_raw": group["experiment_raw"].iloc[0],
            "range_raw": group["range_raw"].iloc[0],
            "row_raw": group["row_raw"].iloc[0],
            "image_count": len(group),
            "TP1_count": counts.get("TP1", 0),
            "TP2_count": counts.get("TP2", 0),
            "TP3_count": counts.get("TP3", 0),
            "one_each_TP1_TP2_TP3": tp_complete,
            "exact_match_count_before_MOValley_alias": base_match_count,
            "full_key_csv_match_count_after_alias": alias_match_count,
            "mapping_status": status,
            "matched_by_MOValley_alias": used_alias,
            "csv_record_number": csv_record_number,
            "csv_index_value": csv_index_value,
            "csv_plotNumber": plot_number,
            "csv_qrCode": qr_code,
            "yieldPerAcre": yield_value,
            "alternate_same_location_range_row_candidate_count_ignoring_experiment": alt_count,
            "alternate_same_location_range_row_candidates_with_numeric_yield": int(alt_yield_count),
            "alternate_candidate_details_REVIEW_ONLY": " || ".join(alt_candidate_details),
            "eligible_for_supervised_multitemporal_training": bool(tp_complete and alias_match_count == 1 and pd.notna(yield_value)),
        })

    plots = pd.DataFrame(plot_rows)
    if not plots.empty:
        save_csv(plots, REPORT_DIR / "plot_level_master_audit.csv")
        save_csv(plots.loc[plots["mapping_status"].eq("no_full_key_match")], REPORT_DIR / "unmatched_plots_after_MOValley_alias.csv")
        save_csv(plots.loc[plots["mapping_status"].eq("ambiguous_full_key_match")], REPORT_DIR / "ambiguous_plot_matches.csv")
        save_csv(plots.loc[plots["mapping_status"].eq("matched_but_yield_missing")], REPORT_DIR / "matched_plots_missing_yield.csv")
        save_csv(plots.loc[plots["matched_by_MOValley_alias"]], REPORT_DIR / "MOValley_alias_recovery_report.csv")
        save_csv(
            plots.loc[(plots["full_key_csv_match_count_after_alias"].eq(0)) & (plots["alternate_same_location_range_row_candidate_count_ignoring_experiment"] > 0)],
            REPORT_DIR / "alternate_matches_ignoring_experiment_REVIEW_ONLY.csv"
        )
    else:
        plots.to_csv(REPORT_DIR / "plot_level_master_audit.csv", index=False)

    # Image-level mapping report: inherits plot-level status/key match.
    plot_lookup = {tuple(r[c] for c in KEY_COLS): r for _, r in plots.iterrows()}
    im_out = []
    for _, ir in images.iterrows():
        rec = ir.to_dict()
        if ir["filename_parsed"]:
            pk = tuple(str(ir[c]) for c in KEY_COLS)
            pr = plot_lookup.get(pk)
            if pr is not None:
                rec.update({
                    "mapping_status": pr["mapping_status"],
                    "csv_record_number": pr["csv_record_number"],
                    "yieldPerAcre": pr["yieldPerAcre"],
                    "matched_by_MOValley_alias": pr["matched_by_MOValley_alias"],
                })
            else:
                rec.update({"mapping_status": "plot_key_not_found_in_plot_audit", "csv_record_number": pd.NA, "yieldPerAcre": pd.NA, "matched_by_MOValley_alias": False})
        else:
            rec.update({"mapping_status": "unparsed_filename", "csv_record_number": pd.NA, "yieldPerAcre": pd.NA, "matched_by_MOValley_alias": False})
        im_out.append(rec)
    image_map = pd.DataFrame(im_out)
    save_csv(image_map, REPORT_DIR / "image_level_master_mapping_audit.csv")

    # ----------------------------- CSV-level presence and yield reports -----------------------------
    plot_key_set = set(tuple(str(v) for v in r) for r in plots[KEY_COLS].itertuples(index=False, name=None)) if not plots.empty else set()
    gt["has_full_key_in_uav_after_alias"] = gt.apply(
        lambda r: bool(r["valid_full_key_alias"] and tuple(str(r[c]) for c in KEY_COLS) in plot_key_set), axis=1
    )
    # Count CSV row mapping based on the alias-aware key lookup. A duplicated key is flagged as ambiguous.
    gt["image_key_match_count_after_alias"] = [len(lookup_alias.get(norm_key_tuple(r, KEY_COLS), [])) if bool(r["valid_full_key_alias"]) else 0 for _, r in gt.iterrows()]
    save_csv(gt, REPORT_DIR / "all_ground_truth_rows_audited.csv")
    save_csv(gt.loc[~gt["has_full_key_in_uav_after_alias"]], REPORT_DIR / "csv_rows_without_full_key_image_match.csv")
    missing_yield = gt.loc[gt["yield_numeric"].isna()].copy()
    save_csv(missing_yield, REPORT_DIR / "all_rows_missing_or_invalid_yield.csv")
    save_csv(missing_yield.loc[missing_yield["has_full_key_in_uav_after_alias"]], REPORT_DIR / "missing_yield_rows_with_full_key_uav_match.csv")
    save_csv(missing_yield.loc[~missing_yield["has_full_key_in_uav_after_alias"]], REPORT_DIR / "missing_yield_rows_without_full_key_uav_match.csv")

    # Exact eligible manifest: unique full-key matched CSV record, numeric yield, one image per TP.
    eligible = plots.loc[plots["eligible_for_supervised_multitemporal_training"]].copy()
    # Safeguard against reusing one CSV row for multiple plot examples.
    duplicate_csv_usage = eligible["csv_record_number"].duplicated(keep=False)
    reused_rows = eligible.loc[duplicate_csv_usage].copy()
    save_csv(reused_rows, REPORT_DIR / "eligible_plots_reusing_csv_row_REVIEW_REQUIRED.csv")
    if not reused_rows.empty:
        eligible = eligible.loc[~eligible["csv_record_number"].isin(reused_rows["csv_record_number"])].copy()

    manifest_rows = []
    for _, pr in eligible.iterrows():
        keymask = pd.Series(True, index=parsed.index)
        for c in KEY_COLS:
            keymask &= parsed[c].astype(str).eq(str(pr[c]))
        group = parsed.loc[keymask]
        paths_for_tp = {}
        ok = True
        for tp in TIMEPOINTS:
            p = group.loc[group["timepoint"].eq(tp), "image_path"].tolist()
            paths_for_tp[f"{tp}_image_path"] = p[0] if len(p) == 1 else ""
            if len(p) != 1:
                ok = False
        if ok:
            manifest_rows.append({
                "location": pr["location_raw"],
                "location_key": pr["location_key"],
                "experiment": pr["experiment_raw"],
                "experiment_key": pr["experiment_key"],
                "range": pr["range_raw"],
                "row": pr["row_raw"],
                **paths_for_tp,
                "csv_record_number": pr["csv_record_number"],
                "csv_index_value": pr["csv_index_value"],
                "yieldPerAcre": pr["yieldPerAcre"],
                "matched_by_MOValley_alias": pr["matched_by_MOValley_alias"],
            })
    manifest = pd.DataFrame(manifest_rows)
    save_csv(manifest, REPORT_DIR / "verified_multitemporal_training_manifest.csv")

    # Plot counts and image counts by location/timepoint.
    if not parsed.empty:
        image_counts = parsed.groupby(["location_raw", "timepoint"]).size().unstack(fill_value=0).reindex(columns=TIMEPOINTS, fill_value=0)
        image_counts.to_csv(REPORT_DIR / "image_counts_by_location_timepoint.csv", encoding="utf-8-sig")
    valid_key_mask = gt["valid_full_key_alias"]
    missing_yield_with_match = int(missing_yield["has_full_key_in_uav_after_alias"].sum())
    missing_yield_without_match = int((~missing_yield["has_full_key_in_uav_after_alias"]).sum())
    base_match_plot_count = int(plots["exact_match_count_before_MOValley_alias"].eq(1).sum()) if not plots.empty else 0
    alias_recovered_plot_count = int(plots["matched_by_MOValley_alias"].sum()) if not plots.empty else 0
    match_counts = plots["full_key_csv_match_count_after_alias"].value_counts() if not plots.empty else pd.Series(dtype=int)
    matched_unique = int((plots["full_key_csv_match_count_after_alias"] == 1).sum()) if not plots.empty else 0
    unmatched_plots_count = int((plots["full_key_csv_match_count_after_alias"] == 0).sum()) if not plots.empty else 0
    ambiguous_plots_count = int((plots["full_key_csv_match_count_after_alias"] > 1).sum()) if not plots.empty else 0
    eligible_by_loc = eligible.groupby("location_raw").size().to_dict() if not eligible.empty else {}

    # ----------------------------- Console summary -----------------------------
    lines = []
    lines.append("MASTER UAV DATASET AUDIT — SUMMARY")
    lines.append(f"Dataset root: {ROOT}")
    lines.append(f"UAV folder: {UAV_ROOT}")
    lines.append(f"Ground-truth CSV: {CSV_PATH}")
    lines.append(f"Reports folder: {REPORT_DIR}")
    lines.append("")
    lines.append("IMAGE INTEGRITY")
    lines.append(f"PNG files discovered: {len(images)}")
    lines.append(f"Filenames parsed: {int(images['filename_parsed'].sum())}")
    lines.append(f"Unreadable images: {int((~images['image_readable']).sum())}")
    lines.append(f"Completely black images: {int(images['all_black'].eq(True).sum())}")
    lines.append(f"Exact duplicate image files by SHA-256: {int(len(dup_hashes)) if CHECK_SHA256_DUPLICATES else 'not checked'}")
    lines.append("")
    lines.append("IMAGE PLOTS AND TIME POINTS")
    lines.append(f"Unique UAV plot keys after location/experiment alias normalization: {len(plots)}")
    lines.append(f"Plots with exactly one TP1, TP2 and TP3: {int(plots['one_each_TP1_TP2_TP3'].sum()) if not plots.empty else 0}")
    lines.append(f"Plots with incomplete/duplicate timepoints: {int((~plots['one_each_TP1_TP2_TP3']).sum()) if not plots.empty else 0}")
    lines.append("")
    lines.append("GROUND-TRUTH CSV")
    lines.append(f"CSV rows: {len(gt)}")
    lines.append(f"Rows with all key fields before alias (location/experiment/range/row): {int(gt['valid_full_key_base'].sum())}")
    lines.append(f"Rows with incomplete key fields: {int((~gt['valid_full_key_base']).sum())}")
    lines.append(f"Numeric yield labels: {int(gt['yield_numeric'].notna().sum())}")
    lines.append(f"Missing/non-numeric yield labels: {int(gt['yield_numeric'].isna().sum())}")
    lines.append(f"Duplicate full-key CSV rows before alias: {int(duplicate_base['csv_record_number'].nunique())}")
    lines.append(f"Duplicate full-key CSV rows after alias: {int(duplicate_alias['csv_record_number'].nunique())}")
    lines.append("")
    lines.append("MATCHING — BASE RULE VS DOCUMENTED MISSOURI VALLEY ALIAS")
    lines.append(f"Unique plot keys matched before spelling alias: {base_match_plot_count}")
    lines.append(f"Additional plots matched by Missouri Valley Hyrbrids/Hybrids alias: {alias_recovered_plot_count}")
    lines.append(f"Unique plot keys matched after alias: {matched_unique}")
    lines.append(f"Image plot keys still without a full-key CSV match: {unmatched_plots_count}")
    lines.append(f"Image plot keys with ambiguous CSV matches: {ambiguous_plots_count}")
    lines.append(f"CSV rows with no full-key image match after alias: {int((~gt['has_full_key_in_uav_after_alias']).sum())}")
    lines.append("")
    lines.append("YIELD AND TRAINING ELIGIBILITY")
    lines.append(f"Plots matched to one CSV row with numeric yield and 3 images: {len(eligible)}")
    lines.append(f"Image files in verified multi-temporal manifest: {len(manifest) * 3}")
    lines.append(f"Matched plots with missing/invalid yield: {int((plots['mapping_status'] == 'matched_but_yield_missing').sum()) if not plots.empty else 0}")
    lines.append(f"Missing-yield CSV rows with full-key UAV match: {missing_yield_with_match}")
    lines.append(f"Missing-yield CSV rows without full-key UAV match: {missing_yield_without_match}")
    lines.append(f"Rows/plots with unique location+range+row candidate when experiment ignored (review only): {int(((plots['full_key_csv_match_count_after_alias'].eq(0)) & (plots['alternate_same_location_range_row_candidate_count_ignoring_experiment'].eq(1))).sum()) if not plots.empty else 0}")
    lines.append("Eligible plots by location:")
    for loc, n in sorted(eligible_by_loc.items()):
        lines.append(f"  {loc}: {n}")
    lines.append("")
    lines.append("IMPORTANT")
    lines.append("- Only the narrow Missouri Valley Hyrbrids/Hybrids alias is automatically applied.")
    lines.append("- Matches that ignore experiment are diagnostic candidates only and are NOT added to the training manifest.")
    lines.append("- Missing yield labels are never filled or invented.")
    lines.append("- Original images and CSV were not modified.")

    summary_text = "\n".join(lines) + "\n"
    (REPORT_DIR / "MASTER_AUDIT_SUMMARY.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print("Detailed reports were saved in:")
    print(REPORT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
