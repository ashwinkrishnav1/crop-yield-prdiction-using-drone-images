from pathlib import Path
import re
import hashlib

import numpy as np
import pandas as pd
from PIL import Image

# --------------------------------------------------
# 1. Set the dataset paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent / "drone_dataset"
UAV_ROOT = ROOT / "UAV"
CSV_PATH = ROOT / "GroundTruth" / "HYBRID_HIPS_V3.5_ALLPLOTS.csv"

if not UAV_ROOT.exists():
    raise FileNotFoundError(f"UAV folder not found: {UAV_ROOT}")

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Ground-truth CSV not found: {CSV_PATH}")

# Expected filename example:
# Crawfordsville-TP1-4351_3_15.PNG
filename_pattern = re.compile(
    r"^(?P<location>.+?)-(?P<timepoint>TP[123])-"
    r"(?P<experiment>[^_]+)_(?P<range>[^_]+)_(?P<row>[^_]+)$",
    flags=re.IGNORECASE
)

# --------------------------------------------------
# 2. Helper function to calculate file checksum
# --------------------------------------------------
def sha256_file(path):
    hasher = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


# --------------------------------------------------
# 3. Scan and check every image
# --------------------------------------------------
image_files = sorted(
    p for p in UAV_ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() == ".png"
)

records = []

for i, path in enumerate(image_files, start=1):
    record = {
        "image_path": str(path.relative_to(ROOT)),
        "filename": path.name,
        "filename_parsed": False,
        "valid_image": False,
        "width": None,
        "height": None,
        "mode": None,
        "all_black": None,
        "black_pixel_percent": None,
        "sha256": None,
        "error": None,
    }

    match = filename_pattern.match(path.stem)

    if match:
        record.update(match.groupdict())
        record["filename_parsed"] = True

    try:
        # Verify that the PNG file is structurally readable.
        with Image.open(path) as img:
            img.verify()

        # Reopen because verify() invalidates the original image object.
        with Image.open(path) as img:
            record["width"], record["height"] = img.size
            record["mode"] = img.mode

            # Convert to RGB for consistent pixel inspection.
            rgb = np.asarray(img.convert("RGB"))

        if record["width"] <= 0 or record["height"] <= 0:
            raise ValueError("Invalid image dimensions")

        record["valid_image"] = True

        # Black masking is expected in this dataset.
        black_mask = np.all(rgb == 0, axis=2)
        record["black_pixel_percent"] = round(
            float(black_mask.mean() * 100), 2
        )
        record["all_black"] = bool(black_mask.all())

        record["sha256"] = sha256_file(path)

    except Exception as exc:
        record["error"] = str(exc)

    records.append(record)

    if i % 500 == 0:
        print(f"Checked {i}/{len(image_files)} images")

images = pd.DataFrame(records)

images.to_csv(ROOT / "image_audit.csv", index=False)

# Save separate lists for inspection.
images.loc[~images["valid_image"]].to_csv(
    ROOT / "unreadable_images.csv", index=False
)

images.loc[~images["filename_parsed"]].to_csv(
    ROOT / "unparsed_filenames.csv", index=False
)

images.loc[images["all_black"].eq(True)].to_csv(
    ROOT / "all_black_images.csv", index=False
)

# --------------------------------------------------
# 4. Report image problems
# --------------------------------------------------
print("\n========== IMAGE AUDIT ==========")
print("Total PNG files:", len(images))
print("Unreadable images:", (~images["valid_image"]).sum())
print("Unparsed filenames:", (~images["filename_parsed"]).sum())
print("Completely black images:", images["all_black"].eq(True).sum())

parsed = images.loc[images["filename_parsed"]]

print("\nImage counts by location and time point:")
print(
    parsed.groupby(["location", "timepoint"], dropna=False)
    .size()
    .unstack(fill_value=0)
)

# Repeated file contents may indicate duplicate copies.
duplicate_files = images[
    images["sha256"].notna()
    & images.duplicated("sha256", keep=False)
]

duplicate_files.to_csv(ROOT / "possible_duplicate_images.csv", index=False)

print("\nImages involved in exact-file duplicates:",
      len(duplicate_files))

# --------------------------------------------------
# 5. Audit ground-truth CSV
# --------------------------------------------------
ground_truth = pd.read_csv(CSV_PATH, low_memory=False)
ground_truth.columns = ground_truth.columns.str.strip()

print("\n========== GROUND-TRUTH AUDIT ==========")
print("CSV rows:", len(ground_truth))
print("CSV columns:")
print(ground_truth.columns.tolist())

print("\nExact duplicate CSV rows:", ground_truth.duplicated().sum())

target_col = next(
    (c for c in ground_truth.columns
     if c.lower() == "yieldperacre"),
    None
)

if target_col is None:
    print("\nWARNING: yieldPerAcre column not found.")
else:
    yield_values = pd.to_numeric(
        ground_truth[target_col], errors="coerce"
    )

    print("Missing or non-numeric yield labels:",
          int(yield_values.isna().sum()))

    print("\nYield summary:")
    print(yield_values.describe())

print("\nAudit completed. Original data files were not modified.")