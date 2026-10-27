from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent

# Check for the extracted plot ID file (valid_plots.csv) first,
# with fallback to verified_multitemporal_training_manifest.csv
DATA_CANDIDATES = [
    ROOT / "drone_dataset" / "valid_plots.csv",
    ROOT / "valid_plots.csv",
    ROOT / "data cleaning" / "valid_plots.csv",
    ROOT / "drone_dataset" / "master_audit_reports" / "verified_multitemporal_training_manifest.csv",
]

DATA_FILE = next((p for p in DATA_CANDIDATES if p.exists()), None)

OUTPUT_DIR = ROOT / "drone_dataset" / "master_audit_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD VALID PLOT DATA
# --------------------------------------------------
if DATA_FILE is None or not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Valid plot file not found. Expected one of:\n"
        + "\n".join(str(p) for p in DATA_CANDIDATES)
    )

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

print(f"Loaded file: {DATA_FILE.name} from {DATA_FILE.parent}")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

# --------------------------------------------------
# 3. FIND LOCATION COLUMN
# --------------------------------------------------
location_candidates = [
    "location", "location_raw", "location_from_filename",
    "location_key", "Location"
]

location_col = next(
    (c for c in location_candidates if c in df.columns),
    None
)

# If no location column exists, derive it from the TP1 filename.
if location_col is None:
    path_col = next(
        (c for c in ["TP1_image_path", "tp1_image_path",
                     "TP1_path", "tp1_path"]
         if c in df.columns),
        None
    )

    if path_col is None:
        raise ValueError(
            "Could not find a location column or TP1 image path. "
            "Check the printed column names and set location_col manually."
        )

    def location_from_path(value):
        match = re.search(
            r"(Ames|Crawfordsville|Lincoln|MOValley|Scottsbluff)"
            r"-TP1-",
            str(value),
            flags=re.IGNORECASE
        )
        return match.group(1) if match else None

    df["location_derived"] = df[path_col].apply(location_from_path)
    location_col = "location_derived"

# Standardize location labels
location_map = {
    "ames": "Ames",
    "crawfordsville": "Crawfordsville",
    "lincoln": "Lincoln",
    "movalley": "MOValley",
    "missouri valley": "MOValley",
    "scottsbluff": "Scottsbluff",
}

df["Location"] = (
    df[location_col].astype("string").str.strip()
    .str.lower().map(location_map)
)

# --------------------------------------------------
# 4. COUNT UNIQUE VALID PLOTS
# --------------------------------------------------
# The manifest/valid_plots should contain one row per valid plot.
# Prefer the extracted 'plot_id' column.
plot_id_candidates = [
    "plot_id", "plot_key", "unique_plot_key",
    "plot_identifier", "plot_id_key"
]

plot_id_col = next(
    (c for c in plot_id_candidates if c in df.columns),
    None
)

if plot_id_col:
    print(f"\nUsing linked plot ID column: '{plot_id_col}'")
    valid = df.dropna(subset=["Location", plot_id_col])
    valid = valid.drop_duplicates(["Location", plot_id_col])
else:
    # If no plot_id column, deduplicate its TP1/TP2/TP3 paths.
    tp1_col = next(
        (c for c in ["TP1_image_path", "tp1_image_path",
                     "TP1_path", "tp1_path"]
         if c in df.columns),
        None
    )

    if tp1_col:
        print(f"\nUsing TP1 image path column: '{tp1_col}'")
        valid = df.dropna(subset=["Location", tp1_col])
        valid = valid.drop_duplicates(["Location", tp1_col])
    else:
        raise ValueError(
            "Could not identify plot IDs or TP1 paths. "
            "Inspect the column names and configure the correct key."
        )

counts = (
    valid.groupby("Location")
    .size()
    .reindex(
        ["Ames", "Crawfordsville", "Lincoln",
         "MOValley", "Scottsbluff"],
        fill_value=0
    )
)

summary = counts.rename("Valid_Plots").reset_index()
summary["Valid_Images"] = summary["Valid_Plots"] * 3

print("\n========== VALID PLOTS BY LOCATION ==========")
print(summary.to_string(index=False))
print("\nTotal valid plots:", int(summary["Valid_Plots"].sum()))
print("Expected images:", int(summary["Valid_Images"].sum()))

# Save exact counts
counts_csv_path = OUTPUT_DIR / "valid_plot_counts_by_location.csv"
summary.to_csv(counts_csv_path, index=False)

# --------------------------------------------------
# 5. BAR CHART
# --------------------------------------------------
plt.figure(figsize=(10, 6))

bars = plt.bar(
    summary["Location"],
    summary["Valid_Plots"],
    color="#2ca02c",
    edgecolor="#1b611b"
)

plt.title("Valid Multi-Temporal Maize Plots by Location (Total = 2,131)")
plt.xlabel("Trial Location")
plt.ylabel("Number of Valid Plots")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Display count above each bar
for bar, value in zip(bars, summary["Valid_Plots"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 10,
        str(int(value)),
        ha="center",
        va="bottom",
        fontweight="bold"
    )

plt.ylim(0, max(summary["Valid_Plots"]) * 1.15)
plt.tight_layout()

chart_path = OUTPUT_DIR / "valid_plots_by_location.png"
plt.savefig(chart_path, dpi=200, bbox_inches="tight")
print("\nChart saved to:", chart_path)
print("Count table saved to:", counts_csv_path)

# Also save a copy of the chart to project__imagesss folder if it exists
proj_img_dir = ROOT / "project__imagesss"
if proj_img_dir.exists():
    copy_chart_path = proj_img_dir / "valid_plots_by_location.png"
    plt.savefig(copy_chart_path, dpi=200, bbox_inches="tight")
    print("Also copied chart to:", copy_chart_path)

plt.close()
