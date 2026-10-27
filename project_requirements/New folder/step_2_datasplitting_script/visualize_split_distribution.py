"""Visual Bar Chart of Dataset Splits by Location.

This script reads:
  - train_plot_ids.csv
  - validation_plot_ids.csv
  - test_plot_ids.csv

It generates a clear, publication-quality grouped bar chart showing the
number of plots in each trial location across Train, Validation, and Test sets,
and saves the figure as an image.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set directory paths
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_FILE = SCRIPT_DIR / "train_plot_ids.csv"
VAL_FILE = SCRIPT_DIR / "validation_plot_ids.csv"
TEST_FILE = SCRIPT_DIR / "test_plot_ids.csv"
OUTPUT_CHART = SCRIPT_DIR / "data_splits_by_location.png"

def main():
    # 1. Load the split files
    for file_path in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required file: {file_path}")

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # 2. Standardize locations list
    locations = ["Ames", "Crawfordsville", "Lincoln", "MOValley", "Scottsbluff"]

    # 3. Count plots per location
    train_counts = train_df["location"].value_counts().reindex(locations, fill_value=0)
    val_counts = val_df["location"].value_counts().reindex(locations, fill_value=0)
    test_counts = test_df["location"].value_counts().reindex(locations, fill_value=0)

    # Summary table in console
    summary_df = pd.DataFrame({
        "Train": train_counts,
        "Validation": val_counts,
        "Test": test_counts,
        "Total": train_counts + val_counts + test_counts
    })
    print("\n" + "="*50)
    print("      DATA SPLIT DISTRIBUTION BY LOCATION")
    print("="*50)
    print(summary_df.to_string())
    print("-" * 50)
    print(f"Total Plots: {summary_df['Total'].sum()} (Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)})")
    print("="*50 + "\n")

    # 4. Create Grouped Bar Chart
    x = np.arange(len(locations))
    bar_width = 0.25

    plt.figure(figsize=(11, 6))

    # Color palette
    color_train = "#2b5c8f"      # Deep Steel Blue
    color_val = "#e67e22"        # Warm Amber / Orange
    color_test = "#27ae60"       # Emerald Green

    bars_train = plt.bar(x - bar_width, train_counts, width=bar_width, label=f"Train ({len(train_df):,})", color=color_train, edgecolor="black", linewidth=0.6)
    bars_val = plt.bar(x, val_counts, width=bar_width, label=f"Validation ({len(val_df):,})", color=color_val, edgecolor="black", linewidth=0.6)
    bars_test = plt.bar(x + bar_width, test_counts, width=bar_width, label=f"Test (Held-out: {len(test_df):,})", color=color_test, edgecolor="black", linewidth=0.6)

    # Add count values above each bar
    for bars in [bars_train, bars_val, bars_test]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 4,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold"
                )

    plt.title("Distribution of Maize Plots Across Data Splits by Location", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Trial Location", fontsize=12, labelpad=10)
    plt.ylabel("Number of Valid Plots", fontsize=12, labelpad=10)
    plt.xticks(x, locations, fontsize=11)
    plt.ylim(0, max(train_counts.max(), test_counts.max()) + 55)

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11, loc="upper right")
    plt.tight_layout()

    # Save chart
    plt.savefig(OUTPUT_CHART, dpi=300, bbox_inches="tight")
    print(f"Bar chart successfully saved to: {OUTPUT_CHART}")

if __name__ == "__main__":
    main()

