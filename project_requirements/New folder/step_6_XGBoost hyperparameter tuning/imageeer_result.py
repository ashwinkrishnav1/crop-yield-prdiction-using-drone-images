from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 1. OUTPUT PATH CONFIGURATION
# ==============================================================================
PROJECT_ROOT = Path(r"C:\Users\ashwi\Music\cropyeildererrrrrrr")
OUTPUT_DIR = PROJECT_ROOT / "step6_xgboost_tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_OUT_PATH = OUTPUT_DIR / "zeroshot_rmse_r2_comparison.png"

# ==============================================================================
# 2. RECORDED ZERO-SHOT METRICS ACROSS ALL RUNS (STEP 3 -> STEP 6)
# ==============================================================================
runs_data = [
    {
        "Run": "Step 3 Winner\n(Replication Pad, Frozen TP3)",
        "ZeroShot_RMSE": 39.9498,
        "ZeroShot_R2": -1.6207,
    },
    {
        "Run": "Step 4 Full\n(Multi-TP PCA + Metadata)",
        "ZeroShot_RMSE": 31.2855,
        "ZeroShot_R2": -0.6072,
    },
    {
        "Run": "Step 4 Raw TP3\n(Fine-Tuned CNN Only)",
        "ZeroShot_RMSE": 29.6455,
        "ZeroShot_R2": -0.4431,
    },
    {
        "Run": "Step 5 (Run 3 Winner)\n(44 Core Untuned + Old Anchor)",
        "ZeroShot_RMSE": 28.9022,
        "ZeroShot_R2": -0.3717,
    },
    {
        "Run": "Step 5 (Run 1 Winner)\n(Top-120 Untuned Physical)",
        "ZeroShot_RMSE": 27.3287,
        "ZeroShot_R2": -0.2264,
    },
    {
        "Run": "Step 6 (Model A)\n(Tuned Physical Baseline)",
        "ZeroShot_RMSE": 24.1202,
        "ZeroShot_R2": 0.0447,
    },
    {
        "Run": "Step 6 (Model B)\n(Tuned Ranker + Anchor)",
        "ZeroShot_RMSE": 23.4758,
        "ZeroShot_R2": 0.0950,
    },
    {
        "Run": "Step 6 FINAL WINNER\n(Tuned LOLO Ensemble)",
        "ZeroShot_RMSE": 23.0871,
        "ZeroShot_R2": 0.1248,
    },
]

df = pd.DataFrame(runs_data)

# ==============================================================================
# 3. BUILD HIGH-RESOLUTION 2-PANEL COMPARISON IMAGE
# ==============================================================================
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.9

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5), sharey=True)
fig.patch.set_facecolor("white")

y_pos = np.arange(len(df))

# Color palette: Red/Orange for Negative R2 runs, Green for Positive R2 (Step 6) runs
colors_rmse = [
    "#c62828" if r2 < -0.5 else ("#ef6c00" if r2 < 0 else "#2e7d32")
    for r2 in df["ZeroShot_R2"]
]
colors_r2 = [
    "#d32f2f" if r2 < 0 else "#2e7d32"
    for r2 in df["ZeroShot_R2"]
]

# Highlight final winner with a darker green & thick border
colors_rmse[-1] = "#1b5e20"
colors_r2[-1] = "#1b5e20"

# ------------------------------------------------------------------------------
# LEFT PANEL: ZERO-SHOT CRAWFORDSVILLE TEST RMSE (LOWER IS BETTER)
# ------------------------------------------------------------------------------
bars1 = ax1.barh(
    y_pos,
    df["ZeroShot_RMSE"],
    color=colors_rmse,
    height=0.62,
    edgecolor="black",
    linewidth=0.8,
    alpha=0.92,
)

# R2 = 0 threshold line at 24.68 bu/ac
r2_zero_threshold = 24.68
ax1.axvline(
    r2_zero_threshold,
    color="#1565c0",
    linestyle="--",
    linewidth=2.0,
    label="R² = 0.0 Threshold (24.68 bu/ac)",
)

for bar, rmse_val in zip(bars1, df["ZeroShot_RMSE"]):
    ax1.text(
        bar.get_width() + 0.45,
        bar.get_y() + bar.get_height() / 2.0,
        f"{rmse_val:.2f} bu/ac",
        va="center",
        ha="left",
        fontsize=10,
        fontweight="bold",
        color="#111111",
    )

ax1.set_yticks(y_pos)
ax1.set_yticklabels(df["Run"], fontsize=10, fontweight="bold")
ax1.invert_yaxis()  # Top-to-bottom chronological progression
ax1.set_xlim(0, 46.5)
ax1.set_xlabel("Zero-Shot Test RMSE (Bushels / Acre) — Lower is Better", fontsize=11, fontweight="bold")
ax1.set_title("1. Zero-Shot Crawfordsville Test RMSE Progression", fontsize=13, fontweight="bold", pad=12)
ax1.legend(loc="lower right", fontsize=9.5, frameon=True)
ax1.grid(axis="x", linestyle=":", alpha=0.6)

# ------------------------------------------------------------------------------
# RIGHT PANEL: ZERO-SHOT CRAWFORDSVILLE TEST R² (HIGHER / POSITIVE IS BETTER)
# ------------------------------------------------------------------------------
bars2 = ax2.barh(
    y_pos,
    df["ZeroShot_R2"],
    color=colors_r2,
    height=0.62,
    edgecolor="black",
    linewidth=0.8,
    alpha=0.92,
)

# Zero reference line
ax2.axvline(0.0, color="black", linestyle="-", linewidth=1.8, label="Baseline Average Guess (R² = 0.0)")

for bar, r2_val in zip(bars2, df["ZeroShot_R2"]):
    if r2_val < 0:
        x_text = r2_val - 0.04
        ha_align = "right"
        txt_str = f"{r2_val:.4f}"
    else:
        x_text = r2_val + 0.03
        ha_align = "left"
        txt_str = f"+{r2_val:.4f}"

    ax2.text(
        x_text,
        bar.get_y() + bar.get_height() / 2.0,
        txt_str,
        va="center",
        ha=ha_align,
        fontsize=10,
        fontweight="bold",
        color="#b71c1c" if r2_val < 0 else "#1b5e20",
    )

ax2.set_xlim(-1.95, 0.45)
ax2.set_xlabel("Zero-Shot Test R² Score — Positive (> 0) Beats Field Average", fontsize=11, fontweight="bold")
ax2.set_title("2. Zero-Shot Crawfordsville Test R² Progression", fontsize=13, fontweight="bold", pad=12)
ax2.legend(loc="lower left", fontsize=9.5, frameon=True)
ax2.grid(axis="x", linestyle=":", alpha=0.6)

# Overall Title & Summary Banner
fig.suptitle(
    "Unseen Crawfordsville Test Set (488 Plots): Zero-Shot RMSE & R² Across Pipeline Runs\n"
    "Total Improvement: RMSE cut from 39.95 -> 23.09 bu/ac (-16.86 bu/ac) | Zero-Shot R² flipped from -1.6207 -> +0.1248",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

plt.tight_layout(rect=[0, 0, 1, 0.91])
plt.savefig(IMAGE_OUT_PATH, dpi=300)
plt.close()

print("=" * 85)
print(f"Saved high-resolution comparison image -> {IMAGE_OUT_PATH}")
print("=" * 85)

# Automatically open the saved PNG image on Windows
try:
    import os
    os.startfile(str(IMAGE_OUT_PATH))
except Exception:
    pass