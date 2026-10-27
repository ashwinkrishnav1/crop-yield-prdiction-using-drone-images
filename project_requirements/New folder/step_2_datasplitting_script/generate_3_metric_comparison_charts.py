import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==============================================================================
# 1. OUTPUT FOLDER CONFIGURATION
# ==============================================================================
PROJECT_ROOT = Path(r"C:\Users\ashwi\Music\cropyeildererrrrrrr")
OUTPUT_DIR = PROJECT_ROOT / "project__imagesss" / "step_comparison_charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

R2_IMG_PATH = OUTPUT_DIR / "1_R2_Comparison_Model1_vs_Model2.png"
RMSE_IMG_PATH = OUTPUT_DIR / "2_RMSE_Comparison_Model1_vs_Model2.png"
MAE_IMG_PATH = OUTPUT_DIR / "3_MAE_Comparison_Model1_vs_Model2.png"

# ==============================================================================
# 2. STEP-BY-STEP METRICS FOR BOTH MODELS (FROM VERIFIED TERMINAL RUNS)
# ==============================================================================
# Model 1: All 5 Locations In-Domain (Ames, Crawfordsville, Lincoln, MOValley, Scottsbluff)
model1_steps = pd.DataFrame(
    [
        {
            "Step": "Step 3 Baseline\n(Raw TP3 CNN Only)",
            "R2": 0.7940,
            "RMSE": 24.1051,
            "MAE": 18.7389,
            "Val_R2": 0.8319,
            "Val_RMSE": 22.9489,
            "Val_MAE": 17.3414,
        },
        {
            "Step": "Step 4A\n(Multi-TP CNN + Deltas PCA)",
            "R2": 0.8087,
            "RMSE": 23.2279,
            "MAE": 18.2848,
            "Val_R2": 0.8425,
            "Val_RMSE": 22.2182,
            "Val_MAE": 16.4322,
        },
        {
            "Step": "Step 4B Full\n(Multi-TP CNN + RGB + Meta)",
            "R2": 0.8178,
            "RMSE": 22.6725,
            "MAE": 18.0380,
            "Val_R2": 0.8494,
            "Val_RMSE": 21.7268,
            "Val_MAE": 16.0979,
        },
        {
            "Step": "Step 5 Feature Selection\n(mRMR + Pedigree + Spatial)",
            "R2": 0.8436,
            "RMSE": 21.0053,
            "MAE": 16.6672,
            "Val_R2": 0.8748,
            "Val_RMSE": 19.8049,
            "Val_MAE": 15.0413,
        },
        {
            "Step": "Step 6 Final Tuned\n(Core Specialist / 3-View Stack)",
            "R2": 0.8454,
            "RMSE": 20.8817,
            "MAE": 16.5349,
            "Val_R2": 0.8721,
            "Val_RMSE": 20.0212,
            "Val_MAE": 15.2008,
        },
    ]
)

# Model 2: 4-Location Train/Val -> 100% Unseen Crawfordsville Test (488 Plots)
model2_steps = pd.DataFrame(
    [
        {
            "Step": "Step 3 Baseline\n(Replication Pad, Frozen TP3)",
            "R2": -1.6207,
            "RMSE": 39.9498,
            "MAE": 33.0106,
            "Stage_Type": "Negative_R2",
        },
        {
            "Step": "Step 4 Full\n(Multi-TP PCA + PlantingDate)",
            "R2": -0.6072,
            "RMSE": 31.2855,
            "MAE": 25.3098,
            "Stage_Type": "Negative_R2",
        },
        {
            "Step": "Step 4 Raw TP3\n(Fine-Tuned CNN Only)",
            "R2": -0.4431,
            "RMSE": 29.6455,
            "MAE": 24.0683,
            "Stage_Type": "Negative_R2",
        },
        {
            "Step": "Step 5 Feature Selection\n(LOLO Prototypes + Spatial + Gen)",
            "R2": -0.2514,
            "RMSE": 27.6063,
            "MAE": 22.0116,
            "Stage_Type": "Step5_Transition",
        },
        {
            "Step": "Step 6 Final Tuned\n(Zero-Shot Unseen Test)",
            "R2": 0.1398,
            "RMSE": 22.8875,
            "MAE": 18.0182,
            "Stage_Type": "Step6_ZeroShot",
        },
        {
            "Step": "Step 6 Final Tuned\n(Field-Calibrated 15-Plot Strip)",
            "R2": 0.2168,
            "RMSE": 21.8926,
            "MAE": 17.4398,
            "Stage_Type": "Step6_Calibrated",
        },
    ]
)

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.9


# ==============================================================================
# 3. IMAGE 1 GENERATOR: R² SCORE COMPARISON (2 SIDE-BY-SIDE BAR CHARTS)
# ==============================================================================
def plot_r2_comparison(m1_df: pd.DataFrame, m2_df: pd.DataFrame, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 7.8))
    fig.patch.set_facecolor("white")

    # --- LEFT CHART: MODEL 1 (ALL 5 LOCATIONS IN-DOMAIN) ---
    y1 = np.arange(len(m1_df))
    m1_colors = ["#90caf9", "#64b5f6", "#42a5f5", "#1e88e5", "#1b5e20"]
    bars1 = ax1.barh(
        y1,
        m1_df["R2"],
        color=m1_colors,
        height=0.56,
        edgecolor="black",
        linewidth=0.9,
        alpha=0.95,
    )

    for bar, r2_val, val_r2 in zip(bars1, m1_df["R2"], m1_df["Val_R2"]):
        ax1.text(
            bar.get_width() + 0.008,
            bar.get_y() + bar.get_height() / 2.0,
            f"Test R² = {r2_val:.4f}  (Val: {val_r2:.4f})",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color="#111111",
        )

    ax1.set_yticks(y1)
    ax1.set_yticklabels(m1_df["Step"], fontsize=10, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlim(0.65, 0.96)
    ax1.set_xlabel("R² Score (Higher is Better — Max = 1.0)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Model 1: All 5 Locations In-Domain (323 Test Plots)\nR² Progression Across Pipeline Steps",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax1.grid(axis="x", linestyle=":", alpha=0.6)

    # --- RIGHT CHART: MODEL 2 (UNSEEN CRAWFORDSVILLE TRANSFER) ---
    y2 = np.arange(len(m2_df))
    m2_colors = []
    for st_type in m2_df["Stage_Type"]:
        if st_type == "Negative_R2":
            m2_colors.append("#d32f2f")
        elif st_type == "Step5_Transition":
            m2_colors.append("#f57c00")
        elif st_type == "Step6_ZeroShot":
            m2_colors.append("#2e7d32")
        else:
            m2_colors.append("#00838f")

    bars2 = ax2.barh(
        y2,
        m2_df["R2"],
        color=m2_colors,
        height=0.56,
        edgecolor="black",
        linewidth=0.9,
        alpha=0.95,
    )
    ax2.axvline(0.0, color="black", linestyle="-", linewidth=1.8, label="R² = 0.0 Baseline Threshold")

    for bar, r2_val in zip(bars2, m2_df["R2"]):
        if r2_val < 0:
            x_pos = r2_val - 0.04
            ha_align = "right"
            txt = f"{r2_val:.4f}"
            col = "#b71c1c"
        else:
            x_pos = r2_val + 0.04
            ha_align = "left"
            txt = f"+{r2_val:.4f}"
            col = "#1b5e20"

        ax2.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2.0,
            txt,
            va="center",
            ha=ha_align,
            fontsize=10,
            fontweight="bold",
            color=col,
        )

    ax2.set_yticks(y2)
    ax2.set_yticklabels(m2_df["Step"], fontsize=10, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlim(-1.95, 0.55)
    ax2.set_xlabel("R² Score (Positive > 0 Beats Field Average)", fontsize=11, fontweight="bold")
    ax2.set_title(
        "Model 2: Unseen Crawfordsville Transfer (488 Test Plots)\nR² Progression Across Pipeline Steps",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    legend_elements = [
        Patch(facecolor="#d32f2f", edgecolor="black", label="Negative R² (Baseline Shift)"),
        Patch(facecolor="#f57c00", edgecolor="black", label="Step 5 Domain-Invariant Selection"),
        Patch(facecolor="#2e7d32", edgecolor="black", label="Step 6 Zero-Shot Positive R² (+0.1398)"),
        Patch(facecolor="#00838f", edgecolor="black", label="Step 6 Calibrated Positive R² (+0.2168)"),
    ]
    ax2.legend(handles=legend_elements, loc="lower left", fontsize=9, frameon=True)
    ax2.grid(axis="x", linestyle=":", alpha=0.6)

    fig.suptitle(
        "IMAGE 1: R² SCORE COMPARISON ACROSS PIPELINE STEPS — MODEL 1 (5-LOC IN-DOMAIN) vs. MODEL 2 (UNSEEN CRAWFORDSVILLE)",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=300)
    plt.close()


# ==============================================================================
# 4. IMAGE 2 GENERATOR: RMSE COMPARISON (2 SIDE-BY-SIDE BAR CHARTS)
# ==============================================================================
def plot_rmse_comparison(m1_df: pd.DataFrame, m2_df: pd.DataFrame, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 7.8))
    fig.patch.set_facecolor("white")

    # --- LEFT CHART: MODEL 1 (ALL 5 LOCATIONS IN-DOMAIN) ---
    y1 = np.arange(len(m1_df))
    m1_colors = ["#ef9a9a", "#ffcc80", "#90caf9", "#66bb6a", "#1b5e20"]
    bars1 = ax1.barh(
        y1,
        m1_df["RMSE"],
        color=m1_colors,
        height=0.56,
        edgecolor="black",
        linewidth=0.9,
        alpha=0.95,
    )

    for bar, rmse_val, val_rmse in zip(bars1, m1_df["RMSE"], m1_df["Val_RMSE"]):
        ax1.text(
            bar.get_width() + 0.25,
            bar.get_y() + bar.get_height() / 2.0,
            f"{rmse_val:.2f} bu/ac  (Val: {val_rmse:.2f})",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color="#111111",
        )

    ax1.set_yticks(y1)
    ax1.set_yticklabels(m1_df["Step"], fontsize=10, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlim(0, 31.0)
    ax1.set_xlabel("Test RMSE (Bushels / Acre) — Lower is Better", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Model 1: All 5 Locations In-Domain (323 Test Plots)\nRMSE Reduction Across Pipeline Steps (24.11 -> 20.88 bu/ac)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax1.grid(axis="x", linestyle=":", alpha=0.6)

    # --- RIGHT CHART: MODEL 2 (UNSEEN CRAWFORDSVILLE TRANSFER) ---
    y2 = np.arange(len(m2_df))
    m2_colors = ["#c62828", "#e65100", "#ef6c00", "#fbc02d", "#2e7d32", "#00838f"]
    bars2 = ax2.barh(
        y2,
        m2_df["RMSE"],
        color=m2_colors,
        height=0.56,
        edgecolor="black",
        linewidth=0.9,
        alpha=0.95,
    )

    ax2.axvline(
        24.68,
        color="#1565c0",
        linestyle="--",
        linewidth=2.0,
        label="Positive R² Threshold (24.68 bu/ac)",
    )

    for bar, rmse_val in zip(bars2, m2_df["RMSE"]):
        ax2.text(
            bar.get_width() + 0.45,
            bar.get_y() + bar.get_height() / 2.0,
            f"{rmse_val:.2f} bu/ac",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#111111",
        )

    ax2.set_yticks(y2)
    ax2.set_yticklabels(m2_df["Step"], fontsize=10, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlim(0, 47.0)
    ax2.set_xlabel("Test RMSE (Bushels / Acre) — Lower is Better", fontsize=11, fontweight="bold")
    ax2.set_title(
        "Model 2: Unseen Crawfordsville Transfer (488 Test Plots)\nRMSE Reduction Across Pipeline Steps (39.95 -> 21.89 bu/ac)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax2.legend(loc="lower right", fontsize=9.5, frameon=True)
    ax2.grid(axis="x", linestyle=":", alpha=0.6)

    fig.suptitle(
        "IMAGE 2: RMSE (ROOT MEAN SQUARED ERROR) COMPARISON ACROSS PIPELINE STEPS — MODEL 1 vs. MODEL 2",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=300)
    plt.close()


# ==============================================================================
# 5. IMAGE 3 GENERATOR: MAE COMPARISON (2 SIDE-BY-SIDE BAR CHARTS)
# ==============================================================================
def plot_mae_comparison(m1_df: pd.DataFrame, m2_df: pd.DataFrame, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 7.8))
    fig.patch.set_facecolor("white")

    # --- LEFT CHART: MODEL 1 (ALL 5 LOCATIONS IN-DOMAIN) ---
    y1 = np.arange(len(m1_df))
    m1_colors = ["#ce93d8", "#b39ddb", "#90caf9", "#4db6ac", "#00695c"]
    bars1 = ax1.barh(
        y1,
        m1_df["MAE"],
        color=m1_colors,
        height=0.56,
        edgecolor="black",
        linewidth=0.9,
        alpha=0.95,
    )

    for bar, mae_val, val_mae in zip(bars1, m1_df["MAE"], m1_df["Val_MAE"]):
        ax1.text(
            bar.get_width() + 0.22,
            bar.get_y() + bar.get_height() / 2.0,
            f"{mae_val:.2f} bu/ac  (Val: {val_mae:.2f})",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color="#111111",
        )

    ax1.set_yticks(y1)
    ax1.set_yticklabels(m1_df["Step"], fontsize=10, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlim(0, 25.0)
    ax1.set_xlabel("Test MAE (Bushels / Acre) — Lower is Better", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Model 1: All 5 Locations In-Domain (323 Test Plots)\nMAE Reduction Across Pipeline Steps (18.74 -> 16.53 bu/ac)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax1.grid(axis="x", linestyle=":", alpha=0.6)

    # --- RIGHT CHART: MODEL 2 (UNSEEN CRAWFORDSVILLE TRANSFER) ---
    y2 = np.arange(len(m2_df))
    m2_colors = ["#b71c1c", "#d84315", "#ef6c00", "#fbc02d", "#2e7d32", "#00838f"]
    bars2 = ax2.barh(
        y2,
        m2_df["MAE"],
        color=m2_colors,
        height=0.56,
        edgecolor="black",
        linewidth=0.9,
        alpha=0.95,
    )

    for bar, mae_val in zip(bars2, m2_df["MAE"]):
        ax2.text(
            bar.get_width() + 0.40,
            bar.get_y() + bar.get_height() / 2.0,
            f"{mae_val:.2f} bu/ac",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#111111",
        )

    ax2.set_yticks(y2)
    ax2.set_yticklabels(m2_df["Step"], fontsize=10, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlim(0, 39.0)
    ax2.set_xlabel("Test MAE (Bushels / Acre) — Lower is Better", fontsize=11, fontweight="bold")
    ax2.set_title(
        "Model 2: Unseen Crawfordsville Transfer (488 Test Plots)\nMAE Reduction Across Pipeline Steps (33.01 -> 17.44 bu/ac)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax2.grid(axis="x", linestyle=":", alpha=0.6)

    fig.suptitle(
        "IMAGE 3: MAE (MEAN ABSOLUTE ERROR) COMPARISON ACROSS PIPELINE STEPS — MODEL 1 vs. MODEL 2",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=300)
    plt.close()


# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================
def main():
    print("=" * 88)
    print("GENERATING 3 TWO-PANEL COMPARISON IMAGES (R², RMSE, MAE) FOR MODEL 1 & MODEL 2...")
    print("=" * 88)

    plot_r2_comparison(model1_steps, model2_steps, R2_IMG_PATH)
    print(f"  [1/3] Saved R² Comparison Image   -> {R2_IMG_PATH}")

    plot_rmse_comparison(model1_steps, model2_steps, RMSE_IMG_PATH)
    print(f"  [2/3] Saved RMSE Comparison Image -> {RMSE_IMG_PATH}")

    plot_mae_comparison(model1_steps, model2_steps, MAE_IMG_PATH)
    print(f"  [3/3] Saved MAE Comparison Image  -> {MAE_IMG_PATH}")

    print("=" * 88)
    print("All 3 images generated successfully! Opening images now...")

    for img_p in (R2_IMG_PATH, RMSE_IMG_PATH, MAE_IMG_PATH):
        try:
            os.startfile(str(img_p))
        except Exception:
            pass


if __name__ == "__main__":
    main()