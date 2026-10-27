import os
import time
import pickle
import warnings
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PROJECT PATHS & CONFIGURATION
# ==============================================================================
PROJECT_ROOT = Path(r"C:\Users\ashwi\Music\cropyeildererrrrrrr")
STEP5_NPZ_PATH = PROJECT_ROOT / "step5_feature_selection" / "step5_selected_features.npz"
STEP6_DIR = PROJECT_ROOT / "step6_xgboost_tuning"
STEP6_PKL_PATH = STEP6_DIR / "step6_tuned_models.pkl"

OUTPUT_DIR = STEP6_DIR / "other_4_locations_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ==============================================================================
# 2. LOAD CACHED FEATURES & STEP 6 TUNED ARTIFACTS
# ==============================================================================
def load_saved_data() -> Tuple[Dict[str, np.ndarray], Dict]:
    if not STEP5_NPZ_PATH.exists():
        raise FileNotFoundError(f"Missing {STEP5_NPZ_PATH}. Please run Step 5 first.")

    s5_data = np.load(STEP5_NPZ_PATH, allow_pickle=True)
    s5_dict = {k: s5_data[k] for k in s5_data.files}

    artifacts = {}
    if STEP6_PKL_PATH.exists():
        with open(STEP6_PKL_PATH, "rb") as f:
            artifacts = pickle.load(f)

    splits = s5_dict["splits"]
    locations = s5_dict["locations"]
    dev_mask = (splits == "train") | (splits == "val")

    print("=" * 105)
    print(
        f"LOADED DATASET: {len(splits)} total plots | "
        f"4 Dev Locations: {np.sum(dev_mask)} plots (Train: {np.sum(splits=='train')}, Val: {np.sum(splits=='val')}) | "
        f"Crawfordsville Test: {np.sum(splits=='test')} plots"
    )
    for loc in sorted(np.unique(locations[dev_mask])):
        n_tr = np.sum((locations == loc) & (splits == "train"))
        n_va = np.sum((locations == loc) & (splits == "val"))
        print(f"  - {loc:<14}: Total = {n_tr + n_va:<4} (Train = {n_tr:<3} | Val = {n_va:<3})")
    print("=" * 105)
    return s5_dict, artifacts


# ==============================================================================
# 3. METRICS & STEP 6 ENSEMBLE BUILDERS
# ==============================================================================
def safe_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_metrics_dict(
    label: str, split_type: str, y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    r = safe_pearson_r(y_true, y_pred)
    mape = float(
        np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1.0, None))) * 100.0
    )
    return {
        "Location": label,
        "Evaluation_Mode": split_type,
        "Plots": int(len(y_true)),
        "Actual_Mean_bu_ac": round(float(np.mean(y_true)), 2),
        "Pred_Mean_bu_ac": round(float(np.mean(y_pred)), 2),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "Pearson_r": round(r, 4),
        "MAPE_%": round(mape, 2),
    }


def compute_location_balanced_weights(locations_subset: np.ndarray) -> np.ndarray:
    unique_locs, counts = np.unique(locations_subset, return_counts=True)
    total = float(len(locations_subset))
    n_locs = float(len(unique_locs))
    weight_map = {
        loc: total / (n_locs * float(cnt)) for loc, cnt in zip(unique_locs, counts)
    }
    return np.array([weight_map[loc] for loc in locations_subset], dtype=np.float32)


def build_xgb_from_cfg(cfg: Dict) -> XGBRegressor:
    params = {k: v for k, v in cfg.items() if k not in ("config_id", "phys_id", "use_loc_weights")}
    return XGBRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)


def get_step6_configs(artifacts: Dict) -> Tuple[Dict, Dict, Dict, Tuple[float, float, float], float]:
    best_ranker_cfg = artifacts.get(
        "best_ranker_cfg",
        {
            "config_id": "R5_Depth4_Balanced",
            "n_estimators": 300,
            "learning_rate": 0.020,
            "max_depth": 4,
            "min_child_weight": 20,
            "subsample": 0.80,
            "colsample_bytree": 0.45,
            "reg_alpha": 4.0,
            "reg_lambda": 20.0,
            "gamma": 0.5,
            "objective": "reg:squarederror",
        },
    )
    best_shallow_cfg = artifacts.get(
        "best_shallow_cfg",
        {
            "config_id": "R2_Shallow_FastLR",
            "n_estimators": 200,
            "learning_rate": 0.030,
            "max_depth": 2,
            "min_child_weight": 25,
            "subsample": 0.70,
            "colsample_bytree": 0.40,
            "reg_alpha": 5.0,
            "reg_lambda": 25.0,
            "gamma": 1.0,
            "objective": "reg:squarederror",
        },
    )
    best_phys_cfg = artifacts.get(
        "best_phys_cfg",
        {
            "phys_id": "P0_Step5_Run3_LocWeighted",
            "use_loc_weights": True,
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 15,
            "subsample": 0.80,
            "colsample_bytree": 0.50,
            "reg_alpha": 2.0,
            "reg_lambda": 12.0,
            "objective": "reg:squarederror",
        },
    )
    blend_weights = artifacts.get("blend_weights", (0.60, 0.30, 0.10))
    alpha_shrink = float(artifacts.get("alpha_shrink", 0.85))
    return best_ranker_cfg, best_shallow_cfg, best_phys_cfg, blend_weights, alpha_shrink


def fit_and_predict_step6_ranker(
    X_fit: np.ndarray,
    y_fit_cent: np.ndarray,
    w_fit: np.ndarray,
    X_eval: np.ndarray,
    cfg1: Dict,
    cfg2: Dict,
    blend_weights: Tuple[float, float, float],
) -> np.ndarray:
    w_x, w_e, w_r = blend_weights

    m_xgb1 = build_xgb_from_cfg(cfg1)
    m_xgb1.fit(X_fit, y_fit_cent, sample_weight=w_fit)

    m_xgb2 = build_xgb_from_cfg(cfg2)
    m_xgb2.fit(X_fit, y_fit_cent, sample_weight=w_fit)

    m_et = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        max_features=0.45,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    m_et.fit(X_fit, y_fit_cent, sample_weight=w_fit)

    m_ridge = RidgeCV(alphas=[10.0, 50.0, 150.0, 400.0, 1000.0])
    m_ridge.fit(X_fit, y_fit_cent, sample_weight=w_fit)

    p_xgb = 0.65 * m_xgb1.predict(X_eval) + 0.35 * m_xgb2.predict(X_eval)
    p_et = m_et.predict(X_eval)
    p_ridge = m_ridge.predict(X_eval)

    return (w_x * p_xgb + w_e * p_et + w_r * p_ridge).astype(np.float32)


# ==============================================================================
# 4. EVALUATION 1: TRAIN & HOLDOUT VALIDATION BREAKDOWN BY LOCATION
# ==============================================================================
def evaluate_train_and_val_by_location(
    X_raw: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    locations: np.ndarray,
    cfg1: Dict,
    cfg2: Dict,
    blend_weights: Tuple[float, float, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    train_mask = splits == "train"
    val_mask = splits == "val"

    scaler = StandardScaler()
    scaler.fit(X_raw[train_mask])
    X = scaler.transform(X_raw).astype(np.float32)

    loc_tr = locations[train_mask]
    y_tr = y[train_mask]
    tr_loc_means = {
        loc: float(np.mean(y_tr[loc_tr == loc])) for loc in np.unique(loc_tr)
    }
    y_tr_cent = np.array(
        [yv - tr_loc_means[l] for yv, l in zip(y_tr, loc_tr)], dtype=np.float32
    )
    w_tr = compute_location_balanced_weights(loc_tr)

    # Predict on both Train (1,313) and Holdout Val (330)
    X_tr_va = np.vstack([X[train_mask], X[val_mask]])
    preds_cent_all = fit_and_predict_step6_ranker(
        X[train_mask], y_tr_cent, w_tr, X_tr_va, cfg1, cfg2, blend_weights
    )
    n_tr = int(np.sum(train_mask))
    tr_preds_cent = preds_cent_all[:n_tr]
    va_preds_cent = preds_cent_all[n_tr:]

    tr_offsets = np.array([tr_loc_means[l] for l in locations[train_mask]], dtype=np.float32)
    va_offsets = np.array([tr_loc_means[l] for l in locations[val_mask]], dtype=np.float32)

    tr_preds_abs = tr_preds_cent + tr_offsets
    va_preds_abs = va_preds_cent + va_offsets

    val_rows = []
    train_rows = []
    dev_locs = sorted(np.unique(loc_tr))

    for loc in dev_locs:
        m_va_loc = locations[val_mask] == loc
        val_rows.append(
            compute_metrics_dict(
                loc, "Holdout Validation (20%)", y[val_mask][m_va_loc], va_preds_abs[m_va_loc]
            )
        )
        m_tr_loc = locations[train_mask] == loc
        train_rows.append(
            compute_metrics_dict(
                loc, "Training Split (80%)", y[train_mask][m_tr_loc], tr_preds_abs[m_tr_loc]
            )
        )

    # Add combined 4-location totals
    val_rows.append(
        compute_metrics_dict(
            "ALL 4 LOCATIONS (Combined Val)",
            "Holdout Validation (20%)",
            y[val_mask],
            va_preds_abs,
        )
    )
    train_rows.append(
        compute_metrics_dict(
            "ALL 4 LOCATIONS (Combined Train)",
            "Training Split (80%)",
            y[train_mask],
            tr_preds_abs,
        )
    )

    return pd.DataFrame(val_rows), pd.DataFrame(train_rows), tr_preds_abs, va_preds_abs


# ==============================================================================
# 5. EVALUATION 2: 5-FOLD OUT-OF-FOLD TEST ACROSS ALL 1,643 PLOTS (4 LOCATIONS)
# ==============================================================================
def evaluate_5fold_cv_all_1643_plots(
    X_raw: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    locations: np.ndarray,
    cfg1: Dict,
    cfg2: Dict,
    blend_weights: Tuple[float, float, float],
) -> Tuple[pd.DataFrame, np.ndarray]:
    dev_mask = (splits == "train") | (splits == "val")
    X_dev_raw = X_raw[dev_mask]
    y_dev = y[dev_mask]
    loc_dev = locations[dev_mask]

    yield_bins = pd.qcut(y_dev, q=5, labels=False, duplicates="drop").astype(str)
    strat_key = pd.Series(loc_dev).astype(str) + "_" + yield_bins
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    oof_preds = np.zeros(len(y_dev), dtype=np.float32)

    for f_tr, f_va in skf.split(X_dev_raw, strat_key):
        scaler = StandardScaler()
        X_f_tr = scaler.fit_transform(X_dev_raw[f_tr]).astype(np.float32)
        X_f_va = scaler.transform(X_dev_raw[f_va]).astype(np.float32)

        loc_f_tr = loc_dev[f_tr]
        y_f_tr = y_dev[f_tr]
        f_means = {
            l: float(np.mean(y_f_tr[loc_f_tr == l])) for l in np.unique(loc_f_tr)
        }
        y_f_cent = np.array(
            [yv - f_means[l] for yv, l in zip(y_f_tr, loc_f_tr)], dtype=np.float32
        )
        w_f = compute_location_balanced_weights(loc_f_tr)

        p_va_cent = fit_and_predict_step6_ranker(
            X_f_tr, y_f_cent, w_f, X_f_va, cfg1, cfg2, blend_weights
        )
        va_offsets = np.array([f_means[l] for l in loc_dev[f_va]], dtype=np.float32)
        oof_preds[f_va] = p_va_cent + va_offsets

    cv_rows = []
    for loc in sorted(np.unique(loc_dev)):
        m_loc = loc_dev == loc
        cv_rows.append(
            compute_metrics_dict(
                loc, "5-Fold Out-of-Fold Test (100% Plots)", y_dev[m_loc], oof_preds[m_loc]
            )
        )

    cv_rows.append(
        compute_metrics_dict(
            "ALL 4 LOCATIONS (1,643 Plots OOF)",
            "5-Fold Out-of-Fold Test (100% Plots)",
            y_dev,
            oof_preds,
        )
    )
    return pd.DataFrame(cv_rows), oof_preds


# ==============================================================================
# 6. EVALUATION 3: BLIND LEAVE-ONE-LOCATION-OUT (LOLO) TEST PER CITY
# ==============================================================================
def evaluate_lolo_blind_city_transfer(
    X_raw: np.ndarray,
    X_phys_raw: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    locations: np.ndarray,
    cfg1: Dict,
    cfg2: Dict,
    phys_cfg: Dict,
    blend_weights: Tuple[float, float, float],
    alpha_shrink: float,
) -> pd.DataFrame:
    """
    Holds out each of the 4 locations (Ames, Lincoln, MOValley, Scottsbluff) one at a time
    as a 100% unseen test farm, plus evaluates Crawfordsville for a complete 5-location table.
    """
    dev_mask = (splits == "train") | (splits == "val")
    dev_locs = sorted(np.unique(locations[dev_mask]))
    all_5_locs = dev_locs + ["Crawfordsville"]

    lolo_rows = []

    for target_loc in all_5_locs:
        test_m = locations == target_loc
        # Train strictly on the remaining dev locations (never using Crawfordsville for training)
        train_m = dev_mask & (locations != target_loc)

        scaler_c = StandardScaler()
        X_tr_c = scaler_c.fit_transform(X_raw[train_m]).astype(np.float32)
        X_te_c = scaler_c.transform(X_raw[test_m]).astype(np.float32)

        scaler_p = StandardScaler()
        X_tr_p = scaler_p.fit_transform(X_phys_raw[train_m]).astype(np.float32)
        X_te_p = scaler_p.transform(X_phys_raw[test_m]).astype(np.float32)

        y_tr = y[train_m]
        loc_tr = locations[train_m]
        y_te = y[test_m]

        w_tr = compute_location_balanced_weights(loc_tr)
        f_means = {l: float(np.mean(y_tr[loc_tr == l])) for l in np.unique(loc_tr)}
        y_tr_cent = np.array(
            [yv - f_means[l] for yv, l in zip(y_tr, loc_tr)], dtype=np.float32
        )

        # 1. Physical Baseline Anchor on Held-Out City
        m_phys = build_xgb_from_cfg(phys_cfg)
        m_phys.fit(X_tr_p, y_tr, sample_weight=w_tr)
        phys_preds = m_phys.predict(X_te_p)
        anchor = float(np.mean(phys_preds))
        phys_dev_zero = phys_preds - anchor

        # 2. Within-Field Multi-Model Ranker Deviations
        rank_preds = fit_and_predict_step6_ranker(
            X_tr_c, y_tr_cent, w_tr, X_te_c, cfg1, cfg2, blend_weights
        )
        rank_dev_zero = rank_preds - float(np.mean(rank_preds))

        zeroshot_preds = anchor + alpha_shrink * (
            0.88 * rank_dev_zero + 0.12 * phys_dev_zero
        )

        # 3. 15-Plot Reference Strip Field-Calibrated Predictions
        rng = np.random.RandomState(RANDOM_SEED)
        cal_idx = rng.choice(len(y_te), size=min(15, len(y_te) // 4), replace=False)
        eval_mask = np.ones(len(y_te), dtype=bool)
        eval_mask[cal_idx] = False
        raw_offset = float(np.median(y_te[cal_idx] - zeroshot_preds[cal_idx]))
        cal_preds = zeroshot_preds + 0.65 * raw_offset

        zs_rmse = float(np.sqrt(mean_squared_error(y_te, zeroshot_preds)))
        zs_mae = float(mean_absolute_error(y_te, zeroshot_preds))
        zs_r2 = float(r2_score(y_te, zeroshot_preds))
        zs_r = safe_pearson_r(y_te, zeroshot_preds)

        cal_rmse = float(np.sqrt(mean_squared_error(y_te[eval_mask], cal_preds[eval_mask])))
        cal_mae = float(mean_absolute_error(y_te[eval_mask], cal_preds[eval_mask]))
        cal_r2 = float(r2_score(y_te[eval_mask], cal_preds[eval_mask]))

        lolo_rows.append(
            {
                "Held_Out_Location": target_loc,
                "Plots": int(len(y_te)),
                "Actual_Mean": round(float(np.mean(y_te)), 2),
                "Actual_Std": round(float(np.std(y_te)), 2),
                "ZeroShot_RMSE": round(zs_rmse, 4),
                "ZeroShot_MAE": round(zs_mae, 4),
                "ZeroShot_R2": round(zs_r2, 4),
                "Calibrated_RMSE": round(cal_rmse, 4),
                "Calibrated_MAE": round(cal_mae, 4),
                "Calibrated_R2": round(cal_r2, 4),
                "Within_City_Pearson_r": round(zs_r, 4),
            }
        )

    return pd.DataFrame(lolo_rows)


# ==============================================================================
# 7. VISUALIZATION DASHBOARD FOR THE 4 LOCATIONS + CRAWFORDSVILLE
# ==============================================================================
def plot_4locations_dashboard(
    val_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    lolo_df: pd.DataFrame,
    y_dev: np.ndarray,
    oof_preds: np.ndarray,
    loc_dev: np.ndarray,
    out_img_path: Path,
):
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12.5))
    fig.patch.set_facecolor("white")

    # Filter out the summary "ALL 4 LOCATIONS" row for per-city bars
    val_cities = val_df[~val_df["Location"].str.startswith("ALL")].reset_index(drop=True)
    cv_cities = cv_df[~cv_df["Location"].str.startswith("ALL")].reset_index(drop=True)

    x_pos = np.arange(len(val_cities))
    width = 0.35

    # Panel 1: Per-Location RMSE (Holdout Val vs 5-Fold OOF Test)
    b1 = ax1.bar(
        x_pos - width / 2,
        val_cities["RMSE"],
        width,
        label="Holdout Validation Split (330 Plots)",
        color="#1565c0",
        edgecolor="black",
    )
    b2 = ax1.bar(
        x_pos + width / 2,
        cv_cities["RMSE"],
        width,
        label="5-Fold Out-of-Fold Test (All 1,643 Plots)",
        color="#2e7d32",
        edgecolor="black",
    )
    for b in list(b1) + list(b2):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + 0.4,
            f"{b.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(val_cities["Location"], fontsize=10.5, fontweight="bold")
    ax1.set_ylim(0, max(float(val_cities["RMSE"].max()), float(cv_cities["RMSE"].max())) + 6.0)
    ax1.set_ylabel("RMSE (Bushels / Acre) — Lower is Better", fontsize=10.5, fontweight="bold")
    ax1.set_title("1. Per-Location Test RMSE Across the 4 Dev Locations", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9.5)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Panel 2: Per-Location Within-City Pearson r & R² (5-Fold OOF Test)
    b3 = ax2.bar(
        x_pos - width / 2,
        cv_cities["Pearson_r"],
        width,
        label="Within-City Pearson r (Plot Ranking)",
        color="#00838f",
        edgecolor="black",
    )
    b4 = ax2.bar(
        x_pos + width / 2,
        cv_cities["R2"],
        width,
        label="Within-City R² Score",
        color="#ef6c00",
        edgecolor="black",
    )
    for b in list(b3) + list(b4):
        v = b.get_height()
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            max(v, 0.0) + 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(cv_cities["Location"], fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Score (Higher is Better)", fontsize=10.5, fontweight="bold")
    ax2.set_title(
        f"2. Within-City Correlation (r) & R² (Combined 4-Loc R² = {cv_df.iloc[-1]['R2']:.4f})",
        fontsize=12,
        fontweight="bold",
    )
    ax2.legend(fontsize=9.5)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    # Panel 3: Actual vs Predicted Scatter Across All 1,643 Plots (Color-Coded by Location)
    palette = {
        "Ames": "#1e88e5",
        "Lincoln": "#43a047",
        "MOValley": "#fb8c00",
        "Scottsbluff": "#8e24aa",
    }
    for loc in sorted(np.unique(loc_dev)):
        m = loc_dev == loc
        r_loc = safe_pearson_r(y_dev[m], oof_preds[m])
        rmse_loc = float(np.sqrt(mean_squared_error(y_dev[m], oof_preds[m])))
        ax3.scatter(
            y_dev[m],
            oof_preds[m],
            alpha=0.55,
            s=26,
            color=palette.get(loc, "#333333"),
            label=f"{loc} (n={np.sum(m)}, RMSE={rmse_loc:.1f}, r={r_loc:.2f})",
        )
    min_v = float(min(y_dev.min(), oof_preds.min()))
    max_v = float(max(y_dev.max(), oof_preds.max()))
    ax3.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="1:1 Ideal Line")
    ax3.set_xlabel("Actual Yield (Bushels / Acre)", fontsize=10.5, fontweight="bold")
    ax3.set_ylabel("Predicted Yield (Bushels / Acre)", fontsize=10.5, fontweight="bold")
    ax3.set_title(
        f"3. All 4 Locations Out-of-Fold Parity Plot (1,643 Plots: RMSE={cv_df.iloc[-1]['RMSE']:.2f}, R²={cv_df.iloc[-1]['R2']:.4f})",
        fontsize=12,
        fontweight="bold",
    )
    ax3.legend(fontsize=9, loc="upper left")
    ax3.grid(True, linestyle=":", alpha=0.6)

    # Panel 4: Blind Leave-One-Location-Out (LOLO) Comparison Across All 5 Locations
    y_lolo = np.arange(len(lolo_df))
    ax4.barh(
        y_lolo - 0.18,
        lolo_df["Calibrated_RMSE"],
        height=0.34,
        color="#2e7d32",
        edgecolor="black",
        label="Field-Calibrated RMSE (15-Plot Strip)",
    )
    ax4.barh(
        y_lolo + 0.18,
        lolo_df["Actual_Std"],
        height=0.34,
        color="#90a4ae",
        edgecolor="black",
        label="Location Natural Yield Std Dev (R²=0 Threshold)",
    )
    for idx_r, r in lolo_df.iterrows():
        ax4.text(
            r["Calibrated_RMSE"] + 0.5,
            idx_r - 0.18,
            f"{r['Calibrated_RMSE']:.2f} bu/ac (r={r['Within_City_Pearson_r']:.2f})",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#1b5e20",
        )
    ax4.set_yticks(y_lolo)
    ax4.set_yticklabels(lolo_df["Held_Out_Location"], fontsize=10.5, fontweight="bold")
    ax4.invert_yaxis()
    ax4.set_xlim(0, max(float(lolo_df["Calibrated_RMSE"].max()), float(lolo_df["Actual_Std"].max())) + 14.0)
    ax4.set_xlabel("RMSE / Yield Spread (Bushels / Acre)", fontsize=10.5, fontweight="bold")
    ax4.set_title("4. Blind Leave-One-Location-Out (LOLO) Transfer Across All 5 Sites", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=9, loc="lower right")
    ax4.grid(axis="x", linestyle=":", alpha=0.6)

    fig.suptitle(
        "Step 6 Final Model Evaluation on the 4 Training/Validation Locations (Ames, Lincoln, MOValley, Scottsbluff)",
        fontsize=14.5,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_img_path, dpi=300)
    plt.close()


# ==============================================================================
# 8. MAIN EXECUTION
# ==============================================================================
def main():
    t0 = time.time()
    s5_dict, artifacts = load_saved_data()

    X_raw = s5_dict["X_raw"].astype(np.float32) if "X_raw" in s5_dict else s5_dict["X"].astype(np.float32)
    X_phys_raw = (
        s5_dict["X_phys_raw"].astype(np.float32)
        if "X_phys_raw" in s5_dict
        else s5_dict["X_phys"].astype(np.float32)
    )
    y = s5_dict["y"].astype(np.float32)
    splits = s5_dict["splits"]
    locations = s5_dict["locations"]
    plot_ids = s5_dict["plot_ids"]
    genotypes = s5_dict["genotypes"] if "genotypes" in s5_dict else np.array(["UNKNOWN"] * len(y))

    cfg1, cfg2, phys_cfg, blend_weights, alpha_shrink = get_step6_configs(artifacts)

    print("\n1. Evaluating Official Holdout Validation (330 Plots) & Train Split (1,313 Plots) per Location...")
    val_df, train_df, tr_preds, va_preds = evaluate_train_and_val_by_location(
        X_raw, y, splits, locations, cfg1, cfg2, blend_weights
    )

    print("2. Running 5-Fold Out-of-Fold Cross-Validation Across All 1,643 Plots of the 4 Locations...")
    cv_df, oof_preds = evaluate_5fold_cv_all_1643_plots(
        X_raw, y, splits, locations, cfg1, cfg2, blend_weights
    )

    print("3. Running Blind Leave-One-Location-Out (LOLO) Transfer Test Across Each Location...")
    lolo_df = evaluate_lolo_blind_city_transfer(
        X_raw,
        X_phys_raw,
        y,
        splits,
        locations,
        cfg1,
        cfg2,
        phys_cfg,
        blend_weights,
        alpha_shrink,
    )

    # Print Table 1: Official Holdout Validation Breakdown
    print("\n" + "=" * 115)
    print("TABLE 1: HOLDOUT VALIDATION SPLIT PERFORMANCE BY LOCATION (330 PLOTS ACROSS THE 4 LOCATIONS)")
    print("=" * 115)
    print(val_df.to_string(index=False))

    # Print Table 2: Training Split Breakdown
    print("\n" + "=" * 115)
    print("TABLE 2: TRAINING SPLIT PERFORMANCE BY LOCATION (1,313 PLOTS ACROSS THE 4 LOCATIONS)")
    print("=" * 115)
    print(train_df.to_string(index=False))

    # Print Table 3: Full 1,643-Plot 5-Fold Out-of-Fold Test Breakdown
    print("\n" + "=" * 115)
    print("TABLE 3: 5-FOLD OUT-OF-FOLD TEST PERFORMANCE ACROSS ALL 1,643 PLOTS OF THE 4 LOCATIONS")
    print("=" * 115)
    print(cv_df.to_string(index=False))

    # Print Table 4: Blind Leave-One-Location-Out (LOLO) Transfer per City
    print("\n" + "=" * 125)
    print("TABLE 4: BLIND LEAVE-ONE-LOCATION-OUT (LOLO) TEST — EACH LOCATION HELD OUT AS AN UNSEEN FARM")
    print("=" * 125)
    print(lolo_df.to_string(index=False))
    print("=" * 125)

    # Export CSV Reports
    val_df.to_csv(OUTPUT_DIR / "table1_holdout_validation_by_location.csv", index=False)
    train_df.to_csv(OUTPUT_DIR / "table2_training_split_by_location.csv", index=False)
    cv_df.to_csv(OUTPUT_DIR / "table3_5fold_oof_all1643_by_location.csv", index=False)
    lolo_df.to_csv(OUTPUT_DIR / "table4_blind_lolo_transfer_by_location.csv", index=False)

    # Save Plot-by-Plot Predictions for all 1,643 plots of the 4 locations
    dev_mask = (splits == "train") | (splits == "val")
    in_split_preds = np.zeros(len(y), dtype=np.float32)
    in_split_preds[splits == "train"] = tr_preds
    in_split_preds[splits == "val"] = va_preds

    dev_preds_df = pd.DataFrame(
        {
            "plot_id": plot_ids[dev_mask],
            "location": locations[dev_mask],
            "split": splits[dev_mask],
            "genotype": genotypes[dev_mask],
            "Actual_Yield_bu_ac": y[dev_mask],
            "TrainVal_Split_Predicted_Yield": in_split_preds[dev_mask],
            "OOF_5Fold_Test_Predicted_Yield": oof_preds,
            "OOF_Absolute_Error_bu_ac": np.abs(y[dev_mask] - oof_preds),
        }
    )
    preds_csv = OUTPUT_DIR / "other_4_locations_1643_plot_predictions.csv"
    dev_preds_df.to_csv(preds_csv, index=False)

    # Generate Dashboard Chart
    chart_path = OUTPUT_DIR / "other_4_locations_evaluation_dashboard.png"
    plot_4locations_dashboard(
        val_df, cv_df, lolo_df, y[dev_mask], oof_preds, locations[dev_mask], chart_path
    )

    print(f"\n[SAVED ARTIFACTS IN {OUTPUT_DIR}]")
    print(f"  -> Plot-by-Plot Predictions (1,643 plots) : {preds_csv}")
    print(f"  -> 4-Location Visual Dashboard (.PNG)     : {chart_path}")
    print(f"  -> Total Evaluation Runtime               : {time.time() - t0:.1f}s")

    try:
        os.startfile(str(chart_path))
    except Exception:
        pass


if __name__ == "__main__":
    main()