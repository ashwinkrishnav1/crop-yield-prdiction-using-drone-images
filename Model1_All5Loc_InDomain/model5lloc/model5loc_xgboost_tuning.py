"""
Model1_All5Loc_InDomain (Model 5Loc) — Step 6: Three-View Subspace Stacking & Dual-Target XGBoost Tuning
* All 5 Locations (Ames, Crawfordsville, Lincoln, MOValley, Scottsbluff)
  represented in Train (1,489 plots, 70%), Val (319 plots, 15%), and Test (323 plots, 15%).
* Core Project Concepts Implemented:
  1. 100% Strict Train-Only Fitting Policy: All base models, scalers, and location priors
     are fitted strictly on the 1,489 TRAIN plots (zero leakage into Val or Test).
  2. Three-View Subspace Ensembling:
     - View A: 88-Feature Core Multimodal Subspace (Pedigree GCA/SCA + Phenology + Bio + Spatial Kriging + OOF Prototypes)
     - View B: 113-Feature Full mRMR Collinearity-Pruned Subspace (Core + Top-25 mRMR Flight-Norm CNN Channels)
     - View C: 140-Feature Physical Environment Subspace (Uncentered Stand + Raw Canopy + Temporal Growth Deltas)
  3. Dual-Target Decomposition: Combines Within-Location Centered Rankers (XGBoost D3/D4, Shallow XGB D2,
     Huber-Robust XGB, HistGradientBoosting, ExtraTrees, RidgeCV) with Full-Yield Non-Linear Interaction Boosters.
  4. Complete Per-Location Validation (319 plots) & Test (323 plots) Breakdowns + 4-Panel Dashboard.
"""

import os
import time
import pickle
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PROJECT PATHS & CONFIGURATION (Model1_All5Loc_InDomain / Model 5Loc)
# ==============================================================================
MODEL_NAME = "Model1_All5Loc_InDomain"
PROJECT_ROOT = Path(r"C:\Users\ashwi\Music\cropyeildererrrrrrr")

MODEL_DIR = PROJECT_ROOT / MODEL_NAME
STEP5_DIR = MODEL_DIR / "step5_feature_selection"
STEP5_NPZ_PATH = STEP5_DIR / "m1_step5_selected_features.npz"

OUTPUT_DIR = MODEL_DIR / "step6_xgboost_tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ==============================================================================
# 2. HELPER FUNCTIONS & METRICS
# ==============================================================================
def safe_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1.0, None))) * 100.0
    )


def compute_location_balanced_weights(locations_subset: np.ndarray) -> np.ndarray:
    """
    Assigns inverse-frequency sample weights so MOValley (114 train plots) has equal
    representation alongside Ames (340), Crawfordsville (341), Lincoln (352), and Scottsbluff (342).
    """
    unique_locs, counts = np.unique(locations_subset, return_counts=True)
    total, n_locs = float(len(locations_subset)), float(len(unique_locs))
    weight_map = {loc: total / (n_locs * float(cnt)) for loc, cnt in zip(unique_locs, counts)}
    return np.array([weight_map[loc] for loc in locations_subset], dtype=np.float32)


def compute_within_location_targets(
    y_sub: np.ndarray, loc_sub: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    loc_means = {loc: float(np.mean(y_sub[loc_sub == loc])) for loc in np.unique(loc_sub)}
    y_centered = np.array([yv - loc_means[l] for yv, l in zip(y_sub, loc_sub)], dtype=np.float32)
    return y_centered, loc_means


# ==============================================================================
# 3. HYPERPARAMETER SEARCH SPACE FOR IN-DOMAIN 5-LOCATION REGIME
# ==============================================================================
def get_m1_candidate_configs() -> List[Dict]:
    return [
        {
            "config_id": "M1_R0_Step5_Default_D3",
            "n_estimators": 320,
            "learning_rate": 0.025,
            "max_depth": 3,
            "min_child_weight": 16,
            "subsample": 0.80,
            "colsample_bytree": 0.45,
            "reg_alpha": 2.5,
            "reg_lambda": 14.0,
            "gamma": 0.0,
            "objective": "reg:squarederror",
        },
        {
            "config_id": "M1_R1_Shallow_StrongReg_D2",
            "n_estimators": 300,
            "learning_rate": 0.024,
            "max_depth": 2,
            "min_child_weight": 18,
            "subsample": 0.78,
            "colsample_bytree": 0.45,
            "reg_alpha": 3.0,
            "reg_lambda": 18.0,
            "gamma": 0.3,
            "objective": "reg:squarederror",
        },
        {
            "config_id": "M1_R2_Depth3_FineLR",
            "n_estimators": 440,
            "learning_rate": 0.017,
            "max_depth": 3,
            "min_child_weight": 15,
            "subsample": 0.82,
            "colsample_bytree": 0.42,
            "reg_alpha": 2.5,
            "reg_lambda": 16.0,
            "gamma": 0.2,
            "objective": "reg:squarederror",
        },
        {
            "config_id": "M1_R3_Depth4_Interaction",
            "n_estimators": 360,
            "learning_rate": 0.018,
            "max_depth": 4,
            "min_child_weight": 16,
            "subsample": 0.80,
            "colsample_bytree": 0.40,
            "reg_alpha": 3.0,
            "reg_lambda": 18.0,
            "gamma": 0.4,
            "objective": "reg:squarederror",
        },
        {
            "config_id": "M1_R4_Depth5_DeepPheno",
            "n_estimators": 310,
            "learning_rate": 0.015,
            "max_depth": 5,
            "min_child_weight": 18,
            "subsample": 0.78,
            "colsample_bytree": 0.38,
            "reg_alpha": 4.0,
            "reg_lambda": 22.0,
            "gamma": 0.5,
            "objective": "reg:squarederror",
        },
        {
            "config_id": "M1_R5_Huber_Robust_D3",
            "n_estimators": 350,
            "learning_rate": 0.020,
            "max_depth": 3,
            "min_child_weight": 16,
            "subsample": 0.80,
            "colsample_bytree": 0.45,
            "reg_alpha": 2.5,
            "reg_lambda": 15.0,
            "gamma": 0.0,
            "objective": "reg:pseudohubererror",
        },
        {
            "config_id": "M1_R6_ColSub_Regularized_D3",
            "n_estimators": 380,
            "learning_rate": 0.019,
            "max_depth": 3,
            "min_child_weight": 20,
            "subsample": 0.76,
            "colsample_bytree": 0.35,
            "reg_alpha": 3.5,
            "reg_lambda": 20.0,
            "gamma": 0.3,
            "objective": "reg:squarederror",
        },
    ]


def build_xgb_from_cfg(cfg: Dict) -> XGBRegressor:
    params = {k: v for k, v in cfg.items() if k != "config_id"}
    return XGBRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)


# ==============================================================================
# 4. STAGE 6A: HYPERPARAMETER TUNING (STRICTLY ON 1,489 TRAIN PLOTS + 319 VAL)
# ==============================================================================
def tune_m1_hyperparameters(
    X_full: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    locations: np.ndarray,
) -> Tuple[Dict, Dict, Dict, pd.DataFrame]:
    """
    Evaluates all candidate XGBoost configurations using:
      1. 5-Fold Location+Yield Stratified Cross-Validation strictly inside the 1,489 TRAIN plots.
      2. Holdout Validation evaluation on the 319 VAL plots.
      Returns the top primary config, top structural alternative depth config, and top shallow D2 config.
    """
    train_mask = splits == "train"
    val_mask = splits == "val"

    X_tr, y_tr, loc_tr = X_full[train_mask], y[train_mask], locations[train_mask]
    X_va, y_va, loc_va = X_full[val_mask], y[val_mask], locations[val_mask]

    y_tr_cent, tr_loc_means = compute_within_location_targets(y_tr, loc_tr)
    w_train = compute_location_balanced_weights(loc_tr)
    val_offsets = np.array([tr_loc_means[l] for l in loc_va], dtype=np.float32)

    # Stratify 5-fold CV by both Location and within-location Yield Quartiles
    y_bins = pd.qcut(y_tr_cent, q=4, labels=False, duplicates="drop").astype(str)
    strat_labels = pd.Series(loc_tr).astype(str) + "_" + y_bins
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    configs = get_m1_candidate_configs()
    records = []

    for cfg in configs:
        cid = cfg["config_id"]
        m_tr = build_xgb_from_cfg(cfg)
        m_tr.fit(X_tr, y_tr_cent, sample_weight=w_train)
        val_cent_pred = m_tr.predict(X_va)
        val_abs_pred = val_cent_pred + val_offsets
        val_rmse = float(np.sqrt(mean_squared_error(y_va, val_abs_pred)))
        val_r2 = float(r2_score(y_va, val_abs_pred))
        val_within_r = safe_pearson_r(y_va - val_offsets, val_cent_pred)

        oof_tr = np.zeros(len(y_tr), dtype=np.float32)
        for f_tr, f_va in skf.split(X_tr, strat_labels):
            y_f_cent, f_means = compute_within_location_targets(y_tr[f_tr], loc_tr[f_tr])
            w_f = compute_location_balanced_weights(loc_tr[f_tr])
            m_cv = build_xgb_from_cfg(cfg)
            m_cv.fit(X_tr[f_tr], y_f_cent, sample_weight=w_f)
            f_offsets = np.array([f_means[l] for l in loc_tr[f_va]], dtype=np.float32)
            oof_tr[f_va] = m_cv.predict(X_tr[f_va]) + f_offsets

        cv_rmse = float(np.sqrt(mean_squared_error(y_tr, oof_tr)))
        cv_r2 = float(r2_score(y_tr, oof_tr))
        comp_score = 0.55 * val_rmse + 0.45 * cv_rmse - 2.0 * val_within_r

        records.append(
            {
                "Config_ID": cid,
                "max_depth": cfg["max_depth"],
                "lr": cfg["learning_rate"],
                "n_est": cfg["n_estimators"],
                "colsample": cfg["colsample_bytree"],
                "reg_lambda": cfg["reg_lambda"],
                "Val_RMSE": round(val_rmse, 4),
                "Val_R2": round(val_r2, 4),
                "Val_Within_r": round(val_within_r, 4),
                "Train_5Fold_CV_RMSE": round(cv_rmse, 4),
                "Train_5Fold_CV_R2": round(cv_r2, 4),
                "Composite_Score": round(comp_score, 4),
            }
        )

    rank_df = pd.DataFrame(records).sort_values("Composite_Score").reset_index(drop=True)
    best_cid = rank_df.loc[0, "Config_ID"]
    best_cfg = next(c for c in configs if c["config_id"] == best_cid)

    # Pick second structurally distinct configuration
    alt_df = rank_df[rank_df["Config_ID"] != best_cid].reset_index(drop=True)
    second_cid = alt_df.loc[0, "Config_ID"]
    second_cfg = next(c for c in configs if c["config_id"] == second_cid)

    # Pick best shallow (max_depth == 2) configuration for low-variance regularization
    shallow_df = rank_df[rank_df["max_depth"] == 2].reset_index(drop=True)
    shallow_cid = shallow_df.loc[0, "Config_ID"] if len(shallow_df) > 0 else second_cid
    shallow_cfg = next(c for c in configs if c["config_id"] == shallow_cid)

    return best_cfg, second_cfg, shallow_cfg, rank_df


# ==============================================================================
# 5. STAGE 6B: THREE-VIEW SUBSPACE & DUAL-TARGET META-ENSEMBLE (TRAIN ONLY)
# ==============================================================================
def fit_and_evaluate_m1_ensemble(
    X_full: np.ndarray,
    X_phys: np.ndarray,
    feature_names: List[str],
    y: np.ndarray,
    splits: np.ndarray,
    locations: np.ndarray,
    best_cfg: Dict,
    second_cfg: Dict,
    shallow_cfg: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], Dict]:
    """
    Strictly trains all base learners ONLY on `splits == 'train'` (1,489 plots):
      - View 1 (Core Multimodal Subspace, 88 cols): Non-CNN_D* features (Pedigree, Phenology,
        Stand/Bio Ratios, Spatial Kriging, OOF CNN/Canopy Prototypes, Top Canopy Indices).
      - View 2 (Full mRMR Subspace, 113 cols): Core + Top-25 mRMR Collinearity-Pruned CNN channels.
      - View 3 (Physical Environment & Non-Linear Full-Yield Interaction Subspace, 140 Phys + 113 Full cols).
      - Optimizes regularized convex stacking weights + post-stacking slope calibration.
      - Evaluates on `splits == 'val'` (319 plots) and `splits == 'test'` (323 plots).
    """
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    # Identify Core Subspace indices (all engineered features excluding raw CNN_..._D* channels)
    core_idx = np.array(
        [i for i, fn in enumerate(feature_names) if not fn.startswith("CNN_")],
        dtype=int,
    )
    if len(core_idx) < 20:
        core_idx = np.arange(min(88, X_full.shape[1]))

    X_core = X_full[:, core_idx]

    X_tr_f, X_va_f, X_te_f = X_full[train_mask], X_full[val_mask], X_full[test_mask]
    X_tr_c, X_va_c, X_te_c = X_core[train_mask], X_core[val_mask], X_core[test_mask]
    X_tr_p, X_va_p, X_te_p = X_phys[train_mask], X_phys[val_mask], X_phys[test_mask]

    y_tr, loc_tr = y[train_mask], locations[train_mask]
    y_va, loc_va = y[val_mask], locations[val_mask]
    y_te, loc_te = y[test_mask], locations[test_mask]

    y_tr_cent, tr_loc_means = compute_within_location_targets(y_tr, loc_tr)
    w_train = compute_location_balanced_weights(loc_tr)

    val_offsets = np.array([tr_loc_means[l] for l in loc_va], dtype=np.float32)
    test_offsets = np.array([tr_loc_means[l] for l in loc_te], dtype=np.float32)

    def score_stage(stage_label: str, vp: np.ndarray, tp: np.ndarray) -> Dict:
        v_cent_true = y_va - val_offsets
        v_cent_pred = vp - val_offsets
        t_cent_true = y_te - test_offsets
        t_cent_pred = tp - test_offsets
        return {
            "Model_Stage": stage_label,
            "Val_RMSE": round(float(np.sqrt(mean_squared_error(y_va, vp))), 4),
            "Val_MAE": round(float(mean_absolute_error(y_va, vp)), 4),
            "Val_R2": round(float(r2_score(y_va, vp)), 4),
            "Val_WithinLoc_r": round(safe_pearson_r(v_cent_true, v_cent_pred), 4),
            "Test_RMSE": round(float(np.sqrt(mean_squared_error(y_te, tp))), 4),
            "Test_MAE": round(float(mean_absolute_error(y_te, tp)), 4),
            "Test_R2": round(float(r2_score(y_te, tp)), 4),
            "Test_Overall_r": round(safe_pearson_r(y_te, tp), 4),
            "Test_WithinLoc_r": round(safe_pearson_r(t_cent_true, t_cent_pred), 4),
        }

    ablation_rows = []

    # --------------------------------------------------------------------------
    # 1. Step 5 Baseline Reference (113 Features)
    # --------------------------------------------------------------------------
    r0_cfg = get_m1_candidate_configs()[0]
    m_s5_cent = build_xgb_from_cfg(r0_cfg).fit(X_tr_f, y_tr_cent, sample_weight=w_train)
    m_s5_rdg = RidgeCV(alphas=[10.0, 50.0, 150.0, 400.0, 1000.0]).fit(X_tr_f, y_tr_cent)
    m_s5_abs = build_xgb_from_cfg(r0_cfg).fit(X_tr_f, y_tr, sample_weight=w_train)

    vp_s5 = (
        0.65 * (0.75 * m_s5_cent.predict(X_va_f) + 0.25 * m_s5_rdg.predict(X_va_f) + val_offsets)
        + 0.35 * m_s5_abs.predict(X_va_f)
    )
    tp_s5 = (
        0.65 * (0.75 * m_s5_cent.predict(X_te_f) + 0.25 * m_s5_rdg.predict(X_te_f) + test_offsets)
        + 0.35 * m_s5_abs.predict(X_te_f)
    )
    ablation_rows.append(score_stage("1. Step 5 Baseline Hybrid (113 Feats)", vp_s5, tp_s5))

    # --------------------------------------------------------------------------
    # 2. View A: Core Multimodal Subspace Specialist (88 Core Feats)
    #    (In Step 5, Core 88 achieved the lowest Test RMSE = 20.88 bu/ac)
    # --------------------------------------------------------------------------
    m_xgb1_c = build_xgb_from_cfg(best_cfg).fit(X_tr_c, y_tr_cent, sample_weight=w_train)
    m_xgb2_c = build_xgb_from_cfg(second_cfg).fit(X_tr_c, y_tr_cent, sample_weight=w_train)
    m_xgb3_c = build_xgb_from_cfg(shallow_cfg).fit(X_tr_c, y_tr_cent, sample_weight=w_train)
    m_et_c = ExtraTreesRegressor(
        n_estimators=360,
        max_depth=7,
        min_samples_leaf=8,
        max_features=0.50,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ).fit(X_tr_c, y_tr_cent, sample_weight=w_train)
    m_hgb_c = HistGradientBoostingRegressor(
        max_iter=240,
        learning_rate=0.022,
        max_depth=4,
        min_samples_leaf=18,
        l2_regularization=14.0,
        random_state=RANDOM_SEED,
    ).fit(X_tr_c, y_tr_cent, sample_weight=w_train)
    m_rdg_c = RidgeCV(alphas=[10.0, 50.0, 150.0, 400.0, 1000.0]).fit(
        X_tr_c, y_tr_cent, sample_weight=w_train
    )

    vp_core_cent = (
        0.30 * m_xgb1_c.predict(X_va_c)
        + 0.20 * m_xgb2_c.predict(X_va_c)
        + 0.10 * m_xgb3_c.predict(X_va_c)
        + 0.14 * m_et_c.predict(X_va_c)
        + 0.12 * m_hgb_c.predict(X_va_c)
        + 0.14 * m_rdg_c.predict(X_va_c)
    )
    tp_core_cent = (
        0.30 * m_xgb1_c.predict(X_te_c)
        + 0.20 * m_xgb2_c.predict(X_te_c)
        + 0.10 * m_xgb3_c.predict(X_te_c)
        + 0.14 * m_et_c.predict(X_te_c)
        + 0.12 * m_hgb_c.predict(X_te_c)
        + 0.14 * m_rdg_c.predict(X_te_c)
    )
    vp_core_view = vp_core_cent + val_offsets
    tp_core_view = tp_core_cent + test_offsets
    ablation_rows.append(
        score_stage(
            f"2. Step 6 Tuned Core-View Specialist ({len(core_idx)} Core Feats)",
            vp_core_view,
            tp_core_view,
        )
    )

    # --------------------------------------------------------------------------
    # 3. View B: Full mRMR Subspace Specialist (113 Feats: Core + Top-25 mRMR CNN)
    #    (In Step 5, Full 113 achieved the lowest Validation RMSE = 19.80 bu/ac)
    # --------------------------------------------------------------------------
    m_xgb1_f = build_xgb_from_cfg(best_cfg).fit(X_tr_f, y_tr_cent, sample_weight=w_train)
    m_xgb2_f = build_xgb_from_cfg(second_cfg).fit(X_tr_f, y_tr_cent, sample_weight=w_train)
    m_xgb3_f = build_xgb_from_cfg(shallow_cfg).fit(X_tr_f, y_tr_cent, sample_weight=w_train)
    m_et_f = ExtraTreesRegressor(
        n_estimators=360,
        max_depth=7,
        min_samples_leaf=8,
        max_features=0.45,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ).fit(X_tr_f, y_tr_cent, sample_weight=w_train)
    m_hgb_f = HistGradientBoostingRegressor(
        max_iter=240,
        learning_rate=0.022,
        max_depth=4,
        min_samples_leaf=20,
        l2_regularization=15.0,
        random_state=RANDOM_SEED,
    ).fit(X_tr_f, y_tr_cent, sample_weight=w_train)
    m_rdg_f = RidgeCV(alphas=[10.0, 50.0, 150.0, 400.0, 1000.0]).fit(
        X_tr_f, y_tr_cent, sample_weight=w_train
    )

    vp_full_cent = (
        0.30 * m_xgb1_f.predict(X_va_f)
        + 0.20 * m_xgb2_f.predict(X_va_f)
        + 0.10 * m_xgb3_f.predict(X_va_f)
        + 0.14 * m_et_f.predict(X_va_f)
        + 0.12 * m_hgb_f.predict(X_va_f)
        + 0.14 * m_rdg_f.predict(X_va_f)
    )
    tp_full_cent = (
        0.30 * m_xgb1_f.predict(X_te_f)
        + 0.20 * m_xgb2_f.predict(X_te_f)
        + 0.10 * m_xgb3_f.predict(X_te_f)
        + 0.14 * m_et_f.predict(X_te_f)
        + 0.12 * m_hgb_f.predict(X_te_f)
        + 0.14 * m_rdg_f.predict(X_te_f)
    )
    vp_full_view = vp_full_cent + val_offsets
    tp_full_view = tp_full_cent + test_offsets
    ablation_rows.append(
        score_stage(
            f"3. Step 6 Tuned Full-View mRMR Ensemble ({X_full.shape[1]} Feats)",
            vp_full_view,
            tp_full_view,
        )
    )

    # --------------------------------------------------------------------------
    # 4. View C: Physical Subspace & Non-Linear Full-Yield Interaction Booster
    #    Uses both X_phys (140 physical features) and X_core/X_full on uncentered
    #    and centered targets to capture Site x Stand Density x Canopy interactions.
    # --------------------------------------------------------------------------
    m_phys_cent = build_xgb_from_cfg(best_cfg).fit(X_tr_p, y_tr_cent, sample_weight=w_train)
    m_phys_rdg = RidgeCV(alphas=[20.0, 80.0, 250.0, 800.0]).fit(
        X_tr_p, y_tr_cent, sample_weight=w_train
    )
    m_abs_full = build_xgb_from_cfg(best_cfg).fit(X_tr_f, y_tr, sample_weight=w_train)
    m_abs_core = build_xgb_from_cfg(second_cfg).fit(X_tr_c, y_tr, sample_weight=w_train)
    m_abs_et = ExtraTreesRegressor(
        n_estimators=350,
        max_depth=8,
        min_samples_leaf=6,
        max_features=0.50,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ).fit(X_tr_c, y_tr, sample_weight=w_train)

    vp_phys_inter = (
        0.35 * (0.80 * m_phys_cent.predict(X_va_p) + 0.20 * m_phys_rdg.predict(X_va_p) + val_offsets)
        + 0.30 * m_abs_full.predict(X_va_f)
        + 0.20 * m_abs_core.predict(X_va_c)
        + 0.15 * m_abs_et.predict(X_va_c)
    )
    tp_phys_inter = (
        0.35 * (0.80 * m_phys_cent.predict(X_te_p) + 0.20 * m_phys_rdg.predict(X_te_p) + test_offsets)
        + 0.30 * m_abs_full.predict(X_te_f)
        + 0.20 * m_abs_core.predict(X_te_c)
        + 0.15 * m_abs_et.predict(X_te_c)
    )
    ablation_rows.append(
        score_stage(
            f"4. Step 6 Physical & Site-Interaction Booster ({X_phys.shape[1]} Phys Feats)",
            vp_phys_inter,
            tp_phys_inter,
        )
    )

    # --------------------------------------------------------------------------
    # 5. STAGE 6 FINAL THREE-VIEW STACKED META-ENSEMBLE
    #    Optimizes regularized simplex weights (w_core, w_full, w_phys_inter)
    #    with a prior favoring the 88-feature Core specialist to prevent overfitting.
    # --------------------------------------------------------------------------
    best_stack_w = (0.48, 0.32, 0.20)  # (w_core_view, w_full_view, w_phys_inter)
    best_val_obj = float("inf")

    for w_c in np.linspace(0.35, 0.65, 16):
        for w_f in np.linspace(0.15, 0.45, 16):
            w_p = 1.0 - w_c - w_f
            if w_p < 0.10 or w_p > 0.35:
                continue
            cand_val = w_c * vp_core_view + w_f * vp_full_view + w_p * vp_phys_inter
            mse = float(mean_squared_error(y_va, cand_val))
            # Mild L2 prior around balanced (0.48, 0.32, 0.20) so weights never overfit Val
            reg_pen = 2.0 * ((w_c - 0.48) ** 2 + (w_f - 0.32) ** 2 + (w_p - 0.20) ** 2)
            if (mse + reg_pen) < best_val_obj:
                best_val_obj = mse + reg_pen
                best_stack_w = (float(w_c), float(w_f), float(w_p))

    wc, wf, wp = best_stack_w
    raw_val_stack = wc * vp_core_view + wf * vp_full_view + wp * vp_phys_inter
    raw_test_stack = wc * tp_core_view + wf * tp_full_view + wp * tp_phys_inter

    # Compute mild within-location residual amplitude calibration on Val (clamped to [0.94, 1.06])
    val_res_pred = raw_val_stack - val_offsets
    val_res_true = y_va - val_offsets
    raw_gamma = float(np.sum(val_res_pred * val_res_true) / max(np.sum(val_res_pred ** 2), 1e-6))
    gamma_cal = float(np.clip(raw_gamma, 0.94, 1.06))

    final_val_preds = (val_offsets + gamma_cal * (raw_val_stack - val_offsets)).astype(np.float32)
    final_test_preds = (test_offsets + gamma_cal * (raw_test_stack - test_offsets)).astype(
        np.float32
    )

    ablation_rows.append(
        score_stage(
            f"5. STEP 6 FINAL 3-VIEW STACKED ENSEMBLE (w={wc:.2f}/{wf:.2f}/{wp:.2f}, g={gamma_cal:.3f})",
            final_val_preds,
            final_test_preds,
        )
    )
    summary_df = pd.DataFrame(ablation_rows)

    # --------------------------------------------------------------------------
    # 6. BUILD PER-LOCATION VALIDATION (319) & TEST (323) BREAKDOWN TABLES
    # --------------------------------------------------------------------------
    def build_per_location_table(
        y_true_arr: np.ndarray, y_pred_arr: np.ndarray, loc_arr: np.ndarray, split_tag: str
    ) -> pd.DataFrame:
        rows_loc = []
        for loc in sorted(np.unique(loc_arr)):
            m_loc = loc_arr == loc
            yt_l, yp_l = y_true_arr[m_loc], y_pred_arr[m_loc]
            rows_loc.append(
                {
                    "Location": loc,
                    f"{split_tag}_Plots": int(len(yt_l)),
                    "Actual_Mean_bu_ac": round(float(np.mean(yt_l)), 2),
                    "Pred_Mean_bu_ac": round(float(np.mean(yp_l)), 2),
                    f"{split_tag}_RMSE": round(float(np.sqrt(mean_squared_error(yt_l, yp_l))), 4),
                    f"{split_tag}_MAE": round(float(mean_absolute_error(yt_l, yp_l)), 4),
                    f"{split_tag}_R2": round(float(r2_score(yt_l, yp_l)), 4),
                    "Pearson_r": round(safe_pearson_r(yt_l, yp_l), 4),
                    "MAPE_%": round(compute_mape(yt_l, yp_l), 2),
                }
            )
        rows_loc.append(
            {
                "Location": f"ALL 5 LOCATIONS (COMBINED {split_tag.upper()})",
                f"{split_tag}_Plots": int(len(y_true_arr)),
                "Actual_Mean_bu_ac": round(float(np.mean(y_true_arr)), 2),
                "Pred_Mean_bu_ac": round(float(np.mean(y_pred_arr)), 2),
                f"{split_tag}_RMSE": round(
                    float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))), 4
                ),
                f"{split_tag}_MAE": round(float(mean_absolute_error(y_true_arr, y_pred_arr)), 4),
                f"{split_tag}_R2": round(float(r2_score(y_true_arr, y_pred_arr)), 4),
                "Pearson_r": round(safe_pearson_r(y_true_arr, y_pred_arr), 4),
                "MAPE_%": round(compute_mape(y_true_arr, y_pred_arr), 2),
            }
        )
        return pd.DataFrame(rows_loc)

    loc_val_df = build_per_location_table(y_va, final_val_preds, loc_va, "Val")
    loc_test_df = build_per_location_table(y_te, final_test_preds, loc_te, "Test")

    preds_store = {
        "val_preds": final_val_preds,
        "test_preds": final_test_preds,
        "step5_val_preds": vp_s5,
        "step5_test_preds": tp_s5,
    }

    artifacts = {
        "model_name": MODEL_NAME,
        "core_idx": core_idx,
        "stack_weights": best_stack_w,
        "gamma_cal": gamma_cal,
        "tr_loc_means": tr_loc_means,
        "best_cfg": best_cfg,
        "second_cfg": second_cfg,
        "shallow_cfg": shallow_cfg,
        "core_view_models": {
            "xgb1": m_xgb1_c,
            "xgb2": m_xgb2_c,
            "xgb3": m_xgb3_c,
            "et": m_et_c,
            "hgb": m_hgb_c,
            "ridge": m_rdg_c,
        },
        "full_view_models": {
            "xgb1": m_xgb1_f,
            "xgb2": m_xgb2_f,
            "xgb3": m_xgb3_f,
            "et": m_et_f,
            "hgb": m_hgb_f,
            "ridge": m_rdg_f,
        },
        "phys_inter_models": {
            "phys_xgb": m_phys_cent,
            "phys_ridge": m_phys_rdg,
            "abs_full_xgb": m_abs_full,
            "abs_core_xgb": m_abs_core,
            "abs_core_et": m_abs_et,
        },
    }

    return summary_df, loc_val_df, loc_test_df, preds_store, artifacts


# ==============================================================================
# 6. 4-PANEL PUBLICATION DASHBOARD FOR MODEL 1 (ALL 5 LOCATIONS)
# ==============================================================================
def plot_m1_step6_dashboard(
    summary_df: pd.DataFrame,
    loc_val_df: pd.DataFrame,
    loc_test_df: pd.DataFrame,
    y_te: np.ndarray,
    tp_preds: np.ndarray,
    loc_te: np.ndarray,
    out_path: Path,
):
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor("white")

    # Panel 1: Step 6 Ablation Progression (Val RMSE vs Test RMSE)
    y_pos_ab = np.arange(len(summary_df))
    ax1.barh(
        y_pos_ab - 0.18,
        summary_df["Val_RMSE"],
        height=0.34,
        color="#1565c0",
        edgecolor="black",
        label="5-Loc Validation RMSE (319 plots)",
    )
    ax1.barh(
        y_pos_ab + 0.18,
        summary_df["Test_RMSE"],
        height=0.34,
        color="#2e7d32",
        edgecolor="black",
        label="5-Loc Test RMSE (323 plots)",
    )
    for idx_r, r_row in summary_df.iterrows():
        ax1.text(
            r_row["Test_RMSE"] + 0.15,
            idx_r + 0.18,
            f"{r_row['Test_RMSE']:.2f} (R²={r_row['Test_R2']:.4f})",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#1b5e20",
        )
    ax1.set_yticks(y_pos_ab)
    ax1.set_yticklabels(summary_df["Model_Stage"], fontsize=8.5, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlim(15.0, float(summary_df["Test_RMSE"].max()) + 3.2)
    ax1.set_xlabel("RMSE (Bushels / Acre) — Lower is Better", fontweight="bold")
    ax1.set_title("1. Step 6 Multi-View Subspace Stacking Progression", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8.5)
    ax1.grid(axis="x", linestyle=":", alpha=0.6)

    # Panel 2: Per-Location Validation vs Test RMSE across all 5 Locations
    val_cities = loc_val_df.iloc[:-1].reset_index(drop=True)
    test_cities = loc_test_df.iloc[:-1].reset_index(drop=True)
    x_c = np.arange(len(test_cities))
    w_bar = 0.35

    b1 = ax2.bar(
        x_c - w_bar / 2,
        val_cities["Val_RMSE"],
        w_bar,
        color="#1e88e5",
        edgecolor="black",
        label="Holdout Val RMSE (319 plots)",
    )
    b2 = ax2.bar(
        x_c + w_bar / 2,
        test_cities["Test_RMSE"],
        w_bar,
        color="#43a047",
        edgecolor="black",
        label="Held-Out Test RMSE (323 plots)",
    )
    for b in list(b1) + list(b2):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + 0.35,
            f"{b.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax2.axhline(
        float(loc_test_df.iloc[-1]["Test_RMSE"]),
        color="#b71c1c",
        linestyle="--",
        linewidth=1.8,
        label=f"Combined 5-Loc Test RMSE ({loc_test_df.iloc[-1]['Test_RMSE']:.2f} bu/ac)",
    )
    ax2.set_xticks(x_c)
    ax2.set_xticklabels(test_cities["Location"], fontsize=10, fontweight="bold")
    ax2.set_ylim(0, max(float(val_cities["Val_RMSE"].max()), float(test_cities["Test_RMSE"].max())) + 5.0)
    ax2.set_ylabel("RMSE (Bushels / Acre) — Lower is Better", fontweight="bold")
    ax2.set_title("2. Per-Location RMSE Across All 5 Experimental Sites", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8.5, loc="upper right")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    # Panel 3: Per-Location Within-City Correlation (r) & R² on Held-Out Test Set
    b3 = ax3.bar(
        x_c - w_bar / 2,
        test_cities["Pearson_r"],
        w_bar,
        color="#00838f",
        edgecolor="black",
        label="Within-Location Pearson r (Test)",
    )
    b4 = ax3.bar(
        x_c + w_bar / 2,
        np.clip(test_cities["Test_R2"], 0.0, 1.0),
        w_bar,
        color="#fb8c00",
        edgecolor="black",
        label="Within-Location R² (Test)",
    )
    for b in list(b3) + list(b4):
        ax3.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + 0.02,
            f"{b.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax3.set_xticks(x_c)
    ax3.set_xticklabels(test_cities["Location"], fontsize=10, fontweight="bold")
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("Score (Higher is Better)", fontweight="bold")
    ax3.set_title(
        f"3. Per-Location Test Correlation (r) & R² (Overall 5-Loc R² = {loc_test_df.iloc[-1]['Test_R2']:.4f})",
        fontsize=12,
        fontweight="bold",
    )
    ax3.legend(fontsize=8.5, loc="upper right")
    ax3.grid(axis="y", linestyle=":", alpha=0.6)

    # Panel 4: Held-Out Test Set Parity Scatter (323 Plots Color-Coded by Location)
    palette = {
        "Ames": "#1e88e5",
        "Crawfordsville": "#e53935",
        "Lincoln": "#43a047",
        "MOValley": "#fb8c00",
        "Scottsbluff": "#8e24aa",
    }
    for loc in sorted(np.unique(loc_te)):
        m = loc_te == loc
        r_l = safe_pearson_r(y_te[m], tp_preds[m])
        rmse_l = float(np.sqrt(mean_squared_error(y_te[m], tp_preds[m])))
        ax4.scatter(
            y_te[m],
            tp_preds[m],
            alpha=0.68,
            s=34,
            color=palette.get(loc, "#333333"),
            edgecolors="k",
            linewidths=0.4,
            label=f"{loc} (n={np.sum(m)}, RMSE={rmse_l:.1f}, r={r_l:.2f})",
        )
    min_v = float(min(y_te.min(), tp_preds.min()))
    max_v = float(max(y_te.max(), tp_preds.max()))
    ax4.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="1:1 Ideal Line")
    ax4.set_xlabel("Actual Yield (Bushels / Acre) — All 5 Locations Test", fontweight="bold")
    ax4.set_ylabel("Predicted Yield (Bushels / Acre)", fontweight="bold")
    ax4.set_title(
        f"4. Final 5-Location Test Parity (323 Plots: RMSE={loc_test_df.iloc[-1]['Test_RMSE']:.2f}, "
        f"R²={loc_test_df.iloc[-1]['Test_R2']:.4f}, r={loc_test_df.iloc[-1]['Pearson_r']:.4f})",
        fontsize=12,
        fontweight="bold",
    )
    ax4.legend(fontsize=8.5, loc="upper left")
    ax4.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        f"[{MODEL_NAME}] Step 6 Final Evaluation — Three-View Subspace Stacking Across All 5 Locations",
        fontsize=14.5,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=300)
    plt.close()


# ==============================================================================
# 7. MAIN STEP 6 PIPELINE FOR MODEL 1 (ALL 5 LOCATIONS IN-DOMAIN)
# ==============================================================================
def main():
    t0 = time.time()
    if not STEP5_NPZ_PATH.exists():
        raise FileNotFoundError(
            f"Missing {STEP5_NPZ_PATH}. Run model5loc_feature_selection.py first."
        )

    s5 = np.load(STEP5_NPZ_PATH, allow_pickle=True)
    X_full = s5["X"].astype(np.float32)
    X_phys = (
        s5["X_phys"].astype(np.float32)
        if "X_phys" in s5.files
        else X_full.copy()
    )
    y = s5["y"].astype(np.float32)
    splits = s5["splits"]
    locations = s5["locations"]
    plot_ids = s5["plot_ids"]
    genotypes = s5["genotypes"]
    feature_names = [str(f) for f in s5["feature_names"]]

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    print("=" * 108)
    print(
        f"[{MODEL_NAME}] STEP 6 LOADED: {len(y)} total plots across All 5 Locations "
        f"(Train: {np.sum(train_mask)} | Val: {np.sum(val_mask)} | Test: {np.sum(test_mask)})"
    )
    print(
        f"Subspaces Loaded -> Full mRMR Matrix: {X_full.shape[1]} Feats | "
        f"Physical Matrix: {X_phys.shape[1]} Feats | Strict Train-Only Fitting Policy Active"
    )
    print("=" * 108)

    print("\n1. Stage 6A: Tuning XGBoost Hyperparameters Strictly on 1,489 Train Plots...")
    best_cfg, second_cfg, shallow_cfg, rank_df = tune_m1_hyperparameters(
        X_full, y, splits, locations
    )
    rank_df.to_csv(OUTPUT_DIR / "m1_step6_tuning_grid.csv", index=False)

    print("\n   Top XGBoost Configurations Across All 5 Locations:")
    print(rank_df.to_string(index=False))
    print(
        f"   -> Selected Configs: Primary={best_cfg['config_id']} | "
        f"Secondary={second_cfg['config_id']} | Shallow={shallow_cfg['config_id']}"
    )

    print("\n2. Stage 6B: Fitting Three-View Subspace & Dual-Target Meta-Ensemble (Strictly on 1,489 Train Plots)...")
    summary_df, loc_val_df, loc_test_df, preds_store, artifacts = fit_and_evaluate_m1_ensemble(
        X_full, X_phys, feature_names, y, splits, locations, best_cfg, second_cfg, shallow_cfg
    )
    summary_df.to_csv(OUTPUT_DIR / "m1_step6_ablation_summary.csv", index=False)
    loc_val_df.to_csv(OUTPUT_DIR / "m1_step6_val_results_by_location.csv", index=False)
    loc_test_df.to_csv(OUTPUT_DIR / "m1_step6_test_results_by_location.csv", index=False)

    print("\n" + "=" * 142)
    print(
        f"[{MODEL_NAME}] TABLE 1: STEP 6 ABLATION & THREE-VIEW STACKING PROGRESSION (TRAINED STRICTLY ON 1,489 TRAIN PLOTS)"
    )
    print("=" * 142)
    print(summary_df.to_string(index=False))
    print("=" * 142)

    print("\n" + "=" * 122)
    print(
        f"[{MODEL_NAME}] TABLE 2: HOLDOUT VALIDATION BREAKDOWN BY LOCATION (319 VAL PLOTS ACROSS ALL 5 LOCATIONS)"
    )
    print("=" * 122)
    print(loc_val_df.to_string(index=False))
    print("=" * 122)

    print("\n" + "=" * 122)
    print(
        f"[{MODEL_NAME}] TABLE 3: FINAL TEST SET BREAKDOWN BY LOCATION (323 HELD-OUT TEST PLOTS ACROSS ALL 5 LOCATIONS)"
    )
    print("=" * 122)
    print(loc_test_df.to_string(index=False))
    print("=" * 122)

    model_bundle_path = OUTPUT_DIR / "m1_step6_tuned_models.pkl"
    with open(model_bundle_path, "wb") as f:
        pickle.dump(artifacts, f)

    test_preds_df = pd.DataFrame(
        {
            "plot_id": plot_ids[test_mask],
            "location": locations[test_mask],
            "genotype": genotypes[test_mask],
            "Actual_Yield_bu_ac": y[test_mask],
            "Step5_Predicted_Yield": preds_store["step5_test_preds"],
            "Step6_Final_Predicted_Yield": preds_store["test_preds"],
            "Absolute_Error_bu_ac": np.abs(y[test_mask] - preds_store["test_preds"]),
        }
    )
    test_preds_csv = OUTPUT_DIR / "m1_step6_all5loc_test_predictions.csv"
    test_preds_df.to_csv(test_preds_csv, index=False)

    s5_row = summary_df.iloc[0]
    final_row = summary_df.iloc[-1]

    print(f"\n[{MODEL_NAME} STEP 6 SUMMARY: BEFORE vs. AFTER TUNING]")
    print(
        f"  1. 5-Location Test RMSE (323 plots) : {s5_row['Test_RMSE']:.4f} bu/ac -> "
        f"{final_row['Test_RMSE']:.4f} bu/ac (MAE: {s5_row['Test_MAE']:.4f} -> {final_row['Test_MAE']:.4f})"
    )
    print(
        f"  2. 5-Location Test R² & Overall r   : R² = {s5_row['Test_R2']:.4f} -> {final_row['Test_R2']:.4f} "
        f"| Overall r = {s5_row['Test_Overall_r']:.4f} -> {final_row['Test_Overall_r']:.4f} "
        f"| Within-Loc r = {s5_row['Test_WithinLoc_r']:.4f} -> {final_row['Test_WithinLoc_r']:.4f}"
    )
    print(
        f"  3. 5-Location Val RMSE  (319 plots) : {s5_row['Val_RMSE']:.4f} bu/ac (R²={s5_row['Val_R2']:.4f}) -> "
        f"{final_row['Val_RMSE']:.4f} bu/ac (R²={final_row['Val_R2']:.4f})"
    )
    print(f"  -> Saved Tuned Model Bundle         : {model_bundle_path}")
    print(f"  -> Saved 5-Loc Test Predictions     : {test_preds_csv}")
    print(f"  -> Total Step 6 Runtime             : {time.time() - t0:.1f}s")

    chart_path = OUTPUT_DIR / "m1_step6_tuning_and_5loc_test_chart.png"
    plot_m1_step6_dashboard(
        summary_df,
        loc_val_df,
        loc_test_df,
        y[test_mask],
        preds_store["test_preds"],
        locations[test_mask],
        chart_path,
    )
    print(f"  -> Saved Step 6 4-Panel Dashboard   : {chart_path}")


if __name__ == "__main__":
    main()