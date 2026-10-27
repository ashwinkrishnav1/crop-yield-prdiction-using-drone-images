import os
import time
import pickle
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PROJECT PATHS & CONFIGURATION
# ==============================================================================
PROJECT_ROOT = Path(r"C:\Users\ashwi\Music\cropyeildererrrrrrr")
VALID_PLOTS_CSV = PROJECT_ROOT / "drone_dataset" / "valid_plots.csv"

STEP4_DIR = PROJECT_ROOT / "step4_feature_engineering"
STEP4_RAW_CACHE = STEP4_DIR / "cache" / "step4_raw_multimodal_features.npz"

STEP5_DIR = PROJECT_ROOT / "step5_feature_selection"
STEP5_NPZ_PATH = STEP5_DIR / "step5_selected_features.npz"

OUTPUT_DIR = PROJECT_ROOT / "step6_xgboost_tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ==============================================================================
# 2. LOAD STEP 4 & STEP 5 CACHED FEATURE MATRICES
# ==============================================================================
def load_all_inputs() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], pd.DataFrame]:
    if not STEP5_NPZ_PATH.exists():
        raise FileNotFoundError(f"Missing {STEP5_NPZ_PATH}. Please run Step 5 first.")
    if not STEP4_RAW_CACHE.exists():
        raise FileNotFoundError(f"Missing {STEP4_RAW_CACHE}. Please run Step 4 first.")

    s5_data = np.load(STEP5_NPZ_PATH, allow_pickle=True)
    s4_raw = np.load(STEP4_RAW_CACHE, allow_pickle=True)

    s5_dict = {k: s5_data[k] for k in s5_data.files}
    raw_dict = {k: s4_raw[k] for k in s4_raw.files}

    plot_ids = [str(pid).strip() for pid in s5_dict["plot_ids"]]
    df_valid = pd.read_csv(VALID_PLOTS_CSV)
    df_valid["plot_id_clean"] = df_valid["plot_id"].astype(str).str.strip()
    df_valid = (
        df_valid.drop_duplicates(subset=["plot_id_clean"])
        .set_index("plot_id_clean")
        .reindex(plot_ids)
        .reset_index()
    )

    splits = s5_dict["splits"]
    print("=" * 98)
    print(
        f"STEP 6 LOADED CACHES: {len(splits)} plots "
        f"(Train: {(splits=='train').sum()} | Val: {(splits=='val').sum()} | "
        f"Unseen Crawfordsville Test: {(splits=='test').sum()})"
    )
    print(f"Step 5 Winner Feature Matrix : {s5_dict['X'].shape[1]} Core Multimodal Features")
    print("=" * 98)
    return s5_dict, raw_dict, df_valid


# ==============================================================================
# 3. RECONSTRUCT FULL PHYSICAL ENVIRONMENT MATRIX (FOR ZERO-SHOT ANCHOR)
# ==============================================================================
def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-6, None)


def build_full_physical_environment_features(
    raw_dict: Dict[str, np.ndarray],
    s5_dict: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    y = s5_dict["y"].astype(np.float32)
    splits = s5_dict["splits"]
    locations = s5_dict["locations"]
    train_mask = splits == "train"
    train_locs = locations[train_mask]
    y_train = y[train_mask]

    meta_raw = raw_dict["meta"]
    stand_count = meta_raw[:, 0]
    stand_density = meta_raw[:, 2]
    log_stand = meta_raw[:, 3]

    c1, c2, c3 = raw_dict["tp1_canopy"], raw_dict["tp2_canopy"], raw_dict["tp3_canopy"]
    fgcc_1, fgcc_2, fgcc_3 = c1[:, 6], c2[:, 6], c3[:, 6]
    yellow_3 = c3[:, 7]
    pure_exg_1, pure_exg_2, pure_exg_3 = c1[:, 8], c2[:, 8], c3[:, 8]
    gap_std_3 = c3[:, 10]

    bio_phys = np.column_stack(
        [
            stand_count,
            stand_density,
            log_stand,
            stand_density * fgcc_2,
            stand_density * fgcc_3,
            log_stand * pure_exg_3,
            fgcc_2 / np.clip(stand_count, 10.0, None),
            fgcc_3 / np.clip(stand_count, 10.0, None),
            (fgcc_3 + 0.02) / (fgcc_2 + 0.02),
            (fgcc_2 + 0.02) / (fgcc_1 + 0.02),
            yellow_3 / (fgcc_3 + 0.02),
            fgcc_3 - gap_std_3,
            log_stand * (0.20 * pure_exg_1 + 0.40 * pure_exg_2 + 0.40 * pure_exg_3),
        ]
    ).astype(np.float32)

    y_train_loc_norm = np.zeros_like(y_train, dtype=np.float32)
    high_mask_train = np.zeros(len(y_train), dtype=bool)
    low_mask_train = np.zeros(len(y_train), dtype=bool)

    for loc in np.unique(train_locs):
        m = train_locs == loc
        y_loc = y_train[m]
        mu, sd = float(np.mean(y_loc)), max(float(np.std(y_loc)), 1.0)
        y_train_loc_norm[m] = (y_loc - mu) / sd
        high_mask_train[m] = y_loc >= np.percentile(y_loc, 75)
        low_mask_train[m] = y_loc <= np.percentile(y_loc, 25)

    cnn_blocks = [
        raw_dict["tp1_mean"],
        raw_dict["tp2_mean"],
        raw_dict["tp3_mean"],
        raw_dict["tp2_mean"] - raw_dict["tp1_mean"],
        raw_dict["tp3_mean"] - raw_dict["tp2_mean"],
        raw_dict["tp3_mean"] - raw_dict["tp1_mean"],
    ]

    proto_list = []
    for mat in cnn_blocks:
        tr_mean = np.mean(mat[train_mask], axis=0, keepdims=True)
        mat_c = l2_normalize_rows(mat - tr_mean)
        mat_tr = mat_c[train_mask]

        hi_p = l2_normalize_rows(np.mean(mat_tr[high_mask_train], axis=0, keepdims=True))[0]
        lo_p = l2_normalize_rows(np.mean(mat_tr[low_mask_train], axis=0, keepdims=True))[0]

        sim_hi = mat_c @ hi_p
        sim_lo = mat_c @ lo_p
        ridge = RidgeCV(alphas=[50.0, 150.0, 300.0, 600.0])
        ridge.fit(mat_tr, y_train_loc_norm)
        vigor = ridge.predict(mat_c).astype(np.float32)
        proto_list.append(np.column_stack([sim_hi, sim_lo, sim_hi - sim_lo, vigor]))

    proto_phys = np.hstack(proto_list).astype(np.float32)

    canopy_all = np.hstack(
        [c1, c2, c3, c2 - c1, c3 - c2, c3 - c1, 0.25 * c1 + 0.50 * c2 + 0.25 * c3]
    ).astype(np.float32)

    cnn_all = np.hstack(
        [
            raw_dict["tp3_mean"],
            raw_dict["tp2_mean"],
            raw_dict["tp1_mean"],
            raw_dict["tp3_std"],
            raw_dict["tp3_mean"] - raw_dict["tp2_mean"],
            raw_dict["tp2_mean"] - raw_dict["tp1_mean"],
            raw_dict["tp3_mean"] - raw_dict["tp1_mean"],
        ]
    ).astype(np.float32)

    candidates = np.hstack([canopy_all, cnn_all])

    X_tr = candidates[train_mask]
    unique_locs = np.unique(train_locs)
    loc_corrs = np.zeros((len(unique_locs), X_tr.shape[1]), dtype=np.float32)
    loc_means = np.zeros((len(unique_locs), X_tr.shape[1]), dtype=np.float32)
    loc_vars = np.zeros((len(unique_locs), X_tr.shape[1]), dtype=np.float32)

    for li, loc in enumerate(unique_locs):
        m = train_locs == loc
        X_s, y_s = X_tr[m], y_train[m]
        yc = y_s - np.mean(y_s)
        Xc = X_s - np.mean(X_s, axis=0, keepdims=True)
        denom = np.sqrt(np.sum(Xc**2, axis=0) * np.sum(yc**2))
        loc_corrs[li] = np.where(denom > 1e-8, (yc @ Xc) / denom, 0.0)
        loc_means[li] = np.mean(X_s, axis=0)
        loc_vars[li] = np.var(X_s, axis=0) + 1e-6

    mean_r = np.mean(loc_corrs, axis=0)
    sign_agree = np.maximum(
        np.sum(loc_corrs > 0, axis=0), np.sum(loc_corrs < 0, axis=0)
    ) / float(len(unique_locs))
    min_r = np.min(loc_corrs * np.sign(mean_r + 1e-12), axis=0)
    city_bias = np.var(loc_means, axis=0) / np.clip(np.mean(loc_vars, axis=0), 1e-6, None)

    score = (np.abs(mean_r) * (sign_agree**1.5) * np.where(min_r > 0, 1.25, 0.75)) / (
        1.0 + 0.35 * city_bias
    )
    valid_mask = (city_bias < 1.25) & (sign_agree >= 0.75)
    valid_indices = np.where(valid_mask)[0]
    sorted_valid = valid_indices[np.argsort(score[valid_indices])[::-1]]

    top_120_idx = sorted_valid[:120]
    X_phys_full_raw = np.hstack(
        [bio_phys, proto_phys, candidates[:, top_120_idx]]
    ).astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(X_phys_full_raw[train_mask])
    X_phys_full_scaled = scaler.transform(X_phys_full_raw).astype(np.float32)

    return X_phys_full_scaled, X_phys_full_raw


# ==============================================================================
# 4. HELPER METRICS & SAMPLE WEIGHTS
# ==============================================================================
def safe_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_location_balanced_weights(locations_subset: np.ndarray) -> np.ndarray:
    unique_locs, counts = np.unique(locations_subset, return_counts=True)
    total = float(len(locations_subset))
    n_locs = float(len(unique_locs))
    weight_map = {loc: total / (n_locs * float(cnt)) for loc, cnt in zip(unique_locs, counts)}
    return np.array([weight_map[loc] for loc in locations_subset], dtype=np.float32)


def compute_within_location_targets(
    y_sub: np.ndarray, loc_sub: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    loc_means = {loc: float(np.mean(y_sub[loc_sub == loc])) for loc in np.unique(loc_sub)}
    y_centered = np.array([yv - loc_means[l] for yv, l in zip(y_sub, loc_sub)], dtype=np.float32)
    return y_centered, loc_means


# ==============================================================================
# 5. STAGE 6A: TUNE WITHIN-FIELD RANKER + DIVERSE MULTI-MODEL ENSEMBLE
# ==============================================================================
def get_ranker_candidate_configs() -> List[Dict]:
    return [
        {
            "config_id": "R0_Step5_Default",
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 15,
            "subsample": 0.80,
            "colsample_bytree": 0.50,
            "reg_alpha": 2.0,
            "reg_lambda": 12.0,
            "gamma": 0.0,
            "objective": "reg:squarederror",
        },
        {
            "config_id": "R1_Shallow_StrongReg",
            "n_estimators": 260,
            "learning_rate": 0.025,
            "max_depth": 2,
            "min_child_weight": 20,
            "subsample": 0.75,
            "colsample_bytree": 0.45,
            "reg_alpha": 4.0,
            "reg_lambda": 20.0,
            "gamma": 0.5,
            "objective": "reg:squarederror",
        },
        {
            "config_id": "R2_Shallow_FastLR",
            "n_estimators": 200,
            "learning_rate": 0.03,
            "max_depth": 2,
            "min_child_weight": 25,
            "subsample": 0.70,
            "colsample_bytree": 0.40,
            "reg_alpha": 5.0,
            "reg_lambda": 25.0,
            "gamma": 1.0,
            "objective": "reg:squarederror",
        },
        {
            "config_id": "R3_Depth3_HeavyL2",
            "n_estimators": 280,
            "learning_rate": 0.022,
            "max_depth": 3,
            "min_child_weight": 22,
            "subsample": 0.75,
            "colsample_bytree": 0.40,
            "reg_alpha": 5.0,
            "reg_lambda": 25.0,
            "gamma": 0.5,
            "objective": "reg:squarederror",
        },
        {
            "config_id": "R4_Depth3_FineLR",
            "n_estimators": 400,
            "learning_rate": 0.018,
            "max_depth": 3,
            "min_child_weight": 18,
            "subsample": 0.80,
            "colsample_bytree": 0.45,
            "reg_alpha": 3.0,
            "reg_lambda": 18.0,
            "gamma": 0.2,
            "objective": "reg:squarederror",
        },
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
        {
            "config_id": "R8_UltraRegularized_D2",
            "n_estimators": 240,
            "learning_rate": 0.020,
            "max_depth": 2,
            "min_child_weight": 28,
            "subsample": 0.68,
            "colsample_bytree": 0.38,
            "reg_alpha": 6.0,
            "reg_lambda": 30.0,
            "gamma": 1.0,
            "objective": "reg:squarederror",
        },
    ]


def build_xgb_from_cfg(cfg: Dict) -> XGBRegressor:
    params = {k: v for k, v in cfg.items() if k != "config_id"}
    return XGBRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)


def tune_within_field_ranker(
    X_core: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    locations: np.ndarray,
) -> Tuple[Dict, Dict, pd.DataFrame, Tuple[float, float, float], float]:
    """
    Tunes XGBoost configurations and builds a 3-family Ranker Ensemble:
      - Top Depth-3/4 XGBoost + Top Depth-2 Shallow XGBoost
      - ExtraTreesRegressor (bagged randomized trees)
      - RidgeCV (L2 linear projection)
    Optimizes blend weights and Gauss-Markov cross-city variance shrinkage alpha_shrink
    strictly via 4-City Leave-One-Location-Out (LOLO) Cross-Validation.
    """
    train_mask = splits == "train"
    val_mask = splits == "val"
    dev_mask = (splits == "train") | (splits == "val")

    X_tr, y_tr, loc_tr = X_core[train_mask], y[train_mask], locations[train_mask]
    X_va, y_va, loc_va = X_core[val_mask], y[val_mask], locations[val_mask]
    X_dev, y_dev, loc_dev = X_core[dev_mask], y[dev_mask], locations[dev_mask]

    y_tr_cent, tr_loc_means = compute_within_location_targets(y_tr, loc_tr)
    w_train = compute_location_balanced_weights(loc_tr)
    val_offsets = np.array([tr_loc_means[l] for l in loc_va], dtype=np.float32)

    logo = LeaveOneGroupOut()
    configs = get_ranker_candidate_configs()
    records = []
    lolo_oof_store = {}

    for cfg in configs:
        cid = cfg["config_id"]
        m_tr = build_xgb_from_cfg(cfg)
        m_tr.fit(X_tr, y_tr_cent, sample_weight=w_train)
        val_cent_pred = m_tr.predict(X_va)
        val_abs_pred = val_cent_pred + val_offsets
        val_rmse = float(np.sqrt(mean_squared_error(y_va, val_abs_pred)))
        val_r2 = float(r2_score(y_va, val_abs_pred))
        val_r = safe_pearson_r(y_va - val_offsets, val_cent_pred)

        oof_lolo = np.zeros(len(y_dev), dtype=np.float32)
        y_dev_cent_all = np.zeros(len(y_dev), dtype=np.float32)
        lolo_rs, lolo_rmses = [], []

        for f_tr, f_va in logo.split(X_dev, y_dev, groups=loc_dev):
            y_f_tr_cent, _ = compute_within_location_targets(y_dev[f_tr], loc_dev[f_tr])
            w_f = compute_location_balanced_weights(loc_dev[f_tr])

            m_lolo = build_xgb_from_cfg(cfg)
            m_lolo.fit(X_dev[f_tr], y_f_tr_cent, sample_weight=w_f)
            p_va = m_lolo.predict(X_dev[f_va])
            p_va_zero = p_va - float(np.mean(p_va))

            y_va_true_cent = y_dev[f_va] - float(np.mean(y_dev[f_va]))
            oof_lolo[f_va] = p_va_zero
            y_dev_cent_all[f_va] = y_va_true_cent

            lolo_rs.append(safe_pearson_r(y_va_true_cent, p_va_zero))
            lolo_rmses.append(float(np.sqrt(mean_squared_error(y_va_true_cent, p_va_zero))))

        lolo_oof_store[cid] = (oof_lolo, y_dev_cent_all)
        mean_lolo_r = float(np.mean(lolo_rs))
        mean_lolo_rmse = float(np.mean(lolo_rmses))
        composite_score = 0.45 * val_rmse + 0.55 * mean_lolo_rmse - 6.0 * mean_lolo_r

        records.append(
            {
                "Config_ID": cid,
                "max_depth": cfg["max_depth"],
                "lr": cfg["learning_rate"],
                "n_est": cfg["n_estimators"],
                "min_child_w": cfg["min_child_weight"],
                "colsample": cfg["colsample_bytree"],
                "reg_lambda": cfg["reg_lambda"],
                "Val_RMSE": round(val_rmse, 4),
                "Val_R2": round(val_r2, 4),
                "Val_Within_r": round(val_r, 4),
                "LOLO_Centered_RMSE": round(mean_lolo_rmse, 4),
                "LOLO_Pearson_r": round(mean_lolo_r, 4),
                "Composite_Score": round(composite_score, 4),
            }
        )

    rank_df = pd.DataFrame(records).sort_values("Composite_Score").reset_index(drop=True)
    best_cid = rank_df.loc[0, "Config_ID"]
    best_cfg = next(c for c in configs if c["config_id"] == best_cid)

    # Also grab best shallow (max_depth == 2) config for structural diversity
    shallow_df = rank_df[rank_df["max_depth"] == 2].reset_index(drop=True)
    best_shallow_cid = shallow_df.loc[0, "Config_ID"] if len(shallow_df) > 0 else best_cid
    best_shallow_cfg = next(c for c in configs if c["config_id"] == best_shallow_cid)

    # Compute LOLO OOF for RidgeCV and ExtraTreesRegressor
    oof_ridge = np.zeros(len(y_dev), dtype=np.float32)
    oof_et = np.zeros(len(y_dev), dtype=np.float32)

    for f_tr, f_va in logo.split(X_dev, y_dev, groups=loc_dev):
        y_f_tr_cent, _ = compute_within_location_targets(y_dev[f_tr], loc_dev[f_tr])
        w_f = compute_location_balanced_weights(loc_dev[f_tr])

        ridge_m = RidgeCV(alphas=[10.0, 50.0, 150.0, 400.0, 1000.0])
        ridge_m.fit(X_dev[f_tr], y_f_tr_cent, sample_weight=w_f)
        p_r = ridge_m.predict(X_dev[f_va])
        oof_ridge[f_va] = p_r - float(np.mean(p_r))

        et_m = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            max_features=0.45,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        et_m.fit(X_dev[f_tr], y_f_tr_cent, sample_weight=w_f)
        p_et = et_m.predict(X_dev[f_va])
        oof_et[f_va] = p_et - float(np.mean(p_et))

    oof_xgb1, y_dev_cent_all = lolo_oof_store[best_cid]
    oof_xgb2, _ = lolo_oof_store[best_shallow_cid]
    oof_xgb_comb = 0.65 * oof_xgb1 + 0.35 * oof_xgb2

    # Grid search optimal ensemble weights (w_xgb, w_et, w_ridge) on 4-City LOLO OOF
    best_weights = (0.65, 0.15, 0.20)
    best_lolo_mse = float("inf")
    for w_r in np.linspace(0.10, 0.45, 8):
        for w_e in np.linspace(0.05, 0.30, 6):
            w_x = 1.0 - w_r - w_e
            if w_x < 0.30:
                continue
            cand_oof = w_x * oof_xgb_comb + w_e * oof_et + w_r * oof_ridge
            # Evaluate variance-calibrated LOLO MSE
            alpha_c = np.clip(
                np.mean(cand_oof * y_dev_cent_all) / max(np.var(cand_oof), 1e-6),
                0.50,
                0.90,
            )
            mse_c = float(np.mean((alpha_c * cand_oof - y_dev_cent_all) ** 2))
            if mse_c < best_lolo_mse:
                best_lolo_mse = mse_c
                best_weights = (float(w_x), float(w_e), float(w_r))

    w_x, w_e, w_r = best_weights
    final_blend_oof = w_x * oof_xgb_comb + w_e * oof_et + w_r * oof_ridge

    # Compute true LOLO Gauss-Markov variance shrinkage factor (alpha_shrink)
    var_pred = float(np.var(final_blend_oof))
    cov_pred_true = float(np.mean(final_blend_oof * y_dev_cent_all))
    raw_alpha = cov_pred_true / max(var_pred, 1e-6)
    alpha_shrink = float(np.clip(raw_alpha, 0.55, 0.85))

    return best_cfg, best_shallow_cfg, rank_df, best_weights, alpha_shrink


# ==============================================================================
# 6. STAGE 6B: TUNE ZERO-SHOT PHYSICAL FIELD ANCHOR MODEL
# ==============================================================================
def tune_physical_baseline_model(
    X_phys_full: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    locations: np.ndarray,
) -> Tuple[Dict, pd.DataFrame]:
    train_mask = splits == "train"
    val_mask = splits == "val"
    dev_mask = (splits == "train") | (splits == "val")

    X_tr, y_tr, loc_tr = X_phys_full[train_mask], y[train_mask], locations[train_mask]
    X_va, y_va = X_phys_full[val_mask], y[val_mask]
    X_dev, y_dev, loc_dev = X_phys_full[dev_mask], y[dev_mask], locations[dev_mask]

    phys_configs = [
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
        {
            "phys_id": "P1_Unweighted_Run1_Style_D3",
            "use_loc_weights": False,
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
        {
            "phys_id": "P2_Unweighted_Shallow_D2_HeavyReg",
            "use_loc_weights": False,
            "n_estimators": 250,
            "learning_rate": 0.025,
            "max_depth": 2,
            "min_child_weight": 25,
            "subsample": 0.75,
            "colsample_bytree": 0.40,
            "reg_alpha": 5.0,
            "reg_lambda": 25.0,
            "objective": "reg:squarederror",
        },
        {
            "phys_id": "P5_SqrtLocWeight_Shallow_D2",
            "use_loc_weights": "sqrt",
            "n_estimators": 250,
            "learning_rate": 0.025,
            "max_depth": 2,
            "min_child_weight": 25,
            "subsample": 0.75,
            "colsample_bytree": 0.40,
            "reg_alpha": 5.0,
            "reg_lambda": 25.0,
            "objective": "reg:squarederror",
        },
    ]

    logo = LeaveOneGroupOut()
    records = []

    for pcfg in phys_configs:
        pid = pcfg["phys_id"]
        w_mode = pcfg["use_loc_weights"]
        xgb_params = {k: v for k, v in pcfg.items() if k not in ("phys_id", "use_loc_weights")}

        def get_weights(locs_arr: np.ndarray):
            if w_mode is True:
                return compute_location_balanced_weights(locs_arr)
            elif w_mode == "sqrt":
                w_full = compute_location_balanced_weights(locs_arr)
                return np.sqrt(w_full).astype(np.float32)
            return None

        m_tr = XGBRegressor(**xgb_params, random_state=RANDOM_SEED, n_jobs=-1)
        m_tr.fit(X_tr, y_tr, sample_weight=get_weights(loc_tr))
        val_preds = m_tr.predict(X_va)
        val_rmse = float(np.sqrt(mean_squared_error(y_va, val_preds)))

        lolo_rmses = []
        lolo_mean_biases = []
        for f_tr, f_va in logo.split(X_dev, y_dev, groups=loc_dev):
            m_lolo = XGBRegressor(**xgb_params, random_state=RANDOM_SEED, n_jobs=-1)
            m_lolo.fit(X_dev[f_tr], y_dev[f_tr], sample_weight=get_weights(loc_dev[f_tr]))
            p_va = m_lolo.predict(X_dev[f_va])
            lolo_rmses.append(float(np.sqrt(mean_squared_error(y_dev[f_va], p_va))))
            lolo_mean_biases.append(abs(float(np.mean(p_va) - np.mean(y_dev[f_va]))))

        mean_lolo_rmse = float(np.mean(lolo_rmses))
        mean_lolo_bias = float(np.mean(lolo_mean_biases))
        selection_metric = 0.60 * mean_lolo_rmse + 0.40 * val_rmse

        records.append(
            {
                "Phys_Config_ID": pid,
                "Loc_Weight_Mode": str(w_mode),
                "max_depth": pcfg["max_depth"],
                "Objective": pcfg["objective"],
                "Val_RMSE": round(val_rmse, 4),
                "LOLO_4City_RMSE": round(mean_lolo_rmse, 4),
                "LOLO_Mean_City_Bias": round(mean_lolo_bias, 4),
                "Selection_Score": round(selection_metric, 4),
            }
        )

    phys_df = pd.DataFrame(records).sort_values("Selection_Score").reset_index(drop=True)
    best_pid = phys_df.loc[0, "Phys_Config_ID"]
    best_pcfg = next(c for c in phys_configs if c["phys_id"] == best_pid)
    return best_pcfg, phys_df


# ==============================================================================
# 7. STAGE 6C: FINAL ENSEMBLE TRAINING & COMPREHENSIVE EVALUATION
# ==============================================================================
def build_phys_model_and_weights(pcfg: Dict, locs_arr: np.ndarray):
    w_mode = pcfg["use_loc_weights"]
    xgb_params = {k: v for k, v in pcfg.items() if k not in ("phys_id", "use_loc_weights")}
    if w_mode is True:
        w = compute_location_balanced_weights(locs_arr)
    elif w_mode == "sqrt":
        w = np.sqrt(compute_location_balanced_weights(locs_arr)).astype(np.float32)
    else:
        w = None
    return XGBRegressor(**xgb_params, random_state=RANDOM_SEED, n_jobs=-1), w


def fit_and_predict_diverse_ranker(
    X_fit: np.ndarray,
    y_fit_cent: np.ndarray,
    w_fit: np.ndarray,
    X_eval: np.ndarray,
    cfg1: Dict,
    cfg2: Dict,
    blend_weights: Tuple[float, float, float],
):
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

    preds = (w_x * p_xgb + w_e * p_et + w_r * p_ridge).astype(np.float32)
    models_dict = {"xgb1": m_xgb1, "xgb2": m_xgb2, "et": m_et, "ridge": m_ridge}
    return preds, models_dict


def evaluate_complete_step6(
    X_core: np.ndarray,
    X_phys_full: np.ndarray,
    X_phys_step5_old: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    locations: np.ndarray,
    best_ranker_cfg: Dict,
    best_shallow_cfg: Dict,
    best_phys_cfg: Dict,
    blend_weights: Tuple[float, float, float],
    alpha_shrink: float,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict]:
    train_mask = splits == "train"
    val_mask = splits == "val"
    dev_mask = (splits == "train") | (splits == "val")
    test_mask = splits == "test"

    y_tr, loc_tr = y[train_mask], locations[train_mask]
    y_va, loc_va = y[val_mask], locations[val_mask]
    y_dev, loc_dev = y[dev_mask], locations[dev_mask]
    y_test = y[test_mask]

    y_tr_cent, tr_loc_means = compute_within_location_targets(y_tr, loc_tr)
    y_dev_cent, dev_loc_means = compute_within_location_targets(y_dev, loc_dev)
    val_offsets = np.array([tr_loc_means[l] for l in loc_va], dtype=np.float32)

    w_tr_bal = compute_location_balanced_weights(loc_tr)
    w_dev_bal = compute_location_balanced_weights(loc_dev)

    rng = np.random.RandomState(RANDOM_SEED)
    cal_idx = rng.choice(len(y_test), size=15, replace=False)
    eval_mask = np.ones(len(y_test), dtype=bool)
    eval_mask[cal_idx] = False

    def compute_row_metrics(
        name: str,
        val_p: np.ndarray,
        test_tr_p: np.ndarray,
        test_full_p: np.ndarray,
    ) -> Tuple[Dict, np.ndarray]:
        # Empirical Bayes robust 15-plot calibration (shrinks sample offset by 0.60)
        raw_offset = float(np.median(y_test[cal_idx] - test_full_p[cal_idx]))
        cal_p = test_full_p + 0.60 * raw_offset
        return {
            "Model_Stage": name,
            "Val_RMSE": round(float(np.sqrt(mean_squared_error(y_va, val_p))), 4),
            "Val_R2": round(float(r2_score(y_va, val_p)), 4),
            "ZeroShot_Test_RMSE": round(
                float(np.sqrt(mean_squared_error(y_test, test_tr_p))), 4
            ),
            "ZeroShot_Test_MAE": round(float(mean_absolute_error(y_test, test_tr_p)), 4),
            "ZeroShot_Test_R2": round(float(r2_score(y_test, test_tr_p)), 4),
            "ZeroShot_Full4Loc_RMSE": round(
                float(np.sqrt(mean_squared_error(y_test, test_full_p))), 4
            ),
            "ZeroShot_Full4Loc_R2": round(float(r2_score(y_test, test_full_p)), 4),
            "Test_Pearson_r": round(safe_pearson_r(y_test, test_tr_p), 4),
            "Test_Full4Loc_r": round(safe_pearson_r(y_test, test_full_p), 4),
            "Calibrated_RMSE": round(
                float(np.sqrt(mean_squared_error(y_test[eval_mask], cal_p[eval_mask]))), 4
            ),
            "Calibrated_MAE": round(
                float(mean_absolute_error(y_test[eval_mask], cal_p[eval_mask])), 4
            ),
            "Calibrated_R2": round(float(r2_score(y_test[eval_mask], cal_p[eval_mask])), 4),
        }, cal_p

    rows = []

    # 1. Step 5 Untuned Reference
    r0_cfg = get_ranker_candidate_configs()[0]
    m_r0_tr = build_xgb_from_cfg(r0_cfg)
    m_r0_tr.fit(X_core[train_mask], y_tr_cent, sample_weight=w_tr_bal)
    m_r0_full = build_xgb_from_cfg(r0_cfg)
    m_r0_full.fit(X_core[dev_mask], y_dev_cent, sample_weight=w_dev_bal)

    m_p0_tr = build_xgb_from_cfg(r0_cfg)
    m_p0_tr.fit(X_phys_step5_old[train_mask], y_tr, sample_weight=w_tr_bal)
    m_p0_full = build_xgb_from_cfg(r0_cfg)
    m_p0_full.fit(X_phys_step5_old[dev_mask], y_dev, sample_weight=w_dev_bal)

    val_p0 = m_r0_tr.predict(X_core[val_mask]) + val_offsets
    dev_t0_tr = m_r0_tr.predict(X_core[test_mask])
    dev_t0_full = m_r0_full.predict(X_core[test_mask])
    test_p0_tr = float(np.mean(m_p0_tr.predict(X_phys_step5_old[test_mask]))) + (
        dev_t0_tr - float(np.mean(dev_t0_tr))
    )
    test_p0_full = float(np.mean(m_p0_full.predict(X_phys_step5_old[test_mask]))) + (
        dev_t0_full - float(np.mean(dev_t0_full))
    )

    row0, _ = compute_row_metrics(
        "1. Step 5 Untuned Baseline (Unshrunk Single XGB)",
        val_p0,
        test_p0_tr,
        test_p0_full,
    )
    rows.append(row0)

    # 2. Step 6 Tuned Physical Baseline Model
    m_phys_tr, w_p_tr = build_phys_model_and_weights(best_phys_cfg, loc_tr)
    m_phys_tr.fit(X_phys_full[train_mask], y_tr, sample_weight=w_p_tr)
    phys_val_preds = m_phys_tr.predict(X_phys_full[val_mask])
    phys_test_preds_tr = m_phys_tr.predict(X_phys_full[test_mask])

    m_phys_full, w_p_full = build_phys_model_and_weights(best_phys_cfg, loc_dev)
    m_phys_full.fit(X_phys_full[dev_mask], y_dev, sample_weight=w_p_full)
    phys_test_preds_full = m_phys_full.predict(X_phys_full[test_mask])

    row_phys, _ = compute_row_metrics(
        "2. Step 6 Tuned Physical Baseline Model",
        phys_val_preds,
        phys_test_preds_tr,
        phys_test_preds_full,
    )
    rows.append(row_phys)

    # 3. Step 6 Diverse Ranker Ensemble (XGB + ExtraTrees + Ridge) + Physical Anchor
    val_rank_cent, rank_models_tr = fit_and_predict_diverse_ranker(
        X_core[train_mask],
        y_tr_cent,
        w_tr_bal,
        X_core[val_mask],
        best_ranker_cfg,
        best_shallow_cfg,
        blend_weights,
    )
    val_tuned_preds = val_rank_cent + val_offsets

    test_rank_tr_raw, _ = fit_and_predict_diverse_ranker(
        X_core[train_mask],
        y_tr_cent,
        w_tr_bal,
        X_core[test_mask],
        best_ranker_cfg,
        best_shallow_cfg,
        blend_weights,
    )
    test_rank_tr_zero = test_rank_tr_raw - float(np.mean(test_rank_tr_raw))

    test_rank_full_raw, rank_models_full = fit_and_predict_diverse_ranker(
        X_core[dev_mask],
        y_dev_cent,
        w_dev_bal,
        X_core[test_mask],
        best_ranker_cfg,
        best_shallow_cfg,
        blend_weights,
    )
    test_rank_full_zero = test_rank_full_raw - float(np.mean(test_rank_full_raw))

    anchor_tr = float(np.mean(phys_test_preds_tr))
    anchor_full = float(np.mean(phys_test_preds_full))

    test_unshrunk_tr = anchor_tr + test_rank_tr_zero
    test_unshrunk_full = anchor_full + test_rank_full_zero

    row_unshrunk, _ = compute_row_metrics(
        "3. Step 6 Diverse Ranker Ensemble + Anchor (Unshrunk)",
        val_tuned_preds,
        test_unshrunk_tr,
        test_unshrunk_full,
    )
    rows.append(row_unshrunk)

    # 4. STEP 6 FINAL TUNED ENSEMBLE (Gauss-Markov LOLO Variance Calibrated)
    phys_dev_tr_zero = phys_test_preds_tr - anchor_tr
    phys_dev_full_zero = phys_test_preds_full - anchor_full

    final_test_tr = anchor_tr + alpha_shrink * (
        0.88 * test_rank_tr_zero + 0.12 * phys_dev_tr_zero
    )
    final_test_full = anchor_full + alpha_shrink * (
        0.88 * test_rank_full_zero + 0.12 * phys_dev_full_zero
    )

    row_final, final_cal_preds = compute_row_metrics(
        "4. STEP 6 FINAL TUNED ENSEMBLE (LOLO-Calibrated)",
        val_tuned_preds,
        final_test_tr,
        final_test_full,
    )
    rows.append(row_final)

    preds_store = {
        "val_preds": val_tuned_preds,
        "test_zeroshot_tr": final_test_tr,
        "test_zeroshot_full": final_test_full,
        "test_calibrated_full": final_cal_preds,
        "step5_test_zeroshot": test_p0_tr,
    }

    artifacts = {
        "ranker_models_train": rank_models_tr,
        "ranker_models_full4loc": rank_models_full,
        "phys_xgb_train": m_phys_tr,
        "phys_xgb_full4loc": m_phys_full,
        "best_ranker_cfg": best_ranker_cfg,
        "best_shallow_cfg": best_shallow_cfg,
        "best_phys_cfg": best_phys_cfg,
        "blend_weights": blend_weights,
        "alpha_shrink": alpha_shrink,
        "tr_loc_means": tr_loc_means,
        "dev_loc_means": dev_loc_means,
    }

    return pd.DataFrame(rows), preds_store, artifacts


# ==============================================================================
# 8. MAIN STEP 6 PIPELINE
# ==============================================================================
def main():
    t0 = time.time()
    s5_dict, raw_dict, df_valid = load_all_inputs()

    X_core = s5_dict["X"].astype(np.float32)
    X_phys_step5_old = s5_dict["X_phys"].astype(np.float32)
    y = s5_dict["y"].astype(np.float32)
    splits = s5_dict["splits"]
    locations = s5_dict["locations"]
    plot_ids = s5_dict["plot_ids"]

    print("\n1. Reconstructing Full Physical + Temporal Delta Environment Matrix...")
    X_phys_full, _ = build_full_physical_environment_features(raw_dict, s5_dict)
    print(f"   -> Full Physical Matrix Ready: {X_phys_full.shape[1]} features")

    print("\n2. Stage 6A: Tuning Within-Field Multi-Model Ranker Ensemble...")
    best_ranker_cfg, best_shallow_cfg, ranker_grid_df, blend_weights, alpha_shrink = (
        tune_within_field_ranker(X_core, y, splits, locations)
    )
    ranker_grid_csv = OUTPUT_DIR / "step6_ranker_tuning_grid.csv"
    ranker_grid_df.to_csv(ranker_grid_csv, index=False)

    print("\n   Top 5 Within-Field Ranker Configurations (Tuned on 4 Dev Locations):")
    print(ranker_grid_df.head(5).to_string(index=False))
    print(
        f"   -> Selected Rankers : {best_ranker_cfg['config_id']} + {best_shallow_cfg['config_id']} "
        f"| Ensemble Weights (XGB, ET, Ridge) = ({blend_weights[0]:.2f}, {blend_weights[1]:.2f}, {blend_weights[2]:.2f}) "
        f"| LOLO Shrinkage alpha = {alpha_shrink:.3f}"
    )

    print("\n3. Stage 6B: Tuning Zero-Shot Physical Field Baseline Model...")
    best_phys_cfg, phys_grid_df = tune_physical_baseline_model(
        X_phys_full, y, splits, locations
    )
    phys_grid_csv = OUTPUT_DIR / "step6_physical_baseline_tuning_grid.csv"
    phys_grid_df.to_csv(phys_grid_csv, index=False)
    print(f"   -> Selected Physical Baseline Config : {best_phys_cfg['phys_id']}")

    print("\n4. Stage 6C: Evaluating Final Tuned Step 6 Models on Validation & Crawfordsville Test...")
    summary_df, preds_store, artifacts = evaluate_complete_step6(
        X_core,
        X_phys_full,
        X_phys_step5_old,
        y,
        splits,
        locations,
        best_ranker_cfg,
        best_shallow_cfg,
        best_phys_cfg,
        blend_weights,
        alpha_shrink,
    )

    summary_csv = OUTPUT_DIR / "step6_final_comparison.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 148)
    print(
        "STEP 6 HYPERPARAMETER TUNING & CALIBRATION RESULTS — 4-LOC VALIDATION + UNSEEN CRAWFORDSVILLE TEST"
    )
    print("=" * 148)
    print(summary_df.to_string(index=False))
    print("=" * 148)

    model_bundle_path = OUTPUT_DIR / "step6_tuned_models.pkl"
    with open(model_bundle_path, "wb") as f:
        pickle.dump(artifacts, f)

    test_mask = splits == "test"
    test_preds_df = pd.DataFrame(
        {
            "plot_id": plot_ids[test_mask],
            "location": locations[test_mask],
            "genotype": s5_dict["genotypes"][test_mask],
            "Actual_Yield_bu_ac": y[test_mask],
            "Step5_ZeroShot_Pred": preds_store["step5_test_zeroshot"],
            "Step6_ZeroShot_TrainSplit_Pred": preds_store["test_zeroshot_tr"],
            "Step6_ZeroShot_Full4Loc_Pred": preds_store["test_zeroshot_full"],
            "Step6_FieldCalibrated_Pred": preds_store["test_calibrated_full"],
        }
    )
    test_preds_csv = OUTPUT_DIR / "step6_crawfordsville_predictions.csv"
    test_preds_df.to_csv(test_preds_csv, index=False)

    final_row = summary_df.iloc[-1]
    step5_row = summary_df.iloc[0]

    print("\n[STEP 6 SUMMARY: BEFORE vs. AFTER UPGRADED TUNING]")
    print(
        f"  1. Zero-Shot Test RMSE (Train Split) : {step5_row['ZeroShot_Test_RMSE']:.4f} bu/ac -> "
        f"{final_row['ZeroShot_Test_RMSE']:.4f} bu/ac "
        f"(Zero-Shot R²: {step5_row['ZeroShot_Test_R2']:+.4f} -> {final_row['ZeroShot_Test_R2']:+.4f})"
    )
    print(
        f"  2. Zero-Shot Test RMSE (Full 4-Loc)  : {step5_row['ZeroShot_Full4Loc_RMSE']:.4f} bu/ac -> "
        f"{final_row['ZeroShot_Full4Loc_RMSE']:.4f} bu/ac "
        f"(Zero-Shot R²: {step5_row['ZeroShot_Full4Loc_R2']:+.4f} -> {final_row['ZeroShot_Full4Loc_R2']:+.4f})"
    )
    print(
        f"  3. Field-Calibrated Test RMSE & R²   : {step5_row['Calibrated_RMSE']:.4f} bu/ac (R²={step5_row['Calibrated_R2']:+.4f}) -> "
        f"{final_row['Calibrated_RMSE']:.4f} bu/ac (R²={final_row['Calibrated_R2']:+.4f})"
    )
    print(
        f"  4. Crawfordsville Plot Ranking (r)   : {step5_row['Test_Pearson_r']:.4f} -> "
        f"{final_row['Test_Pearson_r']:.4f} (Full 4-Loc r = {final_row['Test_Full4Loc_r']:.4f})"
    )
    print(
        f"  5. 4-Location Holdout Val RMSE & R²  : {step5_row['Val_RMSE']:.4f} bu/ac (R²={step5_row['Val_R2']:.4f}) -> "
        f"{final_row['Val_RMSE']:.4f} bu/ac (R²={final_row['Val_R2']:.4f})"
    )
    print(f"  -> Total Step 6 Runtime              : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()