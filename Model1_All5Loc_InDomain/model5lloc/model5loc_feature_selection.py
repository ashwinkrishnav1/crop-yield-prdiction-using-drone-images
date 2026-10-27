"""
Model1_All5Loc_InDomain — Step 5: In-Domain Multi-Location Feature Selection & Scaling
* All 5 Locations (Ames, Crawfordsville, Lincoln, MOValley, Scottsbluff)
  represented in Train (1,489 plots), Val (319 plots), and Test (323 plots).
* Concepts: Train-Only Flight Covariate Alignment, Stratified 5-Fold OOF Pedigree Encoding,
  Train-Only Spatial Grid Kriging, OOF Ridge/Cosine Prototypes, Location-Prior Fusion,
  and mRMR Collinearity Pruning (fit strictly on the 1,489 TRAIN plots).
"""

import os
import re
import time
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PROJECT PATHS & CONFIGURATION (Model1_All5Loc_InDomain)
# ==============================================================================
MODEL_NAME = "Model1_All5Loc_InDomain"
PROJECT_ROOT = Path(r"C:\Users\ashwi\Music\cropyeildererrrrrrr")
VALID_PLOTS_CSV = PROJECT_ROOT / "drone_dataset" / "valid_plots.csv"
RAW_GT_CSV = PROJECT_ROOT / "drone_dataset" / "GroundTruth" / "HYBRID_HIPS_V3.5_ALLPLOTS.csv"

MODEL_DIR = PROJECT_ROOT / MODEL_NAME
STEP4_DIR = MODEL_DIR / "step4_feature_engineering"
STEP4_RAW_CACHE = STEP4_DIR / "cache" / "m1_step4_raw_multimodal_features.npz"
STEP4_MASTER_NPZ = STEP4_DIR / "m1_step4_master_features.npz"

SAT_CACHE_NPZ = PROJECT_ROOT / "satellite_evaluation" / "cache" / "satellite_tp1_tp6_features.npz"

OUTPUT_DIR = MODEL_DIR / "step5_feature_selection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CANOPY_FEATURE_NAMES = [
    "ExG_Mean",
    "ExG_Std",
    "NGRDI_Mean",
    "NGRDI_Std",
    "VARI_Mean",
    "GLI_Mean",
    "Green_Canopy_Cover",
    "Yellow_Senescence_Frac",
    "PureCanopy_ExG_Mean",
    "PureCanopy_NGRDI_Mean",
    "Patch_Canopy_Gap_Std",
    "Patch_Canopy_Min",
]


# ==============================================================================
# 2. LOAD STEP 4 CACHES, PHENOLOGY METADATA & OPTIONAL SATELLITE CACHE
# ==============================================================================
def load_all_step5_sources() -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    pd.DataFrame,
    Optional[Dict[str, np.ndarray]],
]:
    if not STEP4_RAW_CACHE.exists():
        raise FileNotFoundError(f"Missing {STEP4_RAW_CACHE}. Run model5loc_preprocess_feature_extraction.py first.")
    if not STEP4_MASTER_NPZ.exists():
        raise FileNotFoundError(f"Missing {STEP4_MASTER_NPZ}. Run model5loc_preprocess_feature_extraction.py first.")

    raw_data = np.load(STEP4_RAW_CACHE, allow_pickle=True)
    master_data = np.load(STEP4_MASTER_NPZ, allow_pickle=True)

    raw_dict = {k: raw_data[k] for k in raw_data.files}
    master_dict = {k: master_data[k] for k in master_data.files}

    plot_ids = [str(pid).strip() for pid in master_dict["plot_ids"]]
    df_valid = pd.read_csv(VALID_PLOTS_CSV)
    df_valid["plot_id_clean"] = df_valid["plot_id"].astype(str).str.strip()
    df_valid = (
        df_valid.drop_duplicates(subset=["plot_id_clean"])
        .set_index("plot_id_clean")
        .reindex(plot_ids)
        .reset_index()
    )

    if RAW_GT_CSV.exists():
        try:
            df_gt = pd.read_csv(RAW_GT_CSV)
            pheno_cols = [
                c for c in df_gt.columns if c.lower() in ("daystoanthesis", "gddtoanthesis")
            ]
            if pheno_cols and "qrCode" in df_gt.columns and "qrCode" in df_valid.columns:
                df_gt_sub = df_gt[["qrCode"] + pheno_cols].drop_duplicates(subset=["qrCode"])
                df_valid = df_valid.merge(df_gt_sub, on="qrCode", how="left")
                print(f"[METADATA] Merged pre-harvest phenology traits: {pheno_cols}")
        except Exception:
            pass

    sat_dict = None
    if SAT_CACHE_NPZ.exists():
        try:
            sat_npz = np.load(SAT_CACHE_NPZ, allow_pickle=True)
            sat_cand = {
                k: sat_npz[k].astype(np.float32)
                for k in sat_npz.files
                if sat_npz[k].ndim == 2 and sat_npz[k].shape[0] == len(plot_ids)
            }
            if len(sat_cand) > 0:
                sat_dict = sat_cand
                print(f"[SATELLITE CACHE] Loaded {len(sat_dict)} multi-temporal 6-band blocks.")
        except Exception:
            pass

    splits = master_dict["splits"]
    print("=" * 98)
    print(
        f"[{MODEL_NAME}] STEP 5 LOADED: {len(splits)} total plots across All 5 Locations "
        f"(Train: {(splits=='train').sum()} | Val: {(splits=='val').sum()} | Test: {(splits=='test').sum()})"
    )
    print("=" * 98)
    return raw_dict, master_dict, df_valid, sat_dict


# ==============================================================================
# 3. TRAIN-ONLY FLIGHT-LEVEL COVARIATE ALIGNMENT (ACROSS ALL 5 LOCATIONS)
# ==============================================================================
def apply_flight_level_normalization_train_only(
    raw_dict: Dict[str, np.ndarray],
    splits: np.ndarray,
    locations: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Because all 5 locations have 70% of their plots in `splits == 'train'`,
    flight normalization mean and std are computed 100% strictly from the TRAIN plots
    of each location (zero leakage from val or test plots).
    """
    norm_dict = {k: v.copy() for k, v in raw_dict.items()}
    image_keys = [
        k
        for k in raw_dict.keys()
        if k.startswith(("tp1_", "tp2_", "tp3_", "spec_", "cnn_"))
        and raw_dict[k].ndim == 2
    ]
    train_mask = splits == "train"

    for key in image_keys:
        mat = np.nan_to_num(raw_dict[key].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mat_norm = np.zeros_like(mat, dtype=np.float32)

        for loc in np.unique(locations):
            loc_mask = locations == loc
            ref_mask = loc_mask & train_mask  # Strictly TRAIN plots only!
            mu = np.mean(mat[ref_mask], axis=0, keepdims=True)
            sd = np.std(mat[ref_mask], axis=0, keepdims=True)
            sd = np.where(sd < 1e-5, 1.0, sd)
            mat_norm[loc_mask] = np.clip((mat[loc_mask] - mu) / sd, -6.0, 6.0)

        norm_dict[key] = mat_norm

    return norm_dict


# ==============================================================================
# 4. STRATIFIED 5-FOLD TRAIN-ONLY OOF PEDIGREE & EXPERIMENT ENCODING
# ==============================================================================
def parse_hybrid_parents(geno_str: str) -> Tuple[str, str]:
    g = str(geno_str).strip().upper()
    if not g or g == "NAN":
        return "UNKNOWN", "UNKNOWN"
    parts = re.split(r"\s+[X\*\/]\s+|\s*[\*\/]\s*", g)
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()
    return g, g


def stratified_kfold_encode_train_only(
    cat_series: np.ndarray,
    target_vals: np.ndarray,
    splits: np.ndarray,
    locations: np.ndarray,
    smoothing_m: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 5-Fold Location-Stratified OOF Empirical Bayes target encoding strictly
    inside `splits == 'train'` (1,489 plots), and applies the train-fitted mapping
    to `val` (319 plots) and `test` (323 plots).
    """
    train_mask = splits == "train"
    train_idx = np.where(train_mask)[0]
    cats_tr, y_tr, loc_tr = cat_series[train_idx], target_vals[train_idx], locations[train_idx]
    prior = float(np.mean(y_tr))

    def fit_map(c_arr: np.ndarray, y_arr: np.ndarray, l_arr: np.ndarray) -> Dict[str, float]:
        df_tmp = pd.DataFrame({"cat": c_arr, "y": y_arr, "loc": l_arr})
        loc_cat = df_tmp.groupby(["cat", "loc"])["y"].mean().reset_index()
        stats = loc_cat.groupby("cat")["y"].agg(["count", "mean"])
        tot_cnt = df_tmp.groupby("cat")["y"].count()
        eff_cnt = 0.5 * stats["count"] + 0.5 * tot_cnt.reindex(stats.index)
        return ((eff_cnt * stats["mean"] + smoothing_m * prior) / (eff_cnt + smoothing_m)).to_dict()

    encoded = np.full(len(cat_series), prior, dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_tr = np.full(len(train_idx), prior, dtype=np.float32)

    for f_tr, f_va in skf.split(train_idx, loc_tr):
        m_kf = fit_map(cats_tr[f_tr], y_tr[f_tr], loc_tr[f_tr])
        oof_tr[f_va] = [m_kf.get(c, prior) for c in cats_tr[f_va]]

    encoded[train_idx] = oof_tr
    full_train_map = fit_map(cats_tr, y_tr, loc_tr)
    non_train_idx = np.where(~train_mask)[0]
    encoded[non_train_idx] = [full_train_map.get(c, prior) for c in cat_series[non_train_idx]]

    seen_flag = np.array(
        [1.0 if (c in full_train_map and c != "UNKNOWN") else 0.0 for c in cat_series],
        dtype=np.float32,
    )
    return encoded, seen_flag


def build_genetic_and_phenology_features(
    df_valid: pd.DataFrame,
    y: np.ndarray,
    splits: np.ndarray,
    locations: np.ndarray,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    train_mask = splits == "train"
    geno_raw = df_valid["genotype"].fillna("UNKNOWN").astype(str).str.strip().str.upper().values
    exp_raw = (
        df_valid["location"].astype(str).str.strip()
        + "_"
        + df_valid["experiment"].fillna("UNK").astype(str).str.strip()
    ).values

    parents = [parse_hybrid_parents(g) for g in geno_raw]
    p1_raw = np.array([p[0] for p in parents], dtype=object)
    p2_raw = np.array([p[1] for p in parents], dtype=object)

    # Center y strictly using TRAIN split location means across all 5 locations
    y_within_loc = np.zeros_like(y, dtype=np.float32)
    y_z_loc = np.zeros_like(y, dtype=np.float32)
    loc_train_prior = np.zeros_like(y, dtype=np.float32)

    for loc in np.unique(locations):
        m_tr = (locations == loc) & train_mask
        mu = float(np.mean(y[m_tr]))
        sd = max(float(np.std(y[m_tr])), 1.0)
        loc_all = locations == loc
        y_within_loc[loc_all] = y[loc_all] - mu
        y_z_loc[loc_all] = (y[loc_all] - mu) / sd
        loc_train_prior[loc_all] = mu

    geno_kf, geno_seen = stratified_kfold_encode_train_only(geno_raw, y_within_loc, splits, locations, 4.0)
    geno_z_kf, _ = stratified_kfold_encode_train_only(geno_raw, y_z_loc, splits, locations, 4.0)
    p1_kf, _ = stratified_kfold_encode_train_only(p1_raw, y_within_loc, splits, locations, 6.0)
    p2_kf, _ = stratified_kfold_encode_train_only(p2_raw, y_within_loc, splits, locations, 6.0)
    exp_kf, _ = stratified_kfold_encode_train_only(exp_raw, y_within_loc, splits, locations, 8.0)

    parent_gca = 0.5 * (p1_kf + p2_kf)
    comp_adv = np.where(geno_seen > 0.5, geno_kf, parent_gca).astype(np.float32)

    pheno_feats, pheno_names = [], []
    for col in ("daysToAnthesis", "GDDToAnthesis"):
        if col in df_valid.columns:
            raw_v = pd.to_numeric(df_valid[col], errors="coerce")
            if raw_v[train_mask].notnull().sum() > 100:
                med_v = float(raw_v[train_mask].median())
                filled_v = raw_v.fillna(med_v).values.astype(np.float32)
                norm_v = np.zeros_like(filled_v, dtype=np.float32)
                for loc in np.unique(locations):
                    lm = locations == loc
                    rm = lm & train_mask
                    mu_p, sd_p = float(np.mean(filled_v[rm])), max(float(np.std(filled_v[rm])), 1.0)
                    norm_v[lm] = (filled_v[lm] - mu_p) / sd_p
                pheno_feats.append(norm_v)
                pheno_names.append(f"Pheno_FlightNorm_{col}")

    cols_list = [
        loc_train_prior,
        geno_kf,
        geno_z_kf,
        p1_kf,
        p2_kf,
        parent_gca,
        comp_adv,
        exp_kf,
    ] + pheno_feats
    names_list = [
        "Loc_TrainSplit_Mean_Prior",
        "Gen_Hybrid_OOF_Adv",
        "Gen_Hybrid_OOF_Z",
        "Gen_Parent1_OOF_Adv",
        "Gen_Parent2_OOF_Adv",
        "Gen_Parent_GCA_OOF",
        "Gen_Composite_OOF_Adv",
        "Gen_LocExperiment_OOF_Adv",
    ] + pheno_names

    return np.column_stack(cols_list).astype(np.float32), names_list, comp_adv


# ==============================================================================
# 5. TRAIN-ONLY SPATIAL FIELD-GRID KRIGING + BIOLOGICAL RATIOS
# ==============================================================================
def compute_spatial_neighborhood_features(
    df_valid: pd.DataFrame,
    norm_dict: Dict[str, np.ndarray],
    splits: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Strictly uses TRAIN neighbor plots (`loc_tr_idx`) within each of the 5 locations
    to compute spatial field-grid fertility context, keeping val and test plots 100% isolated.
    """
    n = len(df_valid)
    train_mask = splits == "train"
    locs = df_valid["location"].astype(str).str.strip().values
    exps = df_valid["experiment"].astype(str).str.strip().values
    ranges = pd.to_numeric(df_valid["range"], errors="coerce").fillna(0).values.astype(np.float32)
    rows = pd.to_numeric(df_valid["row"], errors="coerce").fillna(0).values.astype(np.float32)

    c2, c3 = norm_dict["tp2_canopy"], norm_dict["tp3_canopy"]
    vis_vigor_tp3 = 0.5 * (c3[:, 6] + c3[:, 8])
    vis_vigor_tp2 = 0.5 * (c2[:, 6] + c2[:, 8])

    neigh_tp3 = np.zeros(n, dtype=np.float32)
    neigh_tp2 = np.zeros(n, dtype=np.float32)
    rel_pos_range = np.zeros(n, dtype=np.float32)
    rel_pos_row = np.zeros(n, dtype=np.float32)

    for loc in np.unique(locs):
        loc_idx = np.where(locs == loc)[0]
        loc_tr_idx = np.where((locs == loc) & train_mask)[0]

        r_mu, r_sd = float(np.mean(ranges[loc_tr_idx])), max(float(np.std(ranges[loc_tr_idx])), 1.0)
        w_mu, w_sd = float(np.mean(rows[loc_tr_idx])), max(float(np.std(rows[loc_tr_idx])), 1.0)
        rel_pos_range[loc_idx] = (ranges[loc_idx] - r_mu) / r_sd
        rel_pos_row[loc_idx] = (rows[loc_idx] - w_mu) / w_sd

        for i in loc_idx:
            dist = np.abs(ranges[loc_tr_idx] - ranges[i]) + np.abs(rows[loc_tr_idx] - rows[i])
            nbr_mask = (dist > 0) & (dist <= 2.5) & (exps[loc_tr_idx] == exps[i])
            if np.sum(nbr_mask) < 2:
                nbr_mask = (dist > 0) & (dist <= 3.5)
            if np.sum(nbr_mask) > 0:
                nbr_indices = loc_tr_idx[nbr_mask]
                neigh_tp3[i] = float(np.mean(vis_vigor_tp3[nbr_indices]))
                neigh_tp2[i] = float(np.mean(vis_vigor_tp2[nbr_indices]))
            else:
                neigh_tp3[i] = vis_vigor_tp3[i]
                neigh_tp2[i] = vis_vigor_tp2[i]

    spatial_X = np.column_stack(
        [
            neigh_tp3,
            vis_vigor_tp3 - neigh_tp3,
            neigh_tp2,
            vis_vigor_tp2 - neigh_tp2,
            rel_pos_range,
            rel_pos_row,
        ]
    ).astype(np.float32)
    spatial_names = [
        "Spatial_Neighbor_CanopyVigor_TP3",
        "Spatial_PlotMinusNeighbor_Vigor_TP3",
        "Spatial_Neighbor_CanopyVigor_TP2",
        "Spatial_PlotMinusNeighbor_Vigor_TP2",
        "Spatial_FieldGrid_Range_Z",
        "Spatial_FieldGrid_Row_Z",
    ]
    return spatial_X, spatial_names


def build_clean_meta_and_bio_features(
    raw_dict: Dict[str, np.ndarray],
    norm_dict: Dict[str, np.ndarray],
    genetic_adv: np.ndarray,
    splits: np.ndarray,
    locations: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    meta_raw = raw_dict["meta"]
    stand_count, planting_doy, stand_density, log_stand = (
        meta_raw[:, 0],
        meta_raw[:, 1],
        meta_raw[:, 2],
        meta_raw[:, 3],
    )

    train_mask = splits == "train"
    stand_flight_z = np.zeros_like(stand_count, dtype=np.float32)
    density_flight_z = np.zeros_like(stand_density, dtype=np.float32)

    for loc in np.unique(locations):
        loc_m = locations == loc
        ref_m = loc_m & train_mask
        s_mu, s_sd = float(np.mean(stand_count[ref_m])), max(float(np.std(stand_count[ref_m])), 1.0)
        d_mu, d_sd = float(np.mean(stand_density[ref_m])), max(float(np.std(stand_density[ref_m])), 1e-3)
        stand_flight_z[loc_m] = (stand_count[loc_m] - s_mu) / s_sd
        density_flight_z[loc_m] = (stand_density[loc_m] - d_mu) / d_sd

    c1, c2, c3 = raw_dict["tp1_canopy"], raw_dict["tp2_canopy"], raw_dict["tp3_canopy"]
    cn1, cn2, cn3 = norm_dict["tp1_canopy"], norm_dict["tp2_canopy"], norm_dict["tp3_canopy"]
    fgcc_1, fgcc_2, fgcc_3 = c1[:, 6], c2[:, 6], c3[:, 6]
    yellow_3, gap_std_3 = c3[:, 7], c3[:, 10]
    pure_exg_1, pure_exg_2, pure_exg_3 = cn1[:, 8], cn2[:, 8], cn3[:, 8]
    season_vigor_index = 0.20 * pure_exg_1 + 0.40 * pure_exg_2 + 0.40 * pure_exg_3

    combined = np.column_stack(
        [
            stand_count,
            planting_doy,
            stand_density,
            log_stand,
            stand_flight_z,
            density_flight_z,
            stand_density * fgcc_2,
            stand_density * fgcc_3,
            stand_flight_z + cn3[:, 6],
            log_stand * pure_exg_3,
            fgcc_2 / np.clip(stand_count, 10.0, None),
            fgcc_3 / np.clip(stand_count, 10.0, None),
            (fgcc_3 + 0.02) / (fgcc_2 + 0.02),
            (fgcc_2 + 0.02) / (fgcc_1 + 0.02),
            yellow_3 / (fgcc_3 + 0.02),
            fgcc_3 - gap_std_3,
            season_vigor_index,
            genetic_adv + 4.0 * season_vigor_index,
        ]
    ).astype(np.float32)

    names = [
        "Meta_TotalStandCount",
        "Meta_Planting_DOY",
        "Meta_Stand_Density_10kPx",
        "Meta_Log_StandCount",
        "Meta_FlightNorm_Stand_Z",
        "Meta_FlightNorm_Density_Z",
        "Bio_Effective_Green_Stand_TP2",
        "Bio_Effective_Green_Stand_TP3",
        "Bio_FlightStand_Plus_TP3Cover",
        "Bio_LogStand_x_TP3_PureExG",
        "Bio_Canopy_Per_Plant_TP2",
        "Bio_Canopy_Per_Plant_TP3",
        "Bio_StayGreen_Ratio_TP3_TP2",
        "Bio_Canopy_Expansion_Ratio_TP2_TP1",
        "Bio_Senescence_To_Green_TP3",
        "Bio_GapAdjusted_GreenCover_TP3",
        "Bio_FlightNorm_Season_Vigor",
        "Bio_Genetics_Plus_CanopyVigor",
    ]
    return combined, names


# ==============================================================================
# 6. TRAIN-ONLY 5-FOLD STRATIFIED OOF CNN & SATELLITE PROTOTYPES
# ==============================================================================
def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    return mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-6, None)


def build_train_oof_cnn_and_sat_prototypes(
    feat_source_dict: Dict[str, np.ndarray],
    sat_norm_dict: Optional[Dict[str, np.ndarray]],
    y: np.ndarray,
    splits: np.ndarray,
    locations: np.ndarray,
    prefix_tag: str = "Proto",
) -> Tuple[np.ndarray, List[str]]:
    """
    Fits CNN (and optional 6-Band Satellite) prototypes and RidgeCV vigor strictly
    on `splits == 'train'` (via 5-Fold Location-Stratified OOF on train, and full-train
    fit applied to val and test).
    """
    train_mask = splits == "train"
    train_idx = np.where(train_mask)[0]
    non_train_idx = np.where(~train_mask)[0]
    train_locs = locations[train_mask]
    y_train = y[train_mask]

    y_train_loc_norm = np.zeros_like(y_train, dtype=np.float32)
    high_mask_train = np.zeros(len(y_train), dtype=bool)
    low_mask_train = np.zeros(len(y_train), dtype=bool)

    for loc in np.unique(train_locs):
        m_sub = train_locs == loc
        y_loc = y_train[m_sub]
        mu, sd = float(np.mean(y_loc)), max(float(np.std(y_loc)), 1.0)
        y_train_loc_norm[m_sub] = (y_loc - mu) / sd
        high_mask_train[m_sub] = y_loc >= np.percentile(y_loc, 75)
        low_mask_train[m_sub] = y_loc <= np.percentile(y_loc, 25)

    blocks = [
        ("TP1_Mean", feat_source_dict["tp1_mean"]),
        ("TP2_Mean", feat_source_dict["tp2_mean"]),
        ("TP3_Mean", feat_source_dict["tp3_mean"]),
        ("TP3_PatchStd", feat_source_dict["tp3_std"]),
        ("Delta_TP2_TP1", feat_source_dict["tp2_mean"] - feat_source_dict["tp1_mean"]),
        ("Delta_TP3_TP2", feat_source_dict["tp3_mean"] - feat_source_dict["tp2_mean"]),
        ("Delta_TP3_TP1", feat_source_dict["tp3_mean"] - feat_source_dict["tp1_mean"]),
        (
            "Joint_AllTP_Canopy",
            np.hstack(
                [
                    feat_source_dict["tp1_canopy"],
                    feat_source_dict["tp2_canopy"],
                    feat_source_dict["tp3_canopy"],
                    feat_source_dict["tp3_canopy"] - feat_source_dict["tp1_canopy"],
                ]
            ),
        ),
    ]

    if sat_norm_dict is not None:
        sat_spec_keys = sorted([k for k in sat_norm_dict.keys() if "spec" in k.lower()])
        sat_cnn_keys = sorted([k for k in sat_norm_dict.keys() if "cnn" in k.lower()])
        if sat_spec_keys:
            all_sat_spec = np.hstack([sat_norm_dict[k] for k in sat_spec_keys])
            blocks.append(("Sat6Band_AllTP_Spec", all_sat_spec))
            if len(sat_spec_keys) >= 3:
                blocks.append(
                    (
                        "Sat6Band_Delta_Late_Early",
                        sat_norm_dict[sat_spec_keys[-2]] - sat_norm_dict[sat_spec_keys[0]],
                    )
                )
        for sk in sat_cnn_keys:
            blocks.append((f"Sat6Band_{sk}", sat_norm_dict[sk]))

    proto_feats, proto_names = [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for blk_label, mat in blocks:
        mat_clean = np.nan_to_num(mat.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        train_mean_vec = np.mean(mat_clean[train_mask], axis=0, keepdims=True)
        mat_unit = l2_normalize_rows(mat_clean - train_mean_vec)
        mat_train = mat_unit[train_mask]

        high_proto = l2_normalize_rows(np.mean(mat_train[high_mask_train], axis=0, keepdims=True))[0]
        low_proto = l2_normalize_rows(np.mean(mat_train[low_mask_train], axis=0, keepdims=True))[0]

        sim_high = mat_unit @ high_proto
        sim_low = mat_unit @ low_proto
        sim_diff = sim_high - sim_low

        oof_vigor = np.zeros(len(mat_unit), dtype=np.float32)
        for f_tr, f_va in skf.split(train_idx, train_locs):
            ridge_kf = RidgeCV(alphas=[20.0, 80.0, 250.0, 600.0, 1500.0])
            ridge_kf.fit(mat_train[f_tr], y_train_loc_norm[f_tr])
            oof_vigor[train_idx[f_va]] = ridge_kf.predict(mat_train[f_va]).astype(np.float32)

        ridge_full = RidgeCV(alphas=[20.0, 80.0, 250.0, 600.0, 1500.0])
        ridge_full.fit(mat_train, y_train_loc_norm)
        oof_vigor[non_train_idx] = ridge_full.predict(mat_unit[non_train_idx]).astype(np.float32)

        proto_feats.append(np.column_stack([sim_high, sim_low, sim_diff, oof_vigor]))
        proto_names.extend(
            [
                f"{prefix_tag}_{blk_label}_SimHigh",
                f"{prefix_tag}_{blk_label}_SimLow",
                f"{prefix_tag}_{blk_label}_NetDiff",
                f"{prefix_tag}_{blk_label}_OOF_RidgeVigor",
            ]
        )

    return np.hstack(proto_feats).astype(np.float32), proto_names


# ==============================================================================
# 7. STABILITY SCORING + mRMR COLLINEARITY PRUNING (STRICTLY ON TRAIN PLOTS)
# ==============================================================================
def score_features_domain_invariant(
    X_candidates: np.ndarray,
    candidate_names: List[str],
    y: np.ndarray,
    splits: np.ndarray,
    locations: np.ndarray,
) -> pd.DataFrame:
    train_mask = splits == "train"
    X_tr = np.nan_to_num(X_candidates[train_mask], nan=0.0, posinf=0.0, neginf=0.0)
    y_tr = y[train_mask]
    loc_tr = locations[train_mask]
    unique_locs = np.unique(loc_tr)

    n_feats = X_tr.shape[1]
    loc_corrs = np.zeros((len(unique_locs), n_feats), dtype=np.float32)
    loc_means = np.zeros((len(unique_locs), n_feats), dtype=np.float32)
    loc_vars = np.zeros((len(unique_locs), n_feats), dtype=np.float32)

    for li, loc in enumerate(unique_locs):
        m = loc_tr == loc
        X_sub, y_sub = X_tr[m], y_tr[m]
        y_c = y_sub - np.mean(y_sub)
        X_c = X_sub - np.mean(X_sub, axis=0, keepdims=True)
        denom = np.sqrt(np.sum(X_c**2, axis=0) * np.sum(y_c**2))
        loc_corrs[li] = np.where(denom > 1e-8, (y_c @ X_c) / denom, 0.0)
        loc_means[li] = np.mean(X_sub, axis=0)
        loc_vars[li] = np.var(X_sub, axis=0) + 1e-6

    mean_within_r = np.mean(loc_corrs, axis=0)
    abs_mean_r = np.abs(mean_within_r)
    sign_agreement = np.maximum(np.sum(loc_corrs > 0, axis=0), np.sum(loc_corrs < 0, axis=0)) / float(
        len(unique_locs)
    )
    min_loc_r = np.min(loc_corrs * np.sign(mean_within_r + 1e-12), axis=0)
    city_bias_ratio = np.var(loc_means, axis=0) / np.clip(np.mean(loc_vars, axis=0), 1e-6, None)

    domain_score = (
        abs_mean_r * (sign_agreement**1.5) * np.where(min_loc_r > 0, 1.25, 0.75)
    ) / (1.0 + 0.35 * city_bias_ratio)

    return (
        pd.DataFrame(
            {
                "Feature_Index": np.arange(n_feats),
                "Feature_Name": candidate_names,
                "Domain_Invariant_Score": domain_score,
                "Mean_Within_City_r": mean_within_r,
                "Sign_Consistency": sign_agreement,
                "City_Bias_Ratio": city_bias_ratio,
            }
        )
        .sort_values(by="Domain_Invariant_Score", ascending=False)
        .reset_index(drop=True)
    )


def select_mrmr_nonredundant_indices(
    X_train: np.ndarray,
    ranked_indices: np.ndarray,
    max_k: int,
    collinearity_threshold: float = 0.88,
) -> np.ndarray:
    if len(ranked_indices) == 0 or max_k <= 0:
        return np.array([], dtype=int)

    pool = ranked_indices[: min(len(ranked_indices), max_k * 5)]
    X_sub = X_train[:, pool].astype(np.float32)
    X_c = X_sub - np.mean(X_sub, axis=0, keepdims=True)
    norms = np.linalg.norm(X_c, axis=0, keepdims=True)
    X_u = X_c / np.clip(norms, 1e-6, None)

    selected_local = [0]
    for i in range(1, len(pool)):
        if len(selected_local) >= max_k:
            break
        vec = X_u[:, i]
        corrs = np.abs(X_u[:, selected_local].T @ vec)
        if np.max(corrs) < collinearity_threshold:
            selected_local.append(i)

    return pool[selected_local]


# ==============================================================================
# 8. EVALUATION ENGINE (5-LOCATION TRAIN -> 5-LOCATION VAL & TEST)
# ==============================================================================
def compute_location_balanced_weights(locations_subset: np.ndarray) -> np.ndarray:
    unique_locs, counts = np.unique(locations_subset, return_counts=True)
    total, n_locs = float(len(locations_subset)), float(len(unique_locs))
    weight_map = {loc: total / (n_locs * float(cnt)) for loc, cnt in zip(unique_locs, counts)}
    return np.array([weight_map[loc] for loc in locations_subset], dtype=np.float32)


def get_step5_xgb() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=320,
        learning_rate=0.025,
        max_depth=3,
        min_child_weight=16,
        subsample=0.80,
        colsample_bytree=0.45,
        reg_alpha=2.5,
        reg_lambda=14.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def evaluate_m1_feature_set(
    label: str,
    X_raw: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    locations: np.ndarray,
    use_location_centered_hybrid: bool = True,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Fits StandardScaler, location baselines, XGBoost, and RidgeCV strictly on the
    1,489 TRAIN plots (`splits == 'train'`) and evaluates on the 319 VAL plots
    and 323 TEST plots across all 5 locations.
    """
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    scaler = StandardScaler()
    scaler.fit(X_raw[train_mask])
    X = scaler.transform(X_raw).astype(np.float32)

    tr_locs = locations[train_mask]
    w_train = compute_location_balanced_weights(tr_locs)

    if not use_location_centered_hybrid:
        model_tr = get_step5_xgb()
        model_tr.fit(X[train_mask], y[train_mask])
        val_preds = model_tr.predict(X[val_mask])
        test_preds = model_tr.predict(X[test_mask])
        tr_loc_means = {loc: float(np.mean(y[train_mask][tr_locs == loc])) for loc in np.unique(tr_locs)}
        test_offsets = np.array([tr_loc_means[loc] for loc in locations[test_mask]], dtype=np.float32)
    else:
        # 1. Location-Centered Within-Field Ranker (XGBoost + RidgeCV) fit strictly on TRAIN
        tr_loc_means = {loc: float(np.mean(y[train_mask][tr_locs == loc])) for loc in np.unique(tr_locs)}
        y_tr_centered = np.array(
            [yv - tr_loc_means[loc] for yv, loc in zip(y[train_mask], tr_locs)],
            dtype=np.float32,
        )

        model_cent = get_step5_xgb()
        model_cent.fit(X[train_mask], y_tr_centered, sample_weight=w_train)
        ridge_cent = RidgeCV(alphas=[10.0, 50.0, 150.0, 400.0, 1000.0])
        ridge_cent.fit(X[train_mask], y_tr_centered)

        # 2. Direct Full-Yield XGBoost Model fit strictly on TRAIN (captures non-linear location x vigor interactions)
        model_abs = get_step5_xgb()
        model_abs.fit(X[train_mask], y[train_mask], sample_weight=w_train)

        val_offsets = np.array([tr_loc_means[loc] for loc in locations[val_mask]], dtype=np.float32)
        test_offsets = np.array([tr_loc_means[loc] for loc in locations[test_mask]], dtype=np.float32)

        val_cent_p = 0.75 * model_cent.predict(X[val_mask]) + 0.25 * ridge_cent.predict(X[val_mask]) + val_offsets
        test_cent_p = 0.75 * model_cent.predict(X[test_mask]) + 0.25 * ridge_cent.predict(X[test_mask]) + test_offsets

        val_preds = 0.65 * val_cent_p + 0.35 * model_abs.predict(X[val_mask])
        test_preds = 0.65 * test_cent_p + 0.35 * model_abs.predict(X[test_mask])

    y_val, y_test = y[val_mask], y[test_mask]
    test_cent_true = y_test - test_offsets
    test_cent_pred = test_preds - test_offsets
    within_r = float(np.corrcoef(test_cent_true, test_cent_pred)[0, 1])

    metrics = {
        "Feature_Set": label,
        "Num_Features": int(X.shape[1]),
        "Val_RMSE": round(float(np.sqrt(mean_squared_error(y_val, val_preds))), 4),
        "Val_MAE": round(float(mean_absolute_error(y_val, val_preds)), 4),
        "Val_R2": round(float(r2_score(y_val, val_preds)), 4),
        "Test_RMSE": round(float(np.sqrt(mean_squared_error(y_test, test_preds))), 4),
        "Test_MAE": round(float(mean_absolute_error(y_test, test_preds)), 4),
        "Test_R2": round(float(r2_score(y_test, test_preds)), 4),
        "Test_Overall_r": round(float(np.corrcoef(y_test, test_preds)[0, 1]), 4),
        "Test_WithinLoc_r": round(within_r, 4),
    }
    return metrics, test_preds


# ==============================================================================
# 9. MAIN STEP 5 PIPELINE FOR MODEL 1 (ALL 5 LOCATIONS IN-DOMAIN)
# ==============================================================================
def main():
    t0 = time.time()
    raw_dict, master_dict, df_valid, sat_dict = load_all_step5_sources()

    X_step4_pca = master_dict["X"]
    y = master_dict["y"].astype(np.float32)
    splits = master_dict["splits"]
    locations = master_dict["locations"]
    plot_ids = master_dict["plot_ids"]
    train_mask = splits == "train"

    print("\n1. Applying Flight-Level Covariate Alignment (fit strictly on 1,489 TRAIN plots)...")
    norm_dict = apply_flight_level_normalization_train_only(raw_dict, splits, locations)
    sat_norm_dict = (
        apply_flight_level_normalization_train_only(sat_dict, splits, locations)
        if sat_dict is not None
        else None
    )

    print("2. Building Multimodal Pool (Stratified OOF Pedigree + Spatial Kriging + OOF Prototypes)...")
    raw_proto, _ = build_train_oof_cnn_and_sat_prototypes(
        raw_dict, None, y, splits, locations, prefix_tag="RawProto"
    )
    c1, c2, c3 = raw_dict["tp1_canopy"], raw_dict["tp2_canopy"], raw_dict["tp3_canopy"]
    raw_phys_pool = np.nan_to_num(
        np.hstack(
            [
                raw_dict["meta"],
                raw_proto,
                c1,
                c2,
                c3,
                c2 - c1,
                c3 - c2,
                c3 - c1,
                raw_dict["tp3_mean"],
                raw_dict["tp2_mean"],
                raw_dict["tp1_mean"],
            ]
        ).astype(np.float32),
        nan=0.0,
    )
    phys_rank = score_features_domain_invariant(
        raw_phys_pool, [f"RawPhys_{i}" for i in range(raw_phys_pool.shape[1])], y, splits, locations
    )
    X_phys_baseline = raw_phys_pool[:, phys_rank["Feature_Index"].values[:140]].astype(np.float32)

    gen_X, gen_names, genetic_adv = build_genetic_and_phenology_features(
        df_valid, y, splits, locations
    )
    meta_bio_X, meta_bio_names = build_clean_meta_and_bio_features(
        raw_dict, norm_dict, genetic_adv, splits, locations
    )
    spatial_X, spatial_names = compute_spatial_neighborhood_features(
        df_valid, norm_dict, splits
    )
    proto_X, proto_names = build_train_oof_cnn_and_sat_prototypes(
        norm_dict, sat_norm_dict, y, splits, locations, prefix_tag="Proto"
    )

    cn1, cn2, cn3 = norm_dict["tp1_canopy"], norm_dict["tp2_canopy"], norm_dict["tp3_canopy"]
    canopy_blocks = [
        ("TP1", cn1),
        ("TP2", cn2),
        ("TP3", cn3),
        ("Delta_TP2_TP1", cn2 - cn1),
        ("Delta_TP3_TP2", cn3 - cn2),
        ("Delta_TP3_TP1", cn3 - cn1),
        ("Season_AUC", 0.25 * cn1 + 0.50 * cn2 + 0.25 * cn3),
    ]
    canopy_X = np.hstack([blk for _, blk in canopy_blocks]).astype(np.float32)
    canopy_names = [
        f"Canopy_FlightNorm_{p}_{n}" for p, _ in canopy_blocks for n in CANOPY_FEATURE_NAMES
    ]

    extra_spec_X = canopy_X
    extra_spec_names = canopy_names
    if sat_norm_dict is not None:
        sat_spec_keys = sorted([k for k in sat_norm_dict.keys() if "spec" in k.lower()])
        if sat_spec_keys:
            sat_s_mat = np.hstack([sat_norm_dict[k] for k in sat_spec_keys])
            sat_s_names = [
                f"SatSpec_{k}_F{i}"
                for k in sat_spec_keys
                for i in range(sat_norm_dict[k].shape[1])
            ]
            extra_spec_X = np.hstack([canopy_X, sat_s_mat]).astype(np.float32)
            extra_spec_names = canopy_names + sat_s_names

    spec_rank = score_features_domain_invariant(
        extra_spec_X, extra_spec_names, y, splits, locations
    )
    top_spec_idx = spec_rank["Feature_Index"].values[:22]

    core_X = np.hstack(
        [gen_X, meta_bio_X, spatial_X, proto_X, extra_spec_X[:, top_spec_idx]]
    ).astype(np.float32)
    core_names = (
        gen_names
        + meta_bio_names
        + spatial_names
        + proto_names
        + [extra_spec_names[i] for i in top_spec_idx]
    )
    n_core = core_X.shape[1]

    cnn_blocks = [
        ("CNN_TP3_Mean", norm_dict["tp3_mean"]),
        ("CNN_TP2_Mean", norm_dict["tp2_mean"]),
        ("CNN_TP1_Mean", norm_dict["tp1_mean"]),
        ("CNN_TP3_PatchStd", norm_dict["tp3_std"]),
        ("CNN_Delta_TP3_TP2", norm_dict["tp3_mean"] - norm_dict["tp2_mean"]),
        ("CNN_Delta_TP2_TP1", norm_dict["tp2_mean"] - norm_dict["tp1_mean"]),
        ("CNN_Delta_TP3_TP1", norm_dict["tp3_mean"] - norm_dict["tp1_mean"]),
    ]
    cnn_X = np.hstack([blk for _, blk in cnn_blocks]).astype(np.float32)
    cnn_names = [f"{bn}_D{i}" for bn, blk in cnn_blocks for i in range(blk.shape[1])]

    all_X = np.nan_to_num(np.hstack([core_X, cnn_X]).astype(np.float32), nan=0.0)
    all_names = core_names + cnn_names

    print("3. Running Stability Selection + mRMR Collinearity Pruning (|r| < 0.88) on 1,489 Train plots...")
    full_ranking_df = score_features_domain_invariant(all_X, all_names, y, splits, locations)
    full_ranking_df.to_csv(OUTPUT_DIR / "m1_step5_feature_ranking_report.csv", index=False)

    print("\n   Top 12 Most Informative Features Across All 5 Locations:")
    print(
        full_ranking_df[
            ["Feature_Name", "Domain_Invariant_Score", "Mean_Within_City_r", "Sign_Consistency", "City_Bias_Ratio"]
        ]
        .head(12)
        .to_string(index=False)
    )

    non_core_df = full_ranking_df[full_ranking_df["Feature_Index"] >= n_core]

    def get_mrmr_subset(k_extra: int) -> Tuple[np.ndarray, List[str]]:
        if k_extra == 0:
            idx = np.arange(n_core)
        else:
            mrmr_extra = select_mrmr_nonredundant_indices(
                all_X[train_mask],
                non_core_df["Feature_Index"].values,
                max_k=k_extra,
                collinearity_threshold=0.88,
            )
            idx = np.concatenate([np.arange(n_core), mrmr_extra])
        return all_X[:, idx], [all_names[i] for i in idx]

    X_core, names_core = get_mrmr_subset(0)
    X_top25, names_top25 = get_mrmr_subset(25)
    X_top50, names_top50 = get_mrmr_subset(50)
    X_top90, names_top90 = get_mrmr_subset(90)

    print("\n4. Evaluating Step 5 Feature Sets on 5-Location Validation (319) & Test (323)...")
    variants = [
        ("0. Step 4 Full Baseline (PCA + Metadata)", X_step4_pca, [f"F_{i}" for i in range(X_step4_pca.shape[1])], False),
        ("1. Step 5 Core Multimodal (Pedigree + Bio + Spatial + OOF Prototypes)", X_core, names_core, True),
        ("2. Step 5 Core + Top-25 mRMR CNN Channels", X_top25, names_top25, True),
        ("3. Step 5 Core + Top-50 mRMR CNN Channels", X_top50, names_top50, True),
        ("4. Step 5 Core + Top-90 mRMR CNN Channels", X_top90, names_top90, True),
    ]

    results, preds_map, matrix_map = [], {}, {}
    for v_label, v_X, v_names, use_hyb in variants:
        m, preds = evaluate_m1_feature_set(v_label, v_X, y, splits, locations, use_location_centered_hybrid=use_hyb)
        results.append(m)
        preds_map[v_label] = preds
        matrix_map[v_label] = (v_X, v_names)

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_DIR / "m1_step5_ablation_comparison.csv", index=False)

    print("\n" + "=" * 136)
    print(f"[{MODEL_NAME}] STEP 5 BENCHMARK (ALL 5 LOCATIONS IN TRAIN / VAL / TEST — FIT STRICTLY ON 1,489 TRAIN PLOTS)")
    print("=" * 136)
    print(res_df.to_string(index=False))
    print("=" * 136)

    step5_candidates = res_df[res_df["Feature_Set"].str.startswith(("1.", "2.", "3.", "4."))]
    best_row = step5_candidates.sort_values(by=["Val_RMSE", "Test_RMSE"]).iloc[0]
    best_label = best_row["Feature_Set"]
    best_X_raw, best_feature_names = matrix_map[best_label]
    best_preds = preds_map[best_label]

    final_scaler = StandardScaler().fit(best_X_raw[train_mask])
    best_X_scaled = final_scaler.transform(best_X_raw).astype(np.float32)

    phys_scaler = StandardScaler().fit(X_phys_baseline[train_mask])
    X_phys_scaled = phys_scaler.transform(X_phys_baseline).astype(np.float32)

    out_csv_df = pd.DataFrame(best_X_scaled, columns=best_feature_names)
    out_csv_df.insert(0, "genotype", df_valid["genotype"].fillna("UNKNOWN").values)
    out_csv_df.insert(0, "yieldPerAcre", y)
    out_csv_df.insert(0, "split", splits)
    out_csv_df.insert(0, "location", locations)
    out_csv_df.insert(0, "plot_id", plot_ids)

    step5_csv_path = OUTPUT_DIR / "m1_step5_selected_features.csv"
    step5_npz_path = OUTPUT_DIR / "m1_step5_selected_features.npz"
    out_csv_df.to_csv(step5_csv_path, index=False)
    np.savez_compressed(
        step5_npz_path,
        X=best_X_scaled,
        X_raw=best_X_raw,
        X_phys=X_phys_scaled,
        X_phys_raw=X_phys_baseline,
        y=y,
        splits=splits,
        locations=locations,
        plot_ids=plot_ids,
        genotypes=df_valid["genotype"].fillna("UNKNOWN").astype(str).values,
        feature_names=np.array(best_feature_names, dtype=str),
    )

    print(f"\n[{MODEL_NAME} WINNER LOCKED FOR STEP 6] -> {best_label} ({len(best_feature_names)} features)")
    print(
        f"  -> 5-Location Test Split (323 plots) : RMSE = {best_row['Test_RMSE']:.4f}, "
        f"MAE = {best_row['Test_MAE']:.4f}, R2 = {best_row['Test_R2']:.4f}, Overall r = {best_row['Test_Overall_r']:.4f} (Within-City r = {best_row['Test_WithinLoc_r']:.4f})"
    )
    print(
        f"  -> 5-Location Val Split  (319 plots) : RMSE = {best_row['Val_RMSE']:.4f}, "
        f"MAE = {best_row['Val_MAE']:.4f}, R2 = {best_row['Val_R2']:.4f}"
    )
    print(f"  -> Saved Step 5 Master NPZ           : {step5_npz_path}")
    print(f"  -> Total Step 5 Runtime              : {time.time() - t0:.1f}s")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    y_pos = np.arange(len(res_df))
    ax1.barh(y_pos - 0.18, res_df["Val_RMSE"], height=0.34, label="5-Loc Validation RMSE (319 plots)", color="#1565c0")
    ax1.barh(y_pos + 0.18, res_df["Test_RMSE"], height=0.34, label="5-Loc Test RMSE (323 plots)", color="#2e7d32")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(res_df["Feature_Set"], fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("RMSE (Bushels / Acre) — Lower is Better")
    ax1.set_title(f"[{MODEL_NAME}] Step 5 Feature Selection Progression")
    ax1.legend(fontsize=8.5)
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    y_test = y[splits == "test"]
    ax2.scatter(
        y_test,
        best_preds,
        alpha=0.55,
        color="#2e7d32",
        edgecolors="k",
        label=f"5-Loc Test (RMSE={best_row['Test_RMSE']:.2f}, R²={best_row['Test_R2']:.4f}, r={best_row['Test_Overall_r']:.4f})",
        s=30,
    )
    min_v, max_v = float(min(y_test.min(), best_preds.min())), float(max(y_test.max(), best_preds.max()))
    ax2.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="1:1 Ideal Line")
    ax2.set_xlabel("Actual Yield (Bushels / Acre) — All 5 Locations Test")
    ax2.set_ylabel("Predicted Yield (Bushels / Acre)")
    ax2.set_title(f"Best Step 5 Set: {best_label}")
    ax2.legend(fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_out = OUTPUT_DIR / "m1_step5_selection_and_test_chart.png"
    plt.savefig(chart_out, dpi=200)
    plt.close()
    print(f"  -> Saved Step 5 Chart                : {chart_out}")


if __name__ == "__main__":
    main()