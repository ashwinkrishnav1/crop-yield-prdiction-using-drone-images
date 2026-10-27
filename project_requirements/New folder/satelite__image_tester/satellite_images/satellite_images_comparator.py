import os
import re
import sys
import math
import time
import subprocess
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import torch
import torch.nn as nn
from torchvision import models

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Ensure imagecodecs and tifffile are available to read LZW-compressed 6-band .TIF satellite images
try:
    import imagecodecs
    import tifffile
except ImportError:
    print("[SETUP] Installing 'tifffile' and 'imagecodecs' to decode LZW-compressed .TIF images...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tifffile", "imagecodecs", "-q"])
    import imagecodecs
    import tifffile
    import importlib
    importlib.reload(tifffile)

# ==============================================================================
# 1. PROJECT PATHS & CONFIGURATION
# ==============================================================================
PROJECT_ROOT = Path(r"C:\Users\ashwi\Music\cropyeildererrrrrrr")
SAT_ROOT_DIR = (
    PROJECT_ROOT / "dataset" / "DataPublication_final" / "DataPublication_final" / "Satellite"
)
VALID_PLOTS_CSV = PROJECT_ROOT / "drone_dataset" / "valid_plots.csv"

SPLIT_DIR = PROJECT_ROOT / "datasplitting_script"
TRAIN_SPLIT_CSV = SPLIT_DIR / "train_plot_ids.csv"
VAL_SPLIT_CSV = SPLIT_DIR / "validation_plot_ids.csv"
TEST_SPLIT_CSV = SPLIT_DIR / "test_plot_ids.csv"

STEP4_CKPT = (
    PROJECT_ROOT / "step4_feature_engineering" / "cache" / "step4_finetuned_effnet_b0.pth"
)
STEP4_RAW_CACHE = (
    PROJECT_ROOT / "step4_feature_engineering" / "cache" / "step4_raw_multimodal_features.npz"
)
STEP5_NPZ_PATH = (
    PROJECT_ROOT / "step5_feature_selection" / "step5_selected_features.npz"
)

OUTPUT_DIR = PROJECT_ROOT / "satellite_evaluation"
CACHE_DIR = OUTPUT_DIR / "cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TIMEPOINTS = ["TP1", "TP2", "TP3", "TP4", "TP5", "TP6"]
CNN_TIMEPOINTS = ["TP2", "TP3", "TP5"]  # Early-mid, Mid-season, and Late grain-fill
PATCH_SIZE = 224
BATCH_SIZE = 64
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


# ==============================================================================
# 2. FAST SATELLITE .TIF DIRECTORY INDEXER & DATASET LOADER
# ==============================================================================
def load_split_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    col = "plot_id" if "plot_id" in df.columns else df.columns[0]
    return set(df[col].astype(str).str.strip().tolist())


def build_satellite_file_index(sat_root: Path) -> Dict[Tuple[str, str], Dict[str, Path]]:
    """
    Indexes all .TIF files across Satellite/{Location}/{TP1..TP6}/ in <0.2 seconds
    for instant, case-insensitive lookup by experiment_range_row or filename stem.
    """
    index: Dict[Tuple[str, str], Dict[str, Path]] = {}
    if not sat_root.exists():
        raise FileNotFoundError(f"Satellite directory not found at: {sat_root}")

    for loc_dir in sat_root.iterdir():
        if not loc_dir.is_dir() or loc_dir.name.lower() == "fieldlevelimages":
            continue
        loc_key = loc_dir.name.lower()
        for tp in TIMEPOINTS:
            tp_dir = loc_dir / tp
            lookup = {}
            if tp_dir.exists():
                for f in tp_dir.iterdir():
                    if f.suffix.lower() in (".tif", ".tiff"):
                        stem_low = f.stem.lower()
                        lookup[stem_low] = f
                        # Also index by {experiment}_{range}_{row} suffix after "{loc}-{tp}-"
                        prefix = f"{loc_key}-{tp.lower()}-"
                        if stem_low.startswith(prefix):
                            suffix_key = stem_low[len(prefix) :]
                            lookup[suffix_key] = f
            index[(loc_key, tp)] = lookup
    return index


def resolve_satellite_tif_path(
    row: pd.Series, tp: str, sat_index: Dict[Tuple[str, str], Dict[str, Path]]
) -> Optional[Path]:
    loc = str(row["location"]).strip()
    loc_key = loc.lower()
    lookup = sat_index.get((loc_key, tp), {})
    if not lookup:
        return None

    exp = str(row["experiment"]).strip().lower()
    rng = str(row["range"]).strip().lower()
    r_num = str(row["row"]).strip().lower()

    # Candidate 1: {location}-{tp}-{experiment}_{range}_{row}
    c1 = f"{loc_key}-{tp.lower()}-{exp}_{rng}_{r_num}"
    if c1 in lookup:
        return lookup[c1]

    # Candidate 2: {experiment}_{range}_{row}
    c2 = f"{exp}_{rng}_{r_num}"
    if c2 in lookup:
        return lookup[c2]

    # Candidate 3: Derive pattern from valid_plots.csv TP3_image_path / TP1_image_path
    for ref_col in ("TP3_image_path", "TP2_image_path", "TP1_image_path"):
        raw_p = str(row.get(ref_col, "")).strip()
        if raw_p and raw_p.lower() != "nan":
            stem = Path(raw_p).stem.lower()
            stem_tp = re.sub(r"-tp[1-6]-", f"-{tp.lower()}-", stem)
            if stem_tp in lookup:
                return lookup[stem_tp]

    # Candidate 4: Match trailing _{range}_{row}
    suffix_target = f"_{rng}_{r_num}"
    for k, path_val in lookup.items():
        if k.endswith(suffix_target):
            return path_val

    return None


def load_satellite_master_dataset() -> pd.DataFrame:
    df = pd.read_csv(VALID_PLOTS_CSV)
    df["location_clean"] = df["location"].astype(str).str.strip()
    df = df.dropna(subset=["yieldPerAcre"]).reset_index(drop=True)

    # If Step 5 NPZ exists, align plot order strictly to Step 5 plot_ids
    if STEP5_NPZ_PATH.exists():
        s5 = np.load(STEP5_NPZ_PATH, allow_pickle=True)
        ordered_ids = [str(pid).strip() for pid in s5["plot_ids"]]
        df["pid_clean"] = df["plot_id"].astype(str).str.strip()
        df = (
            df.drop_duplicates(subset=["pid_clean"])
            .set_index("pid_clean")
            .reindex(ordered_ids)
            .reset_index()
        )

    train_ids = load_split_ids(TRAIN_SPLIT_CSV)
    val_ids = load_split_ids(VAL_SPLIT_CSV)
    test_ids = load_split_ids(TEST_SPLIT_CSV)

    sat_index = build_satellite_file_index(SAT_ROOT_DIR)

    for tp in TIMEPOINTS:
        df[f"sat_{tp}_path"] = [
            resolve_satellite_tif_path(row, tp, sat_index) for _, row in df.iterrows()
        ]

    splits = []
    for _, row in df.iterrows():
        pid = str(row["plot_id"]).strip()
        loc_lower = str(row["location_clean"]).lower()
        if loc_lower == "crawfordsville" or pid in test_ids:
            splits.append("test")
        elif pid in val_ids:
            splits.append("val")
        else:
            splits.append("train")
    df["split"] = splits

    print("=" * 98)
    print(f"SATELLITE MASTER DATASET LOADED: {len(df)} total valid plots")
    print(
        f"Split Breakdown -> Train: {(df['split']=='train').sum()} | "
        f"Val: {(df['split']=='val').sum()} | Unseen Crawfordsville Test: {(df['split']=='test').sum()}"
    )
    tp_counts = " | ".join(
        [f"{tp}: {df[f'sat_{tp}_path'].notnull().sum()}/{len(df)}" for tp in TIMEPOINTS]
    )
    print(f"Satellite .TIF Coverage -> {tp_counts}")
    print("=" * 98)
    return df


# ==============================================================================
# 3. 6-BAND GEOTIFF READER, MULTISPECTRAL INDICES & REPLICATION PADDING
# ==============================================================================
def read_6band_tif(tif_path: Path) -> np.ndarray:
    """
    Reads a 6-band Satellite .TIF image and returns a float32 array of shape (H, W, 6).
    Bands: [0: NIR, 1: RedEdge, 2: Red, 3: Green, 4: Blue, 5: DeepBlue]
    """
    arr = tifffile.imread(str(tif_path))
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 6, axis=2)
    elif arr.ndim == 3:
        # Convert (6, H, W) -> (H, W, 6) if channel dimension is first
        if arr.shape[0] == 6 and arr.shape[2] != 6:
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.shape[2] > 6:
            arr = arr[:, :, :6]
        elif arr.shape[2] < 6:
            pad_c = 6 - arr.shape[2]
            arr = np.concatenate([arr, np.repeat(arr[:, :, -1:], pad_c, axis=2)], axis=2)

    # Replace NaNs or negative nodata values with 0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    return arr


def extract_6band_spectral_features(img_6b: np.ndarray) -> np.ndarray:
    """
    Extracts 24 multispectral vegetation & band features from a 6-band satellite plot chip:
      Bands: 0=NIR, 1=RedEdge, 2=Red, 3=Green, 4=Blue, 5=DeepBlue
      Indices: NDVI, NDRE, GNDVI, CIre (Red-Edge Chlorophyll), NDWI, NGRDI, ExG
      Captures both whole-plot mean/std and 75th-percentile pure-canopy vigor.
    """
    eps = 1e-5
    # Filter out zero/nodata border pixels if present
    valid_px = np.sum(img_6b, axis=2) > eps
    if np.sum(valid_px) >= 4:
        px = img_6b[valid_px]
    else:
        px = img_6b.reshape(-1, 6)

    # Normalize reflectance scale by total pixel radiance to make ratios exposure-invariant
    tot_rad = np.sum(px, axis=1, keepdims=True) + eps
    px_norm = px / tot_rad

    NIR = px[:, 0]
    RE = px[:, 1]
    R = px[:, 2]
    G = px[:, 3]
    B = px[:, 4]
    DB = px[:, 5]

    # 1. Core Multispectral Vegetation Indices
    ndvi = (NIR - R) / (NIR + R + eps)
    ndre = (NIR - RE) / (NIR + RE + eps)
    gndvi = (NIR - G) / (NIR + G + eps)
    cire = np.clip((NIR / (RE + eps)) - 1.0, -2.0, 10.0)
    ngrdi = (G - R) / (G + R + eps)
    ndwi = (G - NIR) / (G + NIR + eps)
    exg = 2.0 * px_norm[:, 3] - px_norm[:, 2] - px_norm[:, 4]
    re_red_ratio = (RE - R) / (RE + R + eps)

    # Active green canopy fraction (pixels with high NDVI)
    green_frac = float(np.mean(ndvi > np.median(ndvi)))

    feats = [
        float(np.mean(ndvi)),
        float(np.std(ndvi)),
        float(np.percentile(ndvi, 75)),
        float(np.percentile(ndvi, 25)),
        float(np.mean(ndre)),
        float(np.std(ndre)),
        float(np.percentile(ndre, 75)),
        float(np.percentile(ndre, 25)),
        float(np.mean(gndvi)),
        float(np.percentile(gndvi, 75)),
        float(np.mean(cire)),
        float(np.percentile(cire, 75)),
        float(np.mean(ngrdi)),
        float(np.mean(ndwi)),
        float(np.mean(exg)),
        float(np.mean(re_red_ratio)),
        green_frac,
        # 6 Normalized Spectral Band Means + NIR/Red-Edge dominance
        float(np.mean(px_norm[:, 0])),  # Norm NIR
        float(np.mean(px_norm[:, 1])),  # Norm RedEdge
        float(np.mean(px_norm[:, 2])),  # Norm Red
        float(np.mean(px_norm[:, 3])),  # Norm Green
        float(np.mean(px_norm[:, 4])),  # Norm Blue
        float(np.mean(px_norm[:, 5])),  # Norm DeepBlue
        float(np.percentile(px_norm[:, 0], 75)),  # Peak NIR
    ]
    return np.array(feats, dtype=np.float32)


SPECTRAL_24_NAMES = [
    "NDVI_Mean",
    "NDVI_Std",
    "NDVI_Q75",
    "NDVI_Q25",
    "NDRE_Mean",
    "NDRE_Std",
    "NDRE_Q75",
    "NDRE_Q25",
    "GNDVI_Mean",
    "GNDVI_Q75",
    "CIre_Mean",
    "CIre_Q75",
    "NGRDI_Mean",
    "NDWI_Mean",
    "ExG_Mean",
    "RERed_Mean",
    "Active_Canopy_Frac",
    "Band1_NIR_Norm",
    "Band2_RedEdge_Norm",
    "Band3_Red_Norm",
    "Band4_Green_Norm",
    "Band5_Blue_Norm",
    "Band6_DeepBlue_Norm",
    "Band1_NIR_Q75",
]


def prepare_satellite_patch_for_effnet(img_6b: np.ndarray) -> np.ndarray:
    """
    Converts a 6-band satellite plot chip into a 3-channel (NIR, RedEdge, Red)
    agronomic composite, scales to [0..255] uint8, and applies Replication Padding
    (with aspect-preserving upsampling if chip is smaller than 224x224) to 224x224.
    """
    # Select Bands [0: NIR, 1: RedEdge, 2: Red] — captures strongest chlorophyll contrast
    comp = img_6b[:, :, :3].copy()
    out_u8 = np.zeros_like(comp, dtype=np.uint8)

    for c in range(3):
        ch = comp[:, :, c]
        lo = float(np.percentile(ch, 2))
        hi = float(np.percentile(ch, 98))
        if hi - lo > 1e-5:
            scaled = np.clip((ch - lo) / (hi - lo), 0.0, 1.0) * 255.0
        else:
            scaled = np.zeros_like(ch)
        out_u8[:, :, c] = scaled.astype(np.uint8)

    h, w, _ = out_u8.shape
    # Satellite plot chips are small (e.g., 10x25 px); resize longer edge to 160 px first,
    # then apply Replication Padding to reach 224x224 so boundary texture is preserved.
    if max(h, w) < 160:
        scale = 160.0 / float(max(h, w, 1))
        new_h = min(PATCH_SIZE, max(1, int(round(h * scale))))
        new_w = min(PATCH_SIZE, max(1, int(round(w * scale))))
        out_u8 = cv2.resize(out_u8, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w, _ = out_u8.shape

    n_h = max(1, math.ceil(h / PATCH_SIZE))
    n_w = max(1, math.ceil(w / PATCH_SIZE))
    pad_b = n_h * PATCH_SIZE - h
    pad_r = n_w * PATCH_SIZE - w

    padded = cv2.copyMakeBorder(
        out_u8, 0, pad_b, 0, pad_r, borderType=cv2.BORDER_REPLICATE
    )
    return padded[:PATCH_SIZE, :PATCH_SIZE]


# ==============================================================================
# 4. EFFICIENTNET-B0 BACKBONE & FULL SATELLITE FEATURE EXTRACTION (WITH CACHE)
# ==============================================================================
def load_effnet_backbone() -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    backbone = models.efficientnet_b0(weights=weights)
    backbone.classifier = nn.Identity()

    if STEP4_CKPT.exists():
        try:
            backbone.load_state_dict(torch.load(STEP4_CKPT, map_location=DEVICE))
            print(f"[BACKBONE] Loaded fine-tuned EfficientNet-B0 weights -> {STEP4_CKPT.name}")
        except Exception:
            print("[BACKBONE] Using ImageNet pretrained EfficientNet-B0 weights.")
    else:
        print("[BACKBONE] Using ImageNet pretrained EfficientNet-B0 weights.")

    backbone.to(DEVICE)
    backbone.eval()
    return backbone


@torch.no_grad()
def extract_all_satellite_features(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    cache_file = CACHE_DIR / "satellite_tp1_tp6_features.npz"
    n_plots = len(df)

    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        if data["spec_TP3"].shape[0] == n_plots:
            print(f"[CACHE] Loaded extracted Satellite TP1–TP6 features -> {cache_file.name}")
            return {k: data[k] for k in data.files}

    backbone = load_effnet_backbone()
    print("\n" + "-" * 98)
    print(
        f"EXTRACTING SATELLITE 6-BAND SPECTRAL (TP1–TP6) + EFFICIENTNET-B0 FEATURES FOR {n_plots} PLOTS"
    )
    print("-" * 98)

    t0 = time.time()
    spec_store = {tp: np.zeros((n_plots, 24), dtype=np.float32) for tp in TIMEPOINTS}
    cnn_patches = {tp: [] for tp in CNN_TIMEPOINTS}
    plot_areas = np.zeros(n_plots, dtype=np.float32)

    for idx, row in df.iterrows():
        last_valid_6b = np.zeros((10, 10, 6), dtype=np.float32)
        for tp in TIMEPOINTS:
            p = row[f"sat_{tp}_path"]
            if p is not None and pd.notnull(p) and Path(p).exists():
                img_6b = read_6band_tif(Path(p))
                last_valid_6b = img_6b
            else:
                img_6b = last_valid_6b

            spec_store[tp][idx] = extract_6band_spectral_features(img_6b)
            if tp == "TP3":
                plot_areas[idx] = float(img_6b.shape[0] * img_6b.shape[1])
            if tp in CNN_TIMEPOINTS:
                cnn_patches[tp].append(prepare_satellite_patch_for_effnet(img_6b))

        if (idx + 1) % 250 == 0 or (idx + 1) == n_plots:
            print(
                f"  -> Read & extracted 6-band indices for {idx + 1}/{n_plots} plots ({time.time() - t0:.1f}s)...",
                end="\r",
                flush=True,
            )
    print()

    # Batch run EfficientNet-B0 on Satellite patches (TP2, TP3, TP5)
    cnn_store = {}
    for tp in CNN_TIMEPOINTS:
        t_c = time.time()
        patches_list = cnn_patches[tp]
        embs = []
        for i in range(0, n_plots, BATCH_SIZE):
            batch_np = np.stack(patches_list[i : i + BATCH_SIZE], axis=0)
            batch_t = (
                torch.from_numpy(batch_np).permute(0, 3, 1, 2).float().to(DEVICE) / 255.0
            )
            batch_t = (batch_t - IMAGENET_MEAN) / IMAGENET_STD
            feats = backbone(batch_t)
            embs.append(feats.cpu().numpy())
        cnn_store[f"cnn_{tp}"] = np.vstack(embs).astype(np.float32)
        print(
            f"  -> Extracted EfficientNet-B0 embeddings for Satellite {tp} in {time.time() - t_c:.1f}s"
        )

    out_dict = {f"spec_{tp}": spec_store[tp] for tp in TIMEPOINTS}
    out_dict.update(cnn_store)
    out_dict["plot_areas"] = plot_areas

    np.savez_compressed(cache_file, **out_dict)
    print(f"[SAVED] Satellite feature cache -> {cache_file}")
    return out_dict


# ==============================================================================
# 5. FLIGHT-LEVEL NORMALIZATION, GENETICS & SATELLITE FEATURE ENGINEERING
# ==============================================================================
def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-6, None)


def parse_hybrid_parents(geno_str: str) -> Tuple[str, str]:
    g = str(geno_str).strip().upper()
    if not g or g == "NAN":
        return "UNKNOWN", "UNKNOWN"
    parts = re.split(r"\s+[X\*\/]\s+|\s*[\*\/]\s*", g)
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()
    return g, g


def oof_empirical_bayes_encode(
    cat_series: np.ndarray,
    target_vals: np.ndarray,
    train_mask: np.ndarray,
    smoothing_m: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    n_total = len(cat_series)
    encoded = np.zeros(n_total, dtype=np.float32)
    train_idx = np.where(train_mask)[0]
    cats_tr = cat_series[train_idx]
    y_tr = target_vals[train_idx]
    prior = float(np.mean(y_tr))

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_tr = np.full(len(train_idx), prior, dtype=np.float32)
    for f_tr, f_va in kf.split(train_idx):
        df_f = pd.DataFrame({"cat": cats_tr[f_tr], "y": y_tr[f_tr]})
        stats = df_f.groupby("cat")["y"].agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + smoothing_m * prior) / (
            stats["count"] + smoothing_m
        )
        mapping = smooth.to_dict()
        oof_tr[f_va] = [mapping.get(c, prior) for c in cats_tr[f_va]]
    encoded[train_idx] = oof_tr

    df_full = pd.DataFrame({"cat": cats_tr, "y": y_tr})
    full_stats = df_full.groupby("cat")["y"].agg(["count", "mean"])
    full_smooth = (full_stats["count"] * full_stats["mean"] + smoothing_m * prior) / (
        full_stats["count"] + smoothing_m
    )
    full_map = full_smooth.to_dict()
    known_cats = set(full_stats.index.tolist())

    non_train_idx = np.where(~train_mask)[0]
    encoded[non_train_idx] = [full_map.get(c, prior) for c in cat_series[non_train_idx]]
    seen_flag = np.array(
        [1.0 if (c in known_cats and c != "UNKNOWN") else 0.0 for c in cat_series],
        dtype=np.float32,
    )
    return encoded, seen_flag


def build_satellite_engineered_matrices(
    df: pd.DataFrame, sat_dict: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Constructs:
      1. X_sat_ranker: Flight-normalized 6-band indices (TP1..TP6) + Season Growth Curves
         + Out-of-Fold Satellite CNN Prototypes + Clean Stand & Genetic Hybrid features.
      2. X_sat_phys: Uncentered physical satellite spectral + CNN + stand matrix for
         estimating the Zero-Shot Crawfordsville field baseline anchor.
    """
    splits = df["split"].values
    locations = df["location_clean"].values
    y = df["yieldPerAcre"].values.astype(np.float32)
    train_mask = splits == "train"
    train_idx = np.where(train_mask)[0]
    non_train_idx = np.where(~train_mask)[0]
    train_locs = locations[train_mask]
    y_train = y[train_mask]

    # 1. Flight-Level Normalization of Satellite Spectral & CNN matrices
    norm_sat = {}
    for k, mat in sat_dict.items():
        if k == "plot_areas":
            continue
        mat_f = mat.astype(np.float32)
        mat_n = np.zeros_like(mat_f, dtype=np.float32)
        for loc in np.unique(locations):
            loc_m = locations == loc
            ref_m = loc_m & train_mask
            if np.sum(ref_m) < 5:
                ref_m = loc_m
            mu = np.mean(mat_f[ref_m], axis=0, keepdims=True)
            sd = np.std(mat_f[ref_m], axis=0, keepdims=True)
            sd = np.where(sd < 1e-5, 1.0, sd)
            mat_n[loc_m] = (mat_f[loc_m] - mu) / sd
        norm_sat[k] = mat_n

    # 2. Genetic Hybrid & Parental OOF Features (100% leak-free on train_mask)
    geno_raw = df["genotype"].fillna("UNKNOWN").astype(str).str.strip().str.upper().values
    exp_raw = df["experiment"].fillna("UNKNOWN").astype(str).str.strip().str.upper().values
    parents = [parse_hybrid_parents(g) for g in geno_raw]
    p1_raw = np.array([p[0] for p in parents], dtype=object)
    p2_raw = np.array([p[1] for p in parents], dtype=object)

    y_within_loc = np.zeros_like(y, dtype=np.float32)
    y_z_loc = np.zeros_like(y, dtype=np.float32)
    for loc in np.unique(train_locs):
        m = (locations == loc) & train_mask
        mu, sd = float(np.mean(y[m])), max(float(np.std(y[m])), 1.0)
        loc_all = locations == loc
        y_within_loc[loc_all] = y[loc_all] - mu
        y_z_loc[loc_all] = (y[loc_all] - mu) / sd

    geno_adv, geno_seen = oof_empirical_bayes_encode(geno_raw, y_within_loc, train_mask, 3.0)
    geno_z, _ = oof_empirical_bayes_encode(geno_raw, y_z_loc, train_mask, 3.0)
    p1_adv, _ = oof_empirical_bayes_encode(p1_raw, y_within_loc, train_mask, 5.0)
    p2_adv, _ = oof_empirical_bayes_encode(p2_raw, y_within_loc, train_mask, 5.0)
    exp_adv, _ = oof_empirical_bayes_encode(exp_raw, y_within_loc, train_mask, 10.0)
    geno_glob, _ = oof_empirical_bayes_encode(geno_raw, y, train_mask, 3.0)

    parent_gca = 0.5 * (p1_adv + p2_adv)
    comp_gen_adv = np.where(geno_seen > 0.5, geno_adv, parent_gca).astype(np.float32)

    gen_X = np.column_stack(
        [geno_adv, geno_z, p1_adv, p2_adv, parent_gca, comp_gen_adv, exp_adv, geno_glob]
    ).astype(np.float32)
    gen_names = [
        "Sat_Gen_Hybrid_Adv",
        "Sat_Gen_Hybrid_Z",
        "Sat_Gen_P1_Adv",
        "Sat_Gen_P2_Adv",
        "Sat_Gen_Parent_GCA",
        "Sat_Gen_Composite_Adv",
        "Sat_Gen_Exp_Adv",
        "Sat_Gen_Global_Yield",
    ]

    # 3. Clean Stand Metadata + Multi-Temporal Satellite Growth Curve Features (TP1..TP6)
    raw_stand = pd.to_numeric(df["totalStandCount"], errors="coerce")
    stand_med = float(raw_stand[train_mask].median())
    stand_count = raw_stand.fillna(stand_med).values.astype(np.float32)
    log_stand = np.log1p(np.clip(stand_count, 0.0, None)).astype(np.float32)

    stand_flight_z = np.zeros_like(stand_count, dtype=np.float32)
    for loc in np.unique(locations):
        loc_m = locations == loc
        ref_m = loc_m & train_mask
        if np.sum(ref_m) < 5:
            ref_m = loc_m
        s_mu, s_sd = float(np.mean(stand_count[ref_m])), max(float(np.std(stand_count[ref_m])), 1.0)
        stand_flight_z[loc_m] = (stand_count[loc_m] - s_mu) / s_sd

    # Extract raw & normalized NDVI (col 0) and NDRE (col 4) trajectories across TP1..TP6
    ndvi_raw_curve = np.column_stack([sat_dict[f"spec_{tp}"][:, 0] for tp in TIMEPOINTS])
    ndre_raw_curve = np.column_stack([sat_dict[f"spec_{tp}"][:, 4] for tp in TIMEPOINTS])
    cire_raw_curve = np.column_stack([sat_dict[f"spec_{tp}"][:, 10] for tp in TIMEPOINTS])

    ndvi_norm_curve = np.column_stack([norm_sat[f"spec_{tp}"][:, 0] for tp in TIMEPOINTS])
    ndre_norm_curve = np.column_stack([norm_sat[f"spec_{tp}"][:, 4] for tp in TIMEPOINTS])

    # Season-Integrated Area Under Curve (AUC), Peak Vigor, Vegetative Ramp & Senescence Drop
    sat_curve_feats = np.column_stack(
        [
            stand_count,
            log_stand,
            stand_flight_z,
            np.mean(ndvi_norm_curve, axis=1),  # Season Mean NDVI Z
            np.max(ndvi_norm_curve, axis=1),   # Season Peak NDVI Z
            np.mean(ndre_norm_curve, axis=1),  # Season Mean NDRE Z
            np.max(ndre_norm_curve, axis=1),   # Season Peak NDRE Z
            ndvi_norm_curve[:, 3] - ndvi_norm_curve[:, 0],  # Early-to-Mid Ramp (TP4 - TP1)
            ndvi_norm_curve[:, 2] - ndvi_norm_curve[:, 0],  # Vegetative Ramp (TP3 - TP1)
            ndvi_norm_curve[:, 5] - ndvi_norm_curve[:, 3],  # Late Stay-Green (TP6 - TP4)
            ndre_norm_curve[:, 4] - ndre_norm_curve[:, 1],  # RedEdge Grain-Fill (TP5 - TP2)
            stand_flight_z + np.mean(ndre_norm_curve[:, 2:5], axis=1),  # Stand + Mid/Late NDRE
            comp_gen_adv + 4.0 * np.mean(ndre_norm_curve[:, 2:5], axis=1),  # Genetics + NDRE
        ]
    ).astype(np.float32)

    curve_names = [
        "Sat_StandCount",
        "Sat_LogStand",
        "Sat_FlightNorm_Stand_Z",
        "Sat_Season_Mean_NDVI_Z",
        "Sat_Season_Peak_NDVI_Z",
        "Sat_Season_Mean_NDRE_Z",
        "Sat_Season_Peak_NDRE_Z",
        "Sat_Ramp_TP4_TP1_NDVI",
        "Sat_Ramp_TP3_TP1_NDVI",
        "Sat_StayGreen_TP6_TP4_NDVI",
        "Sat_RedEdge_TP5_TP2_NDRE",
        "Sat_Stand_Plus_NDRE",
        "Sat_Genetics_Plus_NDRE",
    ]

    # 4. Out-of-Fold Satellite CNN + Multi-Temporal Spectral Prototypes
    y_train_loc_z = y_z_loc[train_mask]
    high_mask = y_train_loc_z >= np.percentile(y_train_loc_z, 75)
    low_mask = y_train_loc_z <= np.percentile(y_train_loc_z, 25)

    all_spec_norm = np.hstack([norm_sat[f"spec_{tp}"] for tp in TIMEPOINTS])
    all_spec_raw = np.hstack([sat_dict[f"spec_{tp}"] for tp in TIMEPOINTS])

    proto_blocks_norm = [
        ("SatSpec_AllTP", all_spec_norm),
        ("SatCNN_TP2", norm_sat["cnn_TP2"]),
        ("SatCNN_TP3", norm_sat["cnn_TP3"]),
        ("SatCNN_TP5", norm_sat["cnn_TP5"]),
        ("SatCNN_Delta_52", norm_sat["cnn_TP5"] - norm_sat["cnn_TP2"]),
        ("SatCNN_Delta_32", norm_sat["cnn_TP3"] - norm_sat["cnn_TP2"]),
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    proto_list = []
    proto_names = []

    for blk_name, mat in proto_blocks_norm:
        mat_u = l2_normalize_rows(mat)
        mat_tr = mat_u[train_mask]

        hi_p = l2_normalize_rows(np.mean(mat_tr[high_mask], axis=0, keepdims=True))[0]
        lo_p = l2_normalize_rows(np.mean(mat_tr[low_mask], axis=0, keepdims=True))[0]
        sim_hi = mat_u @ hi_p
        sim_lo = mat_u @ lo_p
        sim_diff = sim_hi - sim_lo

        oof_vigor = np.zeros(len(mat_u), dtype=np.float32)
        oof_tr = np.zeros(len(train_idx), dtype=np.float32)
        for f_tr, f_va in kf.split(train_idx):
            ridge = RidgeCV(alphas=[20.0, 80.0, 200.0, 500.0, 1000.0])
            ridge.fit(mat_tr[f_tr], y_train_loc_z[f_tr])
            oof_tr[f_va] = ridge.predict(mat_tr[f_va])
        oof_vigor[train_idx] = oof_tr

        ridge_full = RidgeCV(alphas=[20.0, 80.0, 200.0, 500.0, 1000.0])
        ridge_full.fit(mat_tr, y_train_loc_z)
        oof_vigor[non_train_idx] = ridge_full.predict(mat_u[non_train_idx]).astype(np.float32)

        proto_list.append(np.column_stack([sim_hi, sim_lo, sim_diff, oof_vigor]))
        proto_names.extend(
            [
                f"{blk_name}_SimHigh",
                f"{blk_name}_SimLow",
                f"{blk_name}_NetDiff",
                f"{blk_name}_OOF_RidgeVigor",
            ]
        )

    proto_X = np.hstack(proto_list).astype(np.float32)

    # Select Top 25 most domain-invariant individual Flight-Normalized Spectral features
    spec_candidates = np.hstack(
        [
            all_spec_norm,
            norm_sat["spec_TP3"] - norm_sat["spec_TP1"],
            norm_sat["spec_TP5"] - norm_sat["spec_TP3"],
            norm_sat["spec_TP6"] - norm_sat["spec_TP2"],
        ]
    ).astype(np.float32)

    corrs = []
    for j in range(spec_candidates.shape[1]):
        col = spec_candidates[train_mask, j]
        r_locs = [
            np.corrcoef(col[train_locs == l], y_train[train_locs == l])[0, 1]
            for l in np.unique(train_locs)
        ]
        corrs.append(np.mean(np.nan_to_num(r_locs, nan=0.0)))
    top_spec_idx = np.argsort(np.abs(corrs))[::-1][:25]
    top_spec_X = spec_candidates[:, top_spec_idx]
    top_spec_names = [f"SatSpec_Top_{i}" for i in range(len(top_spec_idx))]

    X_sat_ranker_raw = np.hstack([gen_X, sat_curve_feats, proto_X, top_spec_X]).astype(np.float32)
    sat_ranker_names = gen_names + curve_names + proto_names + top_spec_names

    scaler_r = StandardScaler()
    scaler_r.fit(X_sat_ranker_raw[train_mask])
    X_sat_ranker = scaler_r.transform(X_sat_ranker_raw).astype(np.float32)

    # 5. Uncentered Physical Satellite Matrix (for Zero-Shot Field Baseline Anchor)
    raw_deltas = np.hstack(
        [
            sat_dict["spec_TP3"] - sat_dict["spec_TP1"],
            sat_dict["spec_TP4"] - sat_dict["spec_TP1"],
            sat_dict["spec_TP5"] - sat_dict["spec_TP2"],
            sat_dict["spec_TP6"] - sat_dict["spec_TP3"],
            np.mean(ndvi_raw_curve, axis=1, keepdims=True),
            np.max(ndvi_raw_curve, axis=1, keepdims=True),
            np.mean(ndre_raw_curve, axis=1, keepdims=True),
            np.max(ndre_raw_curve, axis=1, keepdims=True),
            np.mean(cire_raw_curve, axis=1, keepdims=True),
        ]
    ).astype(np.float32)

    # Raw CNN uncentered prototypes
    raw_cnn_protos = []
    for c_key in ("cnn_TP2", "cnn_TP3", "cnn_TP5"):
        mat = sat_dict[c_key]
        mu_tr = np.mean(mat[train_mask], axis=0, keepdims=True)
        mat_u = l2_normalize_rows(mat - mu_tr)
        ridge = RidgeCV(alphas=[50.0, 150.0, 500.0])
        ridge.fit(mat_u[train_mask], y_train_loc_z)
        raw_cnn_protos.append(ridge.predict(mat_u)[:, None])

    X_sat_phys_raw = np.hstack(
        [
            np.column_stack([stand_count, log_stand]),
            all_spec_raw,
            raw_deltas,
            np.hstack(raw_cnn_protos),
        ]
    ).astype(np.float32)

    scaler_p = StandardScaler()
    scaler_p.fit(X_sat_phys_raw[train_mask])
    X_sat_phys = scaler_p.transform(X_sat_phys_raw).astype(np.float32)

    return X_sat_ranker, X_sat_phys, sat_ranker_names


# ==============================================================================
# 6. STEP 6 TUNED TWO-STAGE EVALUATION (SATELLITE-ONLY & SATELLITE+UAV FUSION)
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


def get_tuned_ranker_xgb() -> XGBRegressor:
    """Step 6 Winner Ranker Config (R4_Depth3_FineLR)."""
    return XGBRegressor(
        n_estimators=400,
        learning_rate=0.018,
        max_depth=3,
        min_child_weight=18,
        subsample=0.80,
        colsample_bytree=0.45,
        reg_alpha=3.0,
        reg_lambda=18.0,
        gamma=0.2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def get_tuned_phys_xgb() -> XGBRegressor:
    """Step 6 Winner Physical Baseline Config (P0_Step5_Run3_LocWeighted)."""
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=15,
        subsample=0.80,
        colsample_bytree=0.50,
        reg_alpha=2.0,
        reg_lambda=12.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def run_two_stage_evaluation(
    model_label: str,
    X_ranker: np.ndarray,
    X_phys: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    locations: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    train_mask = splits == "train"
    val_mask = splits == "val"
    dev_mask = (splits == "train") | (splits == "val")
    test_mask = splits == "test"

    y_tr, loc_tr = y[train_mask], locations[train_mask]
    y_va, loc_va = y[val_mask], locations[val_mask]
    y_dev, loc_dev = y[dev_mask], locations[dev_mask]
    y_test = y[test_mask]

    tr_loc_means = {loc: float(np.mean(y_tr[loc_tr == loc])) for loc in np.unique(loc_tr)}
    y_tr_cent = np.array([yv - tr_loc_means[l] for yv, l in zip(y_tr, loc_tr)], dtype=np.float32)

    dev_loc_means = {loc: float(np.mean(y_dev[loc_dev == loc])) for loc in np.unique(loc_dev)}
    y_dev_cent = np.array(
        [yv - dev_loc_means[l] for yv, l in zip(y_dev, loc_dev)], dtype=np.float32
    )

    val_offsets = np.array([tr_loc_means[l] for l in loc_va], dtype=np.float32)
    w_tr = compute_location_balanced_weights(loc_tr)
    w_dev = compute_location_balanced_weights(loc_dev)

    # 1. Estimate LOLO optimal variance calibration factor strictly on the 4 Dev Locations
    logo = LeaveOneGroupOut()
    oof_lolo = np.zeros(len(y_dev), dtype=np.float32)
    y_dev_true_cent = np.zeros(len(y_dev), dtype=np.float32)
    X_r_dev = X_ranker[dev_mask]

    for f_tr, f_va in logo.split(X_r_dev, y_dev, groups=loc_dev):
        f_locs = loc_dev[f_tr]
        f_means = {l: float(np.mean(y_dev[f_tr][f_locs == l])) for l in np.unique(f_locs)}
        y_f_c = np.array([yv - f_means[l] for yv, l in zip(y_dev[f_tr], f_locs)], dtype=np.float32)
        m_lolo = get_tuned_ranker_xgb()
        m_lolo.fit(X_r_dev[f_tr], y_f_c, sample_weight=compute_location_balanced_weights(f_locs))
        p_va = m_lolo.predict(X_r_dev[f_va])
        oof_lolo[f_va] = p_va - float(np.mean(p_va))
        y_dev_true_cent[f_va] = y_dev[f_va] - float(np.mean(y_dev[f_va]))

    var_p = float(np.var(oof_lolo))
    cov_pt = float(np.mean(oof_lolo * y_dev_true_cent))
    alpha_shrink = float(np.clip(cov_pt / max(var_p, 1e-6), 0.55, 1.00))

    # 2. Fit Physical Field Baseline Model (Train Split & Full 4-Loc Split)
    m_phys_tr = get_tuned_phys_xgb()
    m_phys_tr.fit(X_phys[train_mask], y_tr, sample_weight=w_tr)
    phys_test_tr = m_phys_tr.predict(X_phys[test_mask])
    anchor_tr = float(np.mean(phys_test_tr))

    m_phys_full = get_tuned_phys_xgb()
    m_phys_full.fit(X_phys[dev_mask], y_dev, sample_weight=w_dev)
    phys_test_full = m_phys_full.predict(X_phys[test_mask])
    anchor_full = float(np.mean(phys_test_full))

    # 3. Fit Within-Field Ranker (Train Split & Full 4-Loc Split)
    m_rank_tr = get_tuned_ranker_xgb()
    m_rank_tr.fit(X_ranker[train_mask], y_tr_cent, sample_weight=w_tr)
    val_preds = m_rank_tr.predict(X_ranker[val_mask]) + val_offsets

    rank_test_tr = m_rank_tr.predict(X_ranker[test_mask])
    rank_test_tr_zero = rank_test_tr - float(np.mean(rank_test_tr))
    phys_dev_tr_zero = phys_test_tr - anchor_tr
    zeroshot_test_tr = anchor_tr + alpha_shrink * (
        0.85 * rank_test_tr_zero + 0.15 * phys_dev_tr_zero
    )

    m_rank_full = get_tuned_ranker_xgb()
    m_rank_full.fit(X_ranker[dev_mask], y_dev_cent, sample_weight=w_dev)
    rank_test_full = m_rank_full.predict(X_ranker[test_mask])
    rank_test_full_zero = rank_test_full - float(np.mean(rank_test_full))
    phys_dev_full_zero = phys_test_full - anchor_full
    zeroshot_test_full = anchor_full + alpha_shrink * (
        0.85 * rank_test_full_zero + 0.15 * phys_dev_full_zero
    )

    # 4. Field-Calibrated (15-plot reference strip)
    rng = np.random.RandomState(RANDOM_SEED)
    cal_idx = rng.choice(len(y_test), size=15, replace=False)
    eval_mask = np.ones(len(y_test), dtype=bool)
    eval_mask[cal_idx] = False
    offset_cal = float(np.mean(y_test[cal_idx] - zeroshot_test_full[cal_idx]))
    cal_preds = zeroshot_test_full + offset_cal

    metrics = {
        "Model_Stage": model_label,
        "Num_Features": int(X_ranker.shape[1]),
        "ZeroShot_Test_RMSE": round(
            float(np.sqrt(mean_squared_error(y_test, zeroshot_test_tr))), 4
        ),
        "ZeroShot_Test_MAE": round(float(mean_absolute_error(y_test, zeroshot_test_tr)), 4),
        "ZeroShot_Test_R2": round(float(r2_score(y_test, zeroshot_test_tr)), 4),
        "ZeroShot_Full4Loc_RMSE": round(
            float(np.sqrt(mean_squared_error(y_test, zeroshot_test_full))), 4
        ),
        "ZeroShot_Full4Loc_R2": round(float(r2_score(y_test, zeroshot_test_full)), 4),
        "Test_Pearson_r": round(safe_pearson_r(y_test, zeroshot_test_tr), 4),
        "Test_Full4Loc_r": round(safe_pearson_r(y_test, zeroshot_test_full), 4),
        "Calibrated_RMSE": round(
            float(np.sqrt(mean_squared_error(y_test[eval_mask], cal_preds[eval_mask]))), 4
        ),
        "Calibrated_R2": round(float(r2_score(y_test[eval_mask], cal_preds[eval_mask])), 4),
        "Val_RMSE": round(float(np.sqrt(mean_squared_error(y_va, val_preds))), 4),
        "Val_R2": round(float(r2_score(y_va, val_preds)), 4),
    }
    return metrics, zeroshot_test_tr


# ==============================================================================
# 7. PLOT COMPARISON CHART WITH HIGHLIGHTED SATELLITE RESULTS
# ==============================================================================
def plot_highlighted_comparison_chart(
    sat_only_metrics: Dict[str, float],
    out_img_path: Path,
):
    runs_data = [
        {
            "Run": "Step 3 UAV Winner\n(Replication Pad, Frozen TP3)",
            "ZeroShot_RMSE": 39.9498,
            "ZeroShot_R2": -1.6207,
            "Group": "UAV_Prev",
        },
        {
            "Run": "Step 4 UAV Full\n(Multi-TP PCA + Metadata)",
            "ZeroShot_RMSE": 31.2855,
            "ZeroShot_R2": -0.6072,
            "Group": "UAV_Prev",
        },
        {
            "Run": "Step 4 UAV Raw TP3\n(Fine-Tuned CNN Only)",
            "ZeroShot_RMSE": 29.6455,
            "ZeroShot_R2": -0.4431,
            "Group": "UAV_Prev",
        },
        {
            "Run": "Step 5 UAV (Run 3)\n(44 Core Untuned + Old Anchor)",
            "ZeroShot_RMSE": 28.9022,
            "ZeroShot_R2": -0.3717,
            "Group": "UAV_Prev",
        },
        {
            "Run": "Step 5 UAV (Run 1)\n(Top-120 Untuned Physical)",
            "ZeroShot_RMSE": 27.3287,
            "ZeroShot_R2": -0.2264,
            "Group": "UAV_Prev",
        },
        {
            "Run": "Step 6 UAV Winner\n(Tuned LOLO UAV Ensemble)",
            "ZeroShot_RMSE": 23.0871,
            "ZeroShot_R2": 0.1248,
            "Group": "UAV_Best",
        },
        {
            "Run": "★ SATELLITE-ONLY (TP1–TP6)\n(6-Band TIF + EffNetB0 + XGB)",
            "ZeroShot_RMSE": sat_only_metrics["ZeroShot_Test_RMSE"],
            "ZeroShot_R2": sat_only_metrics["ZeroShot_Test_R2"],
            "Group": "SAT_ONLY",
        },
    ]

    df_plot = pd.DataFrame(runs_data)

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 8.2), sharey=True)
    fig.patch.set_facecolor("white")

    y_pos = np.arange(len(df_plot))

    bar_colors = []
    edge_colors = []
    line_widths = []

    for _, r in df_plot.iterrows():
        grp = r["Group"]
        r2_v = r["ZeroShot_R2"]
        if grp == "SAT_ONLY":
            bar_colors.append("#7b1fa2")  # Vibrant Royal Purple for Satellite-Only
            edge_colors.append("#ffd600")
            line_widths.append(2.2)
        elif grp == "UAV_Best":
            bar_colors.append("#2e7d32")  # Forest Green for Step 6 UAV Winner
            edge_colors.append("black")
            line_widths.append(1.0)
        else:
            bar_colors.append("#c62828" if r2_v < -0.5 else "#ef6c00")
            edge_colors.append("black")
            line_widths.append(0.8)

    # Left Subplot: Zero-Shot Test RMSE
    bars1 = ax1.barh(y_pos, df_plot["ZeroShot_RMSE"], color=bar_colors, height=0.64, alpha=0.93)
    for b, ec, lw in zip(bars1, edge_colors, line_widths):
        b.set_edgecolor(ec)
        b.set_linewidth(lw)

    ax1.axvline(
        24.68,
        color="#1565c0",
        linestyle="--",
        linewidth=2.0,
        label="Positive R² Threshold (24.68 bu/ac)",
    )

    for bar, (_, r) in zip(bars1, df_plot.iterrows()):
        is_sat = r["Group"] == "SAT_ONLY"
        txt_color = "#4a148c" if is_sat else "#111111"
        ax1.text(
            bar.get_width() + 0.45,
            bar.get_y() + bar.get_height() / 2.0,
            f"{r['ZeroShot_RMSE']:.2f} bu/ac" + ("  ◄ SATELLITE" if is_sat else ""),
            va="center",
            ha="left",
            fontsize=10.5 if is_sat else 9.5,
            fontweight="bold",
            color=txt_color,
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_plot["Run"], fontsize=10, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(48.0, float(df_plot["ZeroShot_RMSE"].max()) + 8.0))
    ax1.set_xlabel("Zero-Shot Test RMSE (Bushels / Acre) — Lower is Better", fontsize=11, fontweight="bold")
    ax1.set_title("1. Zero-Shot Crawfordsville Test RMSE (UAV vs. Satellite)", fontsize=13, fontweight="bold")
    ax1.grid(axis="x", linestyle=":", alpha=0.6)

    # Right Subplot: Zero-Shot Test R²
    bars2 = ax2.barh(y_pos, df_plot["ZeroShot_R2"], color=bar_colors, height=0.64, alpha=0.93)
    for b, ec, lw in zip(bars2, edge_colors, line_widths):
        b.set_edgecolor(ec)
        b.set_linewidth(lw)

    ax2.axvline(0.0, color="black", linestyle="-", linewidth=1.8, label="Baseline Average Guess (R² = 0.0)")

    for bar, (_, r) in zip(bars2, df_plot.iterrows()):
        r2_v = r["ZeroShot_R2"]
        is_sat = r["Group"] == "SAT_ONLY"
        if r2_v < 0:
            x_t = r2_v - 0.04
            ha_t = "right"
            s_t = f"{r2_v:.4f}"
        else:
            x_t = r2_v + 0.03
            ha_t = "left"
            s_t = f"+{r2_v:.4f}"

        txt_color = "#4a148c" if is_sat else ("#b71c1c" if r2_v < 0 else "#1b5e20")
        ax2.text(
            x_t,
            bar.get_y() + bar.get_height() / 2.0,
            s_t + (" ★" if is_sat else ""),
            va="center",
            ha=ha_t,
            fontsize=10.5 if is_sat else 9.5,
            fontweight="bold",
            color=txt_color,
        )

    min_r2 = min(-1.95, float(df_plot["ZeroShot_R2"].min()) - 0.35)
    max_r2 = max(0.45, float(df_plot["ZeroShot_R2"].max()) + 0.22)
    ax2.set_xlim(min_r2, max_r2)
    ax2.set_xlabel("Zero-Shot Test R² Score — Positive (> 0) Beats Field Average", fontsize=11, fontweight="bold")
    ax2.set_title("2. Zero-Shot Crawfordsville Test R² (UAV vs. Satellite)", fontsize=13, fontweight="bold")
    ax2.grid(axis="x", linestyle=":", alpha=0.6)

    legend_elements = [
        Patch(facecolor="#c62828", edgecolor="black", label="Previous UAV Runs (Negative R²)"),
        Patch(facecolor="#2e7d32", edgecolor="black", label="Step 6 Tuned UAV Winner (Positive R²)"),
        Patch(facecolor="#7b1fa2", edgecolor="#ffd600", linewidth=2, label="★ NEW: Satellite-Only Model (TP1–TP6, 6 Bands)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.995),
        fontsize=9.0,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
    )

    fig.suptitle(
        "Unseen Crawfordsville Test Set (488 Plots): Satellite Imagery (.TIF 6-Band) vs. Previous UAV Drone Runs\n"
        f"★ Satellite-Only Zero-Shot RMSE = {sat_only_metrics['ZeroShot_Test_RMSE']:.2f} bu/ac (R² = {sat_only_metrics['ZeroShot_Test_R2']:+.4f})",
        fontsize=13.5,
        fontweight="bold",
        y=0.97,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(out_img_path, dpi=300)
    plt.close()


# ==============================================================================
# 8. MAIN EXECUTION
# ==============================================================================
def main():
    t0 = time.time()
    print(f"Running Satellite Evaluation Pipeline on Device: {DEVICE}")
    df = load_satellite_master_dataset()

    # Step 1: Extract 6-Band Spectral Indices (TP1..TP6) + EfficientNet-B0 Embeddings
    sat_dict = extract_all_satellite_features(df)

    # Step 2: Build Engineered Satellite Feature Matrices
    print("\nBuilding Flight-Normalized Satellite Features, Growth Curves & OOF Prototypes...")
    X_sat_ranker, X_sat_phys, sat_feature_names = build_satellite_engineered_matrices(df, sat_dict)
    print(
        f"  -> Satellite Ranker Matrix : {X_sat_ranker.shape[1]} features | "
        f"Satellite Physical Anchor Matrix : {X_sat_phys.shape[1]} features"
    )

    y = df["yieldPerAcre"].values.astype(np.float32)
    splits = df["split"].values
    locations = df["location_clean"].values

    # Step 3: Evaluate Satellite-Only Model on Unseen Crawfordsville Test Set
    print("\nEvaluating SATELLITE-ONLY Model (TP1–TP6 6-Band .TIF + EfficientNet-B0 + XGBoost)...")
    sat_only_metrics, sat_only_preds = run_two_stage_evaluation(
        "1. SATELLITE-ONLY (TP1–TP6, 6-Band + EffNetB0)",
        X_sat_ranker,
        X_sat_phys,
        y,
        splits,
        locations,
    )
    results_list = [sat_only_metrics]

    res_df = pd.DataFrame(results_list)
    csv_out = OUTPUT_DIR / "satellite_evaluation_metrics.csv"
    res_df.to_csv(csv_out, index=False)

    print("\n" + "=" * 138)
    print("SATELLITE IMAGERY EVALUATION RESULTS — UNSEEN CRAWFORDSVILLE TEST SET (488 PLOTS)")
    print("=" * 138)
    print(res_df.to_string(index=False))
    print("=" * 138)

    print("\n[KEY METRICS TO REPORT TO YOUR TEACHER]")
    print(
        f"  ★ SATELLITE-ONLY ZERO-SHOT TEST RMSE : {sat_only_metrics['ZeroShot_Test_RMSE']:.4f} bu/ac "
        f"(MAE = {sat_only_metrics['ZeroShot_Test_MAE']:.4f} bu/ac)"
    )
    print(
        f"  ★ SATELLITE-ONLY ZERO-SHOT TEST R²   : {sat_only_metrics['ZeroShot_Test_R2']:+.4f} "
        f"(Pearson r = {sat_only_metrics['Test_Pearson_r']:.4f})"
    )
    print(
        f"  ★ SATELLITE-ONLY 4-LOC VAL RMSE & R² : {sat_only_metrics['Val_RMSE']:.4f} bu/ac "
        f"(Val R² = {sat_only_metrics['Val_R2']:.4f})"
    )

    # Step 5: Generate Comparison Chart with Highlighted Colors
    chart_path = OUTPUT_DIR / "satellite_vs_uav_zeroshot_comparison.png"
    plot_highlighted_comparison_chart(sat_only_metrics, chart_path)

    print(f"\nSaved Satellite Metrics CSV -> {csv_out}")
    print(f"Saved Highlighted Comparison Chart -> {chart_path}")
    print(f"Total Runtime: {time.time() - t0:.1f}s")

    try:
        os.startfile(str(chart_path))
    except Exception:
        pass


if __name__ == "__main__":
    main()