"""
Model1_All5Loc_InDomain — Step 3 & 4: Data Preprocessing & Feature Extraction
  * All 5 Locations (Ames, Crawfordsville, Lincoln, MOValley, Scottsbluff)
    are represented in Train (1,489 plots), Val (319 plots), and Test (323 plots).
  * All preprocessing, imputation, CNN fine-tuning, StandardScaler, and PCA are fit strictly on TRAIN plots.
"""
import os
import math
import time
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PROJECT PATHS & CONFIGURATION (Model1_All5Loc_InDomain)
# ==============================================================================
MODEL_NAME = "Model1_All5Loc_InDomain"
PROJECT_ROOT = Path(r"C:\Users\ashwi\Music\cropyeildererrrrrrr")
VALID_PLOTS_CSV = PROJECT_ROOT / "drone_dataset" / "valid_plots.csv"
UAV_DIR = PROJECT_ROOT / "drone_dataset" / "UAV"

SPLIT_DIR = PROJECT_ROOT / MODEL_NAME / "step1_datasplitting"
TRAIN_SPLIT_CSV = SPLIT_DIR / "train_plot_ids.csv"
VAL_SPLIT_CSV = SPLIT_DIR / "validation_plot_ids.csv"
TEST_SPLIT_CSV = SPLIT_DIR / "test_plot_ids.csv"

OUTPUT_DIR = PROJECT_ROOT / MODEL_NAME / "step4_feature_engineering"
CACHE_DIR = OUTPUT_DIR / "cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 224
BATCH_SIZE = 32
FINETUNE_EPOCHS = 6
EARLY_STOP_PATIENCE = 2
FINETUNE_PATCHES_PER_PLOT = 4
PCA_COMPONENTS_PER_BLOCK = 24
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(RANDOM_SEED)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


# ==============================================================================
# 2. DATA LOADING & MULTI-TIMEPOINT (TP1, TP2, TP3) PATH RESOLUTION
# ==============================================================================
def load_split_keys(csv_path: Path) -> set:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split file: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    loc_col = "location" if "location" in df.columns else df.columns[0]
    pid_col = "plot_id" if "plot_id" in df.columns else df.columns[1]
    return set(
        (df[loc_col].astype(str).str.strip() + "::" + df[pid_col].astype(str).str.strip()).tolist()
    )


def resolve_tp_image_path(row: pd.Series, tp_label: str) -> Optional[Path]:
    loc = str(row["location"]).strip()
    loc_tp_dir = UAV_DIR / loc / tp_label
    col_name = f"{tp_label}_image_path"
    raw_path = str(row.get(col_name, "")).strip()

    if raw_path and raw_path.lower() != "nan":
        candidates = [
            Path(raw_path),
            PROJECT_ROOT / raw_path,
            PROJECT_ROOT / "drone_dataset" / raw_path,
            loc_tp_dir / Path(raw_path).name,
        ]
        for cand in candidates:
            if cand.exists():
                return cand

    exp = str(row["experiment"]).strip()
    rng = str(row["range"]).strip()
    r_num = str(row["row"]).strip()
    fallback_name = f"{loc}-{tp_label}-{exp}_{rng}_{r_num}.PNG"
    fallback_path = loc_tp_dir / fallback_name
    if fallback_path.exists():
        return fallback_path

    if loc_tp_dir.exists():
        for p in loc_tp_dir.glob(f"*{exp}_{rng}_{r_num}.*"):
            return p

    return None


def load_master_dataset() -> pd.DataFrame:
    df = pd.read_csv(VALID_PLOTS_CSV)
    df["location_clean"] = df["location"].astype(str).str.strip()
    df["plot_id_clean"] = df["plot_id"].astype(str).str.strip()
    df = df.dropna(subset=["yieldPerAcre"]).reset_index(drop=True)

    train_keys = load_split_keys(TRAIN_SPLIT_CSV)
    val_keys = load_split_keys(VAL_SPLIT_CSV)
    test_keys = load_split_keys(TEST_SPLIT_CSV)

    tp1_paths, tp2_paths, tp3_paths = [], [], []
    valid_rows = []

    for _, row in df.iterrows():
        p3 = resolve_tp_image_path(row, "TP3")
        if p3 is None:
            valid_rows.append(False)
            tp1_paths.append(None)
            tp2_paths.append(None)
            tp3_paths.append(None)
            continue

        p1 = resolve_tp_image_path(row, "TP1")
        p2 = resolve_tp_image_path(row, "TP2")

        valid_rows.append(True)
        tp1_paths.append(p1)
        tp2_paths.append(p2)
        tp3_paths.append(p3)

    df["tp1_path"] = tp1_paths
    df["tp2_path"] = tp2_paths
    df["tp3_path"] = tp3_paths
    df = df[valid_rows].reset_index(drop=True)

    # Strictly match (location, plot_id) to Model1_All5Loc_InDomain splits
    splits = []
    for _, row in df.iterrows():
        key = f"{row['location_clean']}::{row['plot_id_clean']}"
        if key in test_keys:
            splits.append("test")
        elif key in val_keys:
            splits.append("val")
        elif key in train_keys:
            splits.append("train")
        else:
            splits.append("train")

    df["split"] = splits

    n_tp1 = df["tp1_path"].notnull().sum()
    n_tp2 = df["tp2_path"].notnull().sum()
    n_tp3 = df["tp3_path"].notnull().sum()

    print("=" * 92)
    print(f"[{MODEL_NAME}] MASTER DATASET LOADED: {len(df)} total valid plots across All 5 Locations")
    print(
        f"Split Breakdown      -> Train: {(df['split']=='train').sum()} | "
        f"Val: {(df['split']=='val').sum()} | Test (All 5 Locations): {(df['split']=='test').sum()}"
    )
    print(
        f"Timepoint Coverage   -> TP1: {n_tp1}/{len(df)} | TP2: {n_tp2}/{len(df)} | TP3: {n_tp3}/{len(df)}"
    )
    print("=" * 92)
    return df


# ==============================================================================
# 3. IMAGE PREPROCESSING (REPLICATION PADDING) & RGB CANOPY INDEX EXTRACTION
# ==============================================================================
def extract_replication_padded_patches(img_rgb: np.ndarray) -> List[np.ndarray]:
    h, w, _ = img_rgb.shape
    n_h = max(1, math.ceil(h / PATCH_SIZE))
    n_w = max(1, math.ceil(w / PATCH_SIZE))

    target_h = n_h * PATCH_SIZE
    target_w = n_w * PATCH_SIZE

    pad_b = target_h - h
    pad_r = target_w - w

    padded_img = cv2.copyMakeBorder(
        img_rgb, 0, pad_b, 0, pad_r, borderType=cv2.BORDER_REPLICATE
    )

    patches = []
    for r in range(n_h):
        for c in range(n_w):
            y0, x0 = r * PATCH_SIZE, c * PATCH_SIZE
            patches.append(padded_img[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE])
    return patches


def compute_rgb_canopy_features(
    img_rgb: np.ndarray, patches: List[np.ndarray]
) -> np.ndarray:
    eps = 1e-5
    rgb_norm = img_rgb.astype(np.float32) / 255.0
    R = rgb_norm[:, :, 0]
    G = rgb_norm[:, :, 1]
    B = rgb_norm[:, :, 2]

    total_intensity = R + G + B + eps
    r_chrom = R / total_intensity
    g_chrom = G / total_intensity
    b_chrom = B / total_intensity

    exg = 2.0 * g_chrom - r_chrom - b_chrom
    ngrdi = (G - R) / (G + R + eps)
    vari = np.clip((G - R) / (G + R - B + eps), -2.0, 2.0)
    gli = (2.0 * G - R - B) / (2.0 * G + R + B + eps)

    green_mask = (exg > 0.08) & (G > 0.08)
    fgcc = float(np.mean(green_mask))

    yellow_mask = (
        (~green_mask)
        & (R > 0.25)
        & (G > 0.20)
        & (ngrdi > -0.12)
        & (ngrdi <= 0.04)
    )
    yellow_frac = float(np.mean(yellow_mask))

    if np.any(green_mask):
        canopy_exg_mean = float(np.mean(exg[green_mask]))
        canopy_ngrdi_mean = float(np.mean(ngrdi[green_mask]))
    else:
        canopy_exg_mean = float(np.mean(exg))
        canopy_ngrdi_mean = float(np.mean(ngrdi))

    patch_fgccs = []
    for p in patches:
        p_norm = p.astype(np.float32) / 255.0
        p_tot = p_norm[:, :, 0] + p_norm[:, :, 1] + p_norm[:, :, 2] + eps
        p_exg = (2.0 * p_norm[:, :, 1] - p_norm[:, :, 0] - p_norm[:, :, 2]) / p_tot
        patch_fgccs.append(
            float(np.mean((p_exg > 0.08) & (p_norm[:, :, 1] > 0.08)))
        )

    return np.array(
        [
            float(np.mean(exg)),
            float(np.std(exg)),
            float(np.mean(ngrdi)),
            float(np.std(ngrdi)),
            float(np.mean(vari)),
            float(np.mean(gli)),
            fgcc,
            yellow_frac,
            canopy_exg_mean,
            canopy_ngrdi_mean,
            float(np.std(patch_fgccs)),
            float(np.min(patch_fgccs)),
        ],
        dtype=np.float32,
    )


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
# 4. STAGE 4A: FINE-TUNING EFFICIENTNET-B0 STRICTLY ON 1,489 TRAIN PLOTS
# ==============================================================================
class PlotBagDataset(Dataset):
    def __init__(
        self,
        df_sub: pd.DataFrame,
        y_scaled: np.ndarray,
        patches_per_plot: int = 4,
        augment: bool = True,
    ):
        self.paths = df_sub["tp3_path"].tolist()
        self.y = y_scaled.astype(np.float32)
        self.k = patches_per_plot
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def _augment_patch(self, patch: np.ndarray) -> np.ndarray:
        out = patch.astype(np.float32)
        if random.random() < 0.5:
            out = np.fliplr(out)
        if random.random() < 0.5:
            out = np.flipud(out)
        alpha = random.uniform(0.88, 1.12)
        beta = random.uniform(-15.0, 15.0)
        out = np.clip(alpha * out + beta, 0.0, 255.0)
        return out

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_bgr = cv2.imread(str(self.paths[idx]))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        patches = extract_replication_padded_patches(img_rgb)

        if len(patches) >= self.k:
            chosen = (
                random.sample(patches, self.k) if self.augment else patches[: self.k]
            )
        else:
            chosen = [patches[i % len(patches)] for i in range(self.k)]

        if self.augment:
            chosen = [self._augment_patch(p) for p in chosen]
        else:
            chosen = [p.astype(np.float32) for p in chosen]

        bag_np = np.stack(chosen, axis=0)
        bag_t = torch.from_numpy(bag_np).permute(0, 3, 1, 2).float() / 255.0
        return bag_t, torch.tensor(self.y[idx], dtype=torch.float32)


class FineTunableEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)
        self.backbone.classifier = nn.Identity()

        for param in self.backbone.features[:7].parameters():
            param.requires_grad = False
        for param in self.backbone.features[7:].parameters():
            param.requires_grad = True

        self.reg_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        b, k, c, h, w = bag.shape
        flat = bag.view(b * k, c, h, w)
        flat = (flat - IMAGENET_MEAN) / IMAGENET_STD

        with torch.no_grad():
            x = self.backbone.features[:7](flat)

        x = self.backbone.features[7:](x)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)

        feats_bag = feats.view(b, k, 1280)
        plot_emb = feats_bag.mean(dim=1)
        pred = self.reg_head(plot_emb).squeeze(-1)
        return pred


def get_or_finetune_backbone(df: pd.DataFrame) -> nn.Module:
    ckpt_path = CACHE_DIR / "m1_step4_finetuned_effnet_b0.pth"
    model = FineTunableEfficientNet().to(DEVICE)

    if ckpt_path.exists():
        model.backbone.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()
        print(f"[CACHE] Loaded fine-tuned EfficientNet-B0 weights -> {ckpt_path.name}")
        return model.backbone

    print("\n" + "-" * 92)
    print(
        f"[{MODEL_NAME}] STAGE 4A: Fine-Tuning EfficientNet-B0 Strictly on 1,489 TRAIN Plots ({FINETUNE_EPOCHS} Epochs)"
    )
    print("-" * 92)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    # Compute per-location yield mean/std strictly from TRAIN plots across all 5 locations
    loc_stats = (
        train_df.groupby("location_clean")["yieldPerAcre"]
        .agg(["mean", "std"])
        .reset_index()
    )
    loc_mean_map = dict(zip(loc_stats["location_clean"], loc_stats["mean"]))
    loc_std_map = dict(
        zip(loc_stats["location_clean"], loc_stats["std"].clip(lower=1.0))
    )
    global_mean = float(train_df["yieldPerAcre"].mean())
    global_std = max(float(train_df["yieldPerAcre"].std()), 1.0)

    tr_means = train_df["location_clean"].map(loc_mean_map).fillna(global_mean).values
    tr_stds = train_df["location_clean"].map(loc_std_map).fillna(global_std).values
    y_train_norm = (train_df["yieldPerAcre"].values - tr_means) / tr_stds

    va_means = val_df["location_clean"].map(loc_mean_map).fillna(global_mean).values
    va_stds = val_df["location_clean"].map(loc_std_map).fillna(global_std).values
    y_val_norm = (val_df["yieldPerAcre"].values - va_means) / va_stds

    train_ds = PlotBagDataset(
        train_df, y_train_norm, patches_per_plot=FINETUNE_PATCHES_PER_PLOT, augment=True
    )
    val_ds = PlotBagDataset(
        val_df, y_val_norm, patches_per_plot=FINETUNE_PATCHES_PER_PLOT, augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINETUNE_EPOCHS
    )
    criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    best_state = None
    no_improve_epochs = 0

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        t_ep = time.time()
        model.train()
        model.backbone.features[:7].eval()

        train_losses = []
        for b_idx, (bags, targets) in enumerate(train_loader):
            bags, targets = bags.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            preds = model(bags)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if (b_idx + 1) % 20 == 0 or (b_idx + 1) == len(train_loader):
                print(
                    f"  Epoch {epoch}/{FINETUNE_EPOCHS} | Batch {b_idx+1}/{len(train_loader)} "
                    f"| Loss: {np.mean(train_losses):.4f}",
                    end="\r",
                    flush=True,
                )

        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for bags, targets in val_loader:
                bags, targets = bags.to(DEVICE), targets.to(DEVICE)
                preds = model(bags)
                val_losses.append(criterion(preds, targets).item())

        m_tr = float(np.mean(train_losses))
        m_val = float(np.mean(val_losses))
        dt = time.time() - t_ep

        if m_val < best_val_loss:
            best_val_loss = m_val
            best_state = {
                k: v.cpu().clone() for k, v in model.backbone.state_dict().items()
            }
            no_improve_epochs = 0
            status_tag = "*BEST*"
        else:
            no_improve_epochs += 1
            status_tag = f"(patience {no_improve_epochs}/{EARLY_STOP_PATIENCE})"

        print(
            f"  Epoch {epoch}/{FINETUNE_EPOCHS} Completed in {dt:.1f}s | "
            f"Train Huber Loss: {m_tr:.4f} | Val Huber Loss: {m_val:.4f} {status_tag}"
        )

        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(f"  -> Early stopping triggered at Epoch {epoch}.")
            break

    if best_state is not None:
        model.backbone.load_state_dict(best_state)

    torch.save(model.backbone.state_dict(), ckpt_path)
    model.eval()
    print(f"[SAVED] Best fine-tuned EfficientNet-B0 weights -> {ckpt_path}")
    return model.backbone


# ==============================================================================
# 5. STAGE 4B–4D: MULTI-TIMEPOINT CNN + RGB INDICES + METADATA EXTRACTION
# ==============================================================================
@torch.no_grad()
def extract_single_tp_features(
    backbone: nn.Module, img_path: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    patches = extract_replication_padded_patches(img_rgb)
    canopy_feats = compute_rgb_canopy_features(img_rgb, patches)

    all_embs = []
    for i in range(0, len(patches), BATCH_SIZE):
        batch_np = np.stack(patches[i : i + BATCH_SIZE], axis=0)
        batch_t = (
            torch.from_numpy(batch_np).permute(0, 3, 1, 2).float().to(DEVICE) / 255.0
        )
        batch_t = (batch_t - IMAGENET_MEAN) / IMAGENET_STD
        feats = backbone(batch_t)
        all_embs.append(feats.cpu().numpy())

    plot_feats = np.vstack(all_embs)
    emb_mean = np.mean(plot_feats, axis=0).astype(np.float32)
    emb_std = np.std(plot_feats, axis=0).astype(np.float32)
    return emb_mean, emb_std, canopy_feats, (h, w)


def parse_agronomic_metadata(
    df: pd.DataFrame, plot_areas: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    # Strictly impute using Model 1's 1,489 TRAIN plots
    train_mask = df["split"] == "train"

    raw_stand = pd.to_numeric(df["totalStandCount"], errors="coerce")
    stand_median = (
        float(raw_stand[train_mask].median())
        if raw_stand[train_mask].notnull().any()
        else 50.0
    )
    stand_count = raw_stand.fillna(stand_median).values.astype(np.float32)

    parsed_dates = pd.to_datetime(df["plantingDate"], errors="coerce")
    raw_doy = parsed_dates.dt.dayofyear
    doy_median = (
        float(raw_doy[train_mask].median())
        if raw_doy[train_mask].notnull().any()
        else 130.0
    )
    planting_doy = raw_doy.fillna(doy_median).values.astype(np.float32)

    safe_area = np.clip(plot_areas / 10000.0, 1.0, None)
    stand_density = (stand_count / safe_area).astype(np.float32)
    log_stand = np.log1p(np.clip(stand_count, 0.0, None)).astype(np.float32)

    meta_matrix = np.column_stack(
        [stand_count, planting_doy, stand_density, log_stand]
    )
    meta_names = [
        "Meta_TotalStandCount",
        "Meta_Planting_DOY",
        "Meta_Stand_Density_10kPx",
        "Meta_Log_StandCount",
    ]
    return meta_matrix, meta_names


def extract_all_multimodal_features(
    df: pd.DataFrame, backbone: nn.Module
) -> Dict[str, np.ndarray]:
    raw_cache_path = CACHE_DIR / "m1_step4_raw_multimodal_features.npz"
    n_plots = len(df)

    if raw_cache_path.exists():
        data = np.load(raw_cache_path)
        if data["tp3_mean"].shape[0] == n_plots:
            print(
                f"[CACHE] Loaded Model 1 multi-timepoint features -> {raw_cache_path.name}"
            )
            return {k: data[k] for k in data.files}

    print("\n" + "-" * 92)
    print(
        f"[{MODEL_NAME}] STAGE 4B–4D: Extracting TP1, TP2, TP3 Features + RGB Indices for {n_plots} Plots"
    )
    print("-" * 92)

    tp1_means, tp1_stds, tp1_canopy = [], [], []
    tp2_means, tp2_stds, tp2_canopy = [], [], []
    tp3_means, tp3_stds, tp3_canopy = [], [], []
    plot_areas = []

    t0 = time.time()
    for idx, row in df.iterrows():
        m3, s3, c3, (h3, w3) = extract_single_tp_features(backbone, row["tp3_path"])

        if pd.notnull(row["tp2_path"]) and row["tp2_path"] is not None:
            m2, s2, c2, _ = extract_single_tp_features(backbone, row["tp2_path"])
        else:
            m2, s2, c2 = m3.copy(), s3.copy(), c3.copy()

        if pd.notnull(row["tp1_path"]) and row["tp1_path"] is not None:
            m1, s1, c1, _ = extract_single_tp_features(backbone, row["tp1_path"])
        else:
            m1, s1, c1 = m2.copy(), s2.copy(), c2.copy()

        tp1_means.append(m1)
        tp1_stds.append(s1)
        tp1_canopy.append(c1)

        tp2_means.append(m2)
        tp2_stds.append(s2)
        tp2_canopy.append(c2)

        tp3_means.append(m3)
        tp3_stds.append(s3)
        tp3_canopy.append(c3)

        plot_areas.append(float(h3 * w3))

        if (idx + 1) % 100 == 0 or (idx + 1) == n_plots:
            elapsed = time.time() - t0
            print(
                f"  -> Processed {idx + 1}/{n_plots} plots (TP1+TP2+TP3) | Elapsed: {elapsed:.1f}s",
                end="\r",
                flush=True,
            )

    print()
    plot_areas_np = np.array(plot_areas, dtype=np.float32)
    meta_matrix, _ = parse_agronomic_metadata(df, plot_areas_np)

    feature_dict = {
        "tp1_mean": np.vstack(tp1_means).astype(np.float32),
        "tp1_std": np.vstack(tp1_stds).astype(np.float32),
        "tp1_canopy": np.vstack(tp1_canopy).astype(np.float32),
        "tp2_mean": np.vstack(tp2_means).astype(np.float32),
        "tp2_std": np.vstack(tp2_stds).astype(np.float32),
        "tp2_canopy": np.vstack(tp2_canopy).astype(np.float32),
        "tp3_mean": np.vstack(tp3_means).astype(np.float32),
        "tp3_std": np.vstack(tp3_stds).astype(np.float32),
        "tp3_canopy": np.vstack(tp3_canopy).astype(np.float32),
        "meta": meta_matrix.astype(np.float32),
        "plot_areas": plot_areas_np,
    }

    np.savez_compressed(raw_cache_path, **feature_dict)
    print(f"[SAVED] Raw multi-modal feature cache -> {raw_cache_path}")
    return feature_dict


# ==============================================================================
# 6. FEATURE FUSION, TEMPORAL DELTAS & TRAIN-ONLY PCA COMPRESSION
# ==============================================================================
def build_structured_feature_table(
    df: pd.DataFrame, feat_dict: Dict[str, np.ndarray], fit_mask: np.ndarray
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    meta_X, meta_names = parse_agronomic_metadata(df, feat_dict["plot_areas"])

    c1 = feat_dict["tp1_canopy"]
    c2 = feat_dict["tp2_canopy"]
    c3 = feat_dict["tp3_canopy"]

    c_delta_21 = c2 - c1
    c_delta_32 = c3 - c2
    c_delta_31 = c3 - c1
    c_auc = 0.25 * c1 + 0.50 * c2 + 0.25 * c3

    canopy_blocks = [
        ("TP1", c1),
        ("TP2", c2),
        ("TP3", c3),
        ("Delta_TP2_TP1", c_delta_21),
        ("Delta_TP3_TP2", c_delta_32),
        ("Delta_TP3_TP1", c_delta_31),
        ("Season_AUC", c_auc),
    ]

    canopy_X = np.hstack([blk for _, blk in canopy_blocks])
    canopy_cols = [
        f"Canopy_{prefix}_{name}"
        for prefix, _ in canopy_blocks
        for name in CANOPY_FEATURE_NAMES
    ]

    cnn_delta_21 = feat_dict["tp2_mean"] - feat_dict["tp1_mean"]
    cnn_delta_32 = feat_dict["tp3_mean"] - feat_dict["tp2_mean"]
    cnn_delta_31 = feat_dict["tp3_mean"] - feat_dict["tp1_mean"]

    cnn_blocks = [
        ("CNN_TP1_Mean", feat_dict["tp1_mean"], PCA_COMPONENTS_PER_BLOCK),
        ("CNN_TP2_Mean", feat_dict["tp2_mean"], PCA_COMPONENTS_PER_BLOCK),
        ("CNN_TP3_Mean", feat_dict["tp3_mean"], PCA_COMPONENTS_PER_BLOCK),
        ("CNN_TP3_PatchStd", feat_dict["tp3_std"], 12),
        ("CNN_Delta_TP2_TP1", cnn_delta_21, PCA_COMPONENTS_PER_BLOCK),
        ("CNN_Delta_TP3_TP2", cnn_delta_32, PCA_COMPONENTS_PER_BLOCK),
        ("CNN_Delta_TP3_TP1", cnn_delta_31, PCA_COMPONENTS_PER_BLOCK),
    ]

    pca_features_list = []
    pca_col_names = []

    # Fit StandardScaler & PCA strictly on the 1,489 TRAIN plots
    for blk_name, blk_matrix, n_comp in cnn_blocks:
        scaler = StandardScaler()
        blk_train_scaled = scaler.fit_transform(blk_matrix[fit_mask])
        blk_all_scaled = scaler.transform(blk_matrix)

        n_comp_actual = min(
            n_comp, blk_train_scaled.shape[0] - 1, blk_train_scaled.shape[1]
        )
        pca = PCA(n_components=n_comp_actual, random_state=RANDOM_SEED)
        pca.fit(blk_train_scaled)

        blk_pca = pca.transform(blk_all_scaled).astype(np.float32)
        pca_features_list.append(blk_pca)
        pca_col_names.extend([f"{blk_name}_PC{i+1}" for i in range(n_comp_actual)])

    cnn_pca_X = np.hstack(pca_features_list)

    X_full = np.hstack([meta_X, canopy_X, cnn_pca_X]).astype(np.float32)
    all_feature_names = meta_names + canopy_cols + pca_col_names

    out_df = pd.DataFrame(X_full, columns=all_feature_names)
    out_df.insert(0, "yieldPerAcre", df["yieldPerAcre"].values)
    out_df.insert(0, "split", df["split"].values)
    out_df.insert(0, "location", df["location_clean"].values)
    out_df.insert(0, "plot_id", df["plot_id_clean"].values)

    return X_full, all_feature_names, out_df


# ==============================================================================
# 7. EVALUATION BENCHMARK (TRAINED STRICTLY ON 1,489 TRAIN PLOTS)
# ==============================================================================
def safe_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def get_regularized_xgb() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=12,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=2.0,
        reg_lambda=10.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def evaluate_pipeline_variant(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    # Strictly fit ONLY on the 1,489 training plots
    model_tr = get_regularized_xgb()
    model_tr.fit(X[train_mask], y[train_mask])

    val_preds = model_tr.predict(X[val_mask])
    test_preds = model_tr.predict(X[test_mask])

    y_test = y[test_mask]
    y_val = y[val_mask]

    metrics = {
        "Pipeline_Stage": name,
        "Num_Features": X.shape[1],
        "Val_RMSE": round(float(np.sqrt(mean_squared_error(y_val, val_preds))), 4),
        "Val_MAE": round(float(mean_absolute_error(y_val, val_preds)), 4),
        "Val_R2": round(float(r2_score(y_val, val_preds)), 4),
        "Test_RMSE": round(float(np.sqrt(mean_squared_error(y_test, test_preds))), 4),
        "Test_MAE": round(float(mean_absolute_error(y_test, test_preds)), 4),
        "Test_R2": round(float(r2_score(y_test, test_preds)), 4),
        "Test_Pearson_r": round(safe_pearson_r(y_test, test_preds), 4),
    }
    return metrics, test_preds


def main():
    print(f"Running [{MODEL_NAME}] Preprocessing & Feature Extraction on Device: {DEVICE}")
    df = load_master_dataset()

    backbone = get_or_finetune_backbone(df)
    feat_dict = extract_all_multimodal_features(df, backbone)

    splits = df["split"].values
    locations = df["location_clean"].values
    y = df["yieldPerAcre"].values.astype(np.float32)

    train_mask = splits == "train"

    X_step4, feature_names, master_csv_df = build_structured_feature_table(
        df, feat_dict, fit_mask=train_mask
    )

    master_csv_path = OUTPUT_DIR / "m1_step4_master_features.csv"
    master_npz_path = OUTPUT_DIR / "m1_step4_master_features.npz"

    master_csv_df.to_csv(master_csv_path, index=False)
    np.savez_compressed(
        master_npz_path,
        X=X_step4,
        y=y,
        splits=splits,
        locations=locations,
        plot_ids=df["plot_id_clean"].values.astype(str),
        feature_names=np.array(feature_names, dtype=str),
    )

    print("\n" + "=" * 92)
    print(
        f"[{MODEL_NAME}] SAVED MASTER CSV -> {master_csv_path} ({X_step4.shape[0]} rows, {X_step4.shape[1]} features)"
    )
    print(f"[{MODEL_NAME}] SAVED MASTER NPZ -> {master_npz_path}")
    print("=" * 92)

    canopy_all = X_step4[:, : (4 + 7 * len(CANOPY_FEATURE_NAMES))]

    variants = [
        ("1. Raw TP3 CNN Only (1280 dims)", feat_dict["tp3_mean"]),
        (
            "2. TP1+TP2+TP3 CNN + Growth Deltas (PCA)",
            X_step4[:, (4 + 7 * len(CANOPY_FEATURE_NAMES)) :],
        ),
        ("3. RGB Canopy Indices + Agronomic Metadata", canopy_all),
        ("4. FULL STEP 4 (Multi-TP CNN + Deltas + RGB + Metadata)", X_step4),
    ]

    comparison_rows = []
    best_preds = None
    for v_name, v_X in variants:
        m, preds = evaluate_pipeline_variant(v_name, v_X, y, splits)
        comparison_rows.append(m)
        if "FULL STEP 4" in v_name:
            best_preds = preds

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv = OUTPUT_DIR / "m1_step4_ablation_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)

    print("\n" + "=" * 115)
    print(
        f"[{MODEL_NAME}] STEP 4 BENCHMARK RESULTS (FIT STRICTLY ON 1,489 TRAIN PLOTS -> 5-LOC TEST)"
    )
    print("=" * 115)
    print(comp_df.to_string(index=False))
    print("=" * 115)

    # Feature importance strictly from TRAIN split
    model_imp = get_regularized_xgb()
    model_imp.fit(X_step4[train_mask], y[train_mask])
    importances = model_imp.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    top_names = [feature_names[i] for i in top_idx][::-1]
    top_vals = [importances[i] for i in top_idx][::-1]
    ax1.barh(range(len(top_names)), top_vals, color="#1565c0")
    ax1.set_yticks(range(len(top_names)))
    ax1.set_yticklabels(top_names, fontsize=9)
    ax1.set_xlabel("XGBoost Feature Importance (Gain)")
    ax1.set_title(f"[{MODEL_NAME}] Top 15 Features (Train-Only Fit)")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    y_test = y[splits == "test"]
    ax2.scatter(y_test, best_preds, alpha=0.5, color="#2e7d32", edgecolors="k", s=30)
    min_v = float(min(y_test.min(), best_preds.min()))
    max_v = float(max(y_test.max(), best_preds.max()))
    ax2.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="1:1 Ideal Line")
    ax2.set_xlabel("Actual Yield (Bushels / Acre) — All 5 Locations Test")
    ax2.set_ylabel("Predicted Yield (Bushels / Acre)")
    ax2.set_title(f"[{MODEL_NAME}] 5-Location Test Set: Actual vs. Predicted")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "m1_step4_feature_importance_and_test_scatter.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()

    print(f"Saved ablation metrics -> {comp_csv}")
    print(f"Saved Feature Importance & Parity chart -> {chart_path}")


if __name__ == "__main__":
    main()