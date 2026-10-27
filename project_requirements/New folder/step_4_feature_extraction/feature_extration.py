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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PROJECT PATHS & CONFIGURATION
# ==============================================================================
PROJECT_ROOT = Path(r"C:\Users\ashwi\Music\cropyeildererrrrrrr")
VALID_PLOTS_CSV = PROJECT_ROOT / "drone_dataset" / "valid_plots.csv"
UAV_DIR = PROJECT_ROOT / "drone_dataset" / "UAV"

SPLIT_DIR = PROJECT_ROOT / "datasplitting_script"
TRAIN_SPLIT_CSV = SPLIT_DIR / "train_plot_ids.csv"
VAL_SPLIT_CSV = SPLIT_DIR / "validation_plot_ids.csv"
TEST_SPLIT_CSV = SPLIT_DIR / "test_plot_ids.csv"

OUTPUT_DIR = PROJECT_ROOT / "step4_feature_engineering"
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
def load_split_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    col = "plot_id" if "plot_id" in df.columns else df.columns[0]
    return set(df[col].astype(str).str.strip().tolist())


def resolve_tp_image_path(row: pd.Series, tp_label: str) -> Optional[Path]:
    """
    Resolves TP1, TP2, or TP3 image path across any location folder in drone_dataset/UAV.
    Returns None if the image for that timepoint does not exist on disk.
    """
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

    # Explicit naming fallback: {location}-{TP}-{experiment}_{range}_{row}.PNG
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
    """
    Loads all valid plots (4 Dev Locations + Crawfordsville Test), resolves TP1/TP2/TP3
    paths, and assigns each plot strictly to 'train', 'val', or 'test'.
    """
    df = pd.read_csv(VALID_PLOTS_CSV)
    df["location_clean"] = df["location"].astype(str).str.strip()
    df = df.dropna(subset=["yieldPerAcre"]).reset_index(drop=True)

    train_ids = load_split_ids(TRAIN_SPLIT_CSV)
    val_ids = load_split_ids(VAL_SPLIT_CSV)
    test_ids = load_split_ids(TEST_SPLIT_CSV)

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

    # Assign explicit split labels: 'train', 'val', or 'test' (Crawfordsville)
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

    n_tp1 = df["tp1_path"].notnull().sum()
    n_tp2 = df["tp2_path"].notnull().sum()
    n_tp3 = df["tp3_path"].notnull().sum()

    print("=" * 88)
    print(f"MASTER DATASET LOADED: {len(df)} total valid plots")
    print(
        f"Split Breakdown      -> Train: {(df['split']=='train').sum()} | "
        f"Val: {(df['split']=='val').sum()} | Test (Crawfordsville): {(df['split']=='test').sum()}"
    )
    print(
        f"Timepoint Coverage   -> TP1: {n_tp1}/{len(df)} | TP2: {n_tp2}/{len(df)} | TP3: {n_tp3}/{len(df)}"
    )
    print("=" * 88)
    return df


# ==============================================================================
# 3. REPLICATION PADDING & RGB VEGETATION / CANOPY INDEX EXTRACTION
# ==============================================================================
def extract_replication_padded_patches(img_rgb: np.ndarray) -> List[np.ndarray]:
    """
    Applies Replication Padding to the next multiple of 224x224 and slices
    the image into non-overlapping 224x224 patches (Winner of Step 3).
    """
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
    """
    Computes 12 lighting-normalized, domain-invariant RGB canopy & senescence features
    from the real (unpadded) plot pixels plus patch-level green uniformity.
    """
    eps = 1e-5
    rgb_norm = img_rgb.astype(np.float32) / 255.0
    R = rgb_norm[:, :, 0]
    G = rgb_norm[:, :, 1]
    B = rgb_norm[:, :, 2]

    # Chromatic coordinates (normalizes out overall sunlight brightness)
    total_intensity = R + G + B + eps
    r_chrom = R / total_intensity
    g_chrom = G / total_intensity
    b_chrom = B / total_intensity

    # 1. Excess Green (ExG = 2g - r - b)
    exg = 2.0 * g_chrom - r_chrom - b_chrom

    # 2. Normalized Green-Red Difference Index (NGRDI = (G - R) / (G + R))
    ngrdi = (G - R) / (G + R + eps)

    # 3. Visible Atmospherically Resistant Index (VARI = (G - R) / (G + R - B))
    vari = np.clip((G - R) / (G + R - B + eps), -2.0, 2.0)

    # 4. Green Leaf Index (GLI = (2G - R - B) / (2G + R + B))
    gli = (2.0 * G - R - B) / (2.0 * G + R + B + eps)

    # 5. Fractional Green Canopy Cover (FGCC): fraction of pixels that are green leaves
    green_mask = (exg > 0.08) & (G > 0.08)
    fgcc = float(np.mean(green_mask))

    # 6. Senescence / Yellowing Leaf Fraction (late-season grain maturity indicator)
    yellow_mask = (
        (~green_mask)
        & (R > 0.25)
        & (G > 0.20)
        & (ngrdi > -0.12)
        & (ngrdi <= 0.04)
    )
    yellow_frac = float(np.mean(yellow_mask))

    # 7. Vigour of purely green canopy pixels (ignores background soil color completely)
    if np.any(green_mask):
        canopy_exg_mean = float(np.mean(exg[green_mask]))
        canopy_ngrdi_mean = float(np.mean(ngrdi[green_mask]))
    else:
        canopy_exg_mean = float(np.mean(exg))
        canopy_ngrdi_mean = float(np.mean(ngrdi))

    # 8. Patch-to-Patch Green Cover Uniformity (detects bare gaps/missing plants in rows)
    patch_fgccs = []
    for p in patches:
        p_norm = p.astype(np.float32) / 255.0
        p_tot = p_norm[:, :, 0] + p_norm[:, :, 1] + p_norm[:, :, 2] + eps
        p_exg = (2.0 * p_norm[:, :, 1] - p_norm[:, :, 0] - p_norm[:, :, 2]) / p_tot
        patch_fgccs.append(
            float(np.mean((p_exg > 0.08) & (p_norm[:, :, 1] > 0.08)))
        )

    patch_fgcc_std = float(np.std(patch_fgccs))
    patch_fgcc_min = float(np.min(patch_fgccs))

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
            patch_fgcc_std,
            patch_fgcc_min,
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
# 4. STAGE 4A: FINE-TUNING EFFICIENTNET-B0 (FINAL LAYERS ONLY, 6 EPOCHS)
# ==============================================================================
class PlotBagDataset(Dataset):
    """
    Samples K Replication-Padded patches per plot with lighting/flip augmentation
    so EfficientNet-B0 learns soil/lighting-invariant maize canopy representations.
    """

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
        # Random brightness & contrast jitter (simulates sunny vs cloudy drone flights)
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

        bag_np = np.stack(chosen, axis=0)  # (K, 224, 224, 3)
        bag_t = torch.from_numpy(bag_np).permute(0, 3, 1, 2).float() / 255.0
        return bag_t, torch.tensor(self.y[idx], dtype=torch.float32)


class FineTunableEfficientNet(nn.Module):
    """
    Freezes early layers (features[:7]) and unfreezes the final blocks (features[7:])
    with a plot-level regression head for end-to-end bag-of-patches fine-tuning.
    """

    def __init__(self):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)
        self.backbone.classifier = nn.Identity()

        # Freeze early layers (0..6); unfreeze final semantic layers (7..8)
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

        # Run frozen early layers under no_grad for speed & memory efficiency on CPU
        with torch.no_grad():
            x = self.backbone.features[:7](flat)

        # Run trainable top blocks with gradients
        x = self.backbone.features[7:](x)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)  # (B*K, 1280)

        feats_bag = feats.view(b, k, 1280)
        plot_emb = feats_bag.mean(dim=1)  # (B, 1280)
        pred = self.reg_head(plot_emb).squeeze(-1)
        return pred


def get_or_finetune_backbone(df: pd.DataFrame) -> nn.Module:
    """
    Fine-tunes the final layers of EfficientNet-B0 on the 4-location training split
    for 6 epochs (with leak-free target scaling and best-epoch restoration).
    """
    ckpt_path = CACHE_DIR / "step4_finetuned_effnet_b0.pth"
    model = FineTunableEfficientNet().to(DEVICE)

    if ckpt_path.exists():
        model.backbone.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()
        print(f"[CACHE] Loaded fine-tuned EfficientNet-B0 weights -> {ckpt_path.name}")
        return model.backbone

    print("\n" + "-" * 88)
    print(
        f"STAGE 4A: Fine-Tuning EfficientNet-B0 Final Layers (features[7:]) for {FINETUNE_EPOCHS} Epochs"
    )
    print("-" * 88)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    # Compute per-location yield mean/std strictly from TRAIN split (zero val leakage)
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
        # Keep frozen BatchNorm layers in eval mode so running stats stay stable
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
            print(f"  -> Early stopping triggered at Epoch {epoch} to prevent overfitting.")
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
    """
    Given a plot image path:
      1. Slices into 224x224 patches via Replication Padding
      2. Extracts Fine-Tuned EfficientNet-B0 embeddings -> returns (Mean_1280, Std_1280)
      3. Extracts 12 RGB Canopy & Senescence indices -> returns (Canopy_12,)
      4. Returns original (height, width) for plant stand density calculation
    """
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
        feats = backbone(batch_t)  # (B, 1280)
        all_embs.append(feats.cpu().numpy())

    plot_feats = np.vstack(all_embs)
    emb_mean = np.mean(plot_feats, axis=0).astype(np.float32)
    emb_std = np.std(plot_feats, axis=0).astype(np.float32)
    return emb_mean, emb_std, canopy_feats, (h, w)


def parse_agronomic_metadata(
    df: pd.DataFrame, plot_areas: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts clean, location-independent agronomic metadata from valid_plots.csv:
      - totalStandCount (imputed strictly using train split median)
      - planting_doy (Day of Year, imputed strictly using train split median)
      - stand_density_per_10k_px
      - log_stand_count
    """
    train_mask = df["split"] == "train"

    # 1. Total Stand Count
    raw_stand = pd.to_numeric(df["totalStandCount"], errors="coerce")
    stand_median = (
        float(raw_stand[train_mask].median())
        if raw_stand[train_mask].notnull().any()
        else 50.0
    )
    stand_count = raw_stand.fillna(stand_median).values.astype(np.float32)

    # 2. Planting Date converted to Day of Year (1..365)
    parsed_dates = pd.to_datetime(df["plantingDate"], errors="coerce")
    raw_doy = parsed_dates.dt.dayofyear
    doy_median = (
        float(raw_doy[train_mask].median())
        if raw_doy[train_mask].notnull().any()
        else 130.0
    )
    planting_doy = raw_doy.fillna(doy_median).values.astype(np.float32)

    # 3. Plant Density relative to plot pixel area & Log Stand Count
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
    """
    Extracts and caches TP1, TP2, TP3 CNN embeddings (Mean + Std), RGB Canopy Indices,
    and Agronomic Metadata across all 2,131 plots.
    """
    raw_cache_path = CACHE_DIR / "step4_raw_multimodal_features.npz"
    n_plots = len(df)

    if raw_cache_path.exists():
        data = np.load(raw_cache_path)
        if data["tp3_mean"].shape[0] == n_plots:
            print(
                f"[CACHE] Loaded Step 4 multi-timepoint features -> {raw_cache_path.name}"
            )
            return {k: data[k] for k in data.files}

    print("\n" + "-" * 88)
    print(
        f"STAGE 4B–4D: Extracting TP1, TP2, TP3 Features + RGB Indices for {n_plots} Plots"
    )
    print("-" * 88)

    tp1_means, tp1_stds, tp1_canopy = [], [], []
    tp2_means, tp2_stds, tp2_canopy = [], [], []
    tp3_means, tp3_stds, tp3_canopy = [], [], []
    plot_areas = []

    t0 = time.time()
    for idx, row in df.iterrows():
        # TP3 is guaranteed to exist
        m3, s3, c3, (h3, w3) = extract_single_tp_features(backbone, row["tp3_path"])

        # TP2 (fallback to TP3 if missing)
        if pd.notnull(row["tp2_path"]) and row["tp2_path"] is not None:
            m2, s2, c2, _ = extract_single_tp_features(backbone, row["tp2_path"])
        else:
            m2, s2, c2 = m3.copy(), s3.copy(), c3.copy()

        # TP1 (fallback to TP2 if missing)
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
# 6. FEATURE FUSION, TEMPORAL GROWTH DELTAS & LEAK-FREE PCA COMPRESSION
# ==============================================================================
def build_structured_feature_table(
    df: pd.DataFrame, feat_dict: Dict[str, np.ndarray], fit_mask: np.ndarray
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Combines:
      1. Agronomic Metadata (Stand Count, Planting DOY, Stand Density)
      2. RGB Canopy Indices across TP1, TP2, TP3 + Growth Deltas (TP2-TP1, TP3-TP2, TP3-TP1)
      3. PCA-Compressed Fine-Tuned CNN Features (fit strictly on train_mask)
         for TP1, TP2, TP3, Patch Std, and Temporal CNN Growth Deltas
    """
    _, meta_names = parse_agronomic_metadata(df, feat_dict["plot_areas"])
    meta_X = feat_dict["meta"]

    # RGB Canopy Features + Temporal Canopy Deltas
    c1 = feat_dict["tp1_canopy"]
    c2 = feat_dict["tp2_canopy"]
    c3 = feat_dict["tp3_canopy"]

    c_delta_21 = c2 - c1  # Early-to-Mid Canopy Closure Speed
    c_delta_32 = c3 - c2  # Mid-to-Late Stay-Green vs Senescence Rate
    c_delta_31 = c3 - c1  # Full-Season Change
    c_auc = 0.25 * c1 + 0.50 * c2 + 0.25 * c3  # Season-Integrated Canopy Vigour

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

    # Temporal CNN Growth Deltas (Cancels out static background soil color)
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

    # Combine Metadata + Canopy Indices + Compressed CNN Features
    X_full = np.hstack([meta_X, canopy_X, cnn_pca_X]).astype(np.float32)
    all_feature_names = meta_names + canopy_cols + pca_col_names

    # Build clean DataFrame for CSV export
    out_df = pd.DataFrame(X_full, columns=all_feature_names)
    out_df.insert(0, "yieldPerAcre", df["yieldPerAcre"].values)
    out_df.insert(0, "split", df["split"].values)
    out_df.insert(0, "location", df["location_clean"].values)
    out_df.insert(0, "plot_id", df["plot_id"].values)

    return X_full, all_feature_names, out_df


# ==============================================================================
# 7. REAL-WORLD EVALUATION (STEP 3 BASELINE VS. STEP 4 UPGRADED PIPELINE)
# ==============================================================================
def safe_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def get_regularized_xgb() -> XGBRegressor:
    """
    Domain-generalizable XGBoost configuration:
    Shallower trees (max_depth=3) + higher min_child_weight and L2 regularization
    prevent memorizing single-location soil artifacts.
    """
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
    locations: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    train_mask = splits == "train"
    val_mask = splits == "val"
    dev_mask = (splits == "train") | (splits == "val")
    test_mask = splits == "test"

    # 1. Official Train Split (1,313 plots) -> Val (330 plots) & Crawfordsville Test (488 plots)
    model_tr = get_regularized_xgb()
    model_tr.fit(X[train_mask], y[train_mask])

    val_preds = model_tr.predict(X[val_mask])
    test_preds = model_tr.predict(X[test_mask])

    # 2. Full 4-Location Dev Split (1,643 plots) -> Crawfordsville Test (488 plots)
    model_full = get_regularized_xgb()
    model_full.fit(X[dev_mask], y[dev_mask])
    test_full_preds = model_full.predict(X[test_mask])

    # 3. 5-Fold Stratified CV on 4 Locations
    y_dev = y[dev_mask]
    X_dev = X[dev_mask]
    loc_dev = locations[dev_mask]
    yield_bins = pd.qcut(y_dev, q=5, labels=False, duplicates="drop").astype(str)
    strat_labels = pd.Series(loc_dev).astype(str) + "_" + yield_bins
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    cv_rmses = []
    for f_tr, f_va in skf.split(X_dev, strat_labels):
        m_cv = get_regularized_xgb()
        m_cv.fit(X_dev[f_tr], y_dev[f_tr])
        p_cv = m_cv.predict(X_dev[f_va])
        cv_rmses.append(np.sqrt(mean_squared_error(y_dev[f_va], p_cv)))

    y_test = y[test_mask]
    y_val = y[val_mask]

    metrics = {
        "Pipeline_Stage": name,
        "Num_Features": X.shape[1],
        "Test_RMSE": round(float(np.sqrt(mean_squared_error(y_test, test_preds))), 4),
        "Test_MAE": round(float(mean_absolute_error(y_test, test_preds)), 4),
        "Test_R2": round(float(r2_score(y_test, test_preds)), 4),
        "Test_Pearson_r": round(safe_pearson_r(y_test, test_preds), 4),
        "Test_Full4Loc_RMSE": round(
            float(np.sqrt(mean_squared_error(y_test, test_full_preds))), 4
        ),
        "Test_Full4Loc_R2": round(float(r2_score(y_test, test_full_preds)), 4),
        "Test_Full4Loc_r": round(safe_pearson_r(y_test, test_full_preds), 4),
        "Val_RMSE": round(float(np.sqrt(mean_squared_error(y_val, val_preds))), 4),
        "Val_R2": round(float(r2_score(y_val, val_preds)), 4),
        "CV5_RMSE": round(float(np.mean(cv_rmses)), 4),
    }
    return metrics, test_full_preds


def main():
    print(f"Running Step 4 Pipeline on Device: {DEVICE}")
    df = load_master_dataset()

    # Stage 4A: Fine-tune top block of EfficientNet-B0 on 4-location training split (6 Epochs)
    backbone = get_or_finetune_backbone(df)

    # Stage 4B-4D: Extract TP1, TP2, TP3 embeddings + RGB Canopy Indices + Metadata
    feat_dict = extract_all_multimodal_features(df, backbone)

    splits = df["split"].values
    locations = df["location_clean"].values
    y = df["yieldPerAcre"].values.astype(np.float32)

    # Fit StandardScaler & PCA strictly on the 1,313 training plots (zero leakage into Val or Test)
    train_mask = splits == "train"
    dev_mask = (splits == "train") | (splits == "val")

    X_step4, feature_names, master_csv_df = build_structured_feature_table(
        df, feat_dict, fit_mask=train_mask
    )

    # Save Master CSV and NPZ for Step 5 & Step 6
    master_csv_path = OUTPUT_DIR / "step4_master_features.csv"
    master_npz_path = OUTPUT_DIR / "step4_master_features.npz"

    master_csv_df.to_csv(master_csv_path, index=False)
    np.savez_compressed(
        master_npz_path,
        X=X_step4,
        y=y,
        splits=splits,
        locations=locations,
        plot_ids=df["plot_id"].values.astype(str),
        feature_names=np.array(feature_names, dtype=str),
    )

    print("\n" + "=" * 88)
    print(
        f"SAVED MASTER STEP 4 CSV -> {master_csv_path} ({X_step4.shape[0]} rows, {X_step4.shape[1]} features)"
    )
    print(f"SAVED MASTER STEP 4 NPZ -> {master_npz_path}")
    print("=" * 88)

    # Benchmark Progression (Ablation Study from Step 3 Baseline -> Full Step 4)
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
        m, preds = evaluate_pipeline_variant(v_name, v_X, y, splits, locations)
        comparison_rows.append(m)
        if "FULL STEP 4" in v_name:
            best_preds = preds

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv = OUTPUT_DIR / "step4_ablation_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)

    print("\n" + "=" * 118)
    print(
        "STEP 4 BENCHMARK RESULTS — IMPACT OF MULTI-TIMEPOINT + CANOPY INDICES + METADATA ON CRAWFORDSVILLE TEST"
    )
    print("=" * 118)
    cols = [
        "Pipeline_Stage",
        "Num_Features",
        "Test_RMSE",
        "Test_MAE",
        "Test_R2",
        "Test_Pearson_r",
        "Test_Full4Loc_RMSE",
        "Test_Full4Loc_r",
        "Val_RMSE",
        "Val_R2",
    ]
    print(comp_df[cols].to_string(index=False))
    print("=" * 118)

    # Extract and plot Top 15 Most Important Features in Full Step 4 Model
    model_imp = get_regularized_xgb()
    model_imp.fit(X_step4[dev_mask], y[dev_mask])
    importances = model_imp.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Top 15 Features driving real-world yield prediction
    top_names = [feature_names[i] for i in top_idx][::-1]
    top_vals = [importances[i] for i in top_idx][::-1]
    ax1.barh(range(len(top_names)), top_vals, color="#1565c0")
    ax1.set_yticks(range(len(top_names)))
    ax1.set_yticklabels(top_names, fontsize=9)
    ax1.set_xlabel("XGBoost Feature Importance (Gain)")
    ax1.set_title("Top 15 Step 4 Features (Multi-TP + Canopy + Metadata)")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    # Right: Crawfordsville Predicted vs Actual Yield Scatter
    y_test = y[splits == "test"]
    ax2.scatter(y_test, best_preds, alpha=0.5, color="#2e7d32", edgecolors="k", s=30)
    min_v = float(min(y_test.min(), best_preds.min()))
    max_v = float(max(y_test.max(), best_preds.max()))
    ax2.plot(
        [min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="1:1 Ideal Line"
    )
    ax2.set_xlabel("Actual Yield (Bushels / Acre) — Crawfordsville")
    ax2.set_ylabel("Predicted Yield (Bushels / Acre)")
    ax2.set_title("Crawfordsville Unseen Test Set: Actual vs. Predicted")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "step4_feature_importance_and_test_scatter.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()

    print(f"Saved Step 4 ablation metrics -> {comp_csv}")
    print(f"Saved Feature Importance & Parity chart -> {chart_path}")


if __name__ == "__main__":
    main()