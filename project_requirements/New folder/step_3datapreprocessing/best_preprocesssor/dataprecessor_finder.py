import os
import math
import time
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

OUTPUT_DIR = PROJECT_ROOT / "preprocessing_comparison"
CACHE_DIR = OUTPUT_DIR / "feature_cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 224
BATCH_SIZE = 32
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization constants for EfficientNet-B0
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


# ==============================================================================
# 2. DATA LOADING & SPLIT RESOLUTION (4 LOCS TRAIN/VAL + CRAWFORDSVILLE TEST)
# ==============================================================================
def load_split_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    col = "plot_id" if "plot_id" in df.columns else df.columns[0]
    return set(df[col].astype(str).str.strip().tolist())


def resolve_tp3_image_path(row: pd.Series) -> Path:
    """Resolves the TP3 image path across any location folder in drone_dataset/UAV."""
    loc = str(row["location"]).strip()
    loc_tp3_dir = UAV_DIR / loc / "TP3"

    raw_path = str(row.get("TP3_image_path", "")).strip()
    if raw_path and raw_path.lower() != "nan":
        candidates = [
            Path(raw_path),
            PROJECT_ROOT / raw_path,
            PROJECT_ROOT / "drone_dataset" / raw_path,
            loc_tp3_dir / Path(raw_path).name,
        ]
        for cand in candidates:
            if cand.exists():
                return cand

    # Fallback to explicit naming convention: {location}-TP3-{experiment}_{range}_{row}.PNG
    exp = str(row["experiment"]).strip()
    rng = str(row["range"]).strip()
    r_num = str(row["row"]).strip()
    fallback_name = f"{loc}-TP3-{exp}_{rng}_{r_num}.PNG"
    fallback_path = loc_tp3_dir / fallback_name
    if fallback_path.exists():
        return fallback_path

    # Case-insensitive fallback scan inside the location's TP3 directory
    if loc_tp3_dir.exists():
        for p in loc_tp3_dir.glob(f"*{exp}_{rng}_{r_num}.*"):
            return p

    raise FileNotFoundError(
        f"Could not find TP3 image for plot_id={row['plot_id']} (location={loc})"
    )


def filter_resolvable_plots(df_in: pd.DataFrame, label: str) -> pd.DataFrame:
    """Resolves TP3 image paths and drops any row whose image file is missing."""
    df_out = df_in.copy().reset_index(drop=True)
    resolved_paths = []
    valid_mask = []
    missing_count = 0

    for _, row in df_out.iterrows():
        try:
            p = resolve_tp3_image_path(row)
            resolved_paths.append(p)
            valid_mask.append(True)
        except FileNotFoundError:
            resolved_paths.append(None)
            valid_mask.append(False)
            missing_count += 1

    df_out["resolved_tp3_path"] = resolved_paths
    if missing_count > 0:
        print(f"[WARNING] ({label}) Dropped {missing_count} plots due to missing TP3 images.")
        df_out = df_out[valid_mask].reset_index(drop=True)
    return df_out


def load_all_datasets() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Loads:
      1. sub_df (4 Non-Crawfordsville locations) + train_idx & val_idx
      2. test_df (Crawfordsville test location)
    """
    df = pd.read_csv(VALID_PLOTS_CSV)
    df["location_clean"] = df["location"].astype(str).str.strip()
    df = df.dropna(subset=["yieldPerAcre"]).reset_index(drop=True)

    train_ids = load_split_ids(TRAIN_SPLIT_CSV)
    val_ids = load_split_ids(VAL_SPLIT_CSV)
    test_ids = load_split_ids(TEST_SPLIT_CSV)

    # 1. Non-Crawfordsville (4 Locations for Train & Validation)
    sub_df = df[df["location_clean"].str.lower() != "crawfordsville"].copy()
    sub_df = filter_resolvable_plots(sub_df, "4-Locations Train/Val")

    sub_df["in_train"] = sub_df["plot_id"].astype(str).str.strip().isin(train_ids)
    sub_df["in_val"] = sub_df["plot_id"].astype(str).str.strip().isin(val_ids)

    n_train = int(sub_df["in_train"].sum())
    n_val = int(sub_df["in_val"].sum())

    if n_train > 0 and n_val >= 10:
        train_idx = np.where(sub_df["in_train"].values)[0]
        val_idx = np.where(sub_df["in_val"].values)[0]
        split_source = "official train_plot_ids.csv & validation_plot_ids.csv"
    else:
        yield_bins = pd.qcut(
            sub_df["yieldPerAcre"], q=5, labels=False, duplicates="drop"
        ).astype(str)
        strat_key = sub_df["location_clean"] + "_" + yield_bins
        all_idx = np.arange(len(sub_df))
        train_idx, val_idx = train_test_split(
            all_idx, test_size=0.20, random_state=RANDOM_SEED, stratify=strat_key
        )
        split_source = "stratified 80/20 fallback split"

    # 2. Crawfordsville (Held-Out Test Location)
    test_df = df[df["location_clean"].str.lower() == "crawfordsville"].copy()
    if len(test_ids) > 0:
        # Keep Crawfordsville plots that are in test_plot_ids.csv (or all Crawfordsville if none match)
        in_test_csv = test_df["plot_id"].astype(str).str.strip().isin(test_ids)
        if in_test_csv.sum() > 0:
            test_df = test_df[in_test_csv].copy()
    test_df = filter_resolvable_plots(test_df, "Crawfordsville Test")

    print("=" * 85)
    print(f"4-LOCATION DEV DATASET LOADED : {len(sub_df)} plots ({split_source})")
    for loc_name, count in sub_df["location_clean"].value_counts().items():
        print(f"  - {loc_name:<16}: {count} plots")
    print(f"  -> Active Train Split       : {len(train_idx)} plots")
    print(f"  -> Active Validation Split  : {len(val_idx)} plots")
    print("-" * 85)
    print(f"CRAWFORDSVILLE TEST DATASET   : {len(test_df)} plots (100% Unseen Location)")
    print("=" * 85)

    return sub_df, train_idx, val_idx, test_df


# ==============================================================================
# 3. PATCH EXTRACTION & PADDING TECHNIQUES
# ==============================================================================
def get_edge_aligned_starts(length: int, patch_size: int, overlap_pct: float) -> List[int]:
    """
    Computes starting positions for edge-aligned patches along a 1D axis.
    Example: length=500, patch_size=224, overlap_pct=25.0 -> [0, 168, 276]
    """
    if length <= patch_size:
        return [0]

    overlap_px = int(round(patch_size * (overlap_pct / 100.0)))
    stride = max(1, patch_size - overlap_px)

    starts = []
    pos = 0
    while pos + patch_size < length:
        starts.append(pos)
        pos += stride

    # Always align the final patch so it ends exactly at length - 1
    final_start = length - patch_size
    if not starts or starts[-1] != final_start:
        starts.append(final_start)

    return starts


def extract_edge_aligned_patches(img_rgb: np.ndarray, overlap_pct: float) -> List[np.ndarray]:
    """
    Extracts 224x224 edge-aligned patches.
    If an axis is smaller than 224, interpolates only that axis to 224 so all pixels remain real.
    """
    h, w, _ = img_rgb.shape
    if h < PATCH_SIZE or w < PATCH_SIZE:
        new_h = max(h, PATCH_SIZE)
        new_w = max(w, PATCH_SIZE)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w, _ = img_rgb.shape

    y_starts = get_edge_aligned_starts(h, PATCH_SIZE, overlap_pct)
    x_starts = get_edge_aligned_starts(w, PATCH_SIZE, overlap_pct)

    patches = []
    for y in y_starts:
        for x in x_starts:
            patches.append(img_rgb[y : y + PATCH_SIZE, x : x + PATCH_SIZE])
    return patches


def safe_reflect_pad(img: np.ndarray, pad_bottom: int, pad_right: int) -> np.ndarray:
    """
    Safely applies cv2.BORDER_REFLECT_101 even when required padding >= image dimension.
    """
    out = img
    rem_b, rem_r = pad_bottom, pad_right
    while rem_b > 0 or rem_r > 0:
        h, w = out.shape[:2]
        step_b = min(rem_b, max(1, h - 1)) if h > 1 else rem_b
        step_r = min(rem_r, max(1, w - 1)) if w > 1 else rem_r
        border_mode = cv2.BORDER_REFLECT_101 if (h > 1 and w > 1) else cv2.BORDER_REPLICATE
        out = cv2.copyMakeBorder(out, 0, step_b, 0, step_r, borderType=border_mode)
        rem_b -= step_b
        rem_r -= step_r
    return out


def pad_to_multiple_of_224(
    img_rgb: np.ndarray, mode: str
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Pads image to the next multiple of 224x224 and slices into non-overlapping 224x224 patches.
    Returns (patches, validity_masks).
    """
    h, w, _ = img_rgb.shape
    n_h = max(1, math.ceil(h / PATCH_SIZE))
    n_w = max(1, math.ceil(w / PATCH_SIZE))

    target_h = n_h * PATCH_SIZE
    target_w = n_w * PATCH_SIZE

    pad_b = target_h - h
    pad_r = target_w - w

    if mode == "zero" or mode == "pcp":
        padded_img = cv2.copyMakeBorder(
            img_rgb, 0, pad_b, 0, pad_r, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    elif mode == "replication":
        padded_img = cv2.copyMakeBorder(
            img_rgb, 0, pad_b, 0, pad_r, borderType=cv2.BORDER_REPLICATE
        )
    elif mode == "reflection":
        padded_img = safe_reflect_pad(img_rgb, pad_b, pad_r)
    else:
        raise ValueError(f"Unknown padding mode: {mode}")

    # Create binary validity mask (1.0 for real image pixels, 0.0 for padded region)
    mask = np.zeros((target_h, target_w), dtype=np.float32)
    mask[:h, :w] = 1.0

    patches, masks = [], []
    for r in range(n_h):
        for c in range(n_w):
            y0, x0 = r * PATCH_SIZE, c * PATCH_SIZE
            patches.append(padded_img[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE])
            masks.append(mask[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE])

    return patches, masks


# ==============================================================================
# 4. EFFICIENTNET-B0 BACKBONE & PARTIAL CONVOLUTION (PCP) HOOK ENGINE
# ==============================================================================
class PartialConvHookManager:
    """
    Attaches Partial Convolution (Liu et al., 2018) mask propagation and boundary
    re-weighting hooks to all spatial Conv2d layers in a pretrained EfficientNet-B0.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles = []
        self.current_mask = None
        self.active = False
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                h_pre = module.register_forward_pre_hook(self._pre_hook)
                h_post = module.register_forward_hook(self._post_hook)
                self.handles.append(h_pre)
                self.handles.append(h_post)

    def _pre_hook(self, module: nn.Conv2d, inputs: Tuple[torch.Tensor]):
        if not self.active or self.current_mask is None:
            return None
        x = inputs[0]
        # Skip 1x1 global pooled Squeeze-and-Excitation convolutions
        if x.shape[-2] == 1 and x.shape[-1] == 1:
            return None

        if self.current_mask.shape[-2:] != x.shape[-2:]:
            self.current_mask = F.interpolate(
                self.current_mask, size=x.shape[-2:], mode="nearest"
            )

        # Ensure invalid positions are strictly zero before convolution
        return (x * self.current_mask,)

    def _post_hook(self, module: nn.Conv2d, inputs: Tuple[torch.Tensor], output: torch.Tensor):
        if not self.active or self.current_mask is None:
            return output
        x = inputs[0]
        if x.shape[-2] == 1 and x.shape[-1] == 1:
            return output

        kh, kw = module.kernel_size
        window_area = float(kh * kw)

        # If 1x1 conv with no padding, mask simply strides without boundary dilution
        if kh == 1 and kw == 1 and module.padding == (0, 0):
            if module.stride != (1, 1):
                self.current_mask = F.avg_pool2d(
                    self.current_mask, kernel_size=1, stride=module.stride
                )
                self.current_mask = (self.current_mask > 0).float()
            return output * self.current_mask

        # Compute count of valid pixels in each convolution window
        with torch.no_grad():
            ones_kernel = torch.ones(
                (1, 1, kh, kw), device=output.device, dtype=output.dtype
            )
            valid_count = F.conv2d(
                self.current_mask,
                ones_kernel,
                bias=None,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
            )
            new_mask = (valid_count > 0).float()
            scale = torch.where(
                valid_count > 0,
                window_area / torch.clamp(valid_count, min=1e-6),
                torch.zeros_like(valid_count),
            )
            self.current_mask = new_mask

        if module.bias is not None:
            b = module.bias.view(1, -1, 1, 1)
            output = (output - b) * scale + b
            output = output * new_mask
        else:
            output = output * scale

        return output


def build_feature_extractor() -> Tuple[nn.Module, PartialConvHookManager]:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    net = models.efficientnet_b0(weights=weights)
    # Replace classifier with Identity to output 1280-dim pooled features
    net.classifier = nn.Identity()
    net.to(DEVICE)
    net.eval()
    pcp_manager = PartialConvHookManager(net)
    return net, pcp_manager


@torch.no_grad()
def extract_plot_embedding(
    model: nn.Module,
    pcp_manager: PartialConvHookManager,
    patches: List[np.ndarray],
    masks: List[np.ndarray] = None,
    use_pcp: bool = False,
) -> np.ndarray:
    """
    Runs all 224x224 patches of a single plot through EfficientNet-B0 and mean-pools
    the resulting 1280-dim embeddings into a single plot feature vector.
    """
    all_embeddings = []
    pcp_manager.active = use_pcp

    for i in range(0, len(patches), BATCH_SIZE):
        batch_np = np.stack(patches[i : i + BATCH_SIZE], axis=0)  # (B, H, W, 3)
        batch_t = (
            torch.from_numpy(batch_np).permute(0, 3, 1, 2).float().to(DEVICE) / 255.0
        )
        batch_t = (batch_t - IMAGENET_MEAN) / IMAGENET_STD

        if use_pcp and masks is not None:
            mask_np = np.stack(masks[i : i + BATCH_SIZE], axis=0)  # (B, H, W)
            pcp_manager.current_mask = (
                torch.from_numpy(mask_np).unsqueeze(1).float().to(DEVICE)
            )
        else:
            pcp_manager.current_mask = None

        feats = model(batch_t)  # (B, 1280)
        all_embeddings.append(feats.cpu().numpy())

    pcp_manager.active = False
    pcp_manager.current_mask = None

    plot_feats = np.vstack(all_embeddings)
    return np.mean(plot_feats, axis=0)


def get_or_extract_features(
    df_subset: pd.DataFrame,
    exp: Dict,
    cache_file: Path,
    subset_label: str,
    model: nn.Module,
    pcp_manager: PartialConvHookManager,
) -> Tuple[np.ndarray, float, float]:
    """Loads features from cache if available; otherwise extracts and caches them."""
    n_plots = len(df_subset)
    exp_name = exp["name"]

    if cache_file.exists():
        data = np.load(cache_file)
        if data["X"].shape[0] == n_plots:
            X = data["X"]
            avg_patches = float(data["avg_patches"])
            elapsed = float(data["elapsed"])
            print(
                f"[CACHE - {subset_label:<14}] {exp_name:<30} | Avg Patches: {avg_patches:.2f}"
            )
            return X, avg_patches, elapsed

    t0 = time.time()
    features_list = []
    patch_counts = []

    for idx, row in df_subset.iterrows():
        img_bgr = cv2.imread(str(row["resolved_tp3_path"]))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read {row['resolved_tp3_path']}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if exp["type"] == "edge":
            patches = extract_edge_aligned_patches(img_rgb, exp["overlap"])
            emb = extract_plot_embedding(
                model, pcp_manager, patches, masks=None, use_pcp=False
            )
        else:
            mode = exp["mode"]
            patches, masks = pad_to_multiple_of_224(img_rgb, mode)
            emb = extract_plot_embedding(
                model,
                pcp_manager,
                patches,
                masks=masks,
                use_pcp=(mode == "pcp"),
            )

        features_list.append(emb)
        patch_counts.append(len(patches))

        if (idx + 1) % 100 == 0 or (idx + 1) == n_plots:
            print(
                f"  -> [{subset_label} | {exp_name}] Processed {idx + 1}/{n_plots} plots...",
                end="\r",
                flush=True,
            )

    X = np.vstack(features_list).astype(np.float32)
    avg_patches = float(np.mean(patch_counts))
    elapsed = time.time() - t0
    np.savez_compressed(cache_file, X=X, avg_patches=avg_patches, elapsed=elapsed)
    print(
        f"[EXTRACTED - {subset_label:<10}] {exp_name:<30} | Avg Patches: {avg_patches:.2f} | Time: {elapsed:.1f}s"
    )
    return X, avg_patches, elapsed


# ==============================================================================
# 5. VISUAL SANITY CHECK GENERATOR
# ==============================================================================
def save_visual_comparison(sample_img_path: Path, out_path: Path):
    bgr = cv2.imread(str(sample_img_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    ea_patches = extract_edge_aligned_patches(rgb, overlap_pct=25.0)
    zero_patches, _ = pad_to_multiple_of_224(rgb, "zero")
    rep_patches, _ = pad_to_multiple_of_224(rgb, "replication")
    ref_patches, _ = pad_to_multiple_of_224(rgb, "reflection")

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(rgb)
    axes[0].set_title(f"Original Plot\n({rgb.shape[1]}x{rgb.shape[0]})")
    axes[1].imshow(ea_patches[-1])
    axes[1].set_title("Edge-Aligned\n(Final Boundary Patch)")
    axes[2].imshow(zero_patches[-1])
    axes[2].set_title("Zero Padding\n(Final Boundary Patch)")
    axes[3].imshow(rep_patches[-1])
    axes[3].set_title("Replication Padding\n(Final Boundary Patch)")
    axes[4].imshow(ref_patches[-1])
    axes[4].set_title("Reflection Padding\n(Final Boundary Patch)")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ==============================================================================
# 6. XGBOOST EVALUATION (4-LOC TRAIN/VAL + CRAWFORDSVILLE TEST)
# ==============================================================================
def get_xgb_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def safe_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def evaluate_all_splits(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    locations_dev: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    # 1. Fit on 4-Location Official Train Split (1,313 plots)
    model_train = get_xgb_model()
    model_train.fit(X_dev[train_idx], y_dev[train_idx])

    # 1a. Evaluate on 4-Location Holdout Validation Split (330 plots)
    val_preds = model_train.predict(X_dev[val_idx])
    val_rmse = float(np.sqrt(mean_squared_error(y_dev[val_idx], val_preds)))
    val_mae = float(mean_absolute_error(y_dev[val_idx], val_preds))
    val_r2 = float(r2_score(y_dev[val_idx], val_preds))

    # 1b. Evaluate that exact model on Unseen Crawfordsville Test Set (~488 plots)
    test_preds = model_train.predict(X_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
    test_mae = float(mean_absolute_error(y_test, test_preds))
    test_r2 = float(r2_score(y_test, test_preds))
    test_corr = safe_pearson_r(y_test, test_preds)

    # 2. Fit on ALL 4-Location Plots (Train + Val = 1,643 plots) -> Test on Crawfordsville
    model_full_dev = get_xgb_model()
    model_full_dev.fit(X_dev, y_dev)
    test_full_preds = model_full_dev.predict(X_test)
    test_full_rmse = float(np.sqrt(mean_squared_error(y_test, test_full_preds)))
    test_full_mae = float(mean_absolute_error(y_test, test_full_preds))
    test_full_r2 = float(r2_score(y_test, test_full_preds))
    test_full_corr = safe_pearson_r(y_test, test_full_preds)

    # 3. 5-Fold Stratified CV across the 4 Development Locations (1,643 plots)
    yield_bins = pd.qcut(y_dev, q=5, labels=False, duplicates="drop").astype(str)
    strat_labels = pd.Series(locations_dev).astype(str) + "_" + yield_bins
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    cv_rmses, cv_r2s = [], []
    for f_train, f_val in skf.split(X_dev, strat_labels):
        m_cv = get_xgb_model()
        m_cv.fit(X_dev[f_train], y_dev[f_train])
        preds = m_cv.predict(X_dev[f_val])
        cv_rmses.append(np.sqrt(mean_squared_error(y_dev[f_val], preds)))
        cv_r2s.append(r2_score(y_dev[f_val], preds))

    return {
        "Test_RMSE": test_rmse,
        "Test_MAE": test_mae,
        "Test_R2": test_r2,
        "Test_Pearson_r": test_corr,
        "Test_Full4Loc_RMSE": test_full_rmse,
        "Test_Full4Loc_MAE": test_full_mae,
        "Test_Full4Loc_R2": test_full_r2,
        "Test_Full4Loc_r": test_full_corr,
        "Val_RMSE": val_rmse,
        "Val_MAE": val_mae,
        "Val_R2": val_r2,
        "CV5_RMSE_Mean": float(np.mean(cv_rmses)),
        "CV5_R2_Mean": float(np.mean(cv_r2s)),
    }


# ==============================================================================
# 7. MAIN COMPARISON PIPELINE
# ==============================================================================
def main():
    print(f"Running on Device: {DEVICE}")
    sub_df, train_idx, val_idx, test_df = load_all_datasets()

    y_dev = sub_df["yieldPerAcre"].values.astype(np.float32)
    locations_dev = sub_df["location_clean"].values
    y_test = test_df["yieldPerAcre"].values.astype(np.float32)

    # Save visual patch sanity check on first Crawfordsville test plot
    sample_vis_path = OUTPUT_DIR / "patch_visualization_crawfordsville_test.png"
    save_visual_comparison(test_df.loc[0, "resolved_tp3_path"], sample_vis_path)
    print(f"Saved Crawfordsville sample patch visualization -> {sample_vis_path}\n")

    model, pcp_manager = build_feature_extractor()

    experiments = [
        {"name": "Edge-Aligned (25.0% overlap)", "type": "edge", "overlap": 25.0},
        {"name": "Edge-Aligned (12.5% overlap)", "type": "edge", "overlap": 12.5},
        {"name": "Edge-Aligned (6.25% overlap)", "type": "edge", "overlap": 6.25},
        {"name": "Edge-Aligned (3.125% overlap)", "type": "edge", "overlap": 3.125},
        {"name": "Edge-Aligned (1.625% overlap)", "type": "edge", "overlap": 1.625},
        {"name": "Edge-Aligned (0.0% overlap)", "type": "edge", "overlap": 0.0},
        {"name": "Zero Padding", "type": "pad", "mode": "zero"},
        {"name": "Replication Padding", "type": "pad", "mode": "replication"},
        {"name": "Reflection Padding", "type": "pad", "mode": "reflection"},
        {"name": "Partial Conv Padding (PCP)", "type": "pad", "mode": "pcp"},
    ]

    results = []

    for exp in experiments:
        exp_name = exp["name"]
        safe_tag = (
            exp_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("%", "pct")
            .replace(".", "p")
        )

        # Reuses your existing 4-location cache + creates a Crawfordsville test cache
        dev_cache_file = CACHE_DIR / f"non_crawfordsville_tp3_{safe_tag}.npz"
        test_cache_file = CACHE_DIR / f"crawfordsville_test_tp3_{safe_tag}.npz"

        X_dev, dev_patches, dev_time = get_or_extract_features(
            sub_df, exp, dev_cache_file, "4-Loc Dev", model, pcp_manager
        )
        X_test, test_patches, test_time = get_or_extract_features(
            test_df, exp, test_cache_file, "Crawf. Test", model, pcp_manager
        )

        metrics = evaluate_all_splits(
            X_dev, y_dev, locations_dev, train_idx, val_idx, X_test, y_test
        )

        row_summary = {
            "Technique": exp_name,
            "Category": "Edge-Aligned" if exp["type"] == "edge" else "Padding",
            "Overlap_%": exp.get("overlap", np.nan),
            "Dev_Patches": round(dev_patches, 2),
            "Test_Patches": round(test_patches, 2),
            "Total_Time_s": round(dev_time + test_time, 1),
            **{k: round(v, 4) for k, v in metrics.items()},
        }
        results.append(row_summary)

    results_df = pd.DataFrame(results)

    # Sort primarily by Crawfordsville Test RMSE (Train Split -> Test), then Full-4Loc Test RMSE
    results_df = results_df.sort_values(
        by=["Test_RMSE", "Test_Full4Loc_RMSE", "Val_RMSE"]
    ).reset_index(drop=True)

    csv_out = OUTPUT_DIR / "crawfordsville_test_preprocessing_comparison.csv"
    results_df.to_csv(csv_out, index=False)

    print("\n" + "=" * 118)
    print(
        f"FINAL COMPARISON — TRAIN/VAL ON 4 LOCATIONS ({len(sub_df)} PLOTS) | TEST ON CRAWFORDSVILLE ({len(test_df)} PLOTS)"
    )
    print("=" * 118)
    display_cols = [
        "Technique",
        "Test_Patches",
        "Test_RMSE",
        "Test_MAE",
        "Test_R2",
        "Test_Pearson_r",
        "Test_Full4Loc_RMSE",
        "Val_RMSE",
        "CV5_RMSE_Mean",
    ]
    print(results_df[display_cols].to_string(index=False))

    best_test = results_df.iloc[0]
    best_test_full = results_df.sort_values("Test_Full4Loc_RMSE").iloc[0]
    best_val = results_df.sort_values("Val_RMSE").iloc[0]
    best_cv = results_df.sort_values("CV5_RMSE_Mean").iloc[0]
    best_edge_test = (
        results_df[results_df["Category"] == "Edge-Aligned"]
        .sort_values("Test_RMSE")
        .iloc[0]
    )

    print("-" * 118)
    print(
        f"1. BEST ON CRAWFORDSVILLE TEST (Train -> Test)     : {best_test['Technique']} "
        f"(Test RMSE = {best_test['Test_RMSE']:.3f}, MAE = {best_test['Test_MAE']:.3f}, r = {best_test['Test_Pearson_r']:.3f})"
    )
    print(
        f"2. BEST ON CRAWFORDSVILLE TEST (Full 4Loc -> Test) : {best_test_full['Technique']} "
        f"(Test Full4Loc RMSE = {best_test_full['Test_Full4Loc_RMSE']:.3f}, r = {best_test_full['Test_Full4Loc_r']:.3f})"
    )
    print(
        f"3. BEST EDGE-ALIGNED OVERLAP ON TEST               : {best_edge_test['Technique']} "
        f"(Test RMSE = {best_edge_test['Test_RMSE']:.3f}, Val RMSE = {best_edge_test['Val_RMSE']:.3f})"
    )
    print(
        f"4. BEST ON 4-LOCATION HOLDOUT VALIDATION           : {best_val['Technique']} "
        f"(Val RMSE = {best_val['Val_RMSE']:.3f}, Test RMSE = {best_val['Test_RMSE']:.3f})"
    )
    print(
        f"5. BEST ON 4-LOCATION 5-FOLD CV                    : {best_cv['Technique']} "
        f"(CV5 RMSE = {best_cv['CV5_RMSE_Mean']:.3f}, Test RMSE = {best_cv['Test_RMSE']:.3f})"
    )
    print("=" * 118)

    # Plot Comparison Charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.5))

    # Left subplot: All techniques ranked by Crawfordsville Test RMSE
    y_pos = np.arange(len(results_df))
    ax1.barh(
        y_pos - 0.25,
        results_df["Test_RMSE"],
        height=0.25,
        label="Crawfordsville Test RMSE (Train Split)",
        color="#c62828",
    )
    ax1.barh(
        y_pos,
        results_df["Test_Full4Loc_RMSE"],
        height=0.25,
        label="Crawfordsville Test RMSE (Full 4-Loc Train)",
        color="#ef6c00",
    )
    ax1.barh(
        y_pos + 0.25,
        results_df["Val_RMSE"],
        height=0.25,
        label="4-Location Holdout Val RMSE",
        color="#2e7d32",
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(results_df["Technique"])
    ax1.invert_yaxis()
    ax1.set_xlabel("RMSE (Bushels / Acre) — Lower is Better")
    ax1.set_title("Crawfordsville Test vs. 4-Location Val RMSE")
    ax1.legend()
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    # Right subplot: Edge-Aligned Overlap Sensitivity Curve on Crawfordsville Test vs Val
    edge_ordered = results_df[results_df["Category"] == "Edge-Aligned"].sort_values(
        "Overlap_%"
    )
    ax2.plot(
        edge_ordered["Overlap_%"],
        edge_ordered["Test_RMSE"],
        marker="o",
        linewidth=2,
        label="Crawfordsville Test RMSE (Train Split)",
        color="#c62828",
    )
    ax2.plot(
        edge_ordered["Overlap_%"],
        edge_ordered["Test_Full4Loc_RMSE"],
        marker="^",
        linewidth=2,
        linestyle="-.",
        label="Crawfordsville Test RMSE (Full 4-Loc)",
        color="#ef6c00",
    )
    ax2.plot(
        edge_ordered["Overlap_%"],
        edge_ordered["Val_RMSE"],
        marker="s",
        linewidth=2,
        linestyle="--",
        label="4-Location Holdout Val RMSE",
        color="#1565c0",
    )
    for _, r in edge_ordered.iterrows():
        ax2.annotate(
            f"{r['Overlap_%']}%",
            (r["Overlap_%"], r["Test_RMSE"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax2.set_xlabel("Nominal Overlap (%)")
    ax2.set_ylabel("RMSE (Bushels / Acre)")
    ax2.set_title("Edge-Aligned Overlap Sensitivity (Crawfordsville Test vs. 4-Loc Val)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_out = OUTPUT_DIR / "crawfordsville_test_comparison_chart.png"
    plt.savefig(chart_out, dpi=200)
    plt.close()
    print(f"Saved metrics CSV -> {csv_out}")
    print(f"Saved comparison chart -> {chart_out}")


if __name__ == "__main__":
    main()