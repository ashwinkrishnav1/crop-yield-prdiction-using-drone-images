"""
Image -> raw per-timepoint features. Logic is copied from
model5loc_preprocess_feature_extraction.py (steps 4B-4D); do not "improve" it, the
downstream models were fitted on exactly these numbers.

Each model has its OWN fine-tuned EfficientNet-B0 (trained on different plots), so
every image goes through both backbones; the RGB canopy indices are backbone-free and
computed once.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PATCH_SIZE = 224
BATCH_SIZE = 32


def extract_replication_padded_patches(img_rgb: np.ndarray) -> List[np.ndarray]:
    h, w, _ = img_rgb.shape
    n_h, n_w = max(1, math.ceil(h / PATCH_SIZE)), max(1, math.ceil(w / PATCH_SIZE))
    padded = cv2.copyMakeBorder(img_rgb, 0, n_h * PATCH_SIZE - h, 0, n_w * PATCH_SIZE - w,
                                borderType=cv2.BORDER_REPLICATE)
    return [padded[r * PATCH_SIZE:(r + 1) * PATCH_SIZE, c * PATCH_SIZE:(c + 1) * PATCH_SIZE]
            for r in range(n_h) for c in range(n_w)]


def compute_rgb_canopy_features(img_rgb: np.ndarray, patches: List[np.ndarray]) -> np.ndarray:
    eps = 1e-5
    rgb = img_rgb.astype(np.float32) / 255.0
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    tot = R + G + B + eps
    r_c, g_c, b_c = R / tot, G / tot, B / tot
    exg = 2.0 * g_c - r_c - b_c
    ngrdi = (G - R) / (G + R + eps)
    vari = np.clip((G - R) / (G + R - B + eps), -2.0, 2.0)
    gli = (2.0 * G - R - B) / (2.0 * G + R + B + eps)

    green = (exg > 0.08) & (G > 0.08)
    fgcc = float(np.mean(green))
    yellow = (~green) & (R > 0.25) & (G > 0.20) & (ngrdi > -0.12) & (ngrdi <= 0.04)
    yellow_frac = float(np.mean(yellow))
    if np.any(green):
        c_exg, c_ngrdi = float(np.mean(exg[green])), float(np.mean(ngrdi[green]))
    else:
        c_exg, c_ngrdi = float(np.mean(exg)), float(np.mean(ngrdi))

    pf = []
    for p in patches:
        pn = p.astype(np.float32) / 255.0
        pt = pn[:, :, 0] + pn[:, :, 1] + pn[:, :, 2] + eps
        pe = (2.0 * pn[:, :, 1] - pn[:, :, 0] - pn[:, :, 2]) / pt
        pf.append(float(np.mean((pe > 0.08) & (pn[:, :, 1] > 0.08))))

    return np.array([float(np.mean(exg)), float(np.std(exg)), float(np.mean(ngrdi)), float(np.std(ngrdi)),
                     float(np.mean(vari)), float(np.mean(gli)), fgcc, yellow_frac, c_exg, c_ngrdi,
                     float(np.std(pf)), float(np.min(pf))], dtype=np.float32)


def load_backbone(ckpt: Path, device: Optional[str] = None):
    """Fine-tuned EfficientNet-B0 feature extractor (classifier removed -> 1280-d)."""
    import torch
    import torch.nn as nn
    from torchvision import models

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bb = models.efficientnet_b0(weights=None)          # weights come from the checkpoint
    bb.classifier = nn.Identity()
    bb.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    return bb.to(device).eval(), device


class PlotFeatureExtractor:
    def __init__(self, backbone_ckpts: Dict[str, Path], device: Optional[str] = None):
        import torch
        self.torch = torch
        self.backbones, self.device = {}, None
        for name, ck in backbone_ckpts.items():
            self.backbones[name], self.device = load_backbone(ck, device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @staticmethod
    def read_rgb(path) -> np.ndarray:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise RuntimeError(f"Could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def extract_timepoint(self, path) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], np.ndarray, Tuple[int, int]]:
        torch = self.torch
        rgb = self.read_rgb(path)
        h, w, _ = rgb.shape
        patches = extract_replication_padded_patches(rgb)
        canopy = compute_rgb_canopy_features(rgb, patches)
        emb = {}
        with torch.no_grad():
            for name, bb in self.backbones.items():
                outs = []
                for i in range(0, len(patches), BATCH_SIZE):
                    x = torch.from_numpy(np.stack(patches[i:i + BATCH_SIZE])).permute(0, 3, 1, 2).float().to(self.device) / 255.0
                    outs.append(bb((x - self.mean) / self.std).cpu().numpy())
                f = np.vstack(outs)
                emb[name] = (np.mean(f, axis=0).astype(np.float32), np.std(f, axis=0).astype(np.float32))
        return emb, canopy, (h, w)

    def extract_plots(self, tp1: Sequence[Optional[Path]], tp2: Sequence[Optional[Path]],
                      tp3: Sequence[Path]) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[List[str]]]:
        """
        Returns {model_name: raw-cache-style dict} and per-plot flags.
        Missing TP2 is copied from TP3, missing TP1 from TP2 (what training did).
        """
        names = list(self.backbones)
        acc = {n: {f"tp{t}_{k}": [] for t in (1, 2, 3) for k in ("mean", "std", "canopy")} for n in names}
        areas, flags = [], []
        for p1, p2, p3 in zip(tp1, tp2, tp3):
            fl = []
            e3, c3, (h3, w3) = self.extract_timepoint(p3)
            if p2 is not None:
                e2, c2, _ = self.extract_timepoint(p2)
            else:
                e2, c2 = e3, c3
                fl.append("tp2_missing_copied_from_tp3")
            if p1 is not None:
                e1, c1, _ = self.extract_timepoint(p1)
            else:
                e1, c1 = e2, c2
                fl.append("tp1_missing_copied_from_tp2")
            for n in names:
                for t, e, c in ((1, e1, c1), (2, e2, c2), (3, e3, c3)):
                    acc[n][f"tp{t}_mean"].append(e[n][0])
                    acc[n][f"tp{t}_std"].append(e[n][1])
                    acc[n][f"tp{t}_canopy"].append(c)
            areas.append(float(h3 * w3))
            flags.append(fl)
        out = {}
        for n in names:
            d = {k: np.vstack(v).astype(np.float32) for k, v in acc[n].items()}
            d["plot_areas"] = np.array(areas, dtype=np.float32)
            out[n] = d
        return out, flags
