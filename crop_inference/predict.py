"""Blend logic of step 6, reproduced from the training scripts."""
from __future__ import annotations

import pickle
from typing import Dict, Optional, Sequence

import numpy as np


def load_bundle(path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------- Model 1
def predict_m1(B: dict, X_full: np.ndarray, X_phys: np.ndarray, locations: Sequence[str]) -> np.ndarray:
    """Per-plot (no batch dependence). Location mean is added back as the offset."""
    X_core = X_full[:, B["core_idx"]]
    off = np.array([B["tr_loc_means"][l] for l in locations], dtype=np.float32)

    def view(models, X):
        return (0.30 * models["xgb1"].predict(X) + 0.20 * models["xgb2"].predict(X)
                + 0.10 * models["xgb3"].predict(X) + 0.14 * models["et"].predict(X)
                + 0.12 * models["hgb"].predict(X) + 0.14 * models["ridge"].predict(X))

    core_view = view(B["core_view_models"], X_core) + off
    full_view = view(B["full_view_models"], X_full) + off
    pi = B["phys_inter_models"]
    phys_inter = (0.35 * (0.80 * pi["phys_xgb"].predict(X_phys) + 0.20 * pi["phys_ridge"].predict(X_phys) + off)
                  + 0.30 * pi["abs_full_xgb"].predict(X_full)
                  + 0.20 * pi["abs_core_xgb"].predict(X_core)
                  + 0.15 * pi["abs_core_et"].predict(X_core))
    wc, wf, wp = B["stack_weights"]
    stack = wc * core_view + wf * full_view + wp * phys_inter
    return (off + B["gamma_cal"] * (stack - off)).astype(np.float32)


# ----------------------------------------------------------------------------- Model 2
def _m2_rank(B: dict, which: str, X: np.ndarray) -> np.ndarray:
    m = B[f"ranker_models_{which}"]
    w_x, w_e, w_h, w_r = B["blend_weights"]
    p_xgb = 0.65 * m["xgb1"].predict(X) + 0.35 * m["xgb2"].predict(X)
    return (w_x * p_xgb + w_e * m["et"].predict(X) + w_h * m["hgb"].predict(X)
            + w_r * m["ridge"].predict(X)).astype(np.float32)


def predict_m2_group(B: dict, X_core: np.ndarray, X_phys: np.ndarray, which: str = "full4loc") -> np.ndarray:
    """
    ONE field (one location) at a time. The training code defines the output as

        anchor + alpha * (0.86 * (rank - mean(rank)) + 0.14 * (phys - mean(phys)))
        anchor = mean(phys)              # field-level yield level
    so it is only meaningful for a whole field; a single plot collapses to its phys prediction.
    """
    suffix = "full4loc" if which == "full4loc" else "train"
    rank = _m2_rank(B, which, X_core)
    phys = B[f"phys_xgb_{suffix}"].predict(X_phys)
    anchor = float(np.mean(phys))
    return (anchor + B["alpha_shrink"] * (0.86 * (rank - np.mean(rank)) + 0.14 * (phys - anchor))).astype(np.float32)


def calibrate_m2(pred: np.ndarray, cal_idx: Sequence[int], cal_yield: Sequence[float]) -> np.ndarray:
    """Optional: shift the field level using a few plots with known yield (training used 15)."""
    off = float(np.median(np.asarray(cal_yield, dtype=np.float64) - pred[list(cal_idx)]))
    return (pred + 0.65 * off).astype(np.float32)
