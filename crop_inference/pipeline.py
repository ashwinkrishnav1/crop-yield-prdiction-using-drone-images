"""
CropYieldPipeline: plot records (+ their TP1/TP2/TP3 images) -> two yield estimates.

    pipe = CropYieldPipeline(model_root)
    out  = pipe.run(plots_df, image_root=...)       # DataFrame, one row per plot

Model 1  in-domain specialist   : per-plot, valid only for locations it was trained on.
Model 2  zero-shot site model   : per-FIELD (batch), valid for any location, weak absolute accuracy.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import predict as P
from . import tabular as T
from .paths import ModelPaths

SMALL_FIELD = 30   # below this the Model-2 field anchor / centring is unreliable


def resolve_image(p, image_root: Optional[Path]) -> Optional[Path]:
    if p is None or (isinstance(p, float) and np.isnan(p)) or str(p).strip() in ("", "nan", "None"):
        return None
    s = str(p).strip().replace("\\", "/")
    cands = [Path(s)]
    if image_root:
        cands += [Path(image_root) / s, Path(image_root) / "drone_dataset" / s]
    for c in cands:
        if c.exists():
            return c
    return None


class CropYieldPipeline:
    def __init__(self, model_root, device: Optional[str] = None, m2_weights: str = "full4loc",
                 need_images: bool = True):
        self.root = Path(model_root)
        self.mp = {v: ModelPaths(v, self.root) for v in ("m1", "m2")}
        self.state, self.bundle = {}, {}
        for v, mp in self.mp.items():
            if not mp.state_file.exists():
                raise FileNotFoundError(
                    f"{mp.state_file} missing - run once:  python -m crop_inference.export_state "
                    f"--model-root \"{self.root}\" [--gt-csv ...]")
            with open(mp.state_file, "rb") as f:
                self.state[v] = pickle.load(f)
            self.bundle[v] = P.load_bundle(mp.model_bundle)
        self.m2_weights = m2_weights
        self.extractor = None
        if need_images:
            from .extract import PlotFeatureExtractor
            self.extractor = PlotFeatureExtractor({v: mp.backbone_ckpt for v, mp in self.mp.items()}, device)

    # ------------------------------------------------------------------ public
    def run(self, plots: pd.DataFrame, image_root=None, m2_calibration: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """plots: one row per plot (see cli.py for columns). Images are read from the *_image_path columns."""
        if self.extractor is None:
            raise RuntimeError("constructed with need_images=False; use run_from_raw")
        plots = plots.reset_index(drop=True)
        n = len(plots)

        def col(t):
            c = f"TP{t}_image_path"
            return plots[c] if c in plots else pd.Series([None] * n)

        paths = {t: [resolve_image(col(t).iloc[i], image_root) for i in range(n)] for t in (1, 2, 3)}
        per_plot, flags, ok = [], [[] for _ in range(n)], np.zeros(n, dtype=bool)
        for i in range(n):
            if paths[3][i] is None:
                flags[i].append("tp3_image_missing")
                per_plot.append(None)
                continue
            try:
                raw, fl = self.extractor.extract_plots([paths[1][i]], [paths[2][i]], [paths[3][i]])
            except Exception as e:  # unreadable image: skip this plot, keep the batch going
                flags[i].append(f"image_error_{type(e).__name__}")
                per_plot.append(None)
                continue
            flags[i] += fl[0]
            per_plot.append(raw)
            ok[i] = True

        out = self._empty(plots)
        if ok.any():
            good = [r for r in per_plot if r is not None]
            stacked = {v: {k: np.concatenate([r[v][k] for r in good], axis=0) for k in good[0][v]} for v in self.mp}
            sub = self.run_from_raw(plots[ok].reset_index(drop=True), stacked, m2_calibration=m2_calibration)
            for c in sub.columns:
                if c in ("plot_id", "location"):
                    continue
                if c not in out:
                    out[c] = np.nan
                out.loc[ok, c] = sub[c].values
        sub_flags = out["flags"].astype(str).tolist()
        out["flags"] = [";".join(sorted(set(flags[i]) | {f for f in sub_flags[i].split(";") if f})) for i in range(n)]
        return out

    def run_from_raw(self, plots: pd.DataFrame, raw_by_model: Dict[str, Dict[str, np.ndarray]],
                     m2_calibration: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Core logic on pre-extracted raw features (same keys as the training cache)."""
        plots = plots.reset_index(drop=True)
        no_xy = np.zeros(len(plots), dtype=bool)
        for c in ("range", "row"):
            no_xy |= plots[c].isna().values if c in plots else True
        man = T.clean_manifest(plots)
        n = len(man)
        locs = man["location"].values.astype(str)
        out = self._empty(man)
        flags = [["spatial_coords_missing"] if no_xy[i] else [] for i in range(n)]

        # ---------------- Model 1 (per plot, known locations only)
        m1_ok = np.isin(locs, self.state["m1"]["train_locs"])
        for i in np.where(~m1_ok)[0]:
            flags[i].append("m1_unavailable_unseen_location")
        if m1_ok.any():
            idx = np.where(m1_ok)[0]
            t = T.transform(self.state["m1"], man.iloc[idx].reset_index(drop=True),
                            {k: v[idx] for k, v in raw_by_model["m1"].items()})
            out.loc[idx, "model1_yield"] = P.predict_m1(self.bundle["m1"], t["X_full"], t["X_phys"], locs[idx])
            for j, i in enumerate(idx):
                flags[i] += t["flags"][j]

        # ---------------- Model 2 (per field = per location)
        try:
            t2 = T.transform(self.state["m2"], man, raw_by_model["m2"])
        except T.BatchStatsError as e:
            t2 = None
            for i in range(n):
                flags[i].append("m2_unavailable_" + str(e).split("'")[0].strip().replace(" ", "_")[:40])
        if t2 is not None:
            for loc in np.unique(locs):
                g = np.where(locs == loc)[0]
                pred = P.predict_m2_group(self.bundle["m2"], t2["X_full"][g], t2["X_phys"][g], self.m2_weights)
                out.loc[g, "model2_yield"] = pred
                out.loc[g, "model2_field_pctile"] = (pd.Series(pred).rank(pct=True).values * 100.0).round(1) if len(g) > 1 else np.nan
                for i in g:
                    if len(g) < SMALL_FIELD:
                        flags[i].append(f"m2_small_field_n{len(g)}")
                    if len(g) == 1:
                        flags[i].append("m2_single_plot_no_ranking")
                    flags[i] += t2["flags"][i]
                if m2_calibration:
                    ci = [k for k, i in enumerate(g) if man["plot_id"].iloc[i] in m2_calibration]
                    if len(ci) >= 5:
                        cy = [m2_calibration[man["plot_id"].iloc[g[k]]] for k in ci]
                        out.loc[g, "model2_calibrated"] = P.calibrate_m2(pred, ci, cy)
                    else:
                        for i in g:
                            flags[i].append("m2_calibration_needs_5_labelled_plots")

        # ---------------- combined view
        m1, m2 = out["model1_yield"].astype(float), out["model2_yield"].astype(float)
        out["model_gap_m1_minus_m2"] = (m1 - m2).round(2)
        use1 = m1.notna()
        out["recommended_yield"] = np.where(use1, m1, m2)
        out["recommended_source"] = np.where(use1, "model1", np.where(m2.notna(), "model2", "none"))
        out["flags"] = [";".join(sorted(set(f))) for f in flags]
        for c in ("model1_yield", "model2_yield", "model2_calibrated", "recommended_yield"):
            if c in out:
                out[c] = out[c].astype(float).round(2)
        return out

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _empty(plots: pd.DataFrame) -> pd.DataFrame:
        o = pd.DataFrame({"plot_id": plots["plot_id"].astype(str).values, "location": plots["location"].astype(str).values})
        for c in ("model1_yield", "model2_yield", "model2_field_pctile", "model_gap_m1_minus_m2", "recommended_yield"):
            o[c] = np.nan
        o["recommended_source"] = "none"
        o["flags"] = ""
        return o
