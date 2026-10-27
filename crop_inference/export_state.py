"""
One-time step: rebuild + freeze every fitted preprocessing object the training scripts
did not save, and PROVE parity against the feature matrices you already stored.

    python -m crop_inference.export_state --model-root "C:\\...\\final model" \
        [--gt-csv "C:\\...\\GroundTruth\\HYBRID_HIPS_V3.5_ALLPLOTS.csv"]

--gt-csv is needed only for the two phenology features (daysToAnthesis / GDDToAnthesis);
without it those two columns are neutral (0) and flagged `pheno_state_missing`.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import tabular as T
from .paths import ModelPaths


def load_reference(mp: ModelPaths):
    raw = {k: v for k, v in np.load(mp.raw_cache, allow_pickle=True).items()}
    s5 = {k: v for k, v in np.load(mp.step5_npz, allow_pickle=True).items()}
    man = pd.concat([pd.read_csv(mp.split_dir / f"{n}_plot_ids.csv") for n in ("train", "validation", "test")])
    man["plot_id"] = man["plot_id"].astype(str).str.strip()
    man = man.drop_duplicates(subset=["plot_id"]).set_index("plot_id")
    ids = [str(p).strip() for p in s5["plot_ids"]]
    man = T.clean_manifest(man.reindex(ids).reset_index())
    return raw, s5, man


def parity_report(variant: str, S: dict, raw, s5, man) -> dict:
    splits = s5["splits"]
    sel = (splits != "train") if variant == "m1" else (splits == "test")
    idx = np.where(sel)[0]
    sub_raw = {k: v[idx] for k, v in raw.items()}
    out = T.transform(S, man.iloc[idx].reset_index(drop=True), sub_raw)
    names = list(s5["feature_names"])
    ref = s5["X_raw"][idx]
    pheno = [i for i, n in enumerate(names) if n.startswith("Pheno_")]
    ok = [i for i in range(len(names)) if i not in pheno] if S["pheno"] is None else list(range(len(names)))
    diff = np.abs(out["X_raw"][:, ok] - ref[:, ok])
    rel = diff / (np.abs(ref[:, ok]) + 1e-3)
    worst = np.argsort(-rel.max(axis=0))[:3]
    rep = {"rows": len(idx), "cols_checked": len(ok), "max_abs": float(diff.max()),
           "max_rel": float(rel.max()), "worst_cols": [names[ok[j]] for j in worst]}
    if variant == "m1":
        rep["phys_max_abs"] = float(np.abs(out["X_phys_raw"] - s5["X_phys_raw"][idx]).max())
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", required=True, type=Path)
    ap.add_argument("--gt-csv", type=Path, default=None)
    ap.add_argument("--tol", type=float, default=5e-3, help="max relative error accepted in parity check")
    a = ap.parse_args()

    gt = None
    if a.gt_csv:
        g = pd.read_csv(a.gt_csv)
        cols = [c for c in g.columns if c in T.PHENO_COLS]
        gt = g[["qrCode"] + cols].drop_duplicates(subset=["qrCode"])

    for variant in ("m1", "m2"):
        mp = ModelPaths(variant, a.model_root)
        raw, s5, man = load_reference(mp)
        S = T.fit_state(variant, raw, man, s5["y"].astype(np.float32), s5["splits"], list(map(str, s5["feature_names"])),
                        s5["X_raw"], s5["X_phys_raw"] if variant == "m1" else None, gt=gt)
        rep = parity_report(variant, S, raw, s5, man)
        print(f"[{variant}] meta parity max|diff| = {S['meta_parity_maxdiff']:.2e} | "
              f"feature parity on {rep['rows']} held-out plots x {rep['cols_checked']} cols: "
              f"max abs {rep['max_abs']:.3g}, max rel {rep['max_rel']:.3g}, worst {rep['worst_cols']}"
              + (f" | phys cols max abs {rep['phys_max_abs']:.3g}" if "phys_max_abs" in rep else ""))
        if rep["max_rel"] > a.tol:
            raise SystemExit(f"[{variant}] PARITY FAILED (> {a.tol}); state not written")
        mp.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(mp.state_file, "wb") as f:
            pickle.dump(S, f)
        print(f"[{variant}] state written -> {mp.state_file}")


if __name__ == "__main__":
    main()
