"""
Image-stage parity check (needs your UAV images + torch): re-extract features for a few
plots and compare with the cached values the models were trained on.

    python -m crop_inference.verify_images --model-root "C:\\...\\final model" \
        --image-root "C:\\...\\drone_dataset" --n 6
"""
import argparse
from pathlib import Path

import numpy as np

from .export_state import load_reference
from .extract import PlotFeatureExtractor
from .paths import ModelPaths
from .pipeline import resolve_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", required=True, type=Path)
    ap.add_argument("--image-root", required=True, type=Path)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    mps = {v: ModelPaths(v, a.model_root) for v in ("m1", "m2")}
    ref = {v: load_reference(mp) for v, mp in mps.items()}
    raw1, s51, man = ref["m1"]
    ext = PlotFeatureExtractor({v: mp.backbone_ckpt for v, mp in mps.items()}, a.device)

    rng = np.random.RandomState(0)
    worst = {}
    for i in rng.choice(len(man), size=a.n, replace=False):
        p = [resolve_image(man.get(f"TP{t}_image_path").iloc[i], a.image_root) for t in (1, 2, 3)]
        if p[2] is None:
            print(f"skip {man['plot_id'].iloc[i]}: TP3 image not found"); continue
        got, _ = ext.extract_plots([p[0]], [p[1]], [p[2]])
        for v in ("m1", "m2"):
            for k, g in got[v].items():
                d = float(np.abs(g[0] - ref[v][0][k][i]).max())
                worst[(v, k)] = max(worst.get((v, k), 0.0), d)
        print("checked", man["plot_id"].iloc[i])
    bad = False
    for (v, k), d in sorted(worst.items()):
        tol = 1e-3 if "canopy" in k or k == "plot_areas" else 5e-2   # CNN: allow cpu/gpu + cuDNN noise
        flag = "OK " if d <= tol else "BAD"
        bad |= d > tol
        print(f"{flag} {v} {k:12s} max|diff| = {d:.3g}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
