"""
python -m crop_inference.cli --model-root "C:\\...\\final model" --plots plots.csv \
        --image-root "C:\\...\\drone_dataset" --out predictions.csv [--calibration known_yields.csv]

plots.csv columns
  required : plot_id, location, TP3_image_path
  strongly recommended : experiment, range, row, genotype   (spatial + pedigree features)
  optional : TP1_image_path, TP2_image_path, totalStandCount, plantingDate, daysToAnthesis, GDDToAnthesis
known_yields.csv (optional, >=5 plots of a field): plot_id, yield   -> adds model2_calibrated
"""
import argparse
from pathlib import Path

import pandas as pd

from .pipeline import CropYieldPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", required=True, type=Path)
    ap.add_argument("--plots", required=True, type=Path)
    ap.add_argument("--image-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("predictions.csv"))
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--m2-weights", default="full4loc", choices=["full4loc", "train"])
    a = ap.parse_args()

    plots = pd.read_csv(a.plots)
    cal = None
    if a.calibration:
        c = pd.read_csv(a.calibration)
        cal = dict(zip(c["plot_id"].astype(str), c["yield"].astype(float)))
    pipe = CropYieldPipeline(a.model_root, device=a.device, m2_weights=a.m2_weights)
    res = pipe.run(plots, image_root=a.image_root, m2_calibration=cal)
    res.to_csv(a.out, index=False)
    print(res.to_string(max_rows=20))
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
