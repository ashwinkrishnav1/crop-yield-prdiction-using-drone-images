# crop_inference — two-model yield inference

```
plots.csv + TP1/TP2/TP3 images ──► EfficientNet-B0 (M1 weights) ┐
                                 └► EfficientNet-B0 (M2 weights) ┤  + RGB canopy indices (shared)
                                                                  ▼
                    frozen state (flight-norm, pedigree maps, prototypes, ridge, scalers)
                                                                  ▼
            Model 1 (3-view stack, per plot)        Model 2 (ranker + physical anchor, per FIELD)
                       └──────────────► model1_yield, model2_yield, recommended_yield, flags
```

## Setup (once)
Put this folder next to your project and run from its parent directory.

```
python -m crop_inference.export_state --model-root "C:\...\final model" --gt-csv "C:\...\GroundTruth\HYBRID_HIPS_V3.5_ALLPLOTS.csv"
python -m crop_inference.verify_images --model-root "C:\...\final model" --image-root "C:\...\drone_dataset" --n 6
```
`export_state` rebuilds every fitted object the training scripts never saved and **refuses to write
state unless it reproduces your stored feature matrices** (held-out plots, rel. error < 0.5 %).
`--gt-csv` supplies daysToAnthesis / GDDToAnthesis; without it those 2 features are neutral (flagged).

## Run
```
python -m crop_inference.cli --model-root "C:\...\final model" --plots plots.csv --image-root "C:\...\drone_dataset" --out predictions.csv
```
or in Python: `CropYieldPipeline(model_root).run(df, image_root=...)`.

**plots.csv** — required: `plot_id, location, TP3_image_path`. Strongly recommended: `experiment, range, row, genotype`.
Optional: `TP1_image_path, TP2_image_path, totalStandCount, plantingDate, daysToAnthesis, GDDToAnthesis`.
Missing values are imputed exactly as in training and listed in `flags`.
Optional `--calibration yields.csv` (`plot_id,yield`, ≥5 plots of one field) adds `model2_calibrated`.

## What the models really need (why "one image in" is not enough)
* 3 timepoints (missing TP2/TP1 are copied from TP3/TP2, as in training) and plot metadata.
* Model 1 is per plot but only valid for the 5 trained locations (location mean is added back as an offset;
  pedigree/experiment encodings and spatial neighbours come from that location's training plots).
* Model 2 output = field anchor + α·(batch-centred ranking). It is **batch-level**: send a whole field
  (≥30 plots) per location. One plot collapses to its physical-model value (flagged).
  For a location it has never seen it needs ≥5 plots in the batch to estimate label-free field statistics.
* `recommended_yield` = Model 1 if the location is known to it, otherwise Model 2 (policy, easy to change in `pipeline.py`).

## Verified here vs. needs your machine
| stage | status |
|---|---|
| tabular features (M1 111/113 cols + 140 phys; M2 75/77 cols) | rebuilt from state = stored matrices, max abs 2.4e-7 |
| blends vs your saved prediction CSVs | max abs diff 7e-6 (M1) / 2e-5 (M2) with stored X |
| pipeline end-to-end on cached raw features | M1 ≤0.25 bu/ac, M2 ≤0.19 bu/ac from saved (gap = missing phenology) |
| phenology columns (2 per model) | need `--gt-csv` |
| image → raw features (patches, canopy, EfficientNet) | copied verbatim; confirm with `verify_images` |
