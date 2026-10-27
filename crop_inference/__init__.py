"""Two-model crop-yield inference pipeline (Model 1 in-domain + Model 2 zero-shot site)."""


def __getattr__(name):  # lazy: keeps `export_state` usable without torch installed
    if name == "CropYieldPipeline":
        from .pipeline import CropYieldPipeline
        return CropYieldPipeline
    raise AttributeError(name)
