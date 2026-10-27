"""File layout of the training outputs (relative to --model-root, the folder that
contains Model1_All5Loc_InDomain/ and Model2_4Loc_LeaveOutCrawfordsville/)."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelPaths:
    variant: str          # "m1" | "m2"
    root: Path

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root))

    @property
    def base(self) -> Path:
        return self.root / ("Model1_All5Loc_InDomain" if self.variant == "m1"
                            else "Model2_4Loc_LeaveOutCrawfordsville")

    @property
    def _pre(self) -> str:
        return "m1_" if self.variant == "m1" else ""

    @property
    def raw_cache(self) -> Path:
        return self.base / "step4_feature_engineering" / "cache" / f"{self._pre}step4_raw_multimodal_features.npz"

    @property
    def backbone_ckpt(self) -> Path:
        return self.base / "step4_feature_engineering" / "cache" / f"{self._pre}step4_finetuned_effnet_b0.pth"

    @property
    def step5_npz(self) -> Path:
        return self.base / "step5_feature_selection" / f"{self._pre}step5_selected_features.npz"

    @property
    def model_bundle(self) -> Path:
        return self.base / "step6_xgboost_tuning" / f"{self._pre}step6_tuned_models.pkl"

    @property
    def split_dir(self) -> Path:
        return self.base / "step1_datasplitting"

    @property
    def state_file(self) -> Path:
        return self.root / "inference_state" / f"state_{self.variant}.pkl"
