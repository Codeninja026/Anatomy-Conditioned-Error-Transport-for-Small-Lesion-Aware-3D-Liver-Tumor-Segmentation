from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class Config:
    data_root: str = "dataset"
    output_root: str = "runs"
    cache_root: str = "cache/lits_cache"
    seed: int = 42
    epochs: int = 100
    batch_size: int = 2
    workers: int = 0
    lr: float = 2e-4
    weight_decay: float = 1e-4
    dropout: float = 0.10
    cerc_dropout: float = 0.20
    cerc_warmup_epochs: int = 5
    patch_depth: int = 32
    patch_height: int = 128
    patch_width: int = 128
    target_z_spacing: float = 1.5
    base_channels: int = 32
    architecture: str = "acet"
    stage1: bool = False
    roi_manifest: str = ""
    drive_folder_id: str = ""
    drive_path: str = ""
    drive_sync_every: int = 1
    cerc_mode: str = "signed"
    loss_mode: str = "standard"
    anatomy_loss_weight: float = 0.1
    tversky_weight: float = 0.5
    # Mild recall bias for faint/small lesions; anatomy gating controls FP.
    tversky_fp_weight: float = 0.40
    tversky_fn_weight: float = 0.60
    tversky_gamma: float = 0.75
    lr_warmup_epochs: int = 5
    min_lr: float = 1e-6
    litra_tail_weight: float = 0.25
    litra_rho: float = 0.25
    max_train_patches_per_patient: int = 64
    instance_balanced: bool = True
    boundary_focus_weight: float = 0.20
    amp: bool = True
    deterministic: bool = False
    grad_accum: int = 1
    val_every: int = 1
    save_every: int = 5
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    selection_dice_weight: float = 0.50
    selection_precision_weight: float = 0.25
    selection_liver_weight: float = 0.25
    split_seed: int = 42
    # Matches the base paper's 111 train/validation and 20 test patients;
    # 111 is rounded to an integer 83/28 split for the stated 3:1 ratio.
    train_patients: int = 83
    val_patients: int = 28
    test_patients: int = 20
    ablation: str = "full"

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        return cls(**json.loads(path.read_text(encoding="utf-8")))
