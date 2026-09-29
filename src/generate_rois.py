"""Generate predicted, padded ROIs for the cascaded ACET experiment.

Stage-1 predictions create the boxes. Ground-truth masks are used only for the
post-hoc coverage audit and never for crop generation or threshold selection.
"""
from __future__ import annotations
import argparse, json, math, shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .config import Config
from .data import discover_lits, make_splits, PatientPatch3DDataset, load_volume, prepare_cache
from .model import get_model
from .utils import load_checkpoint
from .volume_eval import gaussian_patch_weight

@torch.no_grad()
def predict_patient(model, patient, patch_size, cache_root, cfg, device, threshold, pad_mm):
    image, label, spacing = load_volume(patient, cfg.target_z_spacing, cache_root, 256)
    target_shape = (image.shape[2], image.shape[0], image.shape[1])  # z,y,x
    ds = PatientPatch3DDataset([patient], patch_size, False, cfg.max_train_patches_per_patient, cfg.seed, cfg.target_z_spacing, cache_root, stride=tuple(max(v // 2, 1) for v in patch_size))
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    probabilities = np.zeros(target_shape, dtype=np.float32); weights = np.zeros(target_shape, dtype=np.float32)
    importance = gaussian_patch_weight(patch_size, device).cpu().numpy()
    model.eval()
    for batch in dl:
        logits = model(batch["image"].to(device))["logits"]
        # Stage 1 is a liver-plus-tumor union localizer. Both foreground
        # channels must contribute to ROI probability.
        prob = logits.softmax(1)[:, 1:3].sum(1).cpu().numpy()[0]
        z, y, x = map(int, batch["patch_start"][0].tolist()); pd, ph, pw = patch_size
        vd, vh, vw = min(pd, target_shape[0] - z), min(ph, target_shape[1] - y), min(pw, target_shape[2] - x)
        if min(vd, vh, vw) <= 0: continue
        w = importance[:vd, :vh, :vw]; probabilities[z:z+vd, y:y+vh, x:x+vw] += prob[:vd, :vh, :vw] * w; weights[z:z+vd, y:y+vh, x:x+vw] += w
    prob = probabilities / np.maximum(weights, 1e-6)
    mask = prob > threshold
    if not mask.any():
        mask[:] = True
    coords = np.argwhere(mask)
    pad_z = int(math.ceil(pad_mm / spacing[2])); pad_y = int(math.ceil(pad_mm / spacing[1])); pad_x = int(math.ceil(pad_mm / spacing[0]))
    lo = np.maximum(coords.min(0) - np.array([pad_z, pad_y, pad_x]), 0)
    hi = np.minimum(coords.max(0) + 1 + np.array([pad_z, pad_y, pad_x]), np.array(target_shape))
    box = [int(lo[0]), int(lo[1]), int(lo[2]), int(hi[0]), int(hi[1]), int(hi[2])]
    truth = np.asarray(label).transpose(2, 1, 0) == 2
    inside = truth[box[0]:box[3], box[1]:box[4], box[2]:box[5]]
    total_tumor = int(truth.sum()); covered = int(inside.sum())
    return box, {"patient_id": patient.patient_id, "roi_threshold": threshold, "pad_mm": pad_mm, "roi_box_zyx": box, "tumor_voxels": total_tumor, "tumor_voxels_inside_roi": covered, "coverage": covered / total_tumor if total_tumor else None}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--stage1-run", default="runs/acet_final/stage1"); p.add_argument("--out", default="runs/acet_final/roi_manifest.json"); p.add_argument("--data-root", default="dataset"); p.add_argument("--threshold", type=float, default=0.30); p.add_argument("--pad-mm", type=float, default=50.0); a = p.parse_args()
    stage_run = Path(a.stage1_run); cfg = Config.load(stage_run / "config.json"); patients = discover_lits(Path(a.data_root)); splits = make_splits(patients, cfg.split_seed, cfg.train_patients, cfg.val_patients, cfg.test_patients)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = get_model(1, cfg.base_channels, cfg.architecture, cfg.cerc_mode, cfg.dropout, cfg.cerc_dropout).to(device); model.set_correction_scale(1.0); load_checkpoint(stage_run / "checkpoints" / "best.pt", model, device=device); model.eval()
    patch_size = (cfg.patch_depth, cfg.patch_height, cfg.patch_width); boxes = {}; audits = {}
    for split_index, split in enumerate(splits):
        cache = Path(cfg.cache_root) if split_index < 2 else Path("cache/eval_lits_cache")
        if split_index == 2: shutil.rmtree(cache, ignore_errors=True)
        prepare_cache(split, cache, 256, cfg.target_z_spacing)
        for patient in tqdm(split, desc=("train", "val", "test")[split_index]):
            box, audit = predict_patient(model, patient, patch_size, cache, cfg, device, a.threshold, a.pad_mm); boxes[patient.patient_id] = box; audits[patient.patient_id] = audit
        if split_index == 2: shutil.rmtree(cache, ignore_errors=True)
    output = {"boxes": boxes, "audit": audits, "threshold": a.threshold, "pad_mm": a.pad_mm, "stage1_checkpoint": str(stage_run / "checkpoints" / "best.pt")}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(output, indent=2), encoding="utf-8"); print(json.dumps({"patients": len(boxes), "minimum_coverage": min(v["coverage"] for v in audits.values() if v["coverage"] is not None), "out": str(out)}, indent=2))

if __name__ == "__main__": main()
