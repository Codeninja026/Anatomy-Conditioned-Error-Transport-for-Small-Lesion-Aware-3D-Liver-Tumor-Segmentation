from __future__ import annotations
import argparse, json, shutil, gc
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .config import Config
from .data import discover_lits, make_splits, PatientPatch3DDataset, prepare_cache
from .model import get_model
from .utils import load_checkpoint, empty_metric_counts, summarize_metric_counts
from .volume_eval import evaluate_volume_loader


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--split", choices=("val", "test"), default="test"); p.add_argument("--checkpoint", default="best.pt"); p.add_argument("--data-root", default="dataset"); p.add_argument("--batch-size", type=int, default=1); p.add_argument("--tta", action="store_true", help="average original, x-flip, y-flip, and xy-flip predictions"); p.add_argument("--tumor-threshold", type=float, default=None, help="optional tumor probability threshold; select on validation only before test evaluation"); a = p.parse_args()
    run = Path("runs") / a.run; cfg = Config.load(run / "config.json"); patients = discover_lits(Path(a.data_root)); splits = make_splits(patients, cfg.split_seed, cfg.train_patients, cfg.val_patients, cfg.test_patients)
    roi_boxes = {}
    if getattr(cfg, "roi_manifest", ""):
        manifest_path = Path(cfg.roi_manifest)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            roi_boxes = manifest.get("boxes", manifest)
    patch_size = (cfg.patch_depth, cfg.patch_height, cfg.patch_width); patients_for_eval = splits[1 if a.split == "val" else 2]
    target_transform = (lambda target: (target > 0).long()) if cfg.stage1 else None
    # Test data are never placed in the training cache. For low-disk systems,
    # evaluate one patient at a time and remove only this temporary cache.
    eval_cache = Path(cfg.cache_root) if a.split == "val" else Path("cache/eval_lits_cache")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(1, cfg.base_channels, cfg.architecture, cfg.cerc_mode, cfg.dropout, cfg.cerc_dropout).to(device)
    model.set_correction_scale(1.0)
    load_checkpoint(run / "checkpoints" / a.checkpoint, model, device=device)
    model.eval()
    if a.split == "test":
        combined = empty_metric_counts(); size_rows = {k: {"lesions": 0, "hits": 0, "dice_sum": 0.0} for k in ("small_lt_15mm", "medium_15_to_40mm", "large_gt_40mm")}; lesion_details = []; volumes = 0
        for patient in patients_for_eval:
            prepare_cache([patient], eval_cache, 256, cfg.target_z_spacing)
            ds = PatientPatch3DDataset([patient], patch_size, False, cfg.max_train_patches_per_patient, cfg.seed, cfg.target_z_spacing, eval_cache, stride=tuple(max(v // 2, 1) for v in patch_size), roi_boxes=roi_boxes)
            one = evaluate_volume_loader(model, DataLoader(ds, batch_size=a.batch_size), [patient], patch_size, cfg.target_z_spacing, eval_cache, device, tta=a.tta, roi_boxes=roi_boxes, target_transform=target_transform, tumor_threshold=a.tumor_threshold)
            for key, value in one["_metric_counts"].items(): combined[key] += value
            for name, row in one["lesion_size_metrics"].items():
                if row["lesions"]:
                    size_rows[name]["lesions"] += row["lesions"]; size_rows[name]["hits"] += round(row["detection_recall"] * row["lesions"]); size_rows[name]["dice_sum"] += row["local_dice"] * row["lesions"]
            lesion_details.extend(one.get("lesion_details", []))
            volumes += one["volumes"]
            shutil.rmtree(eval_cache, ignore_errors=True)
            del ds, one
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result = summarize_metric_counts(combined); result["volumes"] = volumes
        result["lesion_size_metrics"] = {k: {"lesions": v["lesions"], "detection_recall": v["hits"] / v["lesions"] if v["lesions"] else None, "local_dice": v["dice_sum"] / v["lesions"] if v["lesions"] else None} for k, v in size_rows.items()}
        result["lesion_details"] = lesion_details
    else:
        prepare_cache(patients_for_eval, eval_cache, 256, cfg.target_z_spacing)
        ds = PatientPatch3DDataset(patients_for_eval, patch_size, False, cfg.max_train_patches_per_patient, cfg.seed, cfg.target_z_spacing, eval_cache, stride=tuple(max(v // 2, 1) for v in patch_size), roi_boxes=roi_boxes); dl = DataLoader(ds, batch_size=a.batch_size)
        result = evaluate_volume_loader(model, dl, patients_for_eval, patch_size, cfg.target_z_spacing, eval_cache, device, tta=a.tta, roi_boxes=roi_boxes, target_transform=target_transform, tumor_threshold=a.tumor_threshold)
        result.pop("_metric_counts", None)
    result.pop("_metric_counts", None)
    result["tta"] = bool(a.tta); result["tumor_threshold"] = a.tumor_threshold
    (run / "metrics").mkdir(exist_ok=True); (run / f"metrics/{a.split}_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
