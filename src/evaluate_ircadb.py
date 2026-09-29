"""Zero-shot 3DIRCADb evaluation.

This script never creates an optimizer, never updates model weights, and never
uses IRCADb to select thresholds or hyperparameters.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .config import Config
from .data import discover_prepared_cohort, PatientPatch3DDataset
from .model import get_model
from .utils import load_checkpoint
from .volume_eval import evaluate_volume_loader


def aggregate_results(rows):
    """Aggregate scalar metrics and lesion-size metrics without averaging lists/dicts."""
    summary = {}
    scalar_keys = [
        key for key, value in rows[0].items()
        if key != "patient_id" and isinstance(value, (int, float))
    ]
    for key in scalar_keys:
        summary[key] = sum(float(row[key]) for row in rows) / len(rows)

    groups = sorted({
        group
        for row in rows
        for group in row.get("lesion_size_metrics", {})
    })
    size_summary = {}
    for group in groups:
        entries = [row.get("lesion_size_metrics", {}).get(group) for row in rows]
        entries = [entry for entry in entries if isinstance(entry, dict)]
        total_lesions = sum(int(entry.get("lesions", 0)) for entry in entries)
        if not entries:
            continue
        if total_lesions == 0:
            size_summary[group] = {
                "lesions": 0,
                "detection_recall": None,
                "local_dice": None,
            }
            continue
        denominator = max(total_lesions, 1)
        size_summary[group] = {
            "lesions": total_lesions,
            "detection_recall": sum(
                float(entry.get("detection_recall") or 0.0) * int(entry.get("lesions", 0))
                for entry in entries
            ) / denominator,
            "local_dice": sum(
                float(entry.get("local_dice") or 0.0) * int(entry.get("lesions", 0))
                for entry in entries
            ) / denominator,
        }
    summary["lesion_size_metrics"] = size_summary
    return summary

@torch.no_grad()
def evaluate_patient(model, patient, cfg, device, batch_size, tta=False, tumor_threshold=None):
    patch_size=(cfg.patch_depth,cfg.patch_height,cfg.patch_width)
    ds=PatientPatch3DDataset([patient],patch_size,False,cfg.max_train_patches_per_patient,cfg.seed,cfg.target_z_spacing,cache_root=None,stride=tuple(max(v // 2, 1) for v in patch_size))
    dl=DataLoader(ds,batch_size=batch_size,shuffle=False)
    return evaluate_volume_loader(
        model, dl, [patient], patch_size, cfg.target_z_spacing, None, device,
        tta=tta, tumor_threshold=tumor_threshold,
    )

def main():
    p=argparse.ArgumentParser(description="Strict zero-shot evaluation on 3DIRCADb-01")
    p.add_argument("--run",required=True); p.add_argument("--prepared-root",default="prepared/3Dircadb1"); p.add_argument("--checkpoint",default="best.pt"); p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--tta", action="store_true", help="average original, x-flip, y-flip, and xy-flip predictions")
    p.add_argument("--tumor-threshold", type=float, default=None, help="frozen tumor probability threshold selected on LiTS validation")
    a=p.parse_args()
    run=Path("runs")/a.run; cfg=Config.load(run/"config.json"); patients=discover_prepared_cohort(Path(a.prepared_root));
    if len(patients)!=20: raise RuntimeError(f"Expected 20 IRCADb patients, found {len(patients)}")
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=get_model(1,cfg.base_channels,cfg.architecture,cfg.cerc_mode,cfg.dropout,cfg.cerc_dropout).to(dev); model.set_correction_scale(1.0); load_checkpoint(run/"checkpoints"/a.checkpoint,model,device=dev); model.eval()
    rows=[]
    for patient in patients:
        r=evaluate_patient(model,patient,cfg,dev,a.batch_size,tta=a.tta,tumor_threshold=a.tumor_threshold); rows.append({"patient_id":patient.patient_id,**r}); print(patient.patient_id,r)
    summary = aggregate_results(rows)
    summary["tta"] = bool(a.tta)
    summary["tumor_threshold"] = a.tumor_threshold
    out=run/"metrics"; out.mkdir(exist_ok=True); (out/"ircadb_zero_shot_summary.json").write_text(json.dumps({"cohort":"3DIRCADb-01","patients":len(rows),"checkpoint":str(a.checkpoint),"summary":summary},indent=2),encoding="utf-8")
    with (out/"ircadb_zero_shot_per_patient.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
