from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from tqdm.auto import tqdm

from .data import load_volume
from .utils import empty_metric_counts, metric_counts_from_prediction, summarize_metric_counts


def gaussian_patch_weight(shape, device):
    """Return an nnU-Net-style separable 3-D importance map for patch blending."""
    axes = []
    for size in shape:
        center = (size - 1) / 2.0
        sigma = max(size * 0.125, 1.0)
        axis = torch.arange(size, device=device, dtype=torch.float32)
        axes.append(torch.exp(-0.5 * ((axis - center) / sigma) ** 2))
    weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return (weight / weight.max()).clamp_min(1e-4)


def lesion_size_metrics(prediction, target, spacing, patient_id=None):
    """Report frozen-evaluation lesion recall and local Dice by size.

    Lesion size is the equivalent spherical diameter in millimetres, computed
    from the ground-truth connected-component volume. This function is used
    only after model selection; it never changes predictions or thresholds.
    """
    structure = ndimage.generate_binary_structure(3, 2)
    labels, count = ndimage.label(target == 2, structure=structure)
    spacing = tuple(float(v) for v in spacing)
    voxel_volume = float(np.prod(spacing))
    buckets = {"small_lt_15mm": [], "medium_15_to_40mm": [], "large_gt_40mm": []}
    for lesion_id in range(1, count + 1):
        lesion = labels == lesion_id
        volume_mm3 = float(lesion.sum()) * voxel_volume
        diameter = 2.0 * (3.0 * volume_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
        name = "small_lt_15mm" if diameter < 15.0 else ("medium_15_to_40mm" if diameter <= 40.0 else "large_gt_40mm")
        # Score each ground-truth lesion inside its own tight bounding box.
        # This includes false positives around that lesion and is a true
        # lesion-local Dice, rather than the intersection-over-GT quantity.
        coords = np.argwhere(lesion)
        lo, hi = coords.min(0), coords.max(0) + 1
        region = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
        predicted_region = prediction[region] == 2
        target_region = lesion[region]
        intersection = int(np.logical_and(predicted_region, target_region).sum())
        denom = int(predicted_region.sum()) + int(target_region.sum())
        local_dice = 2.0 * intersection / denom if denom else 1.0
        buckets[name].append({"diameter_mm": diameter, "hit": bool(intersection > 0), "dice": local_dice})
    result = {}; details = []
    for name, rows in buckets.items():
        result[name] = {
            "lesions": len(rows),
            "detection_recall": sum(row["hit"] for row in rows) / len(rows) if rows else None,
            "local_dice": sum(row["dice"] for row in rows) / len(rows) if rows else None,
        }
    for name in buckets:
        for lesion_id in range(1, count + 1):
            lesion = labels == lesion_id
            volume_mm3 = float(lesion.sum()) * voxel_volume
            diameter = 2.0 * (3.0 * volume_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
            bucket = "small_lt_15mm" if diameter < 15.0 else ("medium_15_to_40mm" if diameter <= 40.0 else "large_gt_40mm")
            if bucket != name: continue
            coords = np.argwhere(lesion)
            lo, hi = coords.min(0), coords.max(0) + 1
            region = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
            pcoords_local = np.argwhere(prediction[region] == 2)
            pcoords = pcoords_local + lo if len(pcoords_local) else pcoords_local
            details.append({"patient_id": patient_id, "size_group": name, "diameter_mm": diameter,
                            "gt_voxel_bbox_zyx": [coords.min(0).tolist(), coords.max(0).tolist()],
                            "gt_centroid_zyx": coords.mean(0).tolist(), "hit": bool(len(pcoords)),
                            "intersection_voxels": int(len(pcoords)),
                            "predicted_inside_gt_bbox_zyx": [pcoords.min(0).tolist(), pcoords.max(0).tolist()] if len(pcoords) else None})
    result["details"] = details
    return result


@torch.no_grad()
def evaluate_volume_loader(model, loader, patients, patch_size, target_z_spacing, cache_root, device, criterion=None, tta=False, roi_boxes=None, target_transform=None, tumor_threshold=None):
    """Reconstruct complete volumes from overlapping patches before scoring."""
    by_id = {patient.patient_id: patient for patient in patients}
    buffers = {}; loss_total = 0.0; loss_samples = 0
    blend_weight = gaussian_patch_weight(patch_size, device).cpu().numpy()
    model.eval()

    expected_patches = {}
    if hasattr(loader, "dataset") and hasattr(loader.dataset, "index") and hasattr(loader.dataset, "patients"):
        for pi, _ in loader.dataset.index:
            pid = loader.dataset.patients[pi].patient_id
            expected_patches[pid] = expected_patches.get(pid, 0) + 1

    totals = empty_metric_counts()
    size_totals = {name: {"lesions": 0, "hits": 0, "dice_sum": 0.0} for name in ("small_lt_15mm", "medium_15_to_40mm", "large_gt_40mm")}
    lesion_details = []
    volumes_processed = 0
    seen_patches = {}

    def finalize_patient(pid):
        nonlocal volumes_processed
        if pid not in buffers:
            return
        item = buffers.pop(pid)
        # In-place normalization to avoid duplicating huge float32 arrays
        count = np.maximum(item["count"], 1e-6)
        np.divide(item["probability"], count[None, ...], out=item["probability"])
        del count
        del item["count"]

        if tumor_threshold is None:
            pred_np = np.argmax(item["probability"], axis=0).astype(np.int64)
        else:
            # Optional evaluation-time tumor threshold. Non-tumor voxels use
            # the background/liver argmax, while tumor wins at the selected
            # probability threshold. The threshold must be selected on val.
            tumor_prob = item["probability"][2]
            non_tumor = np.argmax(item["probability"][:2], axis=0).astype(np.int64)
            pred_np = np.where(tumor_prob >= float(tumor_threshold), 2, non_tumor).astype(np.int64)
        del item["probability"]

        prediction = torch.from_numpy(pred_np).unsqueeze(0)
        target = torch.from_numpy(item["target"]).long().unsqueeze(0)
        eval_target = target_transform(target) if target_transform else target
        # Stage 1 evaluates the liver-plus-tumor union as binary foreground;
        # predicted classes 1 and 2 must both count as foreground.
        if target_transform:
            prediction = (prediction > 0).long()
        counts = metric_counts_from_prediction(prediction, eval_target)
        for key, value in counts.items():
            totals[key] += value
        for name, metrics in lesion_size_metrics(pred_np, item["target"], item["spacing"], item.get("patient_id")).items():
            if name == "details":
                lesion_details.extend(metrics)
                continue
            if metrics["lesions"]:
                size_totals[name]["lesions"] += metrics["lesions"]
                size_totals[name]["hits"] += int(round(metrics["detection_recall"] * metrics["lesions"]))
                size_totals[name]["dice_sum"] += metrics["local_dice"] * metrics["lesions"]
        volumes_processed += 1

    for batch in tqdm(loader, desc="Val volume reconstruction", dynamic_ncols=True, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        if tta:
            predictions = []
            for dims in ((), (3,), (4,), (3, 4)):
                augmented = torch.flip(images, dims=list(dims)) if dims else images
                logits = model(augmented)["logits"]
                predictions.append(torch.flip(logits, dims=list(dims)) if dims else logits)
            probabilities = torch.stack([p.softmax(1) for p in predictions]).mean(0).cpu().numpy()
        else:
            outputs = model(images)
            probabilities = outputs["logits"].softmax(1).cpu().numpy()
        if criterion is not None and not tta:
            loss_total += float(criterion(outputs, targets).detach()) * images.shape[0]
            loss_samples += images.shape[0]
        starts = batch["patch_start"].cpu().numpy()
        for i, patient_id in enumerate(batch["patient_id"]):
            if patient_id not in buffers:
                _, label, spacing = load_volume(by_id[patient_id], target_z_spacing, cache_root, 256)
                target = np.asarray(label).transpose(2, 0, 1).astype(np.int64)
                buffers[patient_id] = {
                    "probability": np.zeros((3,) + target.shape, dtype=np.float32),
                    "count": np.zeros(target.shape, dtype=np.float32),
                    "target": target,
                    "spacing": tuple(float(v) for v in spacing),
                    "patient_id": patient_id,
                }
            item = buffers[patient_id]
            z, y, x = [int(v) for v in starts[i]]
            depth, height, width = item["target"].shape
            pd, ph, pw = patch_size
            vd, vh, vw = min(pd, depth - z), min(ph, height - y), min(pw, width - x)
            if min(vd, vh, vw) > 0:
                weight = blend_weight[:vd, :vh, :vw]
                item["probability"][:, z:z+vd, y:y+vh, x:x+vw] += probabilities[i, :, :vd, :vh, :vw] * weight[None]
                item["count"][z:z+vd, y:y+vh, x:x+vw] += weight

            seen_patches[patient_id] = seen_patches.get(patient_id, 0) + 1
            if patient_id in expected_patches and seen_patches[patient_id] == expected_patches[patient_id]:
                finalize_patient(patient_id)

    for pid in list(buffers.keys()):
        finalize_patient(pid)

    result = summarize_metric_counts(totals)
    # Retained for exact streaming aggregation by evaluate.py; removed from
    # the public summary only after all patients have been combined.
    result["_metric_counts"] = totals
    result["loss"] = loss_total / max(loss_samples, 1) if criterion is not None else None
    result["volumes"] = volumes_processed
    result["lesion_size_metrics"] = {
        name: {
            "lesions": value["lesions"],
            "detection_recall": value["hits"] / value["lesions"] if value["lesions"] else None,
            "local_dice": value["dice_sum"] / value["lesions"] if value["lesions"] else None,
        }
        for name, value in size_totals.items()
    }
    result["lesion_details"] = lesion_details
    return result
