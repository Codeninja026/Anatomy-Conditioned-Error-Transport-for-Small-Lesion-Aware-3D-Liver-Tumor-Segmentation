"""Verify LiTS patient split, cache contents, shapes, intensities, and overlays."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
from .data import discover_lits, make_splits, load_volume

def main():
    p = argparse.ArgumentParser(description="Audit the LiTS cache without training")
    p.add_argument("--data-root", default="dataset"); p.add_argument("--cache-root", default="cache/lits_cache"); p.add_argument("--out", default="cache_check"); p.add_argument("--samples", type=int, default=3); p.add_argument("--seed", type=int, default=42); a = p.parse_args()
    patients = discover_lits(Path(a.data_root)); train, val, test = make_splits(patients, a.seed, 83, 28, 20); cache = Path(a.cache_root); expected = train + val; missing = []; records = []
    for patient in expected:
        try:
            image, label, spacing = load_volume(patient, 1.5, cache, 256)
            if tuple(image.shape) != tuple(label.shape) or image.shape[:2] != (256, 256): raise ValueError(f"shape mismatch image={image.shape} label={label.shape}")
            if not np.isfinite(np.asarray(image)).all(): raise ValueError("non-finite intensity")
            records.append({"patient_id": patient.patient_id, "shape": list(image.shape), "spacing": list(spacing), "min": float(np.min(image)), "max": float(np.max(image)), "foreground_voxels": int(np.count_nonzero(np.asarray(label) > 0))})
        except Exception as error: missing.append({"patient_id": patient.patient_id, "error": str(error)})
    cached_ids = {p.stem.replace("_image", "") for p in cache.glob("*_image.npy")}; expected_ids = {p.patient_id for p in expected}; unexpected = sorted(cached_ids - expected_ids)
    if missing or unexpected: raise RuntimeError(json.dumps({"missing_or_invalid": missing, "unexpected_cached_patients": unexpected}, indent=2))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rng = random.Random(a.seed); chosen = rng.sample(expected, min(a.samples, len(expected)))
    try:
        import matplotlib.pyplot as plt
        for patient in chosen:
            image, label, _ = load_volume(patient, 1.5, cache, 256); label_array = np.asarray(label); z = int(np.argmax(np.sum(label_array > 0, axis=(0, 1))))
            fig, axes = plt.subplots(1, 2, figsize=(8, 4)); axes[0].imshow(np.asarray(image)[:, :, z].T, cmap="gray", origin="lower"); axes[0].set_title(f"{patient.patient_id} CT"); axes[1].imshow(np.asarray(image)[:, :, z].T, cmap="gray", origin="lower"); axes[1].imshow(np.ma.masked_where(label_array[:, :, z].T == 0, label_array[:, :, z].T), alpha=.45, vmin=1, vmax=2, origin="lower"); axes[1].set_title("Label overlay")
            for axis in axes: axis.axis("off")
            fig.tight_layout(); fig.savefig(out / f"{patient.patient_id}_overlay.png", dpi=150); plt.close(fig)
    except ImportError: print("matplotlib unavailable; skipped overlay images")
    report = {"patients_discovered": len(patients), "split_counts": {"train": len(train), "val": len(val), "test": len(test)}, "cache_patients_required": len(expected), "cache_patients_found": len(cached_ids), "sample_records": records[:min(5, len(records))], "overlay_dir": str(out)}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
