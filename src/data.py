from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
import json
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom, binary_erosion, label as connected_components
from tqdm.auto import tqdm

@dataclass(frozen=True)
class Patient:
    patient_id: str
    image: Path
    label: Path
    cohort: str = "LiTS"

def discover_lits(root: Path) -> list[Patient]:
    root = Path(root)
    if (root / "Lits").exists():
        root = root / "Lits"
    elif (root / "lits").exists():
        root = root / "lits"
    pairs = []
    for image in sorted(root.rglob("volume-*.nii")):
        idx = image.stem.split("-")[-1]
        label = image.with_name(f"segmentation-{idx}.nii")
        if not label.exists():
            raise FileNotFoundError(f"Missing label for {image}")
        pairs.append(Patient(f"LiTS-{int(idx):03d}", image, label))
    if len(pairs) != 131:
        raise ValueError(f"Expected 131 LiTS pairs, found {len(pairs)} under {root}")
    return pairs

def discover_prepared_cohort(root: Path, cohort="3DIRCADb") -> list[Patient]:
    pairs=[]
    for image in sorted(root.rglob("volume-*.nii.gz")):
        label=image.with_name(image.name.replace("volume-", "segmentation-"))
        if not label.exists(): raise FileNotFoundError(f"Missing label for {image}")
        pairs.append(Patient(f"{cohort}-{image.parent.name}", image, label, cohort))
    if not pairs: raise ValueError(f"No prepared {cohort} NIfTI pairs found under {root}")
    return pairs

def assert_training_cohort(patients: list[Patient]):
    cohorts={p.cohort for p in patients}
    if cohorts != {"LiTS"}: raise RuntimeError(f"Training may only use LiTS; found {sorted(cohorts)}")

def prepare_cache(patients: list[Patient], cache_root: Path, image_size=256, target_z_spacing=1.5):
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    for patient in tqdm(patients, desc="Preparing LiTS cache", unit="patient"):
        load_volume(patient, target_z_spacing, cache_root, image_size)

def make_splits(patients: list[Patient], seed=42, n_train=83, n_val=28, n_test=20):
    if len({p.patient_id for p in patients}) != len(patients):
        raise ValueError("Duplicate patient IDs detected before splitting")
    if any(p.cohort != "LiTS" for p in patients):
        raise ValueError("make_splits is reserved for LiTS patient partitions")
    ids = list(range(len(patients)))
    random.Random(seed).shuffle(ids)
    if n_train + n_val + n_test != len(ids):
        raise ValueError("Split counts do not cover all patients")
    return ([patients[i] for i in ids[:n_train]],
            [patients[i] for i in ids[n_train:n_train+n_val]],
            [patients[i] for i in ids[n_train+n_val:]])

def save_manifest(root: Path, splits):
    root.mkdir(parents=True, exist_ok=True)
    for name, patients in zip(("train", "val", "test"), splits):
        (root / f"{name}.txt").write_text("\n".join(p.patient_id for p in patients) + "\n", encoding="utf-8")
    all_ids = [p.patient_id for s in splits for p in s]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Patient leakage detected in split manifest")

def _cache_paths(cache_root: Path, patient: Patient):
    key = patient.patient_id.replace("/", "_").replace("\\", "_")
    if (cache_root / f"{key}_image.npy").exists():
        return cache_root / f"{key}_image.npy", cache_root / f"{key}_label.npy", cache_root / f"{key}.json"
    if (cache_root / "lits_cache" / f"{key}_image.npy").exists():
        sub = cache_root / "lits_cache"
        return sub / f"{key}_image.npy", sub / f"{key}_label.npy", sub / f"{key}.json"
    return cache_root / f"{key}_image.npy", cache_root / f"{key}_label.npy", cache_root / f"{key}.json"

def _resample_z_proxy(proxy, shape, factor, order, dtype):
    """Resample only z one output plane at a time to cap evaluation RAM."""
    source_depth = int(shape[2]); target_depth = max(1, int(round(source_depth * factor)))
    out = np.empty((int(shape[0]), int(shape[1]), target_depth), dtype=dtype)
    coordinates = np.linspace(0.0, source_depth - 1.0, target_depth)
    for zi, coordinate in enumerate(coordinates):
        if order == 0:
            out[:, :, zi] = np.asarray(proxy[:, :, int(round(float(coordinate)))], dtype=dtype)
        else:
            lo = int(np.floor(coordinate)); hi = min(lo + 1, source_depth - 1); alpha = float(coordinate - lo)
            first = np.asarray(proxy[:, :, lo], dtype=np.float32)
            if hi == lo:
                out[:, :, zi] = first
            else:
                second = np.asarray(proxy[:, :, hi], dtype=np.float32)
                out[:, :, zi] = first + (second - first) * alpha
    return out

def load_volume(patient: Patient, target_z_spacing: float | None = 1.5, cache_root: Path | None = None, image_size: int = 256):
    if cache_root is not None:
        image_path, label_path, meta_path = _cache_paths(cache_root, patient)
        if image_path.exists() and label_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return np.load(image_path, mmap_mode="r"), np.load(label_path, mmap_mode="r"), tuple(meta["spacing"])
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    image_nii = nib.load(str(patient.image)); label_nii = nib.load(str(patient.label))
    shape = tuple(int(v) for v in image_nii.shape)
    if shape != tuple(int(v) for v in label_nii.shape):
        raise ValueError(f"Shape mismatch for {patient.patient_id}: {shape} vs {tuple(int(v) for v in label_nii.shape)}")
    if len(shape) != 3:
        raise ValueError(f"Expected 3D volume for {patient.patient_id}")
    spacing = tuple(float(x) for x in image_nii.header.get_zooms()[:3])
    original_xy = shape[:2]
    if target_z_spacing is not None and abs(spacing[2] - target_z_spacing) > 1e-3:
        factor = spacing[2] / target_z_spacing
        image = _resample_z_proxy(image_nii.dataobj, shape, factor, 1, np.float32)
        label = _resample_z_proxy(label_nii.dataobj, shape, factor, 0, np.int16)
        spacing = (spacing[0], spacing[1], target_z_spacing)
    else:
        image = np.asarray(image_nii.get_fdata(dtype=np.float32))
        label = np.asarray(label_nii.get_fdata(dtype=np.float32), dtype=np.int16)
    image=window_ct(image)
    image=resize_xy(image, image_size, 1)
    label=resize_xy(label, image_size, 0).astype(np.int64)
    spacing=(spacing[0]*original_xy[0]/image_size, spacing[1]*original_xy[1]/image_size, spacing[2])
    if cache_root is not None:
        np.save(image_path, np.ascontiguousarray(image, dtype=np.float32))
        np.save(label_path, np.ascontiguousarray(label, dtype=np.int16))
        meta_path.write_text(json.dumps({"patient_id":patient.patient_id,"spacing":spacing,"shape":list(image.shape)}),encoding="utf-8")
    return image, label.astype(np.int64), spacing

def window_ct(x, lo=-100.0, hi=250.0):
    # In-place preprocessing avoids allocating another 300–500 MB array for
    # large test volumes during evaluation-cache construction.
    np.clip(x, lo, hi, out=x)
    x -= lo
    x /= (hi - lo)
    return x

def resize_xy(x, size, order):
    if x.shape[:2] == (size, size):
        return x
    factors = (size / x.shape[0], size / x.shape[1], 1.0)
    return zoom(x, factors, order=order)

def boundary_target(mask):
    from scipy.ndimage import binary_dilation, binary_erosion
    tumor = mask == 2
    return (binary_dilation(tumor, iterations=1) ^ binary_erosion(tumor, iterations=1)).astype(np.float32)

class PatientSlabDataset(Dataset):
    """Patient-level dataset; slabs are generated on demand, never pre-sliced."""
    def __init__(self, patients, image_size=256, context=5, train=False, max_slabs=24, tumor_prob=.75, seed=42, target_z_spacing=1.5, cache_root=None):
        self.patients, self.image_size, self.context, self.target_z_spacing, self.cache_root = patients, image_size, context, target_z_spacing, Path(cache_root) if cache_root else None
        self.train, self.max_slabs, self.tumor_prob, self.seed = train, max_slabs, tumor_prob, seed
        self._cache = {}
        self.index = []
        self.original_depths = []
        radius = context // 2
        for pi, patient in enumerate(tqdm(self.patients, desc="Indexing patients", unit="patient")):
            _, lab, _ = load_volume(patient, self.target_z_spacing, self.cache_root, self.image_size)
            if lab.ndim != 3: raise ValueError(f"Expected 3D label for {patient.patient_id}")
            self.original_depths.append(lab.shape[2])
            zs = np.where(np.any(lab == 2, axis=(0, 1)))[0].tolist()
            if not zs:
                zs = list(range(lab.shape[2]))
            if train:
                if len(zs) > max_slabs:
                    chosen = np.asarray(zs)[np.linspace(0, len(zs)-1, max_slabs, dtype=int)].tolist()
                else:
                    uniform = np.linspace(0, lab.shape[2]-1, min(max_slabs, lab.shape[2]), dtype=int).tolist()
                    chosen = sorted(set(zs + uniform))
                self.index.extend((pi, int(z)) for z in chosen)
            else:
                self.index.extend((pi, z) for z in range(radius, lab.shape[2]-radius))

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        pi, z = self.index[i]
        if pi not in self._cache:
            if len(self._cache) >= 2:
                self._cache.pop(next(iter(self._cache)))
            self._cache[pi] = load_volume(self.patients[pi], self.target_z_spacing, self.cache_root, self.image_size)
        image, label, spacing = self._cache[pi]
        original_depth = self.original_depths[pi]
        z = int(round(z * max(image.shape[2] - 1, 1) / max(original_depth - 1, 1)))
        radius = self.context // 2
        zs = [min(max(z + k, 0), image.shape[2]-1) for k in range(-radius, radius+1)]
        x = np.asarray(image[:, :, zs]).transpose(2, 0, 1)
        y = np.asarray(label[:, :, z]).copy()
        y = np.clip(y, 0, 2).astype(np.int64)
        if self.train:
            rng = np.random.default_rng(self.seed + i)
            if rng.random() < .5:
                x, y = x[:, :, ::-1].copy(), y[:, ::-1].copy()
            if rng.random() < .5:
                x, y = x[:, ::-1, :].copy(), y[::-1, :].copy()
            if rng.random() < .35:
                k = int(rng.integers(1, 4))
                x, y = np.rot90(x, k=k, axes=(1, 2)).copy(), np.rot90(y, k=k, axes=(0, 1)).copy()
            if rng.random() < .30:
                x = np.clip(x * float(rng.uniform(.90, 1.10)) + float(rng.uniform(-.05, .05)), 0.0, 1.0)
            if rng.random() < .20:
                x = np.clip(x + rng.normal(0.0, .02, size=x.shape).astype(np.float32), 0.0, 1.0)
        return {"image": torch.from_numpy(x.copy()).float(), "mask": torch.from_numpy(y.copy()).long(),
                "boundary": torch.from_numpy(boundary_target(y)), "patient_id": self.patients[pi].patient_id,
                "slice_index": z, "spacing": torch.tensor(spacing, dtype=torch.float32)}


class PatientPatch3DDataset(Dataset):
    """Patient-level 3D patch dataset with tumor-aware training sampling."""
    def __init__(self, patients, patch_size=(32, 128, 128), train=False, max_patches=64, seed=42, target_z_spacing=1.5, cache_root=None, stride=None, instance_balanced=True, roi_boxes=None, localizer=False):
        self.patients, self.patch_size = patients, tuple(patch_size)
        self.stride = tuple(stride) if stride is not None else tuple(max(v // 2, 1) for v in self.patch_size)
        self.train, self.max_patches, self.seed, self.epoch = train, max_patches, seed, 0
        self.target_z_spacing, self.cache_root, self.instance_balanced, self.localizer = target_z_spacing, Path(cache_root) if cache_root else None, instance_balanced, localizer
        self.roi_boxes = roi_boxes or {}
        self._cache, self.index = {}, []
        for pi, patient in enumerate(tqdm(patients, desc="Indexing 3D patches", unit="patient")):
            _, label, spacing = load_volume(patient, target_z_spacing, self.cache_root, 256)
            box = self.roi_boxes.get(patient.patient_id)
            if box:
                z0, y0, x0, z1, y1, x1 = map(int, box)
                label_for_index = label[y0:y1, x0:x1, z0:z1]
            else:
                z0, y0, x0 = 0, 0, 0; label_for_index = label
            depth, height, width = label_for_index.shape[2], label_for_index.shape[0], label_for_index.shape[1]
            starts = self._starts((depth, height, width), self.stride)
            tumor_voxels = np.argwhere(label_for_index == 2)
            if train and self.localizer:
                # Stage 1 must learn the whole liver/union field and its
                # boundary. Do not inherit Stage-2 lesion/capsule mining.
                # Evenly subsample the complete patch grid per patient.
                step = max(len(starts) // max(max_patches, 1), 1)
                starts = starts[::step][:max_patches]
            elif train and len(tumor_voxels):
                rng = np.random.default_rng(seed + pi)
                extra = []
                # Label-only, patient-local curriculum: 50% lesion-instance
                # centers, 30% liver-capsule hard negatives, 20% uniform.
                # This cannot leak validation/test information because this
                # dataset is constructed only from the already-selected
                # training patients.
                n_tumor = int(round(max_patches * 0.50))
                n_margin = int(round(max_patches * 0.30))
                n_uniform = max_patches - n_tumor - n_margin
                if self.instance_balanced:
                    components, component_count = connected_components(label_for_index == 2, structure=np.ones((3, 3, 3), dtype=np.uint8))
                    instances = [np.argwhere(components == component_id) for component_id in range(1, component_count + 1)]
                    instances = [points for points in instances if len(points)]
                else:
                    instances = [tumor_voxels]
                if self.instance_balanced:
                    groups = {"small": [], "medium": [], "large": []}
                    voxel_volume = float(np.prod(spacing))
                    for points_candidate in instances:
                        diameter = 2.0 * (3.0 * len(points_candidate) * voxel_volume / (4.0 * np.pi)) ** (1.0 / 3.0)
                        group = "small" if diameter < 15.0 else ("medium" if diameter <= 40.0 else "large")
                        groups[group].append(points_candidate)
                    active_groups = [rows for rows in groups.values() if rows]
                    # Explicit small-lesion prior for Stage 2. This is applied
                    # only to already-selected training patients; validation
                    # and test labels never enter this sampler. Renormalize
                    # over groups present in the patient so patients without
                    # large lesions do not lose sampling mass.
                    group_prior = {"small": 0.65, "medium": 0.25, "large": 0.10}
                    active_named = [(name, rows) for name, rows in groups.items() if rows]
                    prior_total = sum(group_prior[name] for name, _ in active_named)
                for _ in range(max(n_tumor, 1)):
                    # Balance diameter groups first, then choose a lesion
                    # instance uniformly within the selected group. This
                    # prevents large lesions with many voxels dominating the
                    # training stream while preserving patient-level splits.
                    if self.instance_balanced:
                        draw = float(rng.random()) * prior_total
                        cumulative = 0.0
                        selected_group = active_groups[0]
                        for name, rows in active_named:
                            cumulative += group_prior[name]
                            if draw <= cumulative:
                                selected_group = rows
                                break
                        points = selected_group[int(rng.integers(len(selected_group)))]
                    else:
                        points = instances[int(rng.integers(len(instances)))]
                    # argwhere returns (height, width, depth); convert once
                    # to the dataset's canonical (depth, height, width).
                    h, w, d = points[int(rng.integers(len(points)))]
                    extra.append(self._start_for((int(d), int(h), int(w)), (depth, height, width)))
                liver = np.asarray(label_for_index > 0)
                capsule = liver & ~binary_erosion(liver, iterations=2, border_value=0)
                margin_voxels = np.argwhere(capsule)
                if len(margin_voxels) == 0:
                    margin_voxels = np.argwhere(liver)
                for _ in range(n_margin):
                    if len(margin_voxels):
                        h, w, d = margin_voxels[int(rng.integers(len(margin_voxels)))]
                        extra.append(self._start_for((int(d), int(h), int(w)), (depth, height, width)))
                uniform = starts[::max(len(starts) // max(n_uniform, 1), 1)][:n_uniform]
                starts = list(dict.fromkeys(extra + uniform))[:max_patches]
            elif train:
                starts = starts[:max_patches]
            self.index.extend((pi, s) for s in starts)

    def _starts(self, shape, stride):
        d, h, w = shape; pd, ph, pw = self.patch_size
        def axis(length, size, step):
            if length <= size: return [0]
            values = list(range(0, length - size + 1, max(step, 1)))
            if values[-1] != length - size: values.append(length - size)
            return values
        return [(z, y, x) for z in axis(d, pd, stride[0]) for y in axis(h, ph, stride[1]) for x in axis(w, pw, stride[2])]

    def _start_for(self, center, shape):
        result = []
        for c, length, size in zip(center, shape, self.patch_size): result.append(min(max(c - size // 2, 0), max(length - size, 0)))
        return tuple(result)

    def __len__(self): return len(self.index)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, i):
        pi, (z, y, x) = self.index[i]
        if pi not in self._cache:
            if len(self._cache) >= 2: self._cache.pop(next(iter(self._cache)))
            self._cache[pi] = load_volume(self.patients[pi], self.target_z_spacing, self.cache_root, 256)
        image, label, spacing = self._cache[pi]
        pd, ph, pw = self.patch_size
        box = self.roi_boxes.get(self.patients[pi].patient_id)
        if box:
            z0, y0, x0, z1, y1, x1 = map(int, box)
        else:
            z0, y0, x0 = 0, 0, 0
            z1, y1, x1 = image.shape[2], image.shape[0], image.shape[1]
        if self.train and not self.localizer:
            # Break fixed-start memorization while keeping every patch inside
            # its full-volume or predicted-ROI support. This uses no labels
            # beyond the already-selected training-patient index.
            rng = np.random.default_rng(self.seed + i + self.epoch * 1000003)
            jitter = rng.integers(-4, 5, size=3)
            z = min(max(z + int(jitter[0]), 0), max(z1 - z0 - pd, 0))
            y = min(max(y + int(jitter[1]), 0), max(y1 - y0 - ph, 0))
            x = min(max(x + int(jitter[2]), 0), max(x1 - x0 - pw, 0))
        z, y, x = z + z0, y + y0, x + x0
        # Cached volumes are (height, width, depth), while model patches are
        # (depth, height, width). Keep starts in canonical (z,y,x) order.
        image = np.asarray(image[y:y+ph, x:x+pw, z:z+pd]).transpose(2, 0, 1)
        label = np.asarray(label[y:y+ph, x:x+pw, z:z+pd]).transpose(2, 0, 1)
        pad = [(0, max(0, pd-image.shape[0])), (0, max(0, ph-image.shape[1])), (0, max(0, pw-image.shape[2]))]
        image = np.pad(image, pad, mode="edge"); label = np.pad(label, pad, mode="constant")
        if self.train:
            rng = np.random.default_rng(self.seed + i + self.epoch * 1000003)
            if rng.random() < .5: image, label = image[:, :, ::-1].copy(), label[:, :, ::-1].copy()
            if rng.random() < .5: image, label = image[:, ::-1, :].copy(), label[:, ::-1, :].copy()
            if rng.random() < .35:
                image = np.clip(image * float(rng.uniform(.90, 1.10)) + float(rng.uniform(-.05, .05)), 0.0, 1.0)
            if rng.random() < .20:
                image = np.clip(image + rng.normal(0.0, .02, size=image.shape).astype(np.float32), 0.0, 1.0)
        return {"image": torch.from_numpy(np.ascontiguousarray(image[None])).float(), "mask": torch.from_numpy(np.ascontiguousarray(np.clip(label, 0, 2))).long(),
                "patient_id": self.patients[pi].patient_id, "patch_start": torch.tensor((z, y, x)), "spacing": torch.tensor(spacing, dtype=torch.float32)}
