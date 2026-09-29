from __future__ import annotations

import argparse, csv, json, time, shutil
from pathlib import Path
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm
from .config import Config
from .data import discover_lits, make_splits, save_manifest, PatientPatch3DDataset, assert_training_cohort, prepare_cache
from .model import get_model
from .losses import total_loss
from .utils import seed_everything, save_checkpoint, load_checkpoint, empty_metric_counts, metric_counts, metric_counts_from_prediction, summarize_metric_counts
from .volume_eval import evaluate_volume_loader

ARCHITECTURES = {"unet": "Plain 3D U-Net", "cers": "CERS-U-Net", "acet": "ACET-Net"}
ABLATIONS = {"full": ("cers", "signed"), "no_signed_direction": ("cers", "absolute"), "no_correction": ("cers", "no_correction"), "unet": ("unet", "signed")}
ACET_ABLATIONS = {
    "full": "Full ACET-Net",
    "anatomy_off": "A1: anatomy gate off",
    "transformer_off": "A2: bottleneck transformer off",
    "verification_off": "A3: self-verification off",
}


def validation_selection_score(metrics, stage1=False):
    """Return a validation-only, multi-metric clinical selection score.

    Stage 1 predicts the liver-plus-tumor union, so tumor-class metrics are
    intentionally excluded there. Stage 2 uses tumor overlap, precision,
    recall, IoU, and liver Dice; pixel accuracy is excluded because abdominal
    background would dominate it.
    """
    if stage1:
        return (0.45 * metrics["liver_dice"]
                + 0.20 * metrics["liver_precision"]
                + 0.20 * metrics["liver_recall"]
                + 0.15 * metrics["liver_iou"])
    return (0.45 * metrics["tumor_dice"]
            + 0.15 * metrics["tumor_precision"]
            + 0.15 * metrics["tumor_recall"]
            + 0.10 * metrics["tumor_iou"]
            + 0.15 * metrics["liver_dice"])


def parse_args():
    p = argparse.ArgumentParser(description="Patient-level resumable ACET-Net/CERS/U-Net training")
    p.add_argument("--data-root", default="dataset"); p.add_argument("--cache-root", default="cache/lits_cache", help="existing preprocessed cache directory"); p.add_argument("--output-root", default="runs", help="persistent directory containing experiment runs (for example a mounted Drive path)"); p.add_argument("--drive-folder-id", default="", help="Google Drive folder ID used for authenticated pull/push"); p.add_argument("--drive-path", default="", help="subfolder under Drive folder; defaults to --run"); p.add_argument("--run", default="acet_final")
    p.add_argument("--architecture", choices=ARCHITECTURES, default="acet"); p.add_argument("--cerc-mode", choices=("signed", "absolute", "no_correction"), default="signed")
    p.add_argument("--loss", dest="loss_mode", choices=("standard", "litra"), default="standard"); p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1); p.add_argument("--workers", type=int, default=0); p.add_argument("--resume", default=""); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-patches", type=int, default=64); p.add_argument("--max-val-patients", type=int, default=0); p.add_argument("--val-batch-size", type=int, default=2); p.add_argument("--patch-depth", type=int, default=32); p.add_argument("--patch-height", type=int, default=128); p.add_argument("--patch-width", type=int, default=128)
    p.add_argument("--base-channels", type=int, default=32); p.add_argument("--cerc-dropout", type=float, default=0.20); p.add_argument("--cerc-warmup-epochs", type=int, default=5); p.add_argument("--checkpoint-every-batches", type=int, default=25); p.add_argument("--no-amp", action="store_true"); p.add_argument("--no-instance-balanced", action="store_true"); p.add_argument("--boundary-focus-weight", type=float, default=0.20); p.add_argument("--stage1", action="store_true", help="train the union-target liver ROI localizer"); p.add_argument("--roi-manifest", default="", help="JSON mapping patient IDs to z0,y0,x0,z1,y1,x1 ROI boxes"); p.add_argument("--ablation", choices=tuple(ACET_ABLATIONS), default="full", help="controlled ACET ablation; use with --architecture acet"); p.add_argument("--patience", type=int, default=10); p.add_argument("--reset-optimizer", action="store_true", help="resume weights/history with fresh AdamW moments"); p.add_argument("--allow-schedule-change", action="store_true", help="explicitly allow changing total epochs when resuming; this changes the LR schedule"); p.add_argument("--deterministic", action="store_true")
    return p.parse_args()


def run_epoch(model, loader, optimizer, scaler, device, cfg, train=True, epoch=0):
    model.train(train); totals = empty_metric_counts(); total = 0.0; started = time.perf_counter(); count = 0; unstable_batches = 0
    bar = tqdm(loader, desc="Train" if train else "Val", dynamic_ncols=True)
    for batch in bar:
        image = batch["image"].to(device, non_blocking=True); mask = batch["mask"].to(device, non_blocking=True)
        if cfg.stage1:
            mask = (mask > 0).long()
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                out = model(image); loss = total_loss(out, mask, mode=cfg.loss_mode, tail_weight=cfg.litra_tail_weight, tail_rho=cfg.litra_rho, anatomy_weight=cfg.anatomy_loss_weight, tversky_weight=cfg.tversky_weight, tversky_fp_weight=cfg.tversky_fp_weight, tversky_fn_weight=cfg.tversky_fn_weight, tversky_gamma=cfg.tversky_gamma, boundary_focus_weight=cfg.boundary_focus_weight, stage1=cfg.stage1)
            if train:
                if torch.isfinite(loss):
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    finite_gradients = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
                    if not finite_gradients:
                        # Preserve usable gradients from a rare unstable path
                        # while neutralizing only invalid entries. If every
                        # gradient is invalid, the batch is discarded.
                        usable = False
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                usable = usable or bool(torch.isfinite(parameter.grad).any())
                                parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                        finite_gradients = usable
                    if finite_gradients:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=False)
                        # Clipping can itself propagate a non-finite total
                        # norm; sanitize once more before AdamW sees it.
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    if finite_gradients:
                        scaler.step(optimizer); scaler.update()
                        # Defensive post-step invariant: no optimizer update
                        # is allowed to leave a non-finite model parameter.
                        for parameter in model.parameters():
                            parameter.data.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
                    else:
                        # unscale_() was already called, so update the scaler
                        # even when this optimizer step is discarded. Without
                        # this, the next batch attempts a second unscale_()
                        # in the same AMP scale cycle and crashes.
                        scaler.update()
                        unstable_batches += 1
                else:
                    print(f"[WARN] NaN/Inf loss detected — skipping this batch")
                optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(out["logits"]).all():
            print("[WARN] NaN/Inf logits detected — excluding this batch from metrics")
            continue
        batch_count = image.shape[0]; count += batch_count; total += float(loss.detach()) * batch_count
        prediction = out["logits"].detach().argmax(1)
        # Stage 1 is a binary union target (liver ∪ tumor). Both foreground
        # output classes must therefore count as the same foreground class.
        if cfg.stage1:
            prediction = (prediction > 0).long()
        counts = metric_counts_from_prediction(prediction, mask)
        for key, value in counts.items(): totals[key] += value
        live = summarize_metric_counts(totals); elapsed = max(time.perf_counter() - started, 1e-6)
        live_text = {"loss": total / max(count, 1), "tumor_dice": live["tumor_dice"], "tumor_precision": live["tumor_precision"], "tumor_recall": live["tumor_recall"], "tumor_iou": live["tumor_iou"], "liver_dice": live["liver_dice"], "liver_precision": live["liver_precision"], "liver_recall": live["liver_recall"], "liver_iou": live["liver_iou"], "acc": live["pixel_accuracy"], "it/s": count / elapsed}
        if device.type == "cuda": live_text["GB"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        bar.set_postfix({k: f"{v:.4f}" if isinstance(v, float) else v for k,v in live_text.items()})
    bar.close()
    if unstable_batches:
        print(f"[WARN] Sanitized invalid gradients in {unstable_batches}/{len(loader)} training batches for epoch {epoch + 1}")
    result = {"loss": total / max(count, 1), "seconds": time.perf_counter() - started, "samples_per_second": count / max(time.perf_counter() - started, 1e-6), **summarize_metric_counts(totals)}
    return result


def main():
    a = parse_args(); run = Path(a.run) if Path(a.run).is_absolute() else Path(a.output_root) / a.run
    if a.drive_folder_id:
        from .drive_sync import download_tree
        try: download_tree(run, a.drive_folder_id, a.drive_path or a.run)
        except Exception as error: raise RuntimeError(f"Drive pull failed; refusing to start without verified remote state: {error}") from error
    run.mkdir(parents=True, exist_ok=True); seed_everything(a.seed, a.deterministic)
    checkpoint_dir = run / "checkpoints"
    if a.resume:
        checkpoint_path = Path(a.resume)
    else:
        # Keep only best.pt and descriptive periodic five-epoch snapshots.
        # Resume from the newest descriptive snapshot, or best.pt before
        # epoch 5 exists. The legacy epoch_*.pt pattern remains a fallback
        # for runs created by older package versions.
        periodic = sorted(checkpoint_dir.glob("model_epoch_*.pt")) if checkpoint_dir.exists() else []
        if not periodic and checkpoint_dir.exists():
            periodic = sorted(checkpoint_dir.glob("epoch_*.pt"))
        checkpoint_path = periodic[-1] if periodic else checkpoint_dir / "best.pt"
    saved_config = run / "config.json"
    if checkpoint_path.exists() and saved_config.exists():
        # A shortened resume command must never silently rebuild a different
        # architecture from argparse defaults (for example ACET -> CERS).
        checkpoint_config = None
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint_config = checkpoint_data.get("config")
        except Exception:
            checkpoint_config = None
        cfg = Config(**checkpoint_config) if checkpoint_config else Config.load(saved_config)
        if checkpoint_config and int(checkpoint_config.get("epochs", a.epochs)) != int(a.epochs) and not a.allow_schedule_change:
            raise RuntimeError(
                f"Checkpoint was created with epochs={checkpoint_config.get('epochs')} but this command requests epochs={a.epochs}. "
                "Restart from epoch 0 with the intended total, or pass --allow-schedule-change knowingly; changing it alters the LR schedule."
            )
        cfg.epochs = a.epochs
        cfg.early_stopping_patience = a.patience
        cfg.workers = a.workers
        # Resume-time CLI safety: the explicit --no-amp choice must override
        # the older checkpoint config instead of being silently ignored.
        if a.no_amp:
            cfg.amp = False
    else:
        cfg = Config(data_root=a.data_root, output_root=str(a.output_root), cache_root=a.cache_root, seed=a.seed, epochs=a.epochs, batch_size=a.batch_size, workers=a.workers,
                     patch_depth=a.patch_depth, patch_height=a.patch_height, patch_width=a.patch_width, base_channels=a.base_channels, cerc_dropout=a.cerc_dropout,
                     cerc_warmup_epochs=a.cerc_warmup_epochs,
                     architecture=a.architecture, cerc_mode=a.cerc_mode, loss_mode=a.loss_mode, max_train_patches_per_patient=a.max_train_patches, instance_balanced=not a.no_instance_balanced, boundary_focus_weight=a.boundary_focus_weight, stage1=a.stage1, roi_manifest=a.roi_manifest, ablation=a.ablation, drive_folder_id=a.drive_folder_id, drive_path=a.drive_path or a.run, drive_sync_every=1,
                     amp=not a.no_amp, deterministic=a.deterministic, early_stopping_patience=a.patience, save_every=5)
    cfg.save(run / "config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); patients = discover_lits(Path(a.data_root)); assert_training_cohort(patients)
    splits = make_splits(patients, cfg.split_seed, cfg.train_patients, cfg.val_patients, cfg.test_patients); save_manifest(run / "splits", splits)
    # Test patients are deliberately excluded from the training/validation cache.
    prepare_cache(splits[0] + splits[1], Path(cfg.cache_root), 256, cfg.target_z_spacing)
    patch_size = (cfg.patch_depth, cfg.patch_height, cfg.patch_width); cache = Path(cfg.cache_root)
    roi_boxes = {}
    if cfg.roi_manifest:
        manifest = json.loads(Path(cfg.roi_manifest).read_text(encoding="utf-8"))
        roi_boxes = manifest.get("boxes", manifest)
    # Stage 2 trains on lesion-focused predicted-ROI patches plus a uniform
    # full-volume safety stream. A Stage-1 miss must not become a permanent
    # blind spot. Stage 1 itself remains a single uniform stream.
    roi_train_ds = PatientPatch3DDataset(splits[0], patch_size, True, cfg.max_train_patches_per_patient, a.seed, cfg.target_z_spacing, cache, instance_balanced=cfg.instance_balanced, roi_boxes=roi_boxes, localizer=cfg.stage1)
    full_train_ds = None
    train_streams = [roi_train_ds]
    if roi_boxes and not cfg.stage1:
        full_train_ds = PatientPatch3DDataset(splits[0], patch_size, True, max(16, cfg.max_train_patches_per_patient // 2), a.seed + 7919, cfg.target_z_spacing, cache, instance_balanced=False, roi_boxes={}, localizer=True)
        train_streams.append(full_train_ds)
    train_ds = ConcatDataset(train_streams) if len(train_streams) > 1 else roi_train_ds
    validation_patients = splits[1] if a.max_val_patients <= 0 else splits[1][:a.max_val_patients]
    # Full-volume validation exposes tumors outside a predicted ROI.
    val_ds = PatientPatch3DDataset(validation_patients, patch_size, False, cfg.max_train_patches_per_patient, a.seed, cfg.target_z_spacing, cache, stride=tuple(max(v // 2, 1) for v in patch_size), roi_boxes={} if (roi_boxes and not cfg.stage1) else roi_boxes)
    loader_kwargs = {"num_workers": a.workers, "pin_memory": device.type == "cuda"}
    if a.workers > 0: loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=a.val_batch_size, shuffle=False, **loader_kwargs)
    model = get_model(1, cfg.base_channels, cfg.architecture, cfg.cerc_mode, cfg.dropout, cfg.cerc_dropout, cfg.ablation).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {ARCHITECTURES.get(cfg.architecture, cfg.architecture)} | Parameters: {n_params:.2f}M")
    # A larger epsilon is standard defensive practice for tiny-lesion 3-D
    # segmentation, where sparse gradients can make Adam's denominator tiny.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, eps=1e-4)
    def lr_schedule(step):
        if step < cfg.lr_warmup_epochs:
            return 0.05 + 0.95 * step / max(cfg.lr_warmup_epochs - 1, 1)
        progress = (step - cfg.lr_warmup_epochs) / max(cfg.epochs - cfg.lr_warmup_epochs - 1, 1)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * min(max(progress, 0.0), 1.0))).item())
        return cfg.min_lr / cfg.lr + (1.0 - cfg.min_lr / cfg.lr) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule); scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    best, stale, history, start = -1.0, 0, [], 0; resume = checkpoint_path
    if resume.exists():
        state = load_checkpoint(resume, model, optimizer, scheduler, scaler, device); start, best, stale, history = state.get("epoch", 0), state.get("best", -1.0), state.get("stale", 0), state.get("history", [])
        if a.reset_optimizer:
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=cfg.weight_decay, eps=1e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
            scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
            print(f"Reset optimizer at epoch {start}; rescue LR=5.00e-05")
        else:
            # Let the restored scheduler continue from where it left off.
            # Gradient sanitization (NaN-to-zero clamping, clip_grad_norm)
            # already guards against instability.  Overriding the cosine
            # schedule with a flat LR silently prevents warmup/decay and
            # stalls early-stage convergence.
            pass
    for epoch in range(start, a.epochs):
        for stream in train_streams:
            stream.set_epoch(epoch)
        model.set_correction_scale(epoch / max(cfg.cerc_warmup_epochs - 1, 1))
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, cfg, True, epoch)
        target_transform = (lambda target: (target > 0).long()) if cfg.stage1 else None
        val_metrics = evaluate_volume_loader(model, val_loader, validation_patients, patch_size, cfg.target_z_spacing, cache, device, lambda out, mask: total_loss(out, target_transform(mask) if target_transform else mask, mode=cfg.loss_mode, tail_weight=cfg.litra_tail_weight, tail_rho=cfg.litra_rho, anatomy_weight=cfg.anatomy_loss_weight, tversky_weight=cfg.tversky_weight, tversky_fp_weight=cfg.tversky_fp_weight, tversky_fn_weight=cfg.tversky_fn_weight, tversky_gamma=cfg.tversky_gamma, boundary_focus_weight=cfg.boundary_focus_weight, stage1=cfg.stage1), roi_boxes=roi_boxes, target_transform=target_transform)
        val_metrics["epoch_seconds"] = time.perf_counter() - epoch_started
        if device.type == "cuda": val_metrics["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        scheduler.step()
        # Select checkpoints using a smoothed clinical score rather than a
        # single noisy tumor-Dice value. This discourages FP-heavy snapshots
        # while retaining tumor overlap as the dominant term.
        # Checkpoint selection uses validation metrics only. The score is
        # multi-metric and stage-aware; training metrics never participate.
        current_score = validation_selection_score(val_metrics, cfg.stage1)
        previous_scores = [float(row["val"].get("selection_score", row["val"]["tumor_dice"]))
                           for row in history[-2:]]
        scores = [current_score] + list(reversed(previous_scores))
        weights = [0.5, 0.3, 0.2][:len(scores)]
        smoothed_score = sum(weight * score for weight, score in zip(weights, scores)) / sum(weights)
        val_metrics["selection_score"] = current_score
        val_metrics["selection_score_smooth"] = smoothed_score
        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics, "lr": optimizer.param_groups[0]["lr"]}); (run / "metrics").mkdir(exist_ok=True)
        (run / "metrics/history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        with (run / "metrics/history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f); writer.writerow(["epoch", "lr", "train_loss", "train_pixel_accuracy", "train_liver_dice", "train_liver_precision", "train_liver_recall", "train_liver_iou", "train_tumor_dice", "train_tumor_precision", "train_tumor_recall", "train_tumor_iou", "val_loss", "val_pixel_accuracy", "val_liver_dice", "val_liver_precision", "val_liver_recall", "val_liver_iou", "val_tumor_dice", "val_tumor_precision", "val_tumor_recall", "val_tumor_iou", "val_selection_score", "val_selection_score_smooth", "epoch_seconds", "peak_gpu_memory_gb"])
            for row in history:
                tr, va = row["train"], row["val"]
                writer.writerow([row["epoch"], row["lr"], tr["loss"], tr["pixel_accuracy"], tr["liver_dice"], tr["liver_precision"], tr["liver_recall"], tr["liver_iou"], tr["tumor_dice"], tr["tumor_precision"], tr["tumor_recall"], tr["tumor_iou"], va.get("loss"), va["pixel_accuracy"], va["liver_dice"], va["liver_precision"], va["liver_recall"], va["liver_iou"], va["tumor_dice"], va["tumor_precision"], va["tumor_recall"], va["tumor_iou"], va.get("selection_score"), va.get("selection_score_smooth"), va.get("epoch_seconds"), va.get("peak_gpu_memory_gb")])
        checkpoint = run / "checkpoints"
        if smoothed_score > best + cfg.early_stopping_min_delta:
            best, stale = smoothed_score, 0
            save_checkpoint(checkpoint / "best.pt", model, optimizer, scheduler, scaler, epoch + 1, best, history, cfg, stale=stale)
            # Keep a descriptive alias for paper/evaluation tooling. Both
            # files are byte-identical and are selected using validation-only
            # multi-metric score, never training metrics.
            shutil.copyfile(checkpoint / "best.pt", checkpoint / "best_model.pt")
        else: stale += 1
        if (epoch + 1) % max(int(cfg.save_every), 1) == 0:
            periodic_path = checkpoint / f"model_epoch_{epoch + 1:03d}_{cfg.ablation}.pt"
            save_checkpoint(periodic_path, model, optimizer, scheduler, scaler, epoch + 1, best, history, cfg, stale=stale)
        if a.drive_folder_id and (epoch + 1) % max(int(cfg.drive_sync_every), 1) == 0:
            from .drive_sync import upload_tree
            try:
                upload_tree(run, a.drive_folder_id, cfg.drive_path or a.run)
                print(f"Drive sync complete after epoch {epoch + 1}")
            except Exception as error:
                raise RuntimeError(f"Drive sync failed after epoch {epoch + 1}; local checkpoint is saved, so resume after fixing Drive credentials: {error}") from error
        print(f"Epoch {epoch + 1}/{a.epochs}: train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['pixel_accuracy']:.4f} val_liver_dice={val_metrics['liver_dice']:.4f} val_tumor_dice={val_metrics['tumor_dice']:.4f} tumor_precision={val_metrics['tumor_precision']:.4f} tumor_recall={val_metrics['tumor_recall']:.4f} tumor_iou={val_metrics['tumor_iou']:.4f} selection={current_score:.4f} smooth={smoothed_score:.4f} best={best:.4f} lr={optimizer.param_groups[0]['lr']:.2e} epoch_s={val_metrics['epoch_seconds']:.1f}")
        if stale >= cfg.early_stopping_patience: break
    (run / "DONE").write_text("training_completed\n", encoding="utf-8")


if __name__ == "__main__": main()
