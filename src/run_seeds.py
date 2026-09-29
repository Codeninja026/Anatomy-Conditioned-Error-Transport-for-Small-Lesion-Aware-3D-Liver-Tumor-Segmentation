"""Sequential runner for the architecture x loss experiment matrix."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--experiment", default="acet_standard"); p.add_argument("--seeds", nargs="+", type=int, default=[42]); p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--architecture", choices=("unet", "cers", "acet"), default="acet"); p.add_argument("--cerc-mode", choices=("signed", "absolute", "no_correction"), default="signed"); p.add_argument("--loss", choices=("standard", "litra"), default="standard")
    p.add_argument("--data-root", default="dataset"); p.add_argument("--batch-size", type=int, default=1); p.add_argument("--val-batch-size", type=int, default=2); p.add_argument("--workers", type=int, default=0); p.add_argument("--patch-depth", type=int, default=16); p.add_argument("--patch-height", type=int, default=128); p.add_argument("--patch-width", type=int, default=128); p.add_argument("--base-channels", type=int, default=32); p.add_argument("--max-train-patches", type=int, default=64); p.add_argument("--patience", type=int, default=10); p.add_argument("--cerc-dropout", type=float, default=.20); p.add_argument("--cerc-warmup-epochs", type=int, default=5); p.add_argument("--no-amp", action="store_true"); a = p.parse_args()
    for seed in a.seeds:
        run = f"{a.experiment}/seed_{seed}"; done = Path("runs") / run / "DONE"
        if done.exists(): print(f"Skipping completed {run}"); continue
        cmd = [sys.executable, "-m", "src.train", "--data-root", a.data_root, "--run", run, "--architecture", a.architecture, "--cerc-mode", a.cerc_mode, "--loss", a.loss, "--epochs", str(a.epochs), "--batch-size", str(a.batch_size), "--val-batch-size", str(a.val_batch_size), "--workers", str(a.workers), "--seed", str(seed), "--patch-depth", str(a.patch_depth), "--patch-height", str(a.patch_height), "--patch-width", str(a.patch_width), "--base-channels", str(a.base_channels), "--max-train-patches", str(a.max_train_patches), "--patience", str(a.patience), "--cerc-dropout", str(a.cerc_dropout), "--cerc-warmup-epochs", str(a.cerc_warmup_epochs)]
        if a.no_amp: cmd.append("--no-amp")
        subprocess.run(cmd, check=True)


if __name__ == "__main__": main()
