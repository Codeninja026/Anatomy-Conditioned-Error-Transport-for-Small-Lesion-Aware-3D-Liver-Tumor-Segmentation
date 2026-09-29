from __future__ import annotations
import argparse
from pathlib import Path
from .data import discover_lits, prepare_cache

def main():
    p=argparse.ArgumentParser(description="Preprocess all LiTS patients once into a reusable cache")
    p.add_argument("--data-root",default="dataset"); p.add_argument("--cache-root",default="cache/lits_cache"); p.add_argument("--image-size",type=int,default=256); p.add_argument("--z-spacing",type=float,default=1.5); a=p.parse_args()
    patients=discover_lits(Path(a.data_root)); root=Path(a.cache_root); root.mkdir(parents=True,exist_ok=True)
    prepare_cache(patients,root,a.image_size,a.z_spacing)
    print(f"Completed cache for {len(patients)} LiTS patients at {root}")
