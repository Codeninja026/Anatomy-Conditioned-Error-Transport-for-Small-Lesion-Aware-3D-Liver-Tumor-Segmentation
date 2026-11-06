# Results index

All reported files are aggregate summaries. Patient-specific lesion coordinates and IDs have been excluded.

- `lits_test_summary.json`: ACET-Net seed 42 on the 20-patient held-out LiTS test split; four-view in-plane flip TTA is enabled.
- `lits_baselines/`: CERS-U-Net and plain 3D U-Net held-out LiTS results, both with and without the separately reported four-view TTA condition. The saved baseline evaluations use the probability threshold recorded in each result JSON.
- `zero_shot_3dircadb/`: every available aggregate 3DIRCADb summary for the ACET seed-42 run, ACET seed-43 run, CERS, plain U-Net, and the Stage 2 result copy. This cohort is external evaluation only. Check each file for its threshold/TTA metadata.
- `ablations/`: sanitized configurations, 18-epoch train/validation aggregate histories (JSON and CSV), and selected/final validation summaries for anatomy-gating-off, transformer-off, and verification-off variants. No held-out test metrics were found for those ablations.
- `threshold_sweep_validation/`: validation-only threshold sweep outputs. Do not use these to claim an independent test result.

Metrics are reproduced from the project run outputs. The experiments vary by seed, inference condition, or evaluation threshold; read per-file metadata before making direct comparisons.
