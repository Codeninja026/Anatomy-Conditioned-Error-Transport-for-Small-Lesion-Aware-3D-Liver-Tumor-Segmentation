from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy import ndimage


def anatomy_consistency_loss(apf_predictions, target, weight=0.1):
    """Supervise the APF heads with the liver ground truth.

    Parameters
    ----------
    apf_predictions : list[Tensor]
        Raw logits from ACET anatomy heads, one per decoder level.
        Each has shape (B, 1, D', H', W').
    target : Tensor
        Ground-truth mask (B, D, H, W) with 0=bg, 1=liver, 2=tumor.
    weight : float
        Scaling factor for the auxiliary loss.

    The liver region is defined as target ≥ 1 (liver + tumor).
    Each APF prediction is supervised against a nearest-downsampled
    version of the binary liver mask at the corresponding scale.
    """
    if not apf_predictions:
        return target.new_zeros(())
    # Liver mask: labels 1 (liver) and 2 (tumor) are both "inside liver"
    liver_mask = (target > 0).float().unsqueeze(1)          # (B, 1, D, H, W)
    loss = target.new_zeros(())
    for pred in apf_predictions:
        pred = pred.float()
        scaled_mask = F.interpolate(liver_mask, size=pred.shape[2:], mode="nearest")
        loss = loss + F.binary_cross_entropy_with_logits(pred, scaled_mask)
    return weight * loss / len(apf_predictions)


def dice_loss(logits, target, eps=1e-5):
    probability = logits.softmax(1)
    one_hot = F.one_hot(target, 3).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    intersection = (probability * one_hot).sum(dims)
    denominator = (probability + one_hot).sum(dims)
    return (1 - (2 * intersection + eps) / (denominator + eps)).mean()


def union_loss(logits, target, eps=1e-5):
    """Binary liver-plus-tumor loss used only by the Stage-1 localizer.

    Uses a single foreground logit (logsumexp of class 1+2 minus class 0)
    so the gradient signal is not diluted across two redundant softmax
    channels.  Boundary weighting ensures liver edges are captured cleanly.
    """
    truth = (target > 0).to(logits.dtype)

    # Single foreground-vs-background logit — clean binary signal
    foreground_logit = torch.logsumexp(logits[:, 1:], dim=1) - logits[:, 0]
    probability = torch.sigmoid(foreground_logit)

    # Binary sigmoid Dice (no softmax competition)
    dims = (0, 1, 2, 3)
    intersection = (probability * truth).sum(dims)
    denominator = (probability + truth).sum(dims)
    dice = 1.0 - (2.0 * intersection + eps) / (denominator + eps)

    # BCE on the binary foreground logit
    bce = F.binary_cross_entropy_with_logits(foreground_logit, truth)

    # Boundary-weighted Dice: upweight liver edges so the localizer learns
    # crisp organ boundaries, not just the interior bulk.
    boundary_weight = _organ_boundary_weight(truth)
    weighted_intersection = (probability * truth * boundary_weight).sum(dims)
    weighted_denominator = (probability * boundary_weight + truth * boundary_weight).sum(dims)
    boundary_dice = 1.0 - (2.0 * weighted_intersection + eps) / (weighted_denominator + eps)

    return dice + bce + 0.25 * boundary_dice


def _organ_boundary_weight(truth, width=2):
    """Create a soft boundary weight map for the binary liver mask.

    Voxels near the organ surface get weight 3.0; interior/exterior get 1.0.
    """
    truth_5d = truth.unsqueeze(1)                      # (B, 1, D, H, W)
    kernel = 2 * width + 1
    dilated = F.max_pool3d(truth_5d, kernel, stride=1, padding=width)
    eroded = -F.max_pool3d(-truth_5d, kernel, stride=1, padding=width)
    boundary = (dilated - eroded).squeeze(1).clamp(0.0, 1.0)
    return 1.0 + 2.0 * boundary


def focal_tversky_tumor_loss(logits, target, fp_weight=0.7, fn_weight=0.3,
                             gamma=0.75, eps=1e-5):
    """Focal Tversky loss for the tumor class.

    ``fp_weight`` explicitly weights false positives and ``fn_weight`` false
    negatives.  The naming avoids the common alpha/beta convention error.
    The larger FP weight is intentional for ACET's out-of-liver false alarms.
    """
    probability = logits.softmax(1)[:, 2]
    truth = (target == 2).to(probability.dtype)
    dims = (0, 1, 2, 3)
    tp = (probability * truth).sum(dims)
    fp = (probability * (1.0 - truth)).sum(dims)
    fn = ((1.0 - probability) * truth).sum(dims)
    tversky = (tp + eps) / (tp + fp_weight * fp + fn_weight * fn + eps)
    # The fractional focal exponent has an infinite derivative at zero.
    # Clamp the residual before the power so late training cannot create
    # Inf/NaN gradients when Tversky approaches one.
    residual = (1.0 - tversky).clamp_min(1e-4)
    return residual.pow(gamma)


def lesion_tail_risk(logits, target, rho=0.25, eps=1e-5):
    """Lesion-level CVaR surrogate: average error of the worst lesions."""
    probability = logits[:, 2].sigmoid(); errors = []
    structure = ndimage.generate_binary_structure(3, 1)
    for batch_index in range(target.shape[0]):
        labels, count = ndimage.label((target[batch_index].detach().cpu().numpy() == 2), structure=structure)
        for lesion_id in range(1, count + 1):
            lesion = torch.as_tensor(labels == lesion_id, device=target.device, dtype=probability.dtype)
            p = probability[batch_index]
            dice = (2 * (p * lesion).sum() + eps) / (p.sum() + lesion.sum() + eps)
            errors.append(1 - dice)
    if not errors: return logits.new_zeros(())
    values = torch.stack(errors); count = max(1, int(torch.ceil(values.new_tensor(values.numel() * rho)).item()))
    return torch.topk(values, k=min(count, values.numel()), largest=True).values.mean()


def total_loss(out, mask, mode="standard", tail_weight=0.25, tail_rho=0.25,
               deep_weight=0.1, anatomy_weight=0.1, tversky_weight=0.5,
               tversky_fp_weight=0.7, tversky_fn_weight=0.3,
               tversky_gamma=0.75, boundary_focus_weight=0.20,
               stage1=False):
    """Combined loss supporting all architecture variants.

    • Dice + weighted CE on main logits (always).
    • Deep supervision on intermediate heads (always).
    • Anatomy consistency loss on APF predictions (ACET only; ignored when
      ``apf_preds`` is absent or empty in *out*).
    • LITRA tail-risk loss (optional, when ``mode='litra'``).
    """
    # AMP remains enabled for the network, but reductions and softmax-based
    # objectives are accumulated in FP32. This prevents rare FP16 overflow
    # from corrupting AdamW state on a later batch.
    main_logits = out["logits"].float()
    if stage1:
        # Stage 1 is not a tumor classifier. Optimize the actual binary
        # union target so the localizer learns complete liver boundaries.
        loss = union_loss(main_logits, mask)
        for deep in out.get("deep_logits", []):
            scaled = F.interpolate(deep.float(), size=mask.shape[1:], mode="trilinear", align_corners=False)
            loss = loss + deep_weight * union_loss(scaled, mask)
        return loss
    class_weight = torch.tensor([0.05, 0.25, 1.0], device=mask.device, dtype=torch.float32)
    loss = dice_loss(main_logits, mask) + F.cross_entropy(main_logits, mask, weight=class_weight)
    if boundary_focus_weight > 0:
        tumor = (mask == 2).float().unsqueeze(1)
        dilation = F.max_pool3d(tumor, 3, stride=1, padding=1)
        erosion = -F.max_pool3d(-tumor, 3, stride=1, padding=1)
        boundary = (dilation - erosion).squeeze(1).clamp(0.0, 1.0)
        per_voxel_ce = F.cross_entropy(main_logits, mask, weight=class_weight, reduction="none")
        boundary_mass = boundary.sum().clamp_min(1.0)
        boundary_ce = (per_voxel_ce * boundary).sum() / boundary_mass
        loss = loss + boundary_focus_weight * boundary_ce
    loss = loss + tversky_weight * focal_tversky_tumor_loss(
        main_logits, mask, tversky_fp_weight, tversky_fn_weight, tversky_gamma,
    )

    deep_weights = [0.1, 0.2, 0.3, 0.4]  # coarse-to-fine: emphasize final detail
    for index, deep in enumerate(out.get("deep_logits", [])):
        scaled = F.interpolate(deep.float(), size=mask.shape[1:], mode="trilinear", align_corners=False)
        weight = deep_weight * deep_weights[min(index, len(deep_weights) - 1)]
        loss = loss + weight * (dice_loss(scaled, mask) + F.cross_entropy(scaled, mask, weight=class_weight))

    # Anatomy consistency — only fires when ACET produces APF predictions
    apf = out.get("apf_preds")
    if apf:
        loss = loss + anatomy_consistency_loss(apf, mask, weight=anatomy_weight)

    if mode == "litra":
        loss = loss + tail_weight * lesion_tail_risk(main_logits, mask, tail_rho)
    elif mode != "standard":
        raise ValueError("loss mode must be 'standard' or 'litra'")
    return loss
