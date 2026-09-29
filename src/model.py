from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


# ══════════════════════════ Building Blocks ══════════════════════════


class ConvBlock3D(nn.Module):
    """Standard double-conv block (kept for CERS backward compatibility)."""
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        regularizer = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.block = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True), nn.SiLU(inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True), nn.SiLU(inplace=True), regularizer,
        )

    def forward(self, x):
        return self.block(x)


class ResConvBlock3D(nn.Module):
    """Double-conv block with a learned residual shortcut.

    Adds a 1×1 projection when *cin ≠ cout*.  Improves gradient flow
    in deeper networks without changing the receptive field.
    """
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            nn.InstanceNorm3d(cout, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False),
            nn.InstanceNorm3d(cout, affine=True),
        )
        self.residual = nn.Conv3d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.act(self.conv(x) + self.residual(x)))


class SE3D(nn.Module):
    """3-D Squeeze-and-Excitation: lightweight channel attention."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, mid, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv3d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class TransformerLayer3D(nn.Module):
    """Pre-norm transformer layer: LN → MHSA → LN → FFN, with residuals."""
    def __init__(self, channels: int, num_heads: int = 8,
                 ffn_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * ffn_ratio, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class BottleneckTransformer3D(nn.Module):
    """Lightweight self-attention at the U-Net bottleneck.

    Flattens 3-D spatial dims into a token sequence, applies one or more
    transformer layers, then reshapes back.  With typical bottleneck sizes
    (e.g. 2 × 10 × 10 = 200 tokens) this is very efficient.
    """
    def __init__(self, channels: int, num_heads: int = 8,
                 num_layers: int = 1, ffn_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer3D(channels, num_heads, ffn_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 2, 8, 8))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        position = F.interpolate(self.pos_embed, size=(D, H, W), mode="trilinear", align_corners=False)
        tokens = (x + position).flatten(2).permute(0, 2, 1)  # (B, N, C)
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens.permute(0, 2, 1).reshape(B, C, D, H, W)


# ══════════════════════════ Error-Transport Cells ══════════════════════════


class CERC(nn.Module):
    """Self-verifying Signed Error Transport cell.

    A direction-specific correction is proposed from delta, then accepted only
    to the extent that its candidate encoder evidence reduces disagreement with
    the decoder hypothesis. This makes the skip update an error-reduction step,
    rather than a generic learned gate.
    """
    def __init__(self, channels: int, mode: str = "signed", dropout: float = 0.0, cross_scale: bool = True):
        super().__init__()
        if mode not in {"signed", "absolute", "no_correction"}:
            raise ValueError(f"Unknown CERC mode: {mode}")
        self.mode = mode
        self.dropout = dropout
        self.cross_scale = cross_scale
        self.context_gain = nn.Parameter(torch.tensor(-2.0)) if cross_scale else None
        self.local_projection = nn.Conv3d(channels, 1, 1)
        self.decoder_projection = nn.Conv3d(channels, 1, 1)
        # The correction is driven only by delta. This constraint prevents the
        # cell from silently becoming an unconstrained feature transform of E.
        self.positive_transport = nn.Sequential(
            nn.Conv3d(1, channels, 1, bias=False), nn.InstanceNorm3d(channels, affine=True), nn.SiLU(inplace=True), nn.Conv3d(channels, channels, 1),
        )
        self.negative_transport = nn.Sequential(
            nn.Conv3d(1, channels, 1, bias=False), nn.InstanceNorm3d(channels, affine=True), nn.SiLU(inplace=True), nn.Conv3d(channels, channels, 1),
        )
        self.verification_temperature = 0.20

    def forward(self, encoder, decoder, prior_delta=None):
        local = torch.sigmoid(self.local_projection(encoder))
        hypothesis = torch.sigmoid(self.decoder_projection(decoder))
        delta = local - hypothesis
        context = None
        if self.cross_scale and prior_delta is not None:
            prior = prior_delta.abs() if self.mode == "absolute" else prior_delta
            context = F.interpolate(prior, size=delta.shape[2:], mode="trilinear", align_corners=False)
            delta = delta + torch.sigmoid(self.context_gain) * context
        correction_input = delta.abs() if self.mode == "absolute" else delta
        positive = F.relu(correction_input)
        negative = F.relu(-correction_input)
        correction = torch.tanh(self.positive_transport(positive) - self.negative_transport(negative))
        if self.training and self.dropout > 0:
            keep = (torch.rand_like(correction[:, :1]) >= self.dropout).to(correction.dtype)
            correction = correction * keep / (1.0 - self.dropout)
        candidate = encoder + correction
        candidate_local = torch.sigmoid(self.local_projection(candidate))
        candidate_delta = candidate_local - hypothesis
        if context is not None:
            candidate_delta = candidate_delta + torch.sigmoid(self.context_gain) * context
        improvement = delta.abs() - candidate_delta.abs()
        acceptance = torch.sigmoid(improvement / self.verification_temperature)
        corrected = encoder if self.mode == "no_correction" else encoder + acceptance * correction
        return corrected, delta


class ACET(nn.Module):
    """Anatomy-Conditioned Error Transport cell  (the novel contribution).

    Extends CERC by gating the error correction signal with the model's own
    soft liver prediction.  Error transport is encouraged inside plausible
    liver tissue and suppressed outside it, directly addressing the false-
    positive problem that afflicts unconditioned error transport.

    Key differences from CERC
    ─────────────────────────
    1.  An *anatomy head* predicts a soft liver probability P_liver from
        decoder features at each skip level.
    2.  The signed disagreement δ is element-wise multiplied by P_liver
        **before** being transported, so corrections fire only inside
        plausible liver tissue.
    3.  P_liver is propagated cross-scale (coarse → fine) alongside the
        error memory, creating hierarchical anatomy-aware context.
    4.  P_liver is supervised by an auxiliary liver-region loss using the
        existing labels (class ≥ 1), requiring no extra annotation.

    Why this is not "attention + gating + boundary combined"
    ────────────────────────────────────────────────────────
    It is a single mechanism: the anatomy prediction *modulates the error
    signal itself*, not the encoder features.  The modulation target is
    fundamentally different from attention gates (Oktay 2018) which weight
    encoder features using decoder features.
    """

    def __init__(self, channels: int, mode: str = "signed",
                 dropout: float = 0.0, cross_scale: bool = True,
                 ablation: str = "full"):
        super().__init__()
        if mode not in {"signed", "absolute", "no_correction"}:
            raise ValueError(f"Unknown ACET mode: {mode}")
        if ablation not in {"full", "anatomy_off", "transformer_off", "verification_off"}:
            raise ValueError(f"Unknown ACET ablation: {ablation}")
        self.mode = mode
        self.dropout = dropout
        self.cross_scale = cross_scale
        self.ablation = ablation
        self.use_anatomy_gate = ablation != "anatomy_off"
        self.use_self_verification = ablation != "verification_off"

        # ── evidence / hypothesis projections (same idea as CERC) ──
        self.local_projection = nn.Conv3d(channels, 1, 1)
        self.decoder_projection = nn.Conv3d(channels, 1, 1)

        # ── NOVEL: anatomy head → soft liver probability ──
        mid = max(channels // 4, 8)
        self.anatomy_head = nn.Sequential(
            nn.Conv3d(channels * 2, mid, 3, padding=1, bias=False),
            nn.InstanceNorm3d(mid, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(mid, 1, 1),
        )

        # ── cross-scale memory gains ──
        self.context_gain = nn.Parameter(torch.tensor(-2.0)) if cross_scale else None
        self.anatomy_memory_gain = nn.Parameter(torch.tensor(-2.0)) if cross_scale else None

        # ── directional transport paths (same idea as CERC) ──
        self.positive_transport = nn.Sequential(
            nn.Conv3d(1, channels, 1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, 1),
        )
        self.negative_transport = nn.Sequential(
            nn.Conv3d(1, channels, 1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, 1),
        )
        self.verification_temperature = nn.Parameter(torch.tensor(0.5))

    def forward(self, encoder, decoder, prior_delta=None, prior_anatomy=None):
        # 1. local evidence  &  decoder hypothesis
        local = torch.sigmoid(self.local_projection(encoder))
        hypothesis = torch.sigmoid(self.decoder_projection(decoder))

        # 2. signed disagreement
        delta = local - hypothesis

        # 3. NOVEL — predict soft liver probability from decoder features
        anatomy_logit = self.anatomy_head(torch.cat((encoder, decoder), dim=1))

        # 4. cross-scale error memory
        context = None
        if self.cross_scale and prior_delta is not None:
            prior = prior_delta.abs() if self.mode == "absolute" else prior_delta
            context = F.interpolate(prior, size=delta.shape[2:],
                                    mode="trilinear", align_corners=False)
            delta = delta + torch.sigmoid(self.context_gain) * context

        # 5. NOVEL — cross-scale anatomy memory
        if self.cross_scale and prior_anatomy is not None:
            prior_anat = F.interpolate(prior_anatomy, size=anatomy_logit.shape[2:],
                                       mode="trilinear", align_corners=False)
            anatomy_logit = anatomy_logit + torch.sigmoid(self.anatomy_memory_gain) * prior_anat
        anatomy_prob = torch.sigmoid(anatomy_logit)

        # 6. ★ ANATOMY-GATED ERROR — the core innovation ★
        correction_input = delta.abs() if self.mode == "absolute" else delta
        if self.use_anatomy_gate:
            correction_input = correction_input * anatomy_prob

        # 7. directional transport
        positive = F.relu(correction_input)
        negative = F.relu(-correction_input)
        correction = torch.tanh(
            self.positive_transport(positive) - self.negative_transport(negative)
        )

        # 8. training dropout
        if self.training and self.dropout > 0:
            keep = (torch.rand_like(correction[:, :1]) >= self.dropout).to(correction.dtype)
            correction = correction * keep / (1.0 - self.dropout)

        # 9. self-verification
        candidate = encoder + correction
        candidate_local = torch.sigmoid(self.local_projection(candidate))
        candidate_delta = candidate_local - hypothesis
        if context is not None:
            candidate_delta = candidate_delta + torch.sigmoid(self.context_gain) * context
        improvement = delta.abs() - candidate_delta.abs()
        acceptance = (torch.sigmoid(improvement / self.verification_temperature.clamp_min(0.05))
                      if self.use_self_verification else torch.ones_like(improvement))

        # 10. apply accepted correction
        corrected = encoder if self.mode == "no_correction" else encoder + acceptance * correction

        return corrected, delta, anatomy_logit, anatomy_prob


# ══════════════════════════ Full Networks ══════════════════════════


class CERSUNet3D(nn.Module):
    """Plain 3D U-Net control or CERS-U-Net with signed CERC skips."""
    def __init__(self, in_channels=1, base=16, architecture="cers", cerc_mode="signed", dropout=0.0, cerc_dropout=0.0):
        super().__init__()
        if architecture not in {"unet", "cers"}:
            raise ValueError("architecture must be 'unet' or 'cers'")
        self.architecture, self.cerc_mode = architecture, cerc_mode
        self.correction_scale = 1.0
        channels = [base, base * 2, base * 4, base * 8]
        self.encoders = nn.ModuleList([
            ConvBlock3D(in_channels, channels[0], dropout), ConvBlock3D(channels[0], channels[1], dropout),
            ConvBlock3D(channels[1], channels[2], dropout), ConvBlock3D(channels[2], channels[3], dropout),
        ])
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock3D(channels[3], channels[3] * 2, dropout)
        bc = channels[3] * 2
        self.ups = nn.ModuleList([
            nn.ConvTranspose3d(bc, channels[3], 2, 2), nn.ConvTranspose3d(channels[3], channels[2], 2, 2),
            nn.ConvTranspose3d(channels[2], channels[1], 2, 2), nn.ConvTranspose3d(channels[1], channels[0], 2, 2),
        ])
        self.decoders = nn.ModuleList([
            ConvBlock3D(channels[3] * 2, channels[3], dropout), ConvBlock3D(channels[2] * 2, channels[2], dropout),
            ConvBlock3D(channels[1] * 2, channels[1], dropout), ConvBlock3D(channels[0] * 2, channels[0], dropout),
        ])
        self.cerc = nn.ModuleList([CERC(channels[3], cerc_mode, cerc_dropout), CERC(channels[2], cerc_mode, cerc_dropout), CERC(channels[1], cerc_mode, cerc_dropout), CERC(channels[0], cerc_mode, cerc_dropout)]) if architecture == "cers" else nn.ModuleList()
        # Feed the finest signed disagreement into the final readout so weak
        # lesion evidence is not discarded after the last skip update. This
        # path exists only for CERS; the plain U-Net remains a matched control.
        self.error_readout = nn.Conv3d(1, channels[0], 1) if architecture == "cers" else None
        self.segmentation = nn.Conv3d(channels[0] * 2 if architecture == "cers" else channels[0], 3, 1)
        self.deep_heads = nn.ModuleList([nn.Conv3d(c, 3, 1) for c in channels])

    @staticmethod
    def _match(x, reference):
        return x if x.shape[2:] == reference.shape[2:] else F.interpolate(x, size=reference.shape[2:], mode="trilinear", align_corners=False)

    def set_correction_scale(self, scale: float):
        self.correction_scale = float(max(0.0, min(1.0, scale)))

    def forward(self, x, spacing=None):
        skips, current = [], x
        for level, encoder in enumerate(self.encoders):
            current = encoder(current); skips.append(current)
            if level < 3: current = self.pool(current)
        current = self.bottleneck(current); deep_logits, deltas = [], []
        prior_delta = None
        for level in range(4):
            current = self._match(self.ups[level](current), skips[3 - level]); skip = skips[3 - level]
            if self.architecture == "cers":
                original_skip = skip
                skip, delta = self.cerc[level](skip, current, prior_delta)
                skip = original_skip + self.correction_scale * (skip - original_skip)
                deltas.append(delta)
                prior_delta = delta
            current = self.decoders[level](torch.cat((current, skip), dim=1)); deep_logits.append(self.deep_heads[3 - level](current))
        readout = current
        if self.architecture == "cers" and deltas:
            error_feature = self.error_readout(deltas[-1]) * self.correction_scale
            readout = torch.cat((current, error_feature), dim=1)
        return {"logits": self.segmentation(readout), "deep_logits": deep_logits, "cerc_deltas": deltas}


class ACETNet3D(nn.Module):
    """ACET-Net — Anatomy-Conditioned Error Transport Network.

    A 3-D U-Net with:
    •  residual encoder / decoder blocks for stable gradient flow,
    •  squeeze-and-excitation channel attention at the bottleneck,
    •  lightweight self-attention (transformer) at the bottleneck,
    •  anatomy-conditioned error transport (ACET) at every skip.

    The model predicts a soft liver probability at each decoder level and
    uses it to gate the signed error correction signal, suppressing false
    positives outside the liver while preserving small-tumor recall inside.
    """

    def __init__(self, in_channels: int = 1, base: int = 32,
                 cerc_mode: str = "signed", dropout: float = 0.0,
                 cerc_dropout: float = 0.0, transformer_heads: int = 8,
                 transformer_layers: int = 1, ablation: str = "full"):
        super().__init__()
        if ablation not in {"full", "anatomy_off", "transformer_off", "verification_off"}:
            raise ValueError(f"Unknown ACET ablation: {ablation}")
        self.architecture = "acet"
        self.cerc_mode = cerc_mode
        self.ablation = ablation
        self.use_anatomy_gate = ablation != "anatomy_off"
        self.use_transformer = ablation != "transformer_off"
        self.use_self_verification = ablation != "verification_off"
        self.correction_scale = 1.0
        channels = [base, base * 2, base * 4, base * 8]

        # ── residual encoder ──
        self.encoders = nn.ModuleList([
            ResConvBlock3D(in_channels, channels[0], dropout),
            ResConvBlock3D(channels[0], channels[1], dropout),
            ResConvBlock3D(channels[1], channels[2], dropout),
            ResConvBlock3D(channels[2], channels[3], dropout),
        ])
        self.pool = nn.MaxPool3d(2)

        # ── bottleneck: residual conv + SE + transformer ──
        bc = channels[3] * 2
        self.bottleneck_conv = ResConvBlock3D(channels[3], bc, dropout)
        self.bottleneck_se = SE3D(bc)
        self.bottleneck_transformer = BottleneckTransformer3D(
            bc, transformer_heads, transformer_layers,
        )

        # ── decoder ──
        self.ups = nn.ModuleList([
            nn.ConvTranspose3d(bc, channels[3], 2, 2),
            nn.ConvTranspose3d(channels[3], channels[2], 2, 2),
            nn.ConvTranspose3d(channels[2], channels[1], 2, 2),
            nn.ConvTranspose3d(channels[1], channels[0], 2, 2),
        ])
        self.decoders = nn.ModuleList([
            ResConvBlock3D(channels[3] * 2, channels[3], dropout),
            ResConvBlock3D(channels[2] * 2, channels[2], dropout),
            ResConvBlock3D(channels[1] * 2, channels[1], dropout),
            ResConvBlock3D(channels[0] * 2, channels[0], dropout),
        ])

        # ── ACET skip connections ──
        self.acet = nn.ModuleList([
            ACET(channels[3], cerc_mode, cerc_dropout, ablation=ablation),
            ACET(channels[2], cerc_mode, cerc_dropout, ablation=ablation),
            ACET(channels[1], cerc_mode, cerc_dropout, ablation=ablation),
            ACET(channels[0], cerc_mode, cerc_dropout, ablation=ablation),
        ])

        # ── error readout from the finest level ──
        self.error_pyramid = nn.ModuleList([nn.Conv3d(1, channels[0], 1) for _ in channels])

        # ── heads ──
        # A single high-resolution residual refinement after multi-scale ACET
        # readout preserves tiny-lesion edges without adding another attention
        # branch or a second model.
        self.fine_refinement = ResConvBlock3D(channels[0] * 2, channels[0], dropout)
        self.segmentation = nn.Conv3d(channels[0], 3, 1)
        self.deep_heads = nn.ModuleList([nn.Conv3d(c, 3, 1) for c in channels])
        # Reuse the finest supervised anatomy field as a soft prior on the
        # tumor output. This suppresses implausible out-of-liver predictions
        # without introducing another prediction branch.

    @staticmethod
    def _match(x, reference):
        if x.shape[2:] == reference.shape[2:]:
            return x
        return F.interpolate(x, size=reference.shape[2:],
                             mode="trilinear", align_corners=False)

    def set_correction_scale(self, scale: float):
        self.correction_scale = float(max(0.0, min(1.0, scale)))

    def forward(self, x, spacing=None):
        # ── encode ──
        skips, current = [], x
        for level, encoder in enumerate(self.encoders):
            current = encoder(current)
            skips.append(current)
            if level < 3:
                current = self.pool(current)

        # ── bottleneck ──
        current = self.bottleneck_conv(current)
        current = self.bottleneck_se(current)
        if self.use_transformer:
            current = self.bottleneck_transformer(current)

        # ── decode with ACET ──
        deep_logits: list[torch.Tensor] = []
        deltas: list[torch.Tensor] = []
        apf_preds: list[torch.Tensor] = []
        prior_delta = None
        prior_anatomy = None

        for level in range(4):
            current = self._match(self.ups[level](current), skips[3 - level])
            skip = skips[3 - level]

            # anatomy-conditioned error transport
            original_skip = skip
            skip, delta, anatomy_logit, anatomy_prob = self.acet[level](
                skip, current, prior_delta, prior_anatomy,
            )
            skip = original_skip + self.correction_scale * (skip - original_skip)
            deltas.append(delta)
            apf_preds.append(anatomy_logit)
            prior_delta = delta
            prior_anatomy = anatomy_logit

            # decode
            current = self.decoders[level](torch.cat((current, skip), dim=1))
            deep_logits.append(self.deep_heads[3 - level](current))

        # ── error readout from the finest level ──
        error_features = []
        for delta, projection in zip(deltas, self.error_pyramid):
            feature = projection(delta)
            if feature.shape[2:] != current.shape[2:]:
                feature = F.interpolate(feature, size=current.shape[2:], mode="trilinear", align_corners=False)
            error_features.append(feature)
        error_feature = sum(error_features) / len(error_features) * self.correction_scale
        readout = torch.cat((current, error_feature), dim=1)

        logits = self.segmentation(self.fine_refinement(readout))
        # Multiplication in probability space is equivalent to adding
        # log(gate) to the tumor logit before softmax.  The gate is detached
        # here so the tumor objective cannot destabilize the anatomy head;
        # APF supervision trains the gate independently.
        liver_gate = (torch.sigmoid(apf_preds[-1] / 0.75).detach().clamp(min=0.05, max=1.0)
                      if self.use_anatomy_gate else torch.ones_like(apf_preds[-1]))
        logits[:, 2:3] = logits[:, 2:3] + 0.35 * torch.log(liver_gate)

        return {
            "logits": logits,
            "deep_logits": deep_logits,
            "cerc_deltas": deltas,
            "apf_preds": apf_preds,
        }


# ══════════════════════════ Factory ══════════════════════════

CERSUNet = CERSUNet3D          # backward-compatible alias


def get_model(in_channels: int = 1, base: int = 16,
              architecture: str = "cers", cerc_mode: str = "signed",
              dropout: float = 0.0, cerc_dropout: float = 0.0,
              ablation: str = "full") -> nn.Module:
    """Create the appropriate model for the requested architecture."""
    if architecture == "acet":
        return ACETNet3D(in_channels, base, cerc_mode, dropout, cerc_dropout, ablation=ablation)
    return CERSUNet3D(in_channels, base, architecture, cerc_mode, dropout, cerc_dropout)
