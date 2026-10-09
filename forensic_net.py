"""
ForensicNet v2.1+ — Unified Multi-Encoder Forgery Detection Model.

Architecture:
    Stream 1 (Semantic):  RGB → EfficientNet-B0 (optional multi-scale FPN)
    Stream 2 (Texture):   RGB → [BayarConv] → MobileNetV3-Small
    Stream 3 (Self-Corr): Stream 1 features → SelfCorrelation (optional)
    Stream 4 (DCT):       [Deferred]
    CLIP Context:         x_clip → frozen CLIP ViT-B/16 → projection (optional)
    Fusion: CBAM attention + BN-Inception reduce (spatial, forensic only)

    Heads:
        cm_head      — binary copy-move evidence
        sp_head      — binary splicing evidence
        tamper_head  — binary tampered-vs-authentic (auxiliary, training only)
        reliability_head — confidence / reliability estimation (post-training)

    Decoder: Forensic features only → segmentation mask (no CLIP)

Expert v2.1+ additions over base v2.1:
    1. tamper_head: uses ambiguous/IMD2020 data without poisoning subtype labels
    2. reliability_head: predicts whether the final decision is trustworthy
    3. source-target consistency score from self-correlation branch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import SemanticEncoder, MultiScaleSemanticEncoder, TextureEncoder
from models.fusion import FeatureFusion
from models.localization import LiteUNetDecoder, ImprovedDecoder
from models.self_correlation import SelfCorrelation


class ForensicNet(nn.Module):
    """
    Production forgery detection model (v2.1+).

    Returns (during training):
        cm_logit:       [B, 1] — copy-move evidence
        sp_logit:       [B, 1] — splicing evidence
        tamper_logit:   [B, 1] — any-manipulation evidence (auxiliary)
        mask:           [B, 1, H, W] — segmentation mask
        extras:         dict with corr_consistency, reliability, etc.

    Returns (during inference, via predict()):
        Full decision dict with calibrated probabilities + reliability.
    """

    def __init__(self, cfg):
        super().__init__()
        self.use_srm = cfg.use_srm
        self.use_ela = False  # Permanently disabled in v2.1
        self.use_localization = cfg.use_localization
        self.use_self_correlation = getattr(cfg, "use_self_correlation", False)
        self.use_bayar = getattr(cfg, "use_bayar", False)
        self.use_dct = getattr(cfg, "use_dct", False)
        self.use_clip = getattr(cfg, "use_clip", False)
        self.use_tamper_head = getattr(cfg, "use_tamper_head", True)
        self.use_reliability_head = getattr(cfg, "use_reliability_head", True)

        backbone_name = getattr(cfg, "backbone_name", "efficientnet_b0")
        drop_path = getattr(cfg, "drop_path_rate", 0.0)
        use_multiscale = getattr(cfg, "use_multiscale", False)
        use_bn_inception = getattr(cfg, "use_bn_inception", False)
        use_improved_decoder = getattr(cfg, "use_improved_decoder", False)
        pretrained_backbones = getattr(cfg, "pretrained_backbones", True)

        # ── Noise preprocessing for texture stream ─────────────────
        if self.use_bayar:
            from models.bayar_conv import BayarFilter
            self.noise_filter = BayarFilter(num_filters=3, kernel_size=5)
        elif self.use_srm:
            from models.srm_filter import SRMFilter
            self.noise_filter = SRMFilter()
        else:
            self.noise_filter = None

        # ── Stream 1: Semantic encoder ─────────────────────────────
        if use_multiscale:
            self.stream1 = MultiScaleSemanticEncoder(
                pretrained=pretrained_backbones,
                model_name=backbone_name,
                drop_path_rate=drop_path,
            )
        else:
            self.stream1 = SemanticEncoder(
                pretrained=pretrained_backbones,
                model_name=backbone_name,
                drop_path_rate=drop_path,
            )

        # ── Stream 2: Texture/noise encoder ────────────────────────
        self.stream2 = TextureEncoder(
            pretrained=pretrained_backbones,
            drop_path_rate=drop_path,
        )

        # ── Stream 3: Self-correlation (applied to stream 1 output)
        if self.use_self_correlation:
            feat_spatial = cfg.image_size // 32
            n_spatial = feat_spatial * feat_spatial
            self.self_corr = SelfCorrelation(
                in_channels=self.stream1.out_channels,
                out_channels=128,
                n_spatial=n_spatial,
                topk_pct=getattr(cfg, "correlation_topk_pct", 0.1),
                dropout=getattr(cfg, "correlation_dropout", 0.3),
            )

        # ── Stream 4: DCT frequency analysis (deferred) ───────────
        dct_ch = 0
        if self.use_dct:
            from models.dct_stream import DCTEncoder
            dct_out = getattr(cfg, "dct_channels", 128)
            self.dct_stream = DCTEncoder(out_channels=dct_out)
            dct_ch = dct_out

        # ── Feature fusion (spatial, forensic only — no CLIP here) ─
        corr_ch = 128 if self.use_self_correlation else 0
        self.fusion = FeatureFusion(
            semantic_ch=self.stream1.out_channels,
            texture_ch=self.stream2.out_channels,
            ela_ch=0,  # ELA removed
            correlation_ch=corr_ch,
            dct_ch=dct_ch,
            fusion_dim=cfg.fusion_dim,
            cbam_reduction=cfg.cbam_reduction,
            use_bn_inception=use_bn_inception,
        )

        # ── Global pooling for image-level features ────────────────
        self.pool = nn.AdaptiveAvgPool2d(1)

        # ── CLIP global context (frozen, optional) ─────────────────
        clip_proj_dim = 0
        if self.use_clip:
            from models.clip_context import FrozenCLIPContext
            self.clip_context = FrozenCLIPContext(
                model_name=getattr(cfg, "clip_model", "ViT-B-16"),
                pretrained=getattr(cfg, "clip_pretrained", "openai"),
                out_dim=getattr(cfg, "clip_proj_dim", 128),
            )
            clip_proj_dim = getattr(cfg, "clip_proj_dim", 128)
        self._clip_proj_dim = clip_proj_dim
        self._cached_semantic = None
        self._cached_corr = None
        self._cached_fused = None

        # ── Evidence heads ─────────────────────────────────────────
        joint_dim = cfg.fusion_dim + clip_proj_dim
        self._joint_dim = joint_dim

        self.cm_head = nn.Sequential(
            nn.Linear(joint_dim, joint_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(joint_dim // 2, 1),
        )
        self.sp_head = nn.Sequential(
            nn.Linear(joint_dim, joint_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(joint_dim // 2, 1),
        )

        # ── [ADDITION 1] Auxiliary tamper head ─────────────────────
        # Binary "any manipulation at all?" — lets us use IMD2020
        # and ambiguous data without poisoning subtype labels.
        # Shares the same joint vector as cm/sp heads.
        if self.use_tamper_head:
            self.tamper_head = nn.Sequential(
                nn.Linear(joint_dim, joint_dim // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout_rate),
                nn.Linear(joint_dim // 4, 1),
            )

        # ── [ADDITION 3] Reliability / confidence head ─────────────
        # Tiny MLP that estimates decision trustworthiness.
        # Input: [forensic_vec, cm_logit, sp_logit, |cm-sp|, max(cm,sp)]
        # Trained post-hoc on held-out "was the decision correct?" label.
        if self.use_reliability_head:
            # 4 scalar features + forensic vector
            reliability_in = cfg.fusion_dim + 4
            self.reliability_head = nn.Sequential(
                nn.Linear(reliability_in, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid(),  # Output in [0, 1] = reliability
            )

        # ── Segmentation decoder (forensic features ONLY) ──────────
        if self.use_localization:
            if use_improved_decoder:
                self.decoder = ImprovedDecoder(
                    in_channels=cfg.fusion_dim,
                    target_size=cfg.image_size,
                    dropout=getattr(cfg, "decoder_dropout", 0.3),
                )
            else:
                self.decoder = LiteUNetDecoder(
                    in_channels=cfg.fusion_dim,
                    target_size=cfg.image_size,
                )

    def _compute_streams(self, x_forensic: torch.Tensor):
        """Compute all forensic stream features."""
        feat_semantic = self.stream1(x_forensic)

        x2 = self.noise_filter(x_forensic) if self.noise_filter is not None else x_forensic
        feat_texture = self.stream2(x2)

        feat_corr = None
        if self.use_self_correlation:
            feat_corr = self.self_corr(feat_semantic)

        feat_dct = None
        if self.use_dct:
            feat_dct = self.dct_stream(x_forensic)

        return feat_semantic, feat_texture, feat_corr, feat_dct

    def _compute_corr_consistency(self, feat_corr, mask_pred):
        """
        [ADDITION 2] Source-target consistency score from self-correlation.

        Rewards high similarity in true source-target regions,
        penalizes similarity in naturally repetitive untampered regions.
        Returns a scalar consistency score per batch element.
        """
        if feat_corr is None or mask_pred is None:
            return None

        # Pool the correlation features
        corr_vec = F.adaptive_avg_pool2d(feat_corr, 1).flatten(1)  # [B, 128]
        corr_score = torch.sigmoid(corr_vec.mean(dim=1))  # [B], bounded for stable loss

        # Decoder already outputs probabilities, so use it directly.
        mask_activation = mask_pred.detach().mean(dim=[1, 2, 3], keepdim=False)  # [B]

        # Consistency = bounded correlation score aligned with predicted mask.
        consistency = corr_score * mask_activation  # [B]
        return consistency

    def forward(
        self,
        x_forensic: torch.Tensor,
        x_clip: torch.Tensor = None,
    ):
        """
        Args:
            x_forensic: RGB image with ImageNet normalization [B, 3, H, W]
            x_clip:     Same image with CLIP preprocessing [B, 3, 224, 224]

        Returns:
            cm_logit:     [B, 1]
            sp_logit:     [B, 1]
            tamper_logit: [B, 1] or None
            mask:         [B, 1, H, W] or None
            extras:       dict with optional additional info
        """
        # ── Forensic streams ───────────────────────────────────────
        feat_semantic, feat_texture, feat_corr, feat_dct = \
            self._compute_streams(x_forensic)
        self._cached_semantic = feat_semantic
        self._cached_corr = feat_corr

        # ── Spatial fusion (forensic only) ─────────────────────────
        feat_map = self.fusion(feat_semantic, feat_texture, None, feat_corr, feat_dct)
        self._cached_fused = feat_map

        # ── Segmentation decoder ───────────────────────────────────
        mask = self.decoder(feat_map) if self.use_localization else None

        # ── Pool forensic features to vector ───────────────────────
        forensic_vec = self.pool(feat_map).flatten(1)  # [B, fusion_dim]

        # ── CLIP global context ────────────────────────────────────
        if self.use_clip:
            if x_clip is None:
                clip_ctx = forensic_vec.new_zeros(forensic_vec.size(0), self._clip_proj_dim)
            else:
                clip_ctx = self.clip_context(x_clip)
            joint = torch.cat([forensic_vec, clip_ctx], dim=1)
        else:
            joint = forensic_vec

        # ── Evidence heads ─────────────────────────────────────────
        cm_logit = self.cm_head(joint)  # [B, 1]
        sp_logit = self.sp_head(joint)  # [B, 1]

        # ── [ADDITION 1] Tamper head ───────────────────────────────
        tamper_logit = None
        if self.use_tamper_head:
            tamper_logit = self.tamper_head(joint)  # [B, 1]

        # ── Extras ─────────────────────────────────────────────────
        extras = {}

        # [ADDITION 2] Source-target consistency
        if self.use_self_correlation and mask is not None:
            corr_consistency = self._compute_corr_consistency(feat_corr, mask)
            extras["corr_consistency"] = corr_consistency

        # [ADDITION 3] Reliability score
        if self.use_reliability_head:
            cm_sig = torch.sigmoid(cm_logit.detach())
            sp_sig = torch.sigmoid(sp_logit.detach())
            margin = (cm_sig - sp_sig).abs()
            max_evidence = torch.max(cm_sig, sp_sig)
            reliability_input = torch.cat([
                forensic_vec.detach(),  # [B, fusion_dim]
                cm_sig,                 # [B, 1]
                sp_sig,                 # [B, 1]
                margin,                 # [B, 1]
                max_evidence,           # [B, 1]
            ], dim=1)
            reliability = self.reliability_head(reliability_input)  # [B, 1]
            extras["reliability"] = reliability

        return cm_logit, sp_logit, tamper_logit, mask, extras

    def get_fusion_features(self, x_forensic: torch.Tensor) -> torch.Tensor:
        """Get fused forensic features for analysis (no CLIP)."""
        feat_semantic, feat_texture, feat_corr, feat_dct = \
            self._compute_streams(x_forensic)
        return self.fusion(feat_semantic, feat_texture, None, feat_corr, feat_dct)
