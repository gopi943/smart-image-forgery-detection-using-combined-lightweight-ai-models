"""
Central configuration for ForensicLens v2.1.
All hyperparameters, paths, and tunable settings in one place.
"""

from typing import Dict, List
from dataclasses import dataclass, field
import os


@dataclass
class Config:
    """Production configuration — every tunable parameter lives here."""

    # ── Project Paths ──────────────────────────────────────────────
    project_root: str = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir: str = "datasets"
    harmonized_dir: str = "dataset_harmonized"
    checkpoint_dir: str = "checkpoints"
    export_dir: str = "exports"
    log_dir: str = "logs"

    # ── Input ──────────────────────────────────────────────────────
    image_size: int = 384
    max_resolution: int = 4096
    supported_formats: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    )

    # ── Architecture ───────────────────────────────────────────────
    backbone_name: str = "efficientnet_b0"   # "efficientnet_b0", "efficientnet_b2"
    backbone_feature_dim: int = 320          # B0=320, B2=408
    pretrained_backbones: bool = True        # Disable in offline tests or cold environments
    fusion_dim: int = 256
    dropout_rate: float = 0.3
    use_srm: bool = False  # BayarConv replaces SRM
    use_localization: bool = True
    use_ela: bool = False           # ← DISABLED in v2.1 (harmful on phone images)
    use_attention_pool: bool = True
    cbam_reduction: int = 16
    drop_path_rate: float = 0.1

    # ── Base Paper Improvements ───────────────────────────────────
    use_self_correlation: bool = True
    correlation_topk_pct: float = 0.1
    correlation_dropout: float = 0.3

    use_multiscale: bool = True
    use_bn_inception: bool = True

    use_improved_decoder: bool = True
    decoder_dropout: float = 0.3
    edge_loss_weight: float = 0.3

    # ── Research Upgrades ─────────────────────────────────────────
    use_bayar: bool = True                  # BayarConv learnable noise filter
    use_dct: bool = False                   # ← DEFERRED in v2.1 (luxury, not needed yet)
    dct_channels: int = 128
    use_boundary_loss: bool = True
    boundary_weight_factor: float = 4.0
    use_ohem: bool = True
    ohem_ratio: float = 0.5
    use_swa: bool = False                   # ← Disabled for v2.1 (simpler training)
    swa_start_epoch: int = 25

    # ── CLIP Integration (v2.1) ───────────────────────────────────
    use_clip: bool = True                   # Frozen CLIP as global context
    clip_model: str = "ViT-B-16"            # Start small, scale if helpful
    clip_pretrained: str = "openai"         # Use OpenAI pretrained weights
    clip_proj_dim: int = 128                # CLIP projection output dimension
    # clip_dim (512 for ViT-B/16) is auto-detected at runtime

    # ── Synthetic Copy-Move Augmentation ──────────────────────────
    use_synthetic_copymove: bool = True
    synthetic_copymove_prob: float = 0.15

    # ── Training (tuned for A6000 48 GB on JarvisLabs) ─────────────
    batch_size: int = 16                    # ← Increased for A6000
    grad_accum_steps: int = 4               # Effective batch = 16 × 4 = 64
    learning_rate: float = 1e-4
    backbone_lr_factor: float = 0.1
    weight_decay: float = 1e-4
    num_epochs: int = 40
    warmup_epochs: int = 3
    label_smoothing: float = 0.05           # ← Lower for BCE heads
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    early_stop_patience: int = 10

    # ── Loss ───────────────────────────────────────────────────────
    # v2.1 uses BCEWithLogitsLoss for both heads (no focal loss needed initially)
    use_focal_loss: bool = False
    focal_gamma: float = 2.0

    # ── Loss Weights (multi-task) ──────────────────────────────────
    cm_loss_weight: float = 1.0     # Weight for copy-move head
    sp_loss_weight: float = 1.0     # Weight for splicing head
    seg_loss_weight: float = 0.3    # Weight for segmentation (Dice+BCE) loss
    tamper_loss_weight: float = 0.5 # Weight for auxiliary tamper head
    corr_consistency_weight: float = 0.2  # Source-target consistency loss
    reliability_loss_weight: float = 0.1  # Reliability head loss (post-hoc)

    # ── Mixup / CutMix ────────────────────────────────────────────
    mixup_alpha: float = 0.3
    cutmix_alpha: float = 1.0
    mixup_prob: float = 0.5
    num_workers: int = 4            # ← JarvisLabs has 8+ CPU cores

    # ── Distillation ───────────────────────────────────────────────
    use_distillation: bool = False
    distill_temperature: float = 4.0
    distill_alpha: float = 0.7
    teacher_checkpoint: str = ""

    # ── TTA ────────────────────────────────────────────────────────
    use_tta: bool = False
    tta_num_views: int = 5

    # ── Dataset ────────────────────────────────────────────────────
    train_dataset: str = "vision+defacto+casia_v2+comofod"
    test_dataset: str = "columbia"
    train_split: float = 0.85       # 85% train, 10% val, 5% calibration
    val_split: float = 0.10
    cal_split: float = 0.05
    random_seed: int = 42

    # ── Evidence Head Thresholds (calibrated in Phase 3, placeholders) ──
    cm_threshold: float = 0.5       # Placeholder — tuned by calibrate_v2.py
    sp_threshold: float = 0.5       # Placeholder — tuned by calibrate_v2.py
    cm_delta: float = 0.1           # Margin required over sp for CM call
    sp_delta: float = 0.1           # Margin required over cm for SP call

    # ── Calibration ────────────────────────────────────────────────
    calibration_method: str = "isotonic"  # "isotonic" or "platt"

    # ── Deployment ─────────────────────────────────────────────────
    onnx_opset: int = 17
    quantize_mode: str = "dynamic"
    inference_timeout_ms: int = 500

    # ── Hard Negative Mining (v2.1) ────────────────────────────────
    hn_mine_every_n_epochs: int = 3         # Mine every 3 epochs
    hn_evidence_threshold: float = 0.4      # FP threshold for mining
    hn_max_pool_size: int = 500             # Cap hard negative pool
    hn_oversample_factor: int = 3           # Oversample 3×
    hn_require_persistent: bool = True      # Only promote persistent FPs

    # ── [ADDITION 1] Auxiliary tamper head ──────────────────────────
    use_tamper_head: bool = True            # Binary tampered-vs-authentic
    # Allows using IMD2020 and ambiguous data for general forgery signal

    # ── [ADDITION 2] Source-target consistency ─────────────────────
    use_corr_consistency: bool = True       # Reward corr alignment with mask

    # ── [ADDITION 3] Reliability head ──────────────────────────────
    use_reliability_head: bool = True       # Confidence estimation

    # ── [ADDITION 4] OSN Robustness Curriculum ─────────────────────
    osn_curriculum: bool = True             # Progressive social-media augs
    osn_stage_epochs: List[int] = field(
        default_factory=lambda: [0, 10, 25]  # Stage transitions at epochs
    )
    # Stage 0 (ep 0-9):  mild JPEG + resize
    # Stage 1 (ep 10-24): heavy social-media chains
    # Stage 2 (ep 25+):  double-compression + screenshot-like degradation

    # ── [ADDITION 5] EMA ───────────────────────────────────────────
    use_ema: bool = True                    # Exponential moving average
    ema_decay: float = 0.999                # EMA decay factor

    # ── [ADDITION 5] Balanced Batch Sampling ───────────────────────
    use_balanced_sampler: bool = True
    batch_ratio_authentic: float = 0.35     # 35% authentic per batch
    batch_ratio_copymove: float = 0.30      # 30% copy-move
    batch_ratio_splicing: float = 0.30      # 30% splicing
    batch_ratio_tamper_only: float = 0.05   # 5% ambiguous/IMD2020

    # ── [ADDITION 5] Label-Quality Weights ─────────────────────────
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "vision": 1.0,        # Trusted: real phone photos
        "defacto_cm": 1.0,    # Trusted: verified subtypes
        "defacto_sp": 1.0,    # Trusted: verified subtypes
        "comofod": 0.9,       # Trusted but academic
        "casia_v2": 0.7,      # Lower: filename-based subtypes
        "imd2020": 0.5,       # Only for tamper_head
        "synthetic": 0.6,     # Generated data
    })

    # ── [ADDITION 7] Inference Optimization ────────────────────────
    use_torch_compile: bool = False         # torch.compile for inference
    use_channels_last: bool = True          # channels_last memory format

    def get_path(self, name: str) -> str:
        """Get absolute path for a named directory."""
        rel = getattr(self, name, name)
        return os.path.join(self.project_root, rel)

    @property
    def joint_dim(self) -> int:
        """Dimension of the joint vector fed to evidence heads.
        = fusion_dim + clip_proj_dim (if CLIP enabled)."""
        return self.fusion_dim + (self.clip_proj_dim if self.use_clip else 0)
