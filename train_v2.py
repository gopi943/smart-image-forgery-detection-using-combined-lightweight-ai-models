"""
ForensicLens v2.1+ — Training Script with Expert Additions.

4-phase training pipeline:
    Phase 0 (Ablation):  Compare stream combinations at 256px
    Phase 1 (Main):      Train winner at 320px, 30-40 epochs
    Phase 2 (Finetune):  Hard-negative mining at 384px, 10-15 epochs
    Phase 3 (Calibrate): Fit calibrators (run scripts/calibrate_v2.py)

Expert Additions integrated:
    1. Auxiliary tamper_head loss (uses IMD2020/ambiguous data)
    2. Source-target consistency loss (aligns self-corr with mask)
    3. OSN curriculum (progressive social-media robustness augmentation)
    4. EMA (exponential moving average of model weights)
    5. Balanced batch sampling (controlled mix per batch)
    6. Label-quality weighting (trusted vs noisy sources)

Anti-flop gates:
    Gate 1: auth_fpr < 0.15 after epoch 5 (else kill)
    Gate 2: Both heads AUROC > 0.6 after epoch 10 (else kill)
    Gate 3: CLIP projection gradient != 0 after epoch 3 (else disable)
    Gate 4: Decoder seg_loss decreasing (else freeze decoder)

Usage:
    python train_v2.py --phase main --data /root/dataset_v2 --resolution 320
"""

import argparse
import copy
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from models.forensic_net import ForensicNet
from data.dataset_v2 import ForensicDatasetV2


CHECKPOINT_CONFIG_KEYS = [
    "backbone_name",
    "fusion_dim",
    "dropout_rate",
    "use_bayar",
    "use_self_correlation",
    "use_dct",
    "use_clip",
    "clip_model",
    "clip_pretrained",
    "clip_proj_dim",
    "use_multiscale",
    "use_bn_inception",
    "use_improved_decoder",
    "use_localization",
    "use_tamper_head",
    "use_reliability_head",
    "use_corr_consistency",
]


def snapshot_model_config(cfg):
    """Persist architecture-critical flags with each checkpoint."""
    return {key: getattr(cfg, key) for key in CHECKPOINT_CONFIG_KEYS}


def apply_model_config(cfg, config_dict):
    """Restore architecture-critical flags from a checkpoint."""
    for key in CHECKPOINT_CONFIG_KEYS:
        if key in config_dict:
            setattr(cfg, key, config_dict[key])


# ═══════════════════════════════════════════════════════════════
#   EMA (Exponential Moving Average)
# ═══════════════════════════════════════════════════════════════

class ModelEMA:
    """
    Exponential moving average of model weights.

    Keeps a shadow copy of weights that's a smoothed version of the
    training weights. Used for evaluation — often gives 0.2-0.5% F1
    improvement over the raw training weights.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """Update shadow weights toward model weights."""
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.copy_(m_buf)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


# ═══════════════════════════════════════════════════════════════
#   BALANCED BATCH SAMPLER
# ═══════════════════════════════════════════════════════════════

def build_balanced_sampler(metadata, cfg):
    """
    Build WeightedRandomSampler that controls per-class mix in each batch.

    Target ratios (from config):
        authentic: 35%, copymove: 30%, splicing: 30%, tamper_only: 5%
    """
    label_counts = defaultdict(int)
    for entry in metadata:
        label_counts[entry["label"]] += 1

    total = len(metadata)

    # Target weights (inverse of ratio to achieve desired batch composition)
    target_ratios = {
        "authentic": cfg.batch_ratio_authentic,
        "copymove": cfg.batch_ratio_copymove,
        "splicing": cfg.batch_ratio_splicing,
        "tamper": cfg.batch_ratio_tamper_only,
    }

    sample_weights = []
    for entry in metadata:
        label = entry["label"]
        count = max(label_counts[label], 1)
        ratio = target_ratios.get(label, 0.05)
        weight = ratio * total / count
        sample_weights.append(weight)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total,
        replacement=True,
    )


# ═══════════════════════════════════════════════════════════════
#   OSN CURRICULUM STAGING
# ═══════════════════════════════════════════════════════════════

def get_osn_stage(epoch, stage_epochs):
    """Determine OSN curriculum stage from epoch and config."""
    stage = 0
    for s, threshold_epoch in enumerate(stage_epochs):
        if epoch >= threshold_epoch:
            stage = s
    return stage


# ═══════════════════════════════════════════════════════════════
#   METRICS
# ═══════════════════════════════════════════════════════════════

def compute_metrics(all_cm_logits, all_sp_logits, all_cm_targets, all_sp_targets):
    """Compute AUROC, AP, F1 for both heads + auth FPR."""
    cm_probs = torch.sigmoid(torch.tensor(all_cm_logits)).numpy()
    sp_probs = torch.sigmoid(torch.tensor(all_sp_logits)).numpy()
    cm_targets = np.array(all_cm_targets)
    sp_targets = np.array(all_sp_targets)

    metrics = {}

    if len(np.unique(cm_targets)) > 1:
        metrics["cm_auroc"] = roc_auc_score(cm_targets, cm_probs)
        metrics["cm_ap"] = average_precision_score(cm_targets, cm_probs)
    else:
        metrics["cm_auroc"] = 0.0
        metrics["cm_ap"] = 0.0

    if len(np.unique(sp_targets)) > 1:
        metrics["sp_auroc"] = roc_auc_score(sp_targets, sp_probs)
        metrics["sp_ap"] = average_precision_score(sp_targets, sp_probs)
    else:
        metrics["sp_auroc"] = 0.0
        metrics["sp_ap"] = 0.0

    cm_preds = (cm_probs > 0.5).astype(int)
    sp_preds = (sp_probs > 0.5).astype(int)
    metrics["cm_f1"] = f1_score(cm_targets, cm_preds, zero_division=0)
    metrics["sp_f1"] = f1_score(sp_targets, sp_preds, zero_division=0)
    
    from sklearn.metrics import accuracy_score
    metrics["cm_accuracy"] = accuracy_score(cm_targets, cm_preds)
    metrics["sp_accuracy"] = accuracy_score(sp_targets, sp_preds)

    # Authentic FPR
    auth_mask = (cm_targets == 0) & (sp_targets == 0)
    if auth_mask.sum() > 0:
        auth_fp = ((cm_preds[auth_mask] == 1) | (sp_preds[auth_mask] == 1)).sum()
        metrics["auth_fpr"] = float(auth_fp) / float(auth_mask.sum())
    else:
        metrics["auth_fpr"] = 0.0

    # Recalls
    cm_pos = (cm_targets == 1)
    metrics["cm_recall"] = float((cm_preds[cm_pos] == 1).sum()) / max(float(cm_pos.sum()), 1)
    sp_pos = (sp_targets == 1)
    metrics["sp_recall"] = float((sp_preds[sp_pos] == 1).sum()) / max(float(sp_pos.sum()), 1)

    # Ablation composite score
    metrics["ablation_score"] = (
        0.5 * (1 - metrics["auth_fpr"])
        + 0.25 * metrics["cm_auroc"]
        + 0.25 * metrics["sp_auroc"]
    )

    return metrics


# ═══════════════════════════════════════════════════════════════
#   DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_metadata(csv_path):
    """Load metadata CSV into list of dicts."""
    entries = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entries.append(row)
    return entries


def make_loader(metadata, cfg, is_train=True, resolution=None, osn_stage=0):
    """Build DataLoader with balanced sampling if training."""
    image_size = resolution or cfg.image_size
    dataset = ForensicDatasetV2(
        metadata=metadata,
        image_size=image_size,
        use_clip=cfg.use_clip,
        is_train=is_train,
        synthetic_cm_prob=cfg.synthetic_copymove_prob if is_train else 0.0,
        osn_stage=osn_stage,
        source_weights=cfg.source_weights,
    )

    sampler = None
    shuffle = is_train
    if is_train and cfg.use_balanced_sampler:
        sampler = build_balanced_sampler(metadata, cfg)
        shuffle = False  # Sampler handles ordering

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=cfg.num_workers > 0,
    )


def _grad_abs_max(module):
    """Maximum absolute gradient magnitude for a module."""
    if module is None:
        return 0.0
    grads = [
        p.grad.detach().abs().max().item()
        for p in module.parameters()
        if p.grad is not None
    ]
    return max(grads) if grads else 0.0


def _to_channels_last(tensor, cfg):
    """Use channels_last layout for 4D image tensors when enabled."""
    if cfg.use_channels_last and tensor.ndim == 4:
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor


# ═══════════════════════════════════════════════════════════════
#   TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scaler, cfg, device, epoch, ema=None):
    """Train for one epoch with all v2.1+ losses."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_cm_loss = 0.0
    total_sp_loss = 0.0
    total_tamper_loss = 0.0
    total_seg_loss = 0.0
    total_corr_loss = 0.0
    total_steps = 0
    grad_norms = []
    clip_proj_grads = []
    cm_head_grads = []
    sp_head_grads = []
    amp_enabled = cfg.use_amp and device.type == "cuda"
    total_batches = len(loader)
    final_window = total_batches % cfg.grad_accum_steps or cfg.grad_accum_steps

    cm_criterion = nn.BCEWithLogitsLoss(reduction='none')
    sp_criterion = nn.BCEWithLogitsLoss(reduction='none')
    tamper_criterion = nn.BCEWithLogitsLoss(reduction='none')

    for step, batch in enumerate(loader):
        x_forensic = batch["x_forensic"].to(device, non_blocking=True)
        x_clip = batch["x_clip"].to(device, non_blocking=True) if cfg.use_clip else None
        cm_target = batch["cm_target"].to(device, non_blocking=True)
        sp_target = batch["sp_target"].to(device, non_blocking=True)
        tamper_target = batch["tamper_target"].to(device, non_blocking=True)
        subtype_known = batch["subtype_known"].to(device, non_blocking=True)
        mask_target = batch["mask"].to(device, non_blocking=True)
        source_weight = batch["source_weight"].to(device, non_blocking=True)
        x_forensic = _to_channels_last(x_forensic, cfg)
        if x_clip is not None:
            x_clip = _to_channels_last(x_clip, cfg)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            cm_logit, sp_logit, tamper_logit, mask_pred, extras = model(
                x_forensic, x_clip
            )

            # ── CM loss (source-quality weighted) ──────────────────
            loss_cm_raw = cm_criterion(cm_logit.squeeze(1), cm_target)
            subtype_weight = source_weight * subtype_known
            subtype_denom = subtype_weight.sum().clamp_min(1e-6)
            loss_cm = (loss_cm_raw * subtype_weight).sum() / subtype_denom

            # ── SP loss (source-quality weighted) ──────────────────
            loss_sp_raw = sp_criterion(sp_logit.squeeze(1), sp_target)
            loss_sp = (loss_sp_raw * subtype_weight).sum() / subtype_denom

            # ── Tamper loss (auxiliary, source-quality weighted) ────
            loss_tamper = torch.tensor(0.0, device=device)
            if tamper_logit is not None:
                loss_tamper_raw = tamper_criterion(
                    tamper_logit.squeeze(1), tamper_target
                )
                loss_tamper = (loss_tamper_raw * source_weight).mean()

            # ── Segmentation loss (forged images only) ─────────────
            loss_seg = torch.tensor(0.0, device=device)
            if mask_pred is not None:
                forged_mask = (cm_target + sp_target) > 0
                if forged_mask.sum() > 0:
                    seg_pred = mask_pred[forged_mask].clamp(1e-6, 1 - 1e-6)
                    seg_tgt = mask_target[forged_mask]
                    bce = nn.functional.binary_cross_entropy(seg_pred, seg_tgt)
                    intersection = (seg_pred * seg_tgt).sum(dim=[1, 2, 3])
                    union = seg_pred.sum(dim=[1, 2, 3]) + seg_tgt.sum(dim=[1, 2, 3])
                    dice = 1 - ((2 * intersection + 1) / (union + 1)).mean()
                    loss_seg = 0.5 * bce + 0.5 * dice

            # ── Source-target consistency loss ──────────────────────
            loss_corr = torch.tensor(0.0, device=device)
            if "corr_consistency" in extras and cfg.use_corr_consistency:
                consistency = extras["corr_consistency"]
                # For CM images: reward high consistency
                # For authentic images: penalize high consistency
                cm_mask = cm_target > 0
                auth_mask = (cm_target + sp_target) == 0
                if cm_mask.sum() > 0:
                    # CM images should have high consistency
                    loss_corr_cm = (1 - consistency[cm_mask]).mean()
                else:
                    loss_corr_cm = torch.tensor(0.0, device=device)
                if auth_mask.sum() > 0:
                    # Auth images should have low consistency
                    loss_corr_auth = consistency[auth_mask].mean()
                else:
                    loss_corr_auth = torch.tensor(0.0, device=device)
                loss_corr = 0.5 * loss_corr_cm + 0.5 * loss_corr_auth

            # ── Total loss ─────────────────────────────────────────
            loss = (
                cfg.cm_loss_weight * loss_cm
                + cfg.sp_loss_weight * loss_sp
                + cfg.tamper_loss_weight * loss_tamper
                + cfg.seg_loss_weight * loss_seg
                + cfg.corr_consistency_weight * loss_corr
            )
            accum_divisor = final_window if step >= total_batches - final_window else cfg.grad_accum_steps
            loss = loss / accum_divisor

        # Backward
        scaler.scale(loss).backward()

        is_update_step = ((step + 1) % cfg.grad_accum_steps == 0) or ((step + 1) == total_batches)
        if is_update_step:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip_norm
            )
            grad_norms.append(grad_norm.item())
            clip_proj_grads.append(
                _grad_abs_max(getattr(getattr(model, "clip_context", None), "proj", None))
            )
            cm_head_grads.append(_grad_abs_max(model.cm_head))
            sp_head_grads.append(_grad_abs_max(model.sp_head))
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)

        total_cm_loss += loss_cm.item()
        total_sp_loss += loss_sp.item()
        total_tamper_loss += loss_tamper.item()
        total_seg_loss += loss_seg.item()
        total_corr_loss += loss_corr.item()
        total_steps += 1

        if (step + 1) % 50 == 0:
            avg_cm = total_cm_loss / total_steps
            avg_sp = total_sp_loss / total_steps
            gn = f" gn={grad_norms[-1]:.2f}" if grad_norms else ""
            print(f"  [{step+1}/{len(loader)}] cm={avg_cm:.4f} sp={avg_sp:.4f}{gn}")

    n = max(total_steps, 1)
    return {
        "cm_loss": total_cm_loss / n,
        "sp_loss": total_sp_loss / n,
        "tamper_loss": total_tamper_loss / n,
        "seg_loss": total_seg_loss / n,
        "corr_loss": total_corr_loss / n,
        "avg_grad_norm": np.mean(grad_norms) if grad_norms else 0.0,
        "clip_proj_grad": np.mean(clip_proj_grads) if clip_proj_grads else 0.0,
        "cm_head_grad": np.mean(cm_head_grads) if cm_head_grads else 0.0,
        "sp_head_grad": np.mean(sp_head_grads) if sp_head_grads else 0.0,
    }


@torch.no_grad()
def validate(model, loader, cfg, device):
    """Run validation. Uses EMA model if available."""
    model.eval()
    all_cm, all_sp, all_cm_t, all_sp_t = [], [], [], []
    amp_enabled = cfg.use_amp and device.type == "cuda"

    for batch in loader:
        x_f = batch["x_forensic"].to(device, non_blocking=True)
        x_c = batch["x_clip"].to(device, non_blocking=True) if cfg.use_clip else None
        known_mask = batch["subtype_known"] > 0
        x_f = _to_channels_last(x_f, cfg)
        if x_c is not None:
            x_c = _to_channels_last(x_c, cfg)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            cm_logit, sp_logit, _, _, _ = model(x_f, x_c)

        if known_mask.any():
            keep = known_mask.nonzero(as_tuple=False).squeeze(1)
            all_cm.extend(cm_logit.squeeze(1)[keep].cpu().tolist())
            all_sp.extend(sp_logit.squeeze(1)[keep].cpu().tolist())
            all_cm_t.extend(batch["cm_target"][keep].tolist())
            all_sp_t.extend(batch["sp_target"][keep].tolist())

    return compute_metrics(all_cm, all_sp, all_cm_t, all_sp_t)


# ═══════════════════════════════════════════════════════════════
#   ANTI-FLOP GATES
# ═══════════════════════════════════════════════════════════════

def check_gates(epoch, metrics, train_stats, model, cfg):
    """Check anti-flop gates. Returns (should_continue, message)."""
    if epoch >= 5 and metrics.get("auth_fpr", 1.0) > 0.15:
        return False, (
            f"GATE 1 FAILED: auth_fpr={metrics['auth_fpr']:.3f} > 0.15. "
            f"Model accuses too many authentic images."
        )
    if epoch >= 10:
        if metrics.get("cm_auroc", 0) < 0.6 or metrics.get("sp_auroc", 0) < 0.6:
            return False, (
                f"GATE 2 FAILED: cm_auroc={metrics.get('cm_auroc', 0):.3f}, "
                f"sp_auroc={metrics.get('sp_auroc', 0):.3f}. Heads not learning."
            )
    if epoch >= 3 and cfg.use_clip and hasattr(model, "clip_context"):
        if train_stats.get("clip_proj_grad", 0.0) <= 0:
            return False, "GATE 3 FAILED: CLIP projection received zero gradient."
    return True, "All gates passed."


# ═══════════════════════════════════════════════════════════════
#   HARD NEGATIVE MINING
# ═══════════════════════════════════════════════════════════════

def mine_hard_negatives(model, train_metadata, cfg, device, resolution, prev_fps=None):
    """Find authentic images model falsely accuses. Only keep persistent FPs."""
    model.eval()
    auth_meta = [e for e in train_metadata if e["label"] == "authentic"]
    if not auth_meta:
        return set(), set()

    loader = make_loader(auth_meta, cfg, is_train=False, resolution=resolution)
    fp_paths = set()
    amp_enabled = cfg.use_amp and device.type == "cuda"

    with torch.no_grad():
        for batch in loader:
            x_f = batch["x_forensic"].to(device, non_blocking=True)
            x_c = batch["x_clip"].to(device, non_blocking=True) if cfg.use_clip else None
            x_f = _to_channels_last(x_f, cfg)
            if x_c is not None:
                x_c = _to_channels_last(x_c, cfg)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                cm_logit, sp_logit, _, _, _ = model(x_f, x_c)
            cm_p = torch.sigmoid(cm_logit.squeeze(1))
            sp_p = torch.sigmoid(sp_logit.squeeze(1))
            for i, path in enumerate(batch["path"]):
                if cm_p[i].item() > cfg.hn_evidence_threshold or \
                   sp_p[i].item() > cfg.hn_evidence_threshold:
                    fp_paths.add(path)

    if prev_fps is not None and cfg.hn_require_persistent:
        persistent = fp_paths & prev_fps
    else:
        persistent = fp_paths

    if len(persistent) > cfg.hn_max_pool_size:
        persistent = set(list(persistent)[:cfg.hn_max_pool_size])

    print(f"  HN: {len(fp_paths)} current FPs, {len(persistent)} persistent")
    return persistent, fp_paths


def oversample_hard_negatives(metadata, hn_paths, factor=3):
    """Add hard negatives to metadata with oversampling."""
    extra = []
    for entry in metadata:
        if entry["path"] in hn_paths:
            for _ in range(factor - 1):
                extra.append(entry.copy())
    max_extra = int(len(metadata) * 0.15)
    if max_extra > 0 and len(extra) > max_extra:
        extra = extra[:max_extra]
    return metadata + extra


# ═══════════════════════════════════════════════════════════════
#   OPTIMIZER & SCHEDULER
# ═══════════════════════════════════════════════════════════════

def build_optimizer(model, cfg):
    """Build AdamW with differential learning rates."""
    backbone_params, other_params = [], []
    for name, param in model.named_parameters():
        if name.startswith("clip_context.clip_model"):
            continue
        if "stream1" in name or "stream2" in name:
            backbone_params.append(param)
        elif param.requires_grad:
            other_params.append(param)

    return torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.learning_rate * cfg.backbone_lr_factor},
        {"params": other_params, "lr": cfg.learning_rate},
    ], weight_decay=cfg.weight_decay)


def build_scheduler(optimizer, cfg):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        progress = (epoch - cfg.warmup_epochs) / max(cfg.num_epochs - cfg.warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ═══════════════════════════════════════════════════════════════
#   MAIN TRAINING PHASE
# ═══════════════════════════════════════════════════════════════

def run_phase(phase, cfg, data_dir, resolution, resume=None, streams=None):
    """Execute a training phase."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Phase: {phase.upper()} | Resolution: {resolution}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"{'='*60}")

    # Phase-specific overrides
    cfg.image_size = resolution
    if phase == "ablation":
        cfg.num_epochs = 15
        cfg.batch_size = 24
    elif phase == "main":
        cfg.num_epochs = 40
    elif phase == "finetune":
        cfg.num_epochs = 15
        cfg.learning_rate = cfg.learning_rate * 0.1

    if streams:
        cfg.use_bayar = "noise" in streams
        cfg.use_self_correlation = "corr" in streams
        cfg.use_clip = "clip" in streams

    # Load data
    train_meta = load_metadata(os.path.join(data_dir, "metadata_train.csv"))
    val_meta = load_metadata(os.path.join(data_dir, "metadata_val.csv"))
    print(f"Train: {len(train_meta)} | Val: {len(val_meta)}")

    resume_ckpt = None
    if resume and os.path.exists(resume):
        resume_ckpt = torch.load(resume, map_location=device, weights_only=False)
        apply_model_config(cfg, resume_ckpt.get("config", {}))
        print(f"Restored model config from: {resume}")

    # Build model
    model = ForensicNet(cfg).to(device)

    # channels_last memory format for speed
    if cfg.use_channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("Using channels_last memory format")

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total_p:,} total, {train_p:,} trainable")

    # Resume
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"], strict=False)
        print(f"Resumed from: {resume}")

    # Freeze lower backbone in main phase
    if phase == "main":
        frozen = 0
        for name, param in model.stream1.named_parameters():
            if "backbone.0" in name or "backbone.1" in name:
                param.requires_grad = False
                frozen += 1
        print(f"Froze {frozen} lower backbone params")

    # Build optimizer, scheduler, scaler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")

    # EMA
    ema = None
    if cfg.use_ema:
        ema = ModelEMA(model, decay=cfg.ema_decay)
        print(f"EMA enabled (decay={cfg.ema_decay})")

    # Checkpointing
    ckpt_dir = os.environ.get("CHECKPOINT_DIR", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_score = 0.0
    best_epoch = 0
    patience_counter = 0
    prev_fps = None
    history = []

    # ── Training loop ──────────────────────────────────────────
    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        print(f"\n--- Epoch {epoch}/{cfg.num_epochs} ---")

        # OSN curriculum staging
        osn_stage = get_osn_stage(epoch, cfg.osn_stage_epochs) if cfg.osn_curriculum else 0
        if cfg.osn_curriculum:
            print(f"  OSN stage: {osn_stage}")

        if phase == "main" and epoch == 6:
            unfrozen = 0
            for name, param in model.stream1.named_parameters():
                if "backbone.0" in name or "backbone.1" in name:
                    param.requires_grad = True
                    unfrozen += 1
            print(f"  Unfroze {unfrozen} lower backbone params")

        # Build train loader (rebuild each epoch to update OSN stage)
        train_loader = make_loader(
            train_meta, cfg, is_train=True, resolution=resolution, osn_stage=osn_stage
        )

        # Hard-negative mining (finetune phase, every N epochs)
        if phase == "finetune" and epoch > 5 and epoch % cfg.hn_mine_every_n_epochs == 0:
            hn_paths, prev_fps = mine_hard_negatives(
                model, train_meta, cfg, device, resolution, prev_fps
            )
            if hn_paths:
                aug_meta = oversample_hard_negatives(
                    train_meta, hn_paths, cfg.hn_oversample_factor
                )
                train_loader = make_loader(
                    aug_meta, cfg, is_train=True, resolution=resolution, osn_stage=osn_stage
                )
                print(f"  Train+HN: {len(aug_meta)}")

        # Train
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scaler, cfg, device, epoch, ema=ema
        )
        scheduler.step()

        dt = time.time() - t0
        print(f"  Train: cm={train_stats['cm_loss']:.4f} sp={train_stats['sp_loss']:.4f} "
              f"tamp={train_stats['tamper_loss']:.4f} seg={train_stats['seg_loss']:.4f} "
              f"corr={train_stats['corr_loss']:.4f} gn={train_stats['avg_grad_norm']:.2f} "
              f"clip_g={train_stats['clip_proj_grad']:.2e} "
              f"t={dt:.0f}s")

        # Validate
        val_interval = 1 if phase == "ablation" else 2
        if epoch % val_interval == 0 or epoch == cfg.num_epochs:
            val_loader = make_loader(val_meta, cfg, is_train=False, resolution=resolution)

            # Validate with EMA model if available
            val_model = ema.shadow if ema is not None else model
            metrics = validate(val_model, val_loader, cfg, device)

            print(f"  Val: cm_auroc={metrics['cm_auroc']:.4f} "
                  f"sp_auroc={metrics['sp_auroc']:.4f} "
                  f"auth_fpr={metrics['auth_fpr']:.4f} "
                  f"cm_f1={metrics['cm_f1']:.4f} sp_f1={metrics['sp_f1']:.4f} "
                  f"score={metrics['ablation_score']:.4f}")

            # Anti-flop gates
            ok, gate_msg = check_gates(epoch, metrics, train_stats, model, cfg)
            if not ok:
                print(f"\n{'!'*60}\n  {gate_msg}\n{'!'*60}")
                break

            # Best model tracking
            score = metrics["ablation_score"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                patience_counter = 0

                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": ema.shadow.state_dict() if ema is not None else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "resolution": resolution,
                    "config": snapshot_model_config(cfg),
                }
                if ema is not None:
                    save_dict["raw_model_state_dict"] = model.state_dict()
                    save_dict["ema_state_dict"] = ema.state_dict()

                ckpt_path = os.path.join(ckpt_dir, f"best_{resolution}.pth")
                torch.save(save_dict, ckpt_path)
                print(f"  ★ New best! score={score:.4f} → {ckpt_path}")
            else:
                patience_counter += 1

            history.append({"epoch": epoch, "phase": phase, **train_stats, **metrics})

            if patience_counter >= cfg.early_stop_patience:
                print(f"\n  Early stop: no improvement for {cfg.early_stop_patience} epochs")
                break

        # Periodic checkpoint
        if epoch % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "resolution": resolution,
                    "config": snapshot_model_config(cfg),
                },
                os.path.join(ckpt_dir, f"epoch_{epoch}_{resolution}.pth"),
            )

    # Save history
    hist_path = os.path.join(ckpt_dir, f"history_{phase}_{resolution}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Phase {phase.upper()} complete! Best: {best_score:.4f} @ epoch {best_epoch}")
    print(f"{'='*60}")

    return best_score, best_epoch


def run_ablation(cfg, data_dir):
    """Run 4 ablation experiments and pick the winner."""
    experiments = {
        "A_rgb_only":           set(),
        "B_rgb_noise":          {"noise"},
        "C_rgb_noise_corr":     {"noise", "corr"},
        "D_rgb_noise_corr_clip": {"noise", "corr", "clip"},
    }
    results = {}
    for name, stream_set in experiments.items():
        print(f"\n{'#'*60}\n# ABLATION: {name}\n{'#'*60}")
        score, epoch = run_phase(
            "ablation", Config(), data_dir, resolution=256, streams=stream_set
        )
        results[name] = {"score": score, "epoch": epoch, "streams": list(stream_set)}

    winner = max(results, key=lambda k: results[k]["score"])
    print(f"\n{'='*60}\nABLATION WINNER: {winner} (score={results[winner]['score']:.4f})")
    print(f"{'='*60}")

    ckpt_dir = os.environ.get("CHECKPOINT_DIR", "checkpoints")
    with open(os.path.join(ckpt_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return winner, set(results[winner]["streams"])


# ═══════════════════════════════════════════════════════════════
#   MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ForensicLens v2.1+ Training")
    parser.add_argument("--phase", required=True,
                        choices=["ablation", "main", "finetune"])
    parser.add_argument("--data", required=True, help="Dataset directory")
    parser.add_argument("--resolution", type=int, default=320)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--streams", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.environ["CHECKPOINT_DIR"] = args.checkpoint_dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    cfg = Config()

    if args.phase == "ablation":
        run_ablation(cfg, args.data)
    elif args.phase == "main":
        stream_set = set(args.streams.split(",")) if args.streams else None
        run_phase("main", cfg, args.data, args.resolution,
                  resume=args.resume, streams=stream_set)
    elif args.phase == "finetune":
        if not args.resume:
            best_320 = os.path.join(args.checkpoint_dir, "best_320.pth")
            if os.path.exists(best_320):
                args.resume = best_320
                print(f"Auto-resuming from: {best_320}")
            else:
                print("[ERROR] Fine-tune requires --resume or best_320.pth")
                sys.exit(1)
        run_phase("finetune", cfg, args.data, args.resolution, resume=args.resume)


if __name__ == "__main__":
    main()
