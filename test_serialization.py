"""Serialization round-trip tests for the v2.1 model path."""

from pathlib import Path

import pytest
import torch

from config import Config
from models.forensic_net import ForensicNet
from utils.runtime_v2 import load_model_bundle


ARTIFACT_DIR = Path("tests") / "_artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


def make_cfg(localization: bool = False):
    cfg = Config()
    cfg.image_size = 96
    cfg.use_clip = False
    cfg.pretrained_backbones = False
    cfg.use_localization = localization
    return cfg


def test_checkpoint_roundtrip_basic():
    cfg = make_cfg(localization=False)
    model = ForensicNet(cfg)
    model.eval()
    x = torch.randn(1, 3, cfg.image_size, cfg.image_size)

    with torch.no_grad():
        outputs_before = model(x)

    ckpt_path = ARTIFACT_DIR / "roundtrip_basic.pth"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    try:
        model2 = ForensicNet(cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model2.load_state_dict(ckpt["model_state_dict"], strict=False)
        model2.eval()

        with torch.no_grad():
            outputs_after = model2(x)

        for before, after in zip(outputs_before[:4], outputs_after[:4]):
                if before is None:
                    assert after is None
                else:
                    assert torch.allclose(before, after, atol=1e-3, rtol=1e-5)
    finally:
        try:
            ckpt_path.unlink(missing_ok=True)
        except PermissionError:
            pass


def test_checkpoint_roundtrip_with_localization():
    cfg = make_cfg(localization=True)
    model = ForensicNet(cfg)
    model.eval()
    x = torch.randn(1, 3, cfg.image_size, cfg.image_size)

    with torch.no_grad():
        outputs_before = model(x)

    ckpt_path = ARTIFACT_DIR / "roundtrip_loc.pth"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    try:
        model2 = ForensicNet(cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model2.load_state_dict(ckpt["model_state_dict"], strict=False)
        model2.eval()

        with torch.no_grad():
            outputs_after = model2(x)

        for before, after in zip(outputs_before[:4], outputs_after[:4]):
                if before is None:
                    assert after is None
                else:
                    assert torch.allclose(before, after, atol=1e-3, rtol=1e-5)
    finally:
        try:
            ckpt_path.unlink(missing_ok=True)
        except PermissionError:
            pass


def test_load_model_bundle_restores_resolution():
    cfg = make_cfg(localization=False)
    model = ForensicNet(cfg)

    ckpt_path = ARTIFACT_DIR / "bundle_restore.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "resolution": cfg.image_size,
            "config": {"use_clip": False, "use_localization": False},
        },
        ckpt_path,
    )

    try:
        load_cfg = Config()
        load_cfg.pretrained_backbones = False
        _, restored_cfg, _, _, _ = load_model_bundle(str(ckpt_path), cfg=load_cfg)
        assert restored_cfg.image_size == cfg.image_size
        assert restored_cfg.use_clip is False
    finally:
        try:
            ckpt_path.unlink(missing_ok=True)
        except PermissionError:
            pass


def test_corrupted_checkpoint_does_not_crash():
    path = ARTIFACT_DIR / "corrupt_checkpoint.pth"
    path.write_bytes(b"this is not a valid pytorch checkpoint")

    try:
        with pytest.raises(Exception):
            torch.load(path, map_location="cpu", weights_only=False)
    finally:
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            pass


def test_truncated_checkpoint_does_not_crash():
    cfg = make_cfg(localization=False)
    model = ForensicNet(cfg)
    path = ARTIFACT_DIR / "truncated_checkpoint.pth"
    torch.save({"model_state_dict": model.state_dict()}, path)

    try:
        path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
        with pytest.raises(Exception):
            torch.load(path, map_location="cpu", weights_only=False)
    finally:
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            pass


def test_nonstrict_loading_with_extra_decoder_keys():
    cfg_with_decoder = make_cfg(localization=True)
    cfg_without_decoder = make_cfg(localization=False)
    model_with_decoder = ForensicNet(cfg_with_decoder)
    model_without_decoder = ForensicNet(cfg_without_decoder)

    path = ARTIFACT_DIR / "nonstrict_decoder.pth"
    torch.save(model_with_decoder.state_dict(), path)

    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        incompatible = model_without_decoder.load_state_dict(state_dict, strict=False)
        assert not incompatible.missing_keys
        assert incompatible.unexpected_keys
    finally:
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            pass
