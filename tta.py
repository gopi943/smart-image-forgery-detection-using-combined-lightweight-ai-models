"""
Test-Time Augmentation (TTA) for improved inference.

Runs multiple augmented views of the same image through the model
and averages evidence logits for more robust results. The v2.1 model uses
two binary subtype heads, so TTA returns averaged copy-move and splicing
logits instead of a legacy multiclass softmax.
"""

import torch


class TTAPredictor:
    """
    Test-Time Augmentation predictor.

    Applies multiple geometric transforms to the input image,
    runs each through the model, and averages logits.

    Transforms:
        1. Original
        2. Horizontal flip
        3. Vertical flip
        4. 90° rotation
        5. 270° rotation
    """

    def __init__(self, num_views: int = 5):
        self.num_views = min(num_views, 5)

    @staticmethod
    def _hflip(x):
        return torch.flip(x, dims=[3])

    @staticmethod
    def _vflip(x):
        return torch.flip(x, dims=[2])

    @staticmethod
    def _rot90(x):
        return torch.rot90(x, k=1, dims=[2, 3])

    @staticmethod
    def _rot270(x):
        return torch.rot90(x, k=3, dims=[2, 3])

    def predict(self, model, x_forensic, x_clip=None):
        """
        Run TTA prediction.

        Args:
            model: ForensicNet model (eval mode)
            x_forensic: Forensic tensor (B, 3, H, W)
            x_clip: Optional CLIP tensor (B, 3, 224, 224)

        Returns:
            avg_cm_logit: Averaged copy-move logits (B, 1)
            avg_sp_logit: Averaged splicing logits (B, 1)
            avg_tamper_logit: Averaged tamper logits (B, 1) or None
        """
        transforms_list = [
            lambda x: x,           # Original
            self._hflip,            # Horizontal flip
            self._vflip,            # Vertical flip
            self._rot90,            # 90° rotation
            self._rot270,           # 270° rotation
        ][:self.num_views]

        all_cm_logits = []
        all_sp_logits = []
        all_tamper_logits = []
        model.eval()

        with torch.no_grad():
            for tfm in transforms_list:
                aug_forensic = tfm(x_forensic)
                aug_clip = tfm(x_clip) if x_clip is not None else None
                cm_logit, sp_logit, tamper_logit, _, _ = model(
                    aug_forensic, aug_clip
                )
                all_cm_logits.append(cm_logit)
                all_sp_logits.append(sp_logit)
                if tamper_logit is not None:
                    all_tamper_logits.append(tamper_logit)

        avg_cm_logit = torch.stack(all_cm_logits).mean(dim=0)
        avg_sp_logit = torch.stack(all_sp_logits).mean(dim=0)
        avg_tamper_logit = (
            torch.stack(all_tamper_logits).mean(dim=0)
            if all_tamper_logits else None
        )
        return avg_cm_logit, avg_sp_logit, avg_tamper_logit
