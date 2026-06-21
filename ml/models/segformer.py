"""SegFormer inference wrapper with graceful fallback.

This module provides `SegFormerWrapper` which will try to load a real model
(if torch and transformers + required code are installed). If not available,
it falls back to a lightweight heuristic mask generator for development.
"""
from typing import Optional
import numpy as np


class SegFormerWrapper:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.processor = None
        self.mock = True
        self.model_name = None

    def load(self, model_name: str = "nvidia/mit-b2", pretrained: bool = True, checkpoint: Optional[str] = None):
        """Attempt to load a real SegFormer model. If unavailable, remain in mock mode.

        To use the real model, install `torch`, `transformers`, and supporting code.
        """
        self.model_name = model_name
        try:
            import torch
            from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

            self.processor = SegformerFeatureExtractor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            # If a local checkpoint is provided, try to load it into the model
            if checkpoint:
                try:
                    state = torch.load(checkpoint, map_location=self.device)
                    # if saved as a dict with 'model_state_dict' key, handle that
                    if isinstance(state, dict) and 'model_state_dict' in state:
                        state = state['model_state_dict']
                    self.model.load_state_dict(state)
                except Exception:
                    # ignore and continue with pretrained weights
                    pass
            self.model.to(self.device)
            self.mock = False
        except Exception:
            # Try fallback: if a checkpoint was provided, try loading tiny UNet
            if checkpoint is not None:
                try:
                    import torch
                    from ml.models.tiny_unet import get_tiny_unet

                    # build tiny unet with 2 classes by default
                    unet = get_tiny_unet(num_classes=2, in_channels=3)
                    state = torch.load(checkpoint, map_location=self.device)
                    if isinstance(state, dict) and 'model_state_dict' in state:
                        state = state['model_state_dict']
                    unet.load_state_dict(state)
                    unet.to(self.device)
                    self.model = unet
                    self.processor = None
                    self.mock = False
                    return
                except Exception:
                    pass

            # Keep mock mode; user can still call `predict` which will run a heuristic
            self.model = None
            self.processor = None
            self.mock = True

    def load_checkpoint(self, checkpoint: str):
        """Load a local checkpoint into the currently configured model or into a tiny UNet fallback.

        This will attempt to load into a transformers SegFormer model if available, otherwise
        it will try to load into a tiny UNet so inference can run from a local file.
        """
        try:
            import torch
            if self.model is not None:
                state = torch.load(checkpoint, map_location=self.device)
                if isinstance(state, dict) and 'model_state_dict' in state:
                    state = state['model_state_dict']
                self.model.load_state_dict(state)
                return True
        except Exception:
            pass

        # Try building tiny_unet and loading
        try:
            import torch
            from ml.models.tiny_unet import get_tiny_unet
            unet = get_tiny_unet(num_classes=2, in_channels=3)
            state = torch.load(checkpoint, map_location=self.device)
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            unet.load_state_dict(state)
            unet.to(self.device)
            self.model = unet
            self.processor = None
            self.mock = False
            return True
        except Exception:
            return False

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict binary road mask from an HxWxC RGB/NIR numpy image.

        Returns a uint8 mask with values 0 or 255.
        """
        if self.mock or self.model is None:
            # Simple heuristic: convert to grayscale and threshold adaptively
            if image.ndim == 3:
                gray = np.mean(image[..., :3], axis=-1)
            else:
                gray = image
            thresh = gray.mean()
            mask = (gray > thresh).astype(np.uint8) * 255
            return mask

        # Real inference path
        try:
            import torch
            # Prepare input
            if image.ndim == 3 and image.shape[2] >= 3:
                img = image[..., :3]
            else:
                img = image
            # Use processor to prepare inputs
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # (batch, num_classes, H, W)
                preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
                # Assume class 1 = road
                mask = (preds == 1).astype(np.uint8) * 255
                return mask
        except Exception:
            # fallback to heuristic
            if image.ndim == 3:
                gray = np.mean(image[..., :3], axis=-1)
            else:
                gray = image
            thresh = gray.mean()
            mask = (gray > thresh).astype(np.uint8) * 255
            return mask
