"""ML inference utilities for Route Resilience.

Provides a simple `predict_mask_from_image` function that accepts a numpy image
or file-like object and returns a binary mask. Uses `SegFormerWrapper` if available.
"""
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image

from ml.models.segformer import SegFormerWrapper


def predict_mask_from_image(img_input: Union[str, Path, np.ndarray, Image.Image], device: str = "cpu", checkpoint: str = None, model_name: str = None) -> np.ndarray:
    """Load image and run segmentation inference, returning uint8 mask (0/255).

    `img_input` can be a path or a numpy array or PIL Image.
    """
    if isinstance(img_input, (str, Path)):
        img = Image.open(str(img_input))
        arr = np.array(img)
    elif isinstance(img_input, Image.Image):
        arr = np.array(img_input)
    elif isinstance(img_input, np.ndarray):
        arr = img_input
    else:
        raise ValueError("Unsupported img_input type")

    wrapper = SegFormerWrapper(device=device)
    # try to load specified model_name and/or checkpoint if provided
    if model_name is None:
        model_name = "nvidia/mit-b2"
    wrapper.load(model_name, checkpoint=checkpoint)
    mask = wrapper.predict(arr)
    return mask


def save_mask_png(mask: np.ndarray, out_path: Union[str, Path]):
    out = Image.fromarray(mask.astype('uint8'))
    out.save(str(out_path), format='PNG')
