import numpy as np
from skimage.morphology import skeletonize as sk_skeletonize


def mask_to_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """Reduce binary mask to 1-pixel wide skeleton.
    Input: binary_mask (H, W) with values 0 or 1
    Output: skeleton (H, W) with values 0 or 1
    """
    if binary_mask.dtype != bool:
        binary_mask = binary_mask.astype(bool)
    skeleton = sk_skeletonize(binary_mask)
    return skeleton.astype(np.uint8)
