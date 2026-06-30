"""
Module 3 - Skeletonization

Converts a binary road mask into a clean
1-pixel wide road skeleton.
"""

import numpy as np
from skimage.morphology import (
    skeletonize as sk_skeletonize,
    remove_small_objects,
)


def mask_to_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    Convert binary road mask to a clean skeleton.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary image (0 or 1)

    Returns
    -------
    np.ndarray
        Skeleton image (0 or 1)
    """

    # Convert to boolean
    binary_mask = binary_mask.astype(bool)

    # Skeletonization
    skeleton = sk_skeletonize(binary_mask)

    # Remove tiny noisy branches
    skeleton = remove_small_objects(
        skeleton,
        min_size=20,
        connectivity=2
    )

    return skeleton.astype(np.uint8)
