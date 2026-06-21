import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A


class RoadSegmentationDataset(Dataset):
    """Simple dataset reading image-mask pairs from directories.

    Expects directory structure:
    - images/xxx.png
    - masks/xxx.png
    """

    def __init__(self, images_dir: str, masks_dir: str, transforms: A.Compose = None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.ids = [p.stem for p in self.images_dir.glob("*.png")] + [p.stem for p in self.images_dir.glob("*.jpg")]
        self.ids = sorted(list(set(self.ids)))
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        id_ = self.ids[idx]
        img_path = self.images_dir / f"{id_}.png"
        if not img_path.exists():
            img_path = self.images_dir / f"{id_}.jpg"
        mask_path = self.masks_dir / f"{id_}.png"
        if not mask_path.exists():
            mask_path = self.masks_dir / f"{id_}.png"
        # Return raw HWC uint8 image and HxW binary mask (values 0 or 1).
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype('uint8')

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = image.astype('uint8')
        mask = mask.astype('uint8')

        return image, mask
