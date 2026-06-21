import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(occlusion_prob: float = 0.4, max_occlusion: float = 0.25):
    """Return an albumentations Compose with synthetic occlusion (CoarseDropout).

    - occlusion_prob: probability to apply coarse dropout
    - max_occlusion: max proportion of image area to drop
    """

    # CoarseDropout parameters: max_holes, max_height, max_width
    # We'll convert max_occlusion (fraction of area) into approximate max_holes and sizes.
    max_holes = 5
    max_size = max_occlusion

    aug_list = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ColorJitter(p=0.3),
        A.GaussNoise(p=0.2),
    ]

    aug_list.append(
        A.OneOf([
            A.CoarseDropout(max_holes=max_holes, max_height=int(32 * max_size * 8), max_width=int(32 * max_size * 8), p=1.0),
            A.Cutout(num_holes=8, max_h_size=int(32 * max_size * 8), max_w_size=int(32 * max_size * 8), p=1.0)
        ], p=occlusion_prob)
    )

    # Keep masks untouched other than geometric transforms
    return A.Compose(aug_list)
