import os
from PIL import Image, ImageDraw
import random


def _draw_road(draw, w, h):
    # draw a random polyline across the image
    n = random.randint(2, 6)
    points = []
    for i in range(n):
        x = int(w * (i / (n - 1)))
        y = random.randint(int(h * 0.2), int(h * 0.8))
        points.append((x, y))
    draw.line(points, fill=(255, 255, 255), width=6)


def generate_sample_dataset(root: str = 'ml/data/sample', num_train: int = 8, num_val: int = 2, size=(256, 256)):
    images_train = os.path.join(root, 'images', 'train')
    masks_train = os.path.join(root, 'masks', 'train')
    images_val = os.path.join(root, 'images', 'val')
    masks_val = os.path.join(root, 'masks', 'val')

    for p in [images_train, masks_train, images_val, masks_val]:
        os.makedirs(p, exist_ok=True)

    def make_example(path_img, path_mask, idx):
        img = Image.new('RGB', size, (30, 30, 30))
        mask = Image.new('L', size, 0)
        draw_img = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)
        _draw_road(draw_img, size[0], size[1])
        _draw_road(draw_mask, size[0], size[1])
        img.save(os.path.join(path_img, f'{idx:04d}.png'))
        mask.save(os.path.join(path_mask, f'{idx:04d}.png'))

    for i in range(num_train):
        make_example(images_train, masks_train, i)
    for i in range(num_val):
        make_example(images_val, masks_val, i + num_train)

    print('Sample dataset generated at', root)


if __name__ == '__main__':
    generate_sample_dataset()
