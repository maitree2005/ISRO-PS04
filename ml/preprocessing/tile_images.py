"""
Module 1 - Satellite Image Ingestion

Reads a GeoTIFF satellite image,
normalizes pixel values,
applies CLAHE enhancement,
splits the image into 512x512 tiles,
and saves them for SegFormer inference.
"""

from pathlib import Path

import cv2
import numpy as np
import rasterio


# ---------------------------------------------
# Project Directories
# ---------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "ml" / "data" / "raw"
TILE_DIR = BASE_DIR / "ml" / "data" / "tiles"

RAW_DIR.mkdir(parents=True, exist_ok=True)
TILE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------
# Read GeoTIFF
# ---------------------------------------------

def read_geotiff(image_path):
    """
    Reads a GeoTIFF image and returns:

    image : RGB numpy array
    profile : raster metadata
    """

    with rasterio.open(image_path) as src:

        profile = src.profile

        image = src.read([1, 2, 3])

        image = np.transpose(image, (1, 2, 0))

    return image, profile
# ---------------------------------------------
# Normalize Image
# ---------------------------------------------

def normalize_image(image):
    """
    Normalize image values to 0-255.
    """

    image = image.astype(np.float32)

    image -= image.min()

    image /= (image.max() + 1e-8)

    image *= 255

    return image.astype(np.uint8)
# ---------------------------------------------
# CLAHE Enhancement
# ---------------------------------------------

def apply_clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to improve road visibility.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))

    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    return enhanced

# ---------------------------------------------
# Tile Image
# ---------------------------------------------

def tile_image(image, tile_size=512):
    """
    Split image into 512x512 tiles.

    Returns:
        list of (tile, row, col)
    """

    tiles = []

    height, width = image.shape[:2]

    for y in range(0, height, tile_size):

        for x in range(0, width, tile_size):

            tile = image[y:y+tile_size, x:x+tile_size]

            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                continue

            tiles.append((tile, y, x))

    return tiles
# ---------------------------------------------
# Save Tiles
# ---------------------------------------------

def save_tiles(tiles, output_dir=TILE_DIR):
    """
    Save image tiles as PNG files.

    Returns:
        List of saved tile paths.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for tile, row, col in tiles:

        filename = f"tile_r{row}_c{col}.png"

        filepath = output_dir / filename

        cv2.imwrite(str(filepath), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

        saved_files.append(filepath)

    return saved_files
# ---------------------------------------------
# Complete Preprocessing Pipeline
# ---------------------------------------------

def process_image(image_path):
    """
    Complete preprocessing pipeline.

    GeoTIFF
        ↓
    Read
        ↓
    Normalize
        ↓
    CLAHE
        ↓
    Tile
        ↓
    Save

    Returns:
        List of saved tile paths.
    """

    print("Reading GeoTIFF...")

    image, profile = read_geotiff(image_path)

    print("Normalizing image...")

    image = normalize_image(image)

    print("Applying CLAHE...")

    image = apply_clahe(image)

    print("Generating tiles...")

    tiles = tile_image(image)

    print(f"Generated {len(tiles)} tiles")

    saved = save_tiles(tiles)

    print(f"Saved {len(saved)} tiles")

    return saved
if __name__ == "__main__":

    sample = RAW_DIR / "sample.tif"

    if sample.exists():

        process_image(sample)

    else:

        print("\nNo sample image found!")

        print(f"Place a GeoTIFF here:\n{RAW_DIR}")