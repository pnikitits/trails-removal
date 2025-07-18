# Satellite Trail Removal Script for Siril
# Uses a trained UNet model to remove satellite trails from astronomical images.
#
# Note:
#   - Supports only single-channel (grayscale) images for now.
#   - Input image must be debayered before running this script.
#   - This is under development and needs more training and testing.
#   - The model path needs to be updated to the model location on your system.
#
# Author: Pierre Nikitits
# GitHub: https://github.com/pnikitits/trails-removal


import sirilpy as s

# === Imports ===
s.ensure_installed("numpy", "torch", "torchvision")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

siril = s.SirilInterface()
siril.connect()

# === Constants to be updated ===
MODEL_PATH = "results/20250717-144407/model_epoch_1500.pth"
DEVICE = "mps"


# === From: utils.py ===
def normalize(img: np.ndarray, min_val: float = None, max_val: float = None) -> tuple:
    img = img.astype(np.float32)
    if min_val is None or max_val is None:
        min_val = np.min(img)
        max_val = np.max(img)
    norm = (img - min_val) / (max_val - min_val + 1e-8)
    return norm, min_val, max_val


def denormalize(norm_img: np.ndarray, min_val: float, max_val: float):
    return norm_img * (max_val - min_val) + min_val


def split_into_tiles(image, tile_size: int = 512, overlap: int = 64) -> tuple:
    """
    Splits a large image into overlapping tiles.

    Returns
    -------
    tiles
        list of image tiles
    coords
        list of (x, y) coordinates for reassembly
    """
    h, w = image.shape
    step = tile_size - overlap
    tiles, coords = [], []

    y_positions = list(range(0, h - tile_size + 1, step))
    x_positions = list(range(0, w - tile_size + 1, step))

    if y_positions[-1] + tile_size < h:
        y_positions.append(h - tile_size)
    if x_positions[-1] + tile_size < w:
        x_positions.append(w - tile_size)

    for y in y_positions:
        for x in x_positions:
            tile = image[y : y + tile_size, x : x + tile_size]
            tiles.append(tile)
            coords.append((x, y))

    return tiles, coords


def reassemble_from_tiles(
    tiles: list,
    coords: list,
    image_shape: tuple,
    tile_size: int = 512,
):
    """
    Reassembles the full image from tiles using averaging in overlaps.

    Parameters
    ----------
    tiles
        list of output tiles (numpy arrays)
    coords
        matching list of (x, y) coordinates
    image_shape
        (H, W) of original image

    Returns
    -------
    Full reassembled image
    """
    output = np.zeros(image_shape, dtype=np.float32)
    weight = np.zeros(image_shape, dtype=np.float32)

    for tile, (x, y) in zip(tiles, coords):
        output[y : y + tile_size, x : x + tile_size] += tile
        weight[y : y + tile_size, x : x + tile_size] += 1.0

    return output / np.maximum(weight, 1e-8)


def rgb_to_grayscale(image):
    """
    Convert an RGB image of shape (H, W, 3) to grayscale (H, W).

    Parameters
    ----------
    image (numpy.ndarray):
        RGB image as a numpy array.

    Returns
    -------
    numpy.ndarray:
        Grayscale image.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must have shape (H, W, 3)")
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])


# === From: model.py ===
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.dec4 = conv_block(512 + 256, 256)
        self.dec3 = conv_block(256 + 128, 128)
        self.dec2 = conv_block(128 + 64, 64)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d4 = F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        out = self.final(d2)
        return torch.sigmoid(out)


def load_model(weights_path: str, device=None):
    model = UNet()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def clean_satellite_trail(
    img: np.ndarray,
    model: torch.nn.Module,
    tile_size: int = 512,
    overlap: int = 64,
    device: str = "mps",
) -> np.ndarray:
    """
    Cleans satellite trails from a .fits image using a trained model and tiling.

    Parameters
    ----------
    fits_path : np.ndarray
        The input image.
    model : torch.nn.Module
        Trained model for trail removal.
    tile_size : int
        Size of tiles to process.
    overlap : int
        Overlap between tiles.
    device : str
        'cpu', 'cuda', or 'mps'.

    Returns
    -------
    np.ndarray
        Cleaned image in [0, 1] range.
    """
    shape = img.shape

    tiles, coords = split_into_tiles(img, tile_size=tile_size, overlap=overlap)

    model.eval()
    model.to(device)

    cleaned_tiles = []
    for tile in tiles:

        with torch.no_grad():
            tile_tensor = (
                torch.tensor(tile, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            predicted_residual = model(tile_tensor)
            cleaned = tile_tensor - predicted_residual
            cleaned_tiles.append(cleaned.cpu().numpy().squeeze(0).squeeze(0))

    cleaned_img = reassemble_from_tiles(cleaned_tiles, coords, shape, tile_size)

    return np.clip(cleaned_img, 0, 1)


if not siril.is_image_loaded():
    print("No image loaded in Siril.")
    exit(1)

with siril.image_lock():
    img = siril.get_image_pixeldata()
    siril.log(f"Starting trail removal script", color=s.LogColor.BLUE)
    print(f"loaded image: {img.shape}, dtype: {img.dtype}")
    img = img.astype(np.float32)

    print(f"original image: {img.shape}, dtype: {img.dtype}")

    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
        print(f"transposed image: {img.shape}, dtype: {img.dtype}")
        img = rgb_to_grayscale(img)
        print(f"grayscale image: {img.shape}, dtype: {img.dtype}")
    elif img.ndim == 2:
        print(f"already grayscale image: {img.shape}, dtype: {img.dtype}")

    img_norm, min_val, max_val = normalize(img)
    print(f"normalized image: {img_norm.shape}, dtype: {img_norm.dtype}")

    model = load_model(
        MODEL_PATH,
        device=DEVICE,
    )

    cleaned_img = clean_satellite_trail(
        img=img_norm, model=model, tile_size=128, overlap=32
    )
    print(f"cleaned image: {cleaned_img.shape}, dtype: {cleaned_img.dtype}")

    cleaned_img = denormalize(cleaned_img, min_val, max_val)

    # add a dim and convert to uint16 for compatibility with Siril
    cleaned_img = np.expand_dims(cleaned_img, axis=-1)
    cleaned_img = np.transpose(cleaned_img, (2, 0, 1)).astype(np.uint16)
    print(f"final image for Siril: {cleaned_img.shape}, dtype: {cleaned_img.dtype}")

    siril.set_image_pixeldata(cleaned_img)

siril.log(f"Trail removal completed", color=s.LogColor.GREEN)
