import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from astropy.io import fits
from utils import normalize, split_into_tiles, reassemble_from_tiles


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.dec3 = conv_block(256 + 128, 128)
        self.dec2 = conv_block(128 + 64, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d3 = F.interpolate(e3, scale_factor=2, mode="bilinear")
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear")
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        return self.final(d2)


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
        tile_tensor = torch.from_numpy(tile[None, None]).float().to(device)
        cleaned = model(tile_tensor).squeeze().cpu().numpy()  # (H, W)
        cleaned_tiles.append(cleaned)

    cleaned_img = reassemble_from_tiles(cleaned_tiles, coords, shape, tile_size)

    return np.clip(cleaned_img, 0, 1)
