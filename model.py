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
