from torch.utils.data import Dataset
from astropy.io import fits
import numpy as np
from utils import *
import torch
import os


class FITSSatelliteTileDataset(Dataset):
    def __init__(
        self,
        directory: str,
        tile_size: int = 512,
        overlap: int = 64,
        augment: bool = True,
        preload: bool = False,
    ):
        """
        Parameters
        ----------
        directory : str
            Path to directory containing .fits images.
        tile_size : int
            Size of each square tile (default: 512).
        overlap : int
            Amount of overlap between tiles (default: 64).
        augment : bool
            Whether to apply fake trail augmentation.
        preload : bool
            Whether to preload all tiles into memory.
        """
        if preload:
            ValueError("Preload not checked !!!")
        self.paths = sorted(
            [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.lower().endswith((".fit", ".fits"))
            ]
        )

        self.tile_size = tile_size
        self.overlap = overlap
        self.augment = augment
        self.preload = preload

        self.preloaded_tiles = []

        if preload:
            for path in self.paths:
                # img = fits.getdata(path).astype(np.float32)
                with fits.open(path) as hdul:
                    img = hdul[0].data.astype(np.float32)

                img = np.transpose(img, (1, 2, 0))
                img = rgb_to_grayscale(img)

                _, min_val, max_val = normalize(img)

                clean_tiles, _ = split_into_tiles(img, tile_size, overlap)
                for clean in clean_tiles:
                    noisy = add_fake_trail(clean.copy()) if augment else clean.copy()
                    noisy, _, _ = normalize(noisy, min_val=min_val, max_val=max_val)
                    clean, _, _ = normalize(clean, min_val=min_val, max_val=max_val)
                    self.preloaded_tiles.append((noisy, clean))

    def __len__(self):
        if self.preload:
            return len(self.preloaded_tiles)
        else:
            est_tiles_per_image = self._estimate_tiles_per_image((3008, 3008))
            return len(self.paths) * est_tiles_per_image

    def __getitem__(self, idx: int):
        if self.preload:
            noisy, clean = self.preloaded_tiles[idx]
        else:
            # select image and generate tile
            tiles_per_image = self._estimate_tiles_per_image((3008, 3008))
            image_idx = idx // tiles_per_image
            tile_idx = idx % tiles_per_image

            with fits.open(self.paths[image_idx]) as hdul:
                img = hdul[0].data.astype(np.float32)

            img = np.transpose(img, (1, 2, 0))
            img = rgb_to_grayscale(img)

            img_max = img.max()
            _, min_val, max_val = normalize(img)

            tiles, _ = split_into_tiles(img, self.tile_size, self.overlap)

            tile_idx = min(tile_idx, len(tiles) - 1)

            clean = tiles[tile_idx]
            noisy = (
                add_fake_trail(clean.copy(), img_max=img_max)
                if self.augment
                else clean.copy()
            )
            noisy, _, _ = normalize(noisy, min_val=min_val, max_val=max_val)
            clean, _, _ = normalize(clean, min_val=min_val, max_val=max_val)

        return (
            torch.tensor(noisy).unsqueeze(0).float(),
            torch.tensor(clean).unsqueeze(0).float(),
        )

    def _estimate_tiles_per_image(self, image_shape: tuple):
        h, w = image_shape
        step = self.tile_size - self.overlap
        ny = ((h - self.tile_size) // step) + 1
        nx = ((w - self.tile_size) // step) + 1
        return ny * nx
