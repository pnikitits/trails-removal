import numpy as np
import cv2
import matplotlib.pyplot as plt
from astropy.io import fits


def auto_stretch(
    img: np.ndarray, percent_low: int = 1, percent_high: int = 99, vmin=None, vmax=None
) -> tuple:
    """
    Auto stretch the image.

    Parameters
    ----------
    img (np.ndarray)
        Input image array.
    percent_low (int)
        Lower percentile for stretching.
    percent_high (int)
        Upper percentile for stretching.
    vmin (float, optional)
        Minimum value for stretching. If None, it will be calculated.
    vmax (float, optional)
        Maximum value for stretching. If None, it will be calculated.

    Returns
    -------
    np.ndarray
        Stretched image array.
    float
        Minimum value used for stretching.
    float
        Maximum value used for stretching.
    """
    clean_img = img[np.isfinite(img)]
    if vmin is None:
        vmin = np.percentile(clean_img, percent_low)
    if vmax is None:
        vmax = np.percentile(clean_img, percent_high)
    stretched = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    return stretched, vmin, vmax


def add_fake_trail(
    img: np.ndarray,
    thickness_range: tuple = (4, 6),
    brightness_range: tuple = (0.35, 0.45),
    alpha: float = 0.01,
    gaussian_sigma_factor: float = 0.55,
    img_max: float = None,
) -> np.ndarray:
    """
    Add a fake satellite trail.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image (float32)
    thickness_range : tuple
        Trail line thickness.
    brightness_range : tuple
        Brightness relative to image max.
    alpha : float
        Alpha blending weight for trail.
    gaussian_sigma_factor : float
        Controls blur strength.

    Returns
    -------
    np.ndarray
        Image with fake trail.
    """
    h, w = img.shape
    x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
    x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
    thickness = np.random.randint(*thickness_range)

    if img_max is None:
        img_max = img.max()
    brightness = np.random.uniform(*brightness_range) * img_max

    line_mask = np.zeros_like(img, dtype=np.float32)
    cv2.line(line_mask, (x1, y1), (x2, y2), 1.0, thickness, lineType=cv2.LINE_8)

    sigma = gaussian_sigma_factor * thickness
    trail_mask = cv2.GaussianBlur(line_mask, (0, 0), sigmaX=sigma, sigmaY=sigma)

    trail_mask = trail_mask / trail_mask.max()
    trail_mask *= brightness

    img_with_trail = img.copy()
    trail_area = trail_mask > 0
    img_with_trail[trail_area] = img[trail_area] + alpha * trail_mask[trail_area]

    return img_with_trail


def show(
    img: np.ndarray | list,
    axis: bool = True,
    title: str | list = None,
    stretch: bool = False,
) -> None:
    """
    Display an image or a list of images.

    Parameters
    ----------
    img (np.ndarray | list)
        Image or list of images to display.
    axis (bool)
        Whether to show the axis or not.
    title (str | list)
        Title for the image.
    stretch (bool)
        Whether to apply auto stretching to the image.
    """
    if isinstance(img, list):
        plt.figure(figsize=(15, 5))
        for i, im in enumerate(img):
            if stretch:
                im, _, _ = auto_stretch(im)
            plt.subplot(1, len(img), i + 1)
            plt.imshow(im, cmap="gray")
            if not axis:
                plt.axis("off")
            if title is not None:
                plt.title(title[i] if isinstance(title, list) else title)
    elif isinstance(img, np.ndarray):
        if stretch:
            img, _, _ = auto_stretch(img)
        plt.imshow(img, cmap="gray")
        if not axis:
            plt.axis("off")
        if title is not None and isinstance(title, str):
            plt.title(title)
    plt.show()


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


def check_image_range(image: np.ndarray):
    if np.all((image >= 0) & (image <= 1)):
        print(f"0-1 ({image.min()} - {image.max()})")
    elif np.all((image >= 0) & (image <= 255)):
        print(f"0-255 ({image.min()} - {image.max()})")
    else:
        print(f"{image.min()} - {image.max()}")

    return image.min(), image.max()


if __name__ == "__main__":
    img_path = "data/debayered_set/deb_00001.fit"
    img = fits.getdata(img_path).astype(np.float32)
    img = np.transpose(img, (1, 2, 0))
    img = rgb_to_grayscale(img)

    img_max = img.max()
    _, min_val, max_val = normalize(img)

    tiles, coords = split_into_tiles(img, tile_size=256, overlap=32)

    tiles = [add_fake_trail(tile, img_max=img_max) for tile in tiles]
    tiles = [normalize(tile, min_val=min_val, max_val=max_val)[0] for tile in tiles]

    # show(tiles[:4])

    # _, min_val, max_val = auto_stretch(img)
    # s_tiles = [auto_stretch(tile, vmin=min_val, vmax=max_val)[0] for tile in tiles]

    # show(s_tiles[:4])
    show(
        [reassemble_from_tiles(tiles, coords, img.shape, tile_size=256)],
        title="Reassembled Image",
        stretch=True,
    )
