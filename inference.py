from model import load_model, clean_satellite_trail
from utils import rgb_to_grayscale, normalize, auto_stretch, show, add_fake_trail
from astropy.io import fits
import numpy as np


def main():
    """
    Main function to run the inference for satellite trail removal.

    Note on the order of operations:
    1. fits.getdata()
    2. np.transpose()
    3. rgb_to_grayscale()
    4. add_fake_trail()
    5. normalize()
    """
    model = load_model(
        "results/20250717-144407/model_epoch_1500.pth",
        device="mps",
    )

    # assume RGB debayered image

    # --- test with synthetic trail ---
    # test_path = "data/debayered_subset/deb_00001.fit"

    # --- test with real trail ---
    test_path = "data/test_image.fit"

    # --- common transformations ---
    img = fits.getdata(test_path).astype(np.float32)
    img = np.transpose(img, (1, 2, 0))
    img = rgb_to_grayscale(img)

    # --- test with synthetic trail ---
    # img_with_trail = add_fake_trail(img.copy())

    # --- test with real trail ---
    img_with_trail = img.copy()

    img_norm, _, _ = normalize(img_with_trail)

    cleaned_img, residual_img = clean_satellite_trail(
        img=img_norm, model=model, tile_size=128, overlap=32
    )

    img_display, _, _ = auto_stretch(img_with_trail)
    cleaned_display, _, _ = auto_stretch(cleaned_img)

    show(
        [img_display, cleaned_display, residual_img],
        title=["Input with real trail", "Cleaned image", "Predicted residual"],
    )


if __name__ == "__main__":
    main()
