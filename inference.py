from model import load_model, clean_satellite_trail
from utils import *
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
        "results/20250704-180452/model_weights.pth",
        device="mps",
    )

    # assume RGB debayered image
    # test_path = "data/test_image.fit"
    test_path = "data/debayered_subset/deb_00001.fit"
    img = fits.getdata(test_path).astype(np.float32)
    img = np.transpose(img, (1, 2, 0))

    img = rgb_to_grayscale(img)
    img_with_trail = add_fake_trail(img.copy())
    img_norm, min_val, max_val = normalize(img_with_trail)

    cleaned_img = clean_satellite_trail(img_norm, model)

    img_display, _, _ = auto_stretch(img_with_trail)
    cleaned_display, _, _ = auto_stretch(cleaned_img)

    show(
        [img_display, cleaned_display],
        title=["Input w/ Fake Trail", "Cleaned Image"],
    )


if __name__ == "__main__":
    main()
