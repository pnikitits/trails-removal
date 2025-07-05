import numpy as np
from astropy.io import fits
from utils import *

img_path = "data/test_image.fit"
img = fits.getdata(img_path).astype(np.float32)
img = np.transpose(img, (1, 2, 0))
img = rgb_to_grayscale(img)

# check_image_range(img)

img_with_trail = add_fake_trail(img.copy())

# check_image_range(img_with_trail)

show(img_with_trail, stretch=True)
