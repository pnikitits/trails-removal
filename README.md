# trails-removal

## Description
A lightweight U-Net implementation to remove satellite trails for astrophotography images.

The goal is to make a script and trained model for use in Siril.

(Ongoing work, not yet ready for use)

## Installation [Training]
Clone and install the requirements:
```bash
git clone https://github.com/pnikitits/trails-removal.git
cd trails-removal
pip install -r requirements.txt
```

Download the dataset [] or use your own:
- the images should be .fit files
- the images should be debayered
- the images used for training should not contain any satellite trails
- the images can be of any size, the script splits them into 128x128 tiles

## Installation [Siril script]
Download the script `siril_script_trails_removal.py` and place it in your Siril scripts directory.
(You can find the scripts directory in Siril by going to `Preferences > Scripts > Scripts directory`)

Download the trained model from the `results` directory and place it in the same directory as the script.

## Usage [Siril script]

1. Open Siril and load your astronomical image.
2. Run the script `siril_script_trails_removal.py` from the Scripts menu.
3. The script will process the image and remove satellite trails using the trained model.

Note: the model only outputs one channel (grayscale) for now.

## Author
Pierre Nikitits