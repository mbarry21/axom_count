import os
from PIL import Image

import properties

for filename in os.listdir(properties.cropped_path):
    if filename.endswith("jpg"):
        im = Image.open(properties.cropped_path + '\\' + filename)

        # TODO: Use greyscale images or not?
        # Convert to black and white image
        # img2 = im.convert('L')
        img2 = im
        img2.save(properties.greyscale_path + "/" + filename, "JPEG")
