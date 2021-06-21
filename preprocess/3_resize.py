
import os
from PIL import Image

import properties

for filename in os.listdir(properties.greyscale_path):
    if filename.endswith("jpg"):
        im = Image.open(properties.greyscale_path + '\\' + filename)

        # Resize image for easier learning
        # orig size: 1295 x 971
        img2 = im.resize((128, 128), Image.ANTIALIAS)
        img2.save(properties.resized_path + "/" + filename, "JPEG")