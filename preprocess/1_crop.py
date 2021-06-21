import os
from PIL import Image

import properties

for filename in os.listdir(properties.orig_images_path):
    if filename.endswith("jpg"):
        im = Image.open(properties.orig_images_path + '\\' + filename)

        # Crop image
        # orig size: 1295 x 971
        # left, top, right, bottom
        img2 = im.crop((140, 220, 1030, 740))
        img2.save(properties.cropped_path + "/" + filename, "JPEG")
