import os
import sys
from PIL import Image

def crop_image(image, left=42, top=0, right=56, bottom=28):
    w, h = image.size
    new_image = Image.new('RGB', (w - left - right, h -top - bottom))
    for i1, i2 in enumerate(range(left, w - right)):
        for j1, j2 in enumerate(range(top, h - bottom)):
            new_pixel = image.getpixel((i2, j2))
            new_image.putpixel((i1, j1), new_pixel)
    return new_image


def crop_and_save_images(files, save_dir):
    for file in files:
        image = Image.open(file)
        file_name = os.path.basename(file)
        crop_image(image).save(os.path.join(save_dir, file_name))
    
label = sys.argv[1]
if not label:
    raise Exception('Without label')

raw_images = os.path.join("novas", label)
save_dir = os.path.join("dataset", label)

files = map(lambda file: os.path.join(raw_images, file), os.listdir(raw_images))

crop_and_save_images(files,  save_dir)


    
