import sys
import os

label = sys.argv[1]
if not label:
    raise Exception("Without label")

raw_images = os.path.join("novas", label)
files = map(lambda file: os.path.join(raw_images, file), os.listdir(raw_images))

destination_dir  = os.path.join("dataset", label)

total_files = len(os.listdir(destination_dir))

for i, file in enumerate(files):
    new_name = os.path.join(raw_images, f'{label[:-1]}{i+1 + total_files}.png')
    old_name = file
    os.rename(old_name, new_name)    
