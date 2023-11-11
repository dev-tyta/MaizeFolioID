import os
import shutil
from random import choice

"""### Oversampling"""

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        num_images = class_counts[class_name]
        images_to_copy = 1100 - num_images
        image_files = os.listdir(class_dir)
        while images_to_copy > 0:
            # Choose a random image to copy
            image_to_copy = choice(image_files)
            # Create a new filename
            new_filename = f"{image_to_copy.split('.')[0]}_copy{images_to_copy}.{image_to_copy.split('.')[1]}"
            # Copy the image
            shutil.copy(os.path.join(class_dir, image_to_copy), os.path.join(class_dir, new_filename))
            images_to_copy -= 1

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        print(f"Class {class_name} has {len(os.listdir(class_dir))} images after oversampling.")
