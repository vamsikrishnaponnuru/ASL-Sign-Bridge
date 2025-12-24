import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

# Path to your dataset folders (A, B, C, ... Z)
dataset_path = r"C:\Users\Sunny\final project\static\project1"

# Augmentation configuration
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    fill_mode='nearest'
)

# Number of augmented images per class
AUG_COUNT = 200

# Go through each folder/class
for folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, folder)

    # Skip if not a folder
    if not os.path.isdir(class_path):
        continue

    original_img_path = os.path.join(class_path, "0.jpg")

    # Check if main image exists
    if not os.path.exists(original_img_path):
        print(f"0.jpg missing in {folder}, skipping...")
        continue

    print(f"Augmenting: {folder}")

    # Load the original image
    img = load_img(original_img_path)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Generate augmented images
    prefix = folder + "_aug"

    i = 0
    for batch in datagen.flow(
        img_array,
        batch_size=1,
        save_to_dir=class_path,
        save_prefix=prefix,
        save_format='jpg'
    ):
        i += 1
        if i >= AUG_COUNT:
            break

print("Dataset augmentation complete! ✔")
