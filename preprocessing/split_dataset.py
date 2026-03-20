# Import required libraries

import os          # Used for file and folder operations (paths, directories)
import random      # Used to shuffle images randomly before splitting
import shutil      # Used to copy files from one folder to another


# Source dataset folder (contains Real and Fake images from preprocessing)
SOURCE_DIR = "processed_faces"

# Destination folder where train/val/test datasets will be created
DEST_DIR = "dataset"


# Dataset split ratios
TRAIN_RATIO = 0.7     # 70% of images for training
VAL_RATIO = 0.15      # 15% for validation
TEST_RATIO = 0.15     # 15% for testing


# Classes in our dataset
# These correspond to the two categories of deepfake detection
classes = ["Real", "Fake"]


# Loop through each class (Real and Fake)
for cls in classes:

    # Create path to the source folder
    # Example: processed_faces/Real or processed_faces/Fake
    src_folder = os.path.join(SOURCE_DIR, cls)


    # Get all images in that folder
    images = os.listdir(src_folder)


    # Shuffle images randomly to avoid bias
    # This ensures training data is well distributed
    random.shuffle(images)


    # Count total number of images
    total = len(images)


    # Calculate splitting indices based on ratios
    train_end = int(TRAIN_RATIO * total)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * total)


    # Split images into train, validation, and test sets
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]


    # Loop through each dataset type
    for split, split_images in zip(
        ["train", "val", "test"],
        [train_images, val_images, test_images]
    ):

        # Create destination folder
        # Example: dataset/train/Real
        dest_folder = os.path.join(DEST_DIR, split, cls)


        # Create folder if it does not exist
        os.makedirs(dest_folder, exist_ok=True)


        # Copy images from source to destination
        for img in split_images:

            src_path = os.path.join(src_folder, img)
            dst_path = os.path.join(dest_folder, img)

            shutil.copy(src_path, dst_path)


# Print message after dataset splitting is completed
print("Dataset split complete!")