import numpy as np
from patch import patchify
from PIL import Image
import matplotlib.pyplot as plt
import os


def patchify_single_image(image_path, patch_size):
    # Load the image from its path
    large_image = Image.open(image_path)
    large_image = np.array(large_image)

    # Extract patches from the image using patchify
    patches = patchify(large_image, patch_size, step=patch_size)

    # Get the number of patches along the height and width dimensions
    num_patches_height, num_patches_width = patches.shape[0], patches.shape[1]

    # Create a single canvas to display all the patches
    canvas_height = num_patches_height * patch_size[0]
    canvas_width = num_patches_width * patch_size[1]
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Arrange the patches on the canvas
    for h in range(num_patches_height):
        for w in range(num_patches_width):
            patch = patches[h, w]
            y_start = h * patch_size[0]
            y_end = y_start + patch_size[0]
            x_start = w * patch_size[1]
            x_end = x_start + patch_size[1]
            canvas[y_start:y_end, x_start:x_end, :] = patch

    return canvas


def patchify_multiple_images(training_folder_path, gt_folder_path, patch_size):
    # Get a list of all image file names in the training folder
    training_files = [f for f in os.listdir(training_folder_path) if f.endswith(".jpg")]

    # Create empty arrays to store all the training set patches and GT patches
    training_patches = []
    gt_patches = []

    # Loop through each image and patchify it
    for image_file in training_files:
        training_image_path = os.path.join(training_folder_path, image_file)
        gt_image_file = image_file.replace(".jpg", "_gt.jpg")
        gt_image_path = os.path.join(gt_folder_path, gt_image_file)

        training_canvas = patchify_single_image(training_image_path, patch_size)
        gt_canvas = patchify_single_image(gt_image_path, patch_size)

        # Append the patches from the current image to the arrays
        training_patches.append(training_canvas)
        gt_patches.append(gt_canvas)

    # Convert the lists of patches into NumPy arrays
    training_patches = np.array(training_patches)
    gt_patches = np.array(gt_patches)

    return training_patches, gt_patches


# Set the patch size (16x16)
patch_size = (16, 16, 3)

# Specify the local folder paths containing the training set images and GT images
training_folder_path = "training/GT_IMAGES"
gt_folder_path = "training/INPUT_IMAGES"

# Patchify multiple images from the folders and get all the patches in separate arrays
training_set_patches, gt_patches = patchify_multiple_images(
    training_folder_path, gt_folder_path, patch_size
)

# Now, 'training_set_patches' contains all the patches from training set images
# and 'gt_patches' contains all the patches from ground truth (GT) images.
# Both are 4D arrays with shape (num_images, num_patches_height, num_patches_width, patch_size[0], patch_size[1], 3)
