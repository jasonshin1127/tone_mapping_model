import os
import cv2
from patchify import patchify


def create_patches(image_path, patch_size, destination_folder):
    # Read the image
    image = cv2.imread(image_path)
    # Convert the image to RGB format (patchify requires RGB images)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create patches using patchify
    patches = patchify(image_rgb, (patch_size, patch_size, 3), step=patch_size)

    # Get the number of rows and columns of patches
    num_rows, num_cols = patches.shape[0], patches.shape[1]

    # Save the patches as separate JPG files in the destination folder
    for r in range(num_rows):
        for c in range(num_cols):
            patch = patches[r, c, 0]
            patch_filename = (
                f"{os.path.splitext(os.path.basename(image_path))[0]}_patch_{r}_{c}.jpg"
            )
            patch_filepath = os.path.join(destination_folder, patch_filename)
            cv2.imwrite(patch_filepath, patch)


folder_path = "_valY"
patch_size = 32
destination_folder = "_valY-patch"


# Get a list of all JPG files in the folder
jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]

# Process each JPG file and create patches
for jpg_file in jpg_files:
    jpg_file_path = os.path.join(folder_path, jpg_file)
    create_patches(jpg_file_path, patch_size, destination_folder)


# import os


# def delete_jpg_files(folder_path):
#     try:
#         for filename in os.listdir(folder_path):
#             if filename.endswith(".jpg"):
#                 file_path = os.path.join(folder_path, filename)
#                 os.remove(file_path)
#                 print(f"Deleted: {filename}")
#     except OSError as e:
#         print(f"Error occurred: {e}")


# # Replace 'folder_path' with the path to the folder containing the JPG files
# folder_path = "_0-patch"
# delete_jpg_files(folder_path)
