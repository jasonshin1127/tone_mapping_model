import os
import shutil


def organize_images_by_suffix(
    source_folder, destination_folder_0, destination_folder_N1
):
    if not os.path.exists(destination_folder_0):
        os.makedirs(destination_folder_0)
    if not os.path.exists(destination_folder_N1):
        os.makedirs(destination_folder_N1)

    for filename in os.listdir(source_folder):
        if filename.endswith("_0.JPG"):
            shutil.move(
                os.path.join(source_folder, filename),
                os.path.join(destination_folder_0, filename),
            )
        elif filename.endswith("_N1.JPG"):
            shutil.move(
                os.path.join(source_folder, filename),
                os.path.join(destination_folder_N1, filename),
            )


source_folder = "INPUT_IMAGES"  # Replace with the actual path of your source folder
destination_folder_0 = "_0"  # Replace with the actual path of the folder for _0 images
destination_folder_N1 = "_N1"

organize_images_by_suffix(source_folder, destination_folder_0, destination_folder_N1)
