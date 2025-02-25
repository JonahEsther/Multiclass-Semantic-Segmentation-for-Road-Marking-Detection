import os
import rasterio
import numpy as np
from tqdm import tqdm
from shutil import copyfile
from torch.utils.data import Dataset

# Define paths
base_path = "../data/training/masks"  # Directory containing class-specific masks
output_folder = "../data/training/masks"  # Merged masks directory
image_base_path = "../data/training/images"  # Directory containing class-specific images
folders = {'Block_gdf', 'Triangle_gdf', 'Solid_gdf', 'Dash_gdf'}  # Class folders
class_values = {'Block_gdf': 1, 'Dash_gdf': 2, 'Solid_gdf': 3, 'Triangle_gdf': 4}  # Class mappings


def assign_class_values():
    """
    Assigns class values to masks and saves them in the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    for folder in folders:
        class_value = class_values.get(folder)
        if class_value is None:
            print(f"Warning: No class value defined for folder {folder}. Skipping.")
            continue

        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist. Skipping.")
            continue

        image_names = os.listdir(folder_path)
        print(f"Processing images for {folder} with class value {class_value}...")

        for image_name in tqdm(image_names, desc=f"Processing {folder}"):
            image_path = os.path.join(folder_path, image_name)

            try:
                with rasterio.open(image_path) as src:
                    image_data = src.read(1)  # Read the single band
                    class_image = np.where(image_data == 1, class_value, 0)  # Assign class value

                new_image_name = f"{os.path.splitext(image_name)[0]}_{folder}.tif"
                new_image_path = os.path.join(output_folder, new_image_name)

                meta = src.meta.copy()
                meta.update(dtype='uint8', count=1)  # Ensure output is a single-band uint8 image

                with rasterio.open(new_image_path, 'w', **meta) as dst:
                    dst.write(class_image, 1)

            except rasterio.errors.RasterioIOError as e:
                print(f"Error processing {image_path}: {e}")

    print(f"Class-assigned images saved in {output_folder}")


def combine_class_images():
    """
    Combines class-specific images into a single directory.
    """
    class_folders = ['Block_gdf', 'Triangle_gdf', 'Solid_gdf', 'Dash_gdf']
    output_folder = os.path.join(image_base_path, 'images')
    os.makedirs(output_folder, exist_ok=True)

    for class_folder in class_folders:
        class_folder_path = os.path.join(image_base_path, class_folder)

        if not os.path.exists(class_folder_path):
            print(f"Folder {class_folder_path} does not exist. Skipping.")
            continue

        print(f"Processing images in {class_folder}...")
        image_names = os.listdir(class_folder_path)

        for image_name in tqdm(image_names, desc=f"Processing {class_folder}"):
            source_path = os.path.join(class_folder_path, image_name)
            destination_path = os.path.join(output_folder, image_name)

            if os.path.exists(destination_path):
                base_name, ext = os.path.splitext(image_name)
                new_name = f"{base_name}_{class_folder}{ext}"
                destination_path = os.path.join(output_folder, new_name)

            copyfile(source_path, destination_path)

    print(f"All images have been combined into {output_folder}")


class RoadDataset(Dataset):
    """
    Custom PyTorch Dataset for road marking detection.
    """
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


if __name__ == "__main__":
    # Execute the preprocessing steps
    assign_class_values()  # Step 1: Assign class values to masks
    combine_class_images()  # Step 2: Combine class-specific images