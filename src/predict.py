import os
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from tqdm import tqdm
import segmentation_models_pytorch as smp
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Relative paths
input_output_pairs = [
    ("../data/clipped_images", "../data/predicted_images"),
]

os.makedirs("../data/predicted_images", exist_ok=True)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
try:
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,  # Weights are loaded from the checkpoint
        in_channels=3,
        classes=5
    )
    model.load_state_dict(torch.load("../data/models/bestUNET_model.pth", map_location=device))
    model.to(device)
    model.eval()
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error("Model checkpoint not found. Ensure 'bestUNET_model.pth' exists in ../data/models/")
    exit(1)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit(1)

# Define the dataset
class RoadDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            with rasterio.open(image_path) as src:
                image = src.read([1, 2, 3]).transpose(1, 2, 0)  # Assuming RGB
                if image.shape[-1] != 3:
                    raise ValueError(f"Image {image_path} does not have 3 channels.")
                image = image / 255.0  # Normalize
            return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), image_path
        except Exception as e:
            logging.warning(f"Error processing {image_path}: {e}")
            return None, None

# Prediction function
def process_images(input_folder, output_folder, batch_size=64):
    os.makedirs(output_folder, exist_ok=True)
    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif')]
    if not image_paths:
        logging.warning(f"No images found in {input_folder}")
        return

    logging.info(f"Found {len(image_paths)} images to process.")
    dataset = RoadDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True, collate_fn=lambda x: list(filter(None, x)))

    progress_bar = tqdm(total=len(image_paths), desc="Processing images", unit="img")
    with torch.no_grad():
        for images, paths in dataloader:
            if not images:  # Skip empty batches
                continue
            images = torch.stack(images).to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            for pred, path in zip(predictions, paths):
                try:
                    with rasterio.open(path) as src:
                        metadata = src.meta.copy()
                        metadata.update({'driver': 'GTiff', 'count': 1, 'dtype': 'uint8', 'nodata': 255})
                    output_path = os.path.join(output_folder, os.path.basename(path))
                    with rasterio.open(output_path, 'w', **metadata) as dst:
                        dst.write(pred.astype(np.uint8), 1)
                    progress_bar.update(1)
                except Exception as e:
                    logging.warning(f"Error saving prediction for {path}: {e}")

    progress_bar.close()
    logging.info(f"Predictions saved successfully in {output_folder}!")

if __name__ == "__main__":
    for input_folder, output_folder in input_output_pairs:
        process_images(input_folder, output_folder)