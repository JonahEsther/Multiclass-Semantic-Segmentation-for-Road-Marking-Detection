import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .dataset import RoadDataset

# Updated paths
image_dirs = ["../data/training/images"]
mask_dirs = ["../data/training/masks"]
target_size = (224, 224)


def load_data_with_padding(image_dirs, mask_dirs, target_size):
    """
    Loads and preprocesses images and masks with padding.
    """
    images, masks = [], []
    
    # Define transformations
    image_transform = A.Compose([
        A.PadIfNeeded(min_height=target_size[0], min_width=target_size[1], p=1.0, border_mode=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    mask_transform = A.Compose([
        A.PadIfNeeded(min_height=target_size[0], min_width=target_size[1], p=1.0, border_mode=0),
        ToTensorV2()
    ])
    
    # Load and preprocess data
    for image_dir, mask_dir in zip(image_dirs, mask_dirs):
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])
        
        for img_file, mask_file in zip(image_files, mask_files):
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)
            
            try:
                with rasterio.open(img_path) as src:
                    image = src.read([1, 2, 3]).transpose(1, 2, 0)
                
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)
                
                # Apply transformations
                augmented_image = image_transform(image=image)
                augmented_mask = mask_transform(image=mask)
                
                images.append(augmented_image['image'])
                masks.append(augmented_mask['image'].long())
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    return images, masks


# Load data
images, masks = load_data_with_padding(image_dirs, mask_dirs, target_size)

# Split data into training and validation sets
train_data, val_data = train_test_split(list(zip(images, masks)), test_size=0.2, random_state=42)
train_images, train_masks = zip(*train_data)
val_images, val_masks = zip(*val_data)

# Create datasets and dataloaders
train_dataset = RoadDataset(list(train_images), list(train_masks))
val_dataset = RoadDataset(list(val_images), list(val_masks))

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Data augmentation pipeline
augmentation_pipeline = A.Compose([
    A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definition
num_classes = 5
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes
).to(device)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Early stopping parameters
num_epochs = 50
patience = 10
best_val_loss = float('inf')
epochs_without_improvement = 0
train_losses, val_losses = [], []


def train():
    """
    Trains the model with early stopping.
    """
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        # Training phase
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_masks).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "../data/models/bestUNET_model.pth")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
    
    print("Training complete.")


if __name__ == "__main__":
    train()