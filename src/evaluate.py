import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import logging
from .model import model, val_loader, device, num_classes

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def evaluate():
    """
    Evaluates the model on the validation dataset and computes metrics.
    """
    if not val_loader:
        logging.error("Validation loader is empty. Ensure the dataset is properly loaded.")
        return

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            val_outputs = model(val_images)
            preds = torch.argmax(val_outputs, dim=1)
            all_preds.append(preds)
            all_targets.append(val_masks)

    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_targets = torch.cat(all_targets, dim=0).cpu().numpy()

    # Compute confusion matrix and metrics
    cm = confusion_matrix(all_targets, all_preds, labels=np.arange(num_classes))
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average=None, labels=np.arange(num_classes)
    )
    accuracy = np.sum(np.diagonal(cm)) / cm.sum()

    logging.info(f"Overall Accuracy: {accuracy:.4f}")
    for i in range(num_classes):
        logging.info(f"Class {i} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1 Score: {f1[i]:.4f}")

    logging.info(f"Average Precision: {precision.mean():.4f}")
    logging.info(f"Average Recall: {recall.mean():.4f}")
    logging.info(f"Average F1 Score: {f1.mean():.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


def apply_color_map(mask, class_colors):
    """
    Applies a color map to the mask for visualization.
    """
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for c in range(len(class_colors)):
        colored_mask[mask == c] = class_colors[c]
    return colored_mask


def visualize_batch(batch_idx, val_loader, model, class_colors):
    """
    Visualizes a batch of images, ground truth masks, and predicted masks.
    """
    if not val_loader:
        logging.error("Validation loader is empty. Ensure the dataset is properly loaded.")
        return

    for idx, (images, masks) in enumerate(val_loader):
        if idx == batch_idx:
            break

    if idx != batch_idx:
        logging.warning(f"Batch index {batch_idx} not found in the validation loader.")
        return

    images, masks = images.to(device), masks.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

    for i in range(len(images)):
        plt.figure(figsize=(12, 4))

        # Input image
        plt.subplot(1, 3, 1)
        plt.imshow(images[i].cpu().permute(1, 2, 0))
        plt.title(f"Image - Batch {batch_idx + 1}")
        plt.axis('off')

        # Ground truth
        plt.subplot(1, 3, 2)
        colored_mask = apply_color_map(masks[i].cpu().numpy(), class_colors)
        plt.imshow(colored_mask)
        plt.title("Ground Truth")
        plt.axis('off')

        # Prediction
        plt.subplot(1, 3, 3)
        colored_pred = apply_color_map(preds[i].cpu().numpy(), class_colors)
        plt.imshow(colored_pred)
        plt.title("Prediction")
        plt.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    evaluate()

    # Define class colors dynamically
    class_colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    if len(class_colors) < num_classes:
        logging.warning("Not enough class colors defined. Extending with random colors.")
        for _ in range(num_classes - len(class_colors)):
            class_colors.append(tuple(np.random.randint(0, 256, size=3)))

    # Visualize a specific batch
    visualize_batch(2, val_loader, model, class_colors)