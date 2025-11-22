#!/usr/bin/env python3
"""
Train a ResNet model (e.g., ResNet-18) on the CIFAKE dataset (Real vs Fake images).
This script loads the CIFAKE dataset, applies data augmentation, trains the model with 
validation after each epoch, and tracks metrics and artifacts using MLflow.
"""

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import mlflow
import mlflow.pytorch

# Check if this variables exists in the environment, else exit code with error
if "MLFLOW_TRACKING_URI" not in os.environ:
    raise EnvironmentError("MLFLOW_TRACKING_URI environment variable not set.")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("MLOps_Demo_Experiment")

def export_probs_csv(model, data_loader, device, csv_path, idx_real, idx_fake):
    """
    Run model on data_loader and save a CSV with columns:
    prob_real, prob_fake, predicted_label, true_label
    """
    model.eval()
    rows = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = softmax(outputs)  # shape: [batch, num_classes]

            # Extract probabilities for REAL and FAKE using class indices
            prob_real_batch = probs[:, idx_real].cpu().numpy()
            prob_fake_batch = probs[:, idx_fake].cpu().numpy()
            preds_batch     = probs.argmax(dim=1).cpu().numpy()
            labels_batch    = labels.cpu().numpy()

            for pr, pf, pred, y in zip(prob_real_batch, prob_fake_batch, preds_batch, labels_batch):
                rows.append({
                    "prob_real":       float(pr),
                    "prob_fake":       float(pf),
                    "predicted_label": int(pred),
                    "true_label":      int(y),
                })

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved probabilities CSV to {csv_path}")


def main():
    # Parse command-line arguments for training configuration
    parser = argparse.ArgumentParser(description="Train ResNet on CIFAKE dataset (Real vs Fake image classification)")
    parser.add_argument('--data-dir', type=str, default='./data/cifake_data',
                        help='Path to CIFAKE dataset root containing "train" and "test" subdirectories')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training and validation')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay (L2 regularization) for optimizer')
    parser.add_argument('--model-name', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Which ResNet architecture to use (default: resnet18)')
    parser.add_argument('--experiment-name', type=str, default='CIFAKE_ResNet_Run',
                        help='MLflow experiment name for this run')
    args = parser.parse_args()

    # Check that dataset directories exist
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "test")  # Using the provided test set as validation
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError("Could not find 'train' and 'test' directories in the specified data path.")

    # Define data transformations:
    # For training: random crop and horizontal flip (data augmentation) + tensor conversion + normalization
    # Using typical CIFAR-10 augmentations and normalization to CIFAR-10 mean/std.
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2470, 0.2435, 0.2616])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # Random crop with padding (augmenting 32x32 images)
        transforms.RandomHorizontalFlip(),          # Random horizontal flip for augmentation
        transforms.ToTensor(),                      # Convert PIL image to PyTorch tensor
        normalize                                   # Normalize to CIFAR-10 statistics for stable training
    ])
    # For validation/test: no augmentation, just tensor and normalization
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # Load datasets using ImageFolder (expects subdirs for classes "REAL" and "FAKE")
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=val_dir, transform=val_transform)

    # Create data loaders for train and val sets
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Determine which index corresponds to REAL and FAKE
    class_to_idx = train_dataset.class_to_idx   # e.g. {'FAKE': 0, 'REAL': 1} (alphabetical)
    idx_real = class_to_idx['REAL']
    idx_fake = class_to_idx['FAKE']

    # Initialize the ResNet model (ResNet-18 or chosen variant) from torchvision
    num_classes = 2  # Two classes: REAL and FAKE
    if args.model_name == 'resnet18':
        model = models.resnet18(weights=None, num_classes=num_classes)  # initialize ResNet18
    elif args.model_name == 'resnet34':
        model = models.resnet34(weights=None, num_classes=num_classes)
    elif args.model_name == 'resnet50':
        model = models.resnet50(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    # **Optional**: Modify ResNet for CIFAR-sized input (32x32) by using a smaller kernel and removing pooling.
    # This can improve performance on small images by preserving more spatial information.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")

    # Define loss function and optimizer (SGD with momentum and weight decay)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    # Set up MLflow experiment and start a run for tracking
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        # Log hyperparameters to MLflow
        mlflow.log_param("model", args.model_name)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("momentum", args.momentum)
        mlflow.log_param("weight_decay", args.weight_decay)

        # Training loop
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            running_correct = 0
            total_train = 0

            # Iterate over training data in batches
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # accumulate training loss and accuracy
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, dim=1)
                running_correct += (preds == labels).sum().item()
                total_train += labels.size(0)

            # Compute average train loss and accuracy for the epoch
            avg_train_loss = running_loss / total_train
            train_accuracy = running_correct / total_train

            # Validation loop (evaluate on validation/test set)
            model.eval()
            val_loss = 0.0
            val_correct = 0
            total_val = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    # accumulate validation loss and accuracy
                    val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, dim=1)
                    val_correct += (preds == labels).sum().item()
                    total_val += labels.size(0)
            avg_val_loss = val_loss / total_val
            val_accuracy = val_correct / total_val

            # Print epoch summary
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            # Log metrics for this epoch to MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

        # After training, log the model to MLflow as an artifact
        mlflow.pytorch.log_model(model, artifact_path="model")

        # (Optional) Save model state_dict locally as well
        os.makedirs("outputs", exist_ok=True)
        model_path = os.path.join("outputs", f"{args.model_name}_cifake.pth")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)  # log the saved model file to MLflow


    # Paths for monitoring CSVs (you can change these if you prefer another location)
    monitor_dir = os.path.join(args.data_dir, "monitoring")
    os.makedirs(monitor_dir, exist_ok=True)
    reference_csv = os.path.join(monitor_dir, "reference.csv")
    current_csv   = os.path.join(monitor_dir, "current.csv")

    # Generate reference.csv from TRAIN set
    export_probs_csv(model, train_loader, device, reference_csv, idx_real, idx_fake)

    # Generate current.csv from TEST set (or from recent prod data in the future)
    export_probs_csv(model, val_loader, device, current_csv, idx_real, idx_fake)

    # Optionally log to MLflow as artifacts
    if mlflow:
        mlflow.log_artifact(reference_csv)
        mlflow.log_artifact(current_csv)

if __name__ == "__main__":
    main()
