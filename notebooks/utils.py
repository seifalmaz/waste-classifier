"""
utils.py — Shared Utilities for Waste Classifier Project
=========================================================
Used by all model notebooks (03–07).
Place this file in: D:/HNU/waste_classifier/notebooks/utils.py

Provides:
  - get_transforms()         : train + val/test transforms
  - get_dataloaders()        : DataLoaders with WeightedRandomSampler
  - train_one_epoch()        : single epoch training step
  - evaluate()               : loss + accuracy on any split
  - train_model()            : full training loop with scheduler + early stopping
  - get_classification_report(): per-class metrics dict
  - plot_confusion_matrix()  : saves confusion matrix PNG to results/
  - plot_history()           : saves loss + accuracy curves PNG to results/
  - save_checkpoint()        : saves best model .pth to models/
  - load_checkpoint()        : loads a saved .pth checkpoint
  - get_class_names()        : returns ordered class list
  - set_seed()               : reproducibility
"""

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────
# 0. Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# 1. Constants
# ─────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224
CLASS_NAMES   = ["Glass", "Metal", "Organic", "Paper", "Plastic"]  # alphabetical (ImageFolder order)


def get_class_names():
    return CLASS_NAMES


# ─────────────────────────────────────────────
# 2. Transforms
# ─────────────────────────────────────────────

def get_transforms(augment: bool = True):
    """
    Returns a dict with 'train' and 'val' transform pipelines.

    Args:
        augment: If True, applies data augmentation to the train transform.
                 Set to False for the Baseline CNN (Phase 4).
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Baseline CNN: no augmentation
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    return {"train": train_transform, "val": val_transform}


# ─────────────────────────────────────────────
# 3. DataLoaders
# ─────────────────────────────────────────────

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    augment: bool = True,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
):
    """
    Builds train, val, and test DataLoaders from a processed/ directory.

    Expected structure:
        data_dir/
          train/  Glass/  Metal/  Organic/  Paper/  Plastic/
          val/    ...
          test/   ...

    Args:
        data_dir           : path to processed/ folder
        batch_size         : images per batch (default 32)
        augment            : whether to apply augmentation on train set
        num_workers        : DataLoader workers (set 0 on Windows if errors occur)
        use_weighted_sampler: balance class frequencies during training

    Returns:
        dataloaders : dict with keys 'train', 'val', 'test'
        dataset_sizes: dict with split sizes
        class_names  : list of class name strings
    """
    tf = get_transforms(augment=augment)

    image_datasets = {
        "train": datasets.ImageFolder(os.path.join(data_dir, "train"), tf["train"]),
        "val":   datasets.ImageFolder(os.path.join(data_dir, "val"),   tf["val"]),
        "test":  datasets.ImageFolder(os.path.join(data_dir, "test"),  tf["val"]),
    }

    # ── WeightedRandomSampler for class imbalance ──────────────────────────
    if use_weighted_sampler:
        train_targets = image_datasets["train"].targets
        class_counts  = np.bincount(train_targets)
        class_weights = 1.0 / class_counts.astype(np.float32)
        sample_weights = class_weights[train_targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False  # sampler is mutually exclusive with shuffle
    else:
        sampler       = None
        train_shuffle = True

    dataloaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=batch_size,
            sampler=sampler,
            shuffle=train_shuffle if sampler is None else False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            image_datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            image_datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    dataset_sizes = {split: len(ds) for split, ds in image_datasets.items()}
    class_names   = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names


# ─────────────────────────────────────────────
# 4. Training & Evaluation
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model on a DataLoader. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds         = outputs.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

    return running_loss / total, correct / total


def train_model(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs: int = 25,
    patience: int = 7,
    model_name: str = "model",
    results_dir: str = "../results",
    models_dir: str = "../models",
):
    """
    Full training loop with:
      - ReduceLROnPlateau scheduling (pass scheduler=None to disable)
      - Early stopping on val loss
      - Best-model checkpoint saving

    Args:
        model        : PyTorch model
        dataloaders  : dict with 'train' and 'val' DataLoaders
        dataset_sizes: dict with split sizes (for display only)
        criterion    : loss function (e.g. nn.CrossEntropyLoss())
        optimizer    : optimizer (e.g. Adam)
        scheduler    : LR scheduler or None
        device       : torch.device
        num_epochs   : maximum training epochs
        patience     : early-stopping patience (epochs without val improvement)
        model_name   : used for checkpoint filename and plot titles
        results_dir  : where to save history plots
        models_dir   : where to save .pth checkpoints

    Returns:
        history: dict with lists 'train_loss', 'val_loss',
                 'train_acc', 'val_acc'
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir,  exist_ok=True)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    best_val_loss    = float("inf")
    epochs_no_improve = 0
    best_model_path  = os.path.join(models_dir, f"{model_name}_best.pth")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, dataloaders["val"], criterion, device
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch:>3}/{num_epochs}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
            print(f"  ✔ Best model saved → {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n  ⏹ Early stopping triggered after {epoch} epochs "
                      f"(no improvement for {patience} epochs).")
                break

    # Save history JSON
    history_path = os.path.join(results_dir, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved → {history_path}")

    return history


# ─────────────────────────────────────────────
# 5. Metrics
# ─────────────────────────────────────────────

def get_all_preds_labels(model, loader, device):
    """Collect all predictions and ground-truth labels from a DataLoader."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def get_classification_report(model, loader, device, class_names):
    """
    Returns sklearn classification_report as a dict.
    Also prints the human-readable report.
    """
    preds, labels = get_all_preds_labels(model, loader, device)
    report = classification_report(
        labels, preds,
        target_names=class_names,
        output_dict=True,
    )
    print(classification_report(labels, preds, target_names=class_names))
    return report


# ─────────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    model, loader, device, class_names,
    model_name: str = "model",
    results_dir: str = "../results",
):
    """
    Plots and saves a confusion matrix heatmap.

    Returns:
        cm: numpy confusion matrix
    """
    os.makedirs(results_dir, exist_ok=True)
    preds, labels = get_all_preds_labels(model, loader, device)
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(results_dir, f"{model_name}_confusion_matrix.png")
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"Confusion matrix saved → {save_path}")

    return cm


def plot_history(
    history: dict,
    model_name: str = "model",
    results_dir: str = "../results",
):
    """
    Plots training vs validation loss and accuracy curves side by side.
    Saves to results_dir.
    """
    os.makedirs(results_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss",  color="steelblue")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",    color="tomato", linestyle="--")
    ax1.set_title(f"Loss — {model_name}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train Acc",  color="steelblue")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc",    color="tomato", linestyle="--")
    ax2.set_title(f"Accuracy — {model_name}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(results_dir, f"{model_name}_history.png")
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"Training curves saved → {save_path}")


# ─────────────────────────────────────────────
# 7. Checkpointing
# ─────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, val_loss, path: str):
    """Save model + optimizer state to a .pth file."""
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss":             val_loss,
    }, path)


def load_checkpoint(model, path: str, device, optimizer=None):
    """
    Load a checkpoint into model (and optionally optimizer).

    Args:
        model    : instantiated model with the same architecture
        path     : path to .pth checkpoint file
        device   : torch.device
        optimizer: pass optimizer to also restore its state (for resuming training)

    Returns:
        epoch, val_loss from the checkpoint
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch    = checkpoint.get("epoch", -1)
    val_loss = checkpoint.get("val_loss", float("inf"))
    print(f"Loaded checkpoint: epoch={epoch}, val_loss={val_loss:.4f}  ← {path}")
    return epoch, val_loss


# ─────────────────────────────────────────────
# 8. Device Helper
# ─────────────────────────────────────────────

def get_device():
    """Returns CUDA device if available, otherwise CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    return device


# ─────────────────────────────────────────────
# 9. Metrics CSV Logger
# ─────────────────────────────────────────────

def log_metrics_to_csv(
    model_name: str,
    metrics: dict,
    csv_path: str = "../results/all_models_metrics.csv",
):
    """
    Appends a row of metrics to a shared CSV file for cross-model comparison.

    Args:
        model_name: e.g. "baseline_cnn"
        metrics   : dict with keys like accuracy, precision, recall, f1
        csv_path  : shared CSV path (created if not exists)

    Example:
        log_metrics_to_csv("baseline_cnn", {
            "test_accuracy": 0.72,
            "macro_f1": 0.71,
            "macro_precision": 0.72,
            "macro_recall": 0.71,
        })
    """
    import csv

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    row = {"model": model_name, **metrics}

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Metrics logged → {csv_path}")