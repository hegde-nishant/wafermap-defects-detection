import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional
import cv2
from pathlib import Path
import yaml


class MetricsTracker:
    """Track and compute training metrics."""

    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: float):
        """Update metrics with batch results."""
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        if len(preds.shape) > 1:
            preds = np.argmax(preds, axis=1)

        self.predictions.extend(preds)
        self.targets.extend(targets)
        self.losses.append(loss)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)

        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        avg_loss = np.mean(self.losses) if self.losses else 0.0

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': avg_loss
        }

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(self.targets, self.predictions)

    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        return classification_report(
            self.targets,
            self.predictions,
            target_names=self.class_names,
            zero_division=0
        )


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False


class AverageMeter:
    """Compute and store the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True
):
    """Plot confusion matrix."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """Plot training and validation metrics over epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    metrics = ['loss', 'accuracy', 'precision', 'recall']
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        train_key = f'train_{metric}'
        val_key = f'val_{metric}'

        if train_key in history:
            ax.plot(history[train_key], label=f'Train {metric.capitalize()}', marker='o')
        if val_key in history:
            ax.plot(history[val_key], label=f'Val {metric.capitalize()}', marker='s')

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_wafer_predictions(
    images: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    num_samples: int = 16
):
    """Visualize wafer maps with predictions."""
    num_samples = min(num_samples, len(images))
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for idx in range(grid_size * grid_size):
        ax = axes[idx]
        if idx < num_samples:
            img = images[idx]
            if img.shape[0] == 3:
                img = img[0]

            ax.imshow(img, cmap='viridis')

            true_class = class_names[true_labels[idx]]
            pred_class = class_names[pred_labels[idx]]

            color = 'green' if true_labels[idx] == pred_labels[idx] else 'red'
            ax.set_title(f'True: {true_class}\nPred: {pred_class}', color=color, fontsize=8)
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_attention_map(
    image: np.ndarray,
    attention: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Attention Map"
):
    """Visualize attention map overlaid on wafer map."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if image.shape[0] == 3:
        image = image[0]

    axes[0].imshow(image, cmap='viridis')
    axes[0].set_title('Original Wafer Map')
    axes[0].axis('off')

    attention_resized = cv2.resize(attention, (image.shape[1], image.shape[0]))
    axes[1].imshow(attention_resized, cmap='hot')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')

    axes[2].imshow(image, cmap='viridis', alpha=0.6)
    axes[2].imshow(attention_resized, cmap='hot', alpha=0.4)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention map saved to {save_path}")
    else:
        plt.show()

    plt.close()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config: Dict) -> str:
    """Get device for training."""
    device_config = config.get('device', 'cpu')

    # Check if CUDA/GPU is requested
    if 'cuda' in device_config.lower():
        if torch.cuda.is_available():
            # Parse device index if specified (e.g., "cuda:3")
            if ':' in device_config:
                device_idx = int(device_config.split(':')[1])
                num_gpus = torch.cuda.device_count()

                if device_idx < num_gpus:
                    device = device_config
                    print(f"Using CUDA device {device_idx}: {torch.cuda.get_device_name(device_idx)}")
                else:
                    print(f"Warning: Requested {device_config} but only {num_gpus} GPU(s) available")
                    device = 'cuda:0'
                    print(f"Falling back to cuda:0: {torch.cuda.get_device_name(0)}")
            else:
                # Just "cuda" without index, use default
                device = 'cuda'
                print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print(f"Warning: CUDA requested but not available")
            device = 'cpu'
            print("Falling back to CPU")
    else:
        # CPU or other device requested
        device = device_config
        print(f"Using device: {device}")

    return device


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_output_directories(config: Dict):
    """Create output directories if they don't exist."""
    directories = [
        config['logging']['log_dir'],
        config['logging']['checkpoint_dir'],
        config['logging']['visualization_dir']
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Directory ready: {directory}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def save_results(
    results: Dict,
    save_path: str
):
    """Save results to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    print("Utility functions loaded successfully!")

    class_names = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
                   'Loc', 'Near-full', 'Random', 'Scratch', 'none']

    metrics_tracker = MetricsTracker(num_classes=9, class_names=class_names)

    dummy_preds = torch.randn(32, 9)
    dummy_targets = torch.randint(0, 9, (32,))

    metrics_tracker.update(dummy_preds, dummy_targets, 0.5)

    metrics = metrics_tracker.compute()
    print("\nSample metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

