import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import yaml
from typing import Dict, Tuple, Optional

from data import WaferDataProcessor, create_dataloaders
from model import create_model, load_checkpoint
from utils import (
    MetricsTracker, plot_confusion_matrix, visualize_wafer_predictions,
    visualize_attention_map, load_config, get_device, save_results
)


class InferenceEngine:
    """Handles model inference and evaluation."""

    def __init__(self, config: Dict, checkpoint_path: str, device: str):
        self.config = config
        self.device = device
        self.checkpoint_path = checkpoint_path

        self.class_names = [
            'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
            'Loc', 'Near-full', 'Random', 'Scratch', 'none'
        ]

        self._setup_data()
        self._setup_model()

    def _setup_data(self):
        """Setup data loaders."""
        print("\n" + "="*50)
        print("Setting up data loaders...")
        print("="*50)

        processor = WaferDataProcessor(self.config)
        df = processor.load_data()

        self.train_df, self.val_df, self.test_df = processor.split_data(df)

        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            self.config, self.train_df, self.val_df, self.test_df
        )

        print(f"Test set size: {len(self.test_df)}")

    def _setup_model(self):
        """Setup and load model."""
        print("\n" + "="*50)
        print("Loading model...")
        print("="*50)

        self.model = create_model(self.config, self.device)
        self.model, epoch, best_acc = load_checkpoint(
            self.model, self.checkpoint_path, self.device
        )
        self.model.eval()

        print(f"Model loaded from epoch {epoch} with accuracy {best_acc:.4f}")

    @torch.no_grad()
    def evaluate(self, dataloader=None) -> Dict:
        """Evaluate model on test set."""
        if dataloader is None:
            dataloader = self.test_loader

        print("\n" + "="*50)
        print("Evaluating model...")
        print("="*50)

        self.model.eval()
        metrics_tracker = MetricsTracker(
            num_classes=self.config['data']['num_classes'],
            class_names=self.class_names
        )

        all_images = []
        all_predictions = []
        all_targets = []

        pbar = tqdm(dataloader, desc='Evaluating')

        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            preds = torch.argmax(outputs, dim=1)

            metrics_tracker.update(outputs, labels, 0.0)

            all_images.extend(images.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

        metrics = metrics_tracker.compute()
        cm = metrics_tracker.get_confusion_matrix()
        report = metrics_tracker.get_classification_report()

        print("\n" + "="*50)
        print("Evaluation Results:")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print("\n" + "="*50)
        print("Classification Report:")
        print("="*50)
        print(report)

        viz_dir = Path(self.config['logging']['visualization_dir'])

        cm_path = viz_dir / 'confusion_matrix.png'
        plot_confusion_matrix(cm, self.class_names, str(cm_path))

        num_viz = min(
            self.config['evaluation']['num_visualization_samples'],
            len(all_images)
        )
        pred_path = viz_dir / 'predictions.png'
        visualize_wafer_predictions(
            np.array(all_images[:num_viz]),
            np.array(all_targets[:num_viz]),
            np.array(all_predictions[:num_viz]),
            self.class_names,
            str(pred_path),
            num_samples=num_viz
        )

        results = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1']),
            'classification_report': report
        }

        results_path = viz_dir / 'evaluation_results.yaml'
        save_results(results, str(results_path))

        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'images': all_images,
            'predictions': all_predictions,
            'targets': all_targets
        }

    @torch.no_grad()
    def predict_single(self, image: torch.Tensor) -> Tuple[int, np.ndarray]:
        """Predict single wafer map."""
        self.model.eval()

        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)
        output = self.model(image)

        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1)

        return prediction.item(), probabilities.cpu().numpy()[0]

    def visualize_attention(self, num_samples: int = 10):
        """Visualize attention maps for sample predictions."""
        print("\n" + "="*50)
        print("Generating attention visualizations...")
        print("="*50)

        self.model.eval()
        viz_dir = Path(self.config['logging']['visualization_dir']) / 'attention_maps'
        viz_dir.mkdir(exist_ok=True)

        sample_count = 0
        for images, labels, metadata in self.test_loader:
            if sample_count >= num_samples:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            for idx in range(min(images.size(0), num_samples - sample_count)):
                image = images[idx:idx+1]
                label = labels[idx].item()

                attention_map = self._extract_attention_map(image)

                if attention_map is not None:
                    image_np = image.cpu().numpy()[0]

                    save_path = viz_dir / f'attention_sample_{sample_count}_class_{self.class_names[label]}.png'

                    visualize_attention_map(
                        image_np,
                        attention_map,
                        str(save_path),
                        title=f'Attention Map - True: {self.class_names[label]}'
                    )

                    sample_count += 1

                if sample_count >= num_samples:
                    break

        print(f"Generated {sample_count} attention visualizations in {viz_dir}")

    def _extract_attention_map(self, image: torch.Tensor) -> Optional[np.ndarray]:
        """Extract attention map from ViT model."""
        try:
            attention_maps = []

            def attention_hook(module, input, output):
                if isinstance(output, tuple):
                    attn = output[1] if len(output) > 1 else output[0]
                else:
                    attn = output

                if hasattr(attn, 'detach'):
                    attention_maps.append(attn.detach())

            hooks = []
            for name, module in self.model.named_modules():
                if 'attn' in name.lower() and hasattr(module, 'forward'):
                    hook = module.register_forward_hook(attention_hook)
                    hooks.append(hook)

            with torch.no_grad():
                _ = self.model(image)

            for hook in hooks:
                hook.remove()

            if attention_maps:
                attn = attention_maps[-1]

                if len(attn.shape) == 4:
                    attn = attn[0].mean(0)
                elif len(attn.shape) == 3:
                    attn = attn[0]

                attn = attn.cpu().numpy()

                if len(attn.shape) == 2:
                    grid_size = int(np.sqrt(attn.shape[0] - 1))
                    attn = attn[1:, 1:].reshape(grid_size, grid_size, -1).mean(-1)
                elif len(attn.shape) == 1:
                    grid_size = int(np.sqrt(len(attn) - 1))
                    attn = attn[1:].reshape(grid_size, grid_size)

                attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

                return attn

            return None

        except Exception as e:
            print(f"Error extracting attention map: {e}")
            return None

    def predict_and_visualize(self, num_samples: int = 5):
        """Make predictions and visualize results for random samples."""
        print("\n" + "="*50)
        print(f"Predicting {num_samples} random samples...")
        print("="*50)

        self.model.eval()
        viz_dir = Path(self.config['logging']['visualization_dir']) / 'sample_predictions'
        viz_dir.mkdir(exist_ok=True)

        sample_count = 0
        for images, labels, _ in self.test_loader:
            if sample_count >= num_samples:
                break

            images = images.to(self.device)

            with torch.no_grad():
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)
                probabilities = torch.softmax(outputs, dim=1)

            for idx in range(min(images.size(0), num_samples - sample_count)):
                image = images[idx].cpu().numpy()
                true_label = labels[idx].item()
                pred_label = predictions[idx].item()
                probs = probabilities[idx].cpu().numpy()

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                if image.shape[0] == 3:
                    image_display = image[0]
                else:
                    image_display = image

                axes[0].imshow(image_display, cmap='viridis')
                axes[0].set_title(f'True: {self.class_names[true_label]}\n'
                                  f'Pred: {self.class_names[pred_label]}',
                                  color='green' if true_label == pred_label else 'red')
                axes[0].axis('off')

                axes[1].barh(self.class_names, probs)
                axes[1].set_xlabel('Probability')
                axes[1].set_title('Class Probabilities')
                axes[1].set_xlim([0, 1])

                plt.tight_layout()

                save_path = viz_dir / f'prediction_{sample_count}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

                sample_count += 1

            if sample_count >= num_samples:
                break

        print(f"Saved {sample_count} prediction visualizations to {viz_dir}")


def main():
    parser = argparse.ArgumentParser(description='Inference for wafer defect classification')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--visualize-attention', action='store_true',
                        help='Generate attention visualizations')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of samples for visualization')
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config)

    inference_engine = InferenceEngine(config, args.checkpoint, device)

    results = inference_engine.evaluate()

    if args.visualize_attention:
        inference_engine.visualize_attention(num_samples=args.num_samples)

    inference_engine.predict_and_visualize(num_samples=min(10, args.num_samples))

    print("\n" + "="*50)
    print("Inference completed!")
    print("="*50)


if __name__ == "__main__":
    main()
