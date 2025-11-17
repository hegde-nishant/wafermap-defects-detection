"""
Safe inference script that handles memory and data loading issues.
This version processes the test set in smaller chunks to avoid bus errors.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm
import yaml
import gc

from data import WaferDataProcessor, WaferMapDataset, get_transforms
from torch.utils.data import DataLoader
from model import create_model, load_checkpoint
from utils import (
    MetricsTracker, plot_confusion_matrix, visualize_wafer_predictions,
    visualize_attention_map, load_config, get_device, save_results
)


class SafeInferenceEngine:
    """Handles model inference with safe data loading."""

    def __init__(self, config, checkpoint_path, device):
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
        """Setup data with safe loading."""
        print("\n" + "="*50)
        print("Setting up data loaders (safe mode)...")
        print("="*50)

        processor = WaferDataProcessor(self.config)
        df = processor.load_data()

        self.train_df, self.val_df, self.test_df = processor.split_data(df)

        print(f"Test set size: {len(self.test_df)}")
        print("Using safe batch processing mode")

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
    def evaluate_in_chunks(self, chunk_size=5000):
        """Evaluate model by processing test set in chunks."""
        print("\n" + "="*50)
        print(f"Evaluating model in chunks of {chunk_size}...")
        print("="*50)

        self.model.eval()
        metrics_tracker = MetricsTracker(
            num_classes=self.config['data']['num_classes'],
            class_names=self.class_names
        )

        all_predictions = []
        all_targets = []
        sample_images = []
        sample_preds = []
        sample_targets = []

        test_transform = get_transforms(self.config, training=False)

        # Process in chunks
        total_samples = len(self.test_df)
        num_chunks = (total_samples + chunk_size - 1) // chunk_size

        print(f"Total samples: {total_samples}")
        print(f"Processing in {num_chunks} chunks")

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_samples)

            print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} (samples {start_idx}-{end_idx})...")

            # Create dataset for this chunk
            chunk_df = self.test_df.iloc[start_idx:end_idx].copy()

            try:
                chunk_dataset = WaferMapDataset(
                    chunk_df,
                    image_size=self.config['data']['image_size'],
                    transform=test_transform
                )

                # Use small batch size and no workers for safety
                chunk_loader = DataLoader(
                    chunk_dataset,
                    batch_size=32,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False
                )

                # Process chunk
                chunk_preds = []
                chunk_targets = []

                for batch_idx, (images, labels, _) in enumerate(tqdm(chunk_loader, desc=f"Chunk {chunk_idx + 1}")):
                    try:
                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        outputs = self.model(images)
                        preds = torch.argmax(outputs, dim=1)

                        metrics_tracker.update(outputs, labels, 0.0)

                        chunk_preds.extend(preds.cpu().numpy())
                        chunk_targets.extend(labels.cpu().numpy())

                        # Save some samples for visualization
                        if len(sample_images) < 100 and batch_idx % 5 == 0:
                            sample_images.extend(images.cpu().numpy()[:5])
                            sample_preds.extend(preds.cpu().numpy()[:5])
                            sample_targets.extend(labels.cpu().numpy()[:5])

                        # Free memory
                        del images, labels, outputs, preds
                        if batch_idx % 10 == 0:
                            torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"Error in batch {batch_idx}: {e}")
                        continue

                all_predictions.extend(chunk_preds)
                all_targets.extend(chunk_targets)

                # Clean up
                del chunk_dataset, chunk_loader, chunk_df
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing chunk {chunk_idx}: {e}")
                print("Continuing with next chunk...")
                continue

        # Compute final metrics
        print("\n" + "="*50)
        print("Computing final metrics...")
        print("="*50)

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

        # Save visualizations
        viz_dir = Path(self.config['logging']['visualization_dir'])

        cm_path = viz_dir / 'confusion_matrix.png'
        plot_confusion_matrix(cm, self.class_names, str(cm_path))

        if len(sample_images) > 0:
            num_viz = min(20, len(sample_images))
            pred_path = viz_dir / 'predictions.png'
            visualize_wafer_predictions(
                np.array(sample_images[:num_viz]),
                np.array(sample_targets[:num_viz]),
                np.array(sample_preds[:num_viz]),
                self.class_names,
                str(pred_path),
                num_samples=num_viz
            )

        # Save results
        results = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1']),
            'total_samples_processed': len(all_predictions),
            'classification_report': report
        }

        results_path = viz_dir / 'evaluation_results.yaml'
        save_results(results, str(results_path))

        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets
        }

    @torch.no_grad()
    def predict_samples(self, num_samples=20):
        """Generate predictions for sample images."""
        print("\n" + "="*50)
        print(f"Generating predictions for {num_samples} samples...")
        print("="*50)

        self.model.eval()
        viz_dir = Path(self.config['logging']['visualization_dir']) / 'sample_predictions'
        viz_dir.mkdir(exist_ok=True, parents=True)

        # Take random samples
        sample_df = self.test_df.sample(n=min(num_samples, len(self.test_df)))

        test_transform = get_transforms(self.config, training=False)

        sample_dataset = WaferMapDataset(
            sample_df,
            image_size=self.config['data']['image_size'],
            transform=test_transform
        )

        sample_loader = DataLoader(
            sample_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        sample_count = 0
        for images, labels, _ in sample_loader:
            if sample_count >= num_samples:
                break

            try:
                images = images.to(self.device)

                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)
                probabilities = torch.softmax(outputs, dim=1)

                image = images[0].cpu().numpy()
                true_label = labels[0].item()
                pred_label = predictions[0].item()
                probs = probabilities[0].cpu().numpy()

                # Visualize
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

            except Exception as e:
                print(f"Error processing sample {sample_count}: {e}")
                continue

        print(f"Saved {sample_count} prediction visualizations to {viz_dir}")


def main():
    parser = argparse.ArgumentParser(description='Safe inference for wafer defect classification')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--chunk-size', type=int, default=5000,
                        help='Chunk size for processing (default: 5000)')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of samples for visualization')
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config)

    inference_engine = SafeInferenceEngine(config, args.checkpoint, device)

    # Run evaluation in chunks
    results = inference_engine.evaluate_in_chunks(chunk_size=args.chunk_size)

    # Generate sample predictions
    inference_engine.predict_samples(num_samples=args.num_samples)

    print("\n" + "="*50)
    print("Safe inference completed!")
    print("="*50)


if __name__ == "__main__":
    main()
