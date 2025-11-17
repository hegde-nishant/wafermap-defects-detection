import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import argparse
from pathlib import Path
from tqdm import tqdm
import yaml
from datetime import datetime
from typing import Dict

from data import WaferDataProcessor, create_dataloaders
from model import create_model, save_checkpoint
from utils import (
    MetricsTracker, EarlyStopping, AverageMeter,
    plot_confusion_matrix, plot_training_history,
    set_seed, get_device, load_config, create_output_directories,
    save_results
)

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.utils.data.dataloader')


class Trainer:
    """Handles model training and validation."""

    def __init__(self, config: Dict, device: str):
        self.config = config
        self.device = device

        set_seed(config['seed'])
        create_output_directories(config)

        self.class_names = [
            'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
            'Loc', 'Near-full', 'Random', 'Scratch', 'none'
        ]

        self._setup_data()
        self._setup_model()
        self._setup_training()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path(config['logging']['log_dir']) / timestamp
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs: {log_dir}")

        self.history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': [],
            'train_precision': [], 'train_recall': [],
            'val_precision': [], 'val_recall': []
        }

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

        print(f"\nDataset sizes:")
        print(f"  Train: {len(self.train_df)}")
        print(f"  Validation: {len(self.val_df)}")
        print(f"  Test: {len(self.test_df)}")

        if self.config['class_weights']['use_weights']:
            self.class_weights = processor.compute_class_weights(self.train_df)
            self.class_weights = self.class_weights.to(self.device)
        else:
            self.class_weights = None

    def _setup_model(self):
        """Setup model."""
        print("\n" + "="*50)
        print("Setting up model...")
        print("="*50)

        self.model = create_model(self.config, self.device)

    def _setup_training(self):
        """Setup training components."""
        print("\n" + "="*50)
        print("Setting up training components...")
        print("="*50)

        if self.class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        optimizer_name = self.config['training']['optimizer'].lower()
        if optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['training']['weight_decay']
            )

        scheduler_name = self.config['training']['scheduler'].lower()
        if scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs']
            )
        elif scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            self.scheduler = None

        self.early_stopping = EarlyStopping(
            patience=self.config['training']['early_stopping_patience'],
            mode='max'
        )

        if self.config['training']['mixed_precision'] and self.device == 'cuda':
            self.scaler = GradScaler()
            self.use_amp = True
            print("Using mixed precision training")
        else:
            self.scaler = None
            self.use_amp = False

        print(f"Optimizer: {optimizer_name}")
        print(f"Scheduler: {scheduler_name}")
        print(f"Learning rate: {self.config['training']['learning_rate']}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics_tracker = MetricsTracker(
            num_classes=self.config['data']['num_classes'],
            class_names=self.class_names
        )
        loss_meter = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["training"]["num_epochs"]}')

        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()

                if self.config['training']['gradient_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                if self.config['training']['gradient_clip'] > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )

                self.optimizer.step()

            loss_meter.update(loss.item(), images.size(0))
            metrics_tracker.update(outputs, labels, loss.item())

            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

            if batch_idx % self.config['logging']['log_interval'] == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), step)

        metrics = metrics_tracker.compute()
        return metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        metrics_tracker = MetricsTracker(
            num_classes=self.config['data']['num_classes'],
            class_names=self.class_names
        )
        loss_meter = AverageMeter()

        pbar = tqdm(self.val_loader, desc='Validation')

        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            loss_meter.update(loss.item(), images.size(0))
            metrics_tracker.update(outputs, labels, loss.item())

            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

        metrics = metrics_tracker.compute()
        return metrics

    def train(self):
        """Main training loop."""
        print("\n" + "="*50)
        print("Starting training...")
        print("="*50)

        best_acc = 0.0
        checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])

        for epoch in range(self.config['training']['num_epochs']):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
            print(f"{'='*50}")

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['train_precision'].append(train_metrics['precision'])
            self.history['train_recall'].append(train_metrics['recall'])

            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])

            self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)

            print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")

            is_best = val_metrics['accuracy'] > best_acc
            if is_best:
                best_acc = val_metrics['accuracy']
                print(f"\nNew best accuracy: {best_acc:.4f}")

            if (epoch + 1) % self.config['logging']['save_interval'] == 0 or is_best:
                checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
                save_checkpoint(
                    self.model, self.optimizer, epoch, best_acc,
                    str(checkpoint_path), is_best
                )

            if self.early_stopping(val_metrics['accuracy']):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        self.writer.close()

        history_path = Path(self.config['logging']['visualization_dir']) / 'training_history.png'
        plot_training_history(self.history, str(history_path))

        results = {
            'best_accuracy': float(best_acc),
            'final_train_loss': float(self.history['train_loss'][-1]),
            'final_val_loss': float(self.history['val_loss'][-1]),
            'total_epochs': len(self.history['train_loss'])
        }

        results_path = Path(self.config['logging']['log_dir']) / 'training_results.yaml'
        save_results(results, str(results_path))

        print("\n" + "="*50)
        print("Training completed!")
        print(f"Best validation accuracy: {best_acc:.4f}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Train wafer defect classification model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config)

    trainer = Trainer(config, device)
    trainer.train()


if __name__ == "__main__":
    main()
