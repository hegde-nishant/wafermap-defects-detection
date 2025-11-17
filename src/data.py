import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Dict, Optional, List
import yaml
from pathlib import Path
import os


class WaferDataProcessor:
    """Handles loading and preprocessing of WM-811K wafer dataset."""

    def __init__(self, config: Dict):
        self.config = config
        self.data_path = config['data']['data_path']
        self.image_size = config['data']['image_size']
        self.use_existing_split = config['data']['use_existing_split']

        self.class_names = [
            'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
            'Loc', 'Near-full', 'Random', 'Scratch', 'none'
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def load_data(self) -> pd.DataFrame:
        """Load LSWMD.pkl file and perform initial processing."""
        # Handle relative paths - convert to absolute based on project root
        data_path = Path(self.data_path)
        if not data_path.is_absolute():
            # Try to find the project root (contains configs/ and data/ directories)
            current_dir = Path.cwd()

            # Check if we're already in project root
            if (current_dir / 'configs').exists() and (current_dir / 'data').exists():
                data_path = current_dir / self.data_path
            # Check if we're in src/ or notebooks/ subdirectory
            elif (current_dir.parent / 'configs').exists() and (current_dir.parent / 'data').exists():
                data_path = current_dir.parent / self.data_path
            # Otherwise use as-is
            else:
                data_path = Path(self.data_path)

        print(f"Loading data from {data_path}...")
        df = pd.read_pickle(str(data_path))
        print(f"Initial data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        df = self._clean_data(df)
        print(f"Data shape after cleaning: {df.shape}")

        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter the dataset."""
        def convert_failure_type(ftype):
            # Handle None first
            if ftype is None:
                return None

            # Handle numpy arrays
            if isinstance(ftype, np.ndarray):
                if ftype.size == 0:
                    return None
                ftype = ftype.tolist()

            # Handle NaN for non-array types
            try:
                if pd.isna(ftype):
                    return None
            except (ValueError, TypeError):
                # pd.isna can fail on some types, continue processing
                pass

            # Handle lists
            if isinstance(ftype, list):
                if len(ftype) == 0:
                    return None
                # Get first element
                first_elem = ftype[0]
                if isinstance(first_elem, list):
                    return first_elem[0] if len(first_elem) > 0 else None
                return first_elem

            # Handle strings directly
            if isinstance(ftype, str):
                return ftype if ftype else None

            return None

        df['failureType_clean'] = df['failureType'].apply(convert_failure_type)
        df_cleaned = df[df['failureType_clean'].notna()].copy()
        df_cleaned['label'] = df_cleaned['failureType_clean'].map(self.class_to_idx)
        df_cleaned = df_cleaned[df_cleaned['label'].notna()].copy()

        return df_cleaned

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        if self.use_existing_split and 'trianTestLabel' in df.columns:
            train_df = df[df['trianTestLabel'] == 'Training'].copy()
            test_df = df[df['trianTestLabel'] == 'Test'].copy()

            train_split = int(len(train_df) * self.config['data']['train_val_split'])
            val_df = train_df.iloc[train_split:].copy()
            train_df = train_df.iloc[:train_split].copy()

            print(f"Using existing split:")
        else:
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(
                df, test_size=0.2, stratify=df['label'], random_state=self.config['seed']
            )
            train_df, val_df = train_test_split(
                train_df, test_size=0.15, stratify=train_df['label'], random_state=self.config['seed']
            )
            print(f"Using custom split:")

        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    def compute_class_weights(self, df: pd.DataFrame) -> torch.Tensor:
        """Compute class weights for handling imbalanced dataset."""
        from sklearn.utils.class_weight import compute_class_weight

        labels = df['label'].values
        class_weights = compute_class_weight(
            'balanced',
            classes=np.arange(self.config['data']['num_classes']),
            y=labels
        )

        print("Class weights:")
        for name, weight in zip(self.class_names, class_weights):
            print(f"  {name}: {weight:.4f}")

        return torch.FloatTensor(class_weights)

    def get_class_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get class distribution statistics."""
        distribution = {}
        for idx, name in enumerate(self.class_names):
            count = len(df[df['label'] == idx])
            distribution[name] = count
        return distribution


class WaferMapDataset(Dataset):
    """PyTorch Dataset for wafer maps."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        aspect_ratio_threshold: float = 0.1
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_size = image_size
        self.transform = transform
        self.aspect_ratio_threshold = aspect_ratio_threshold

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        try:
            row = self.df.iloc[idx]

            # Safely extract wafer map with error handling
            wafer_map = row['waferMap']
            if wafer_map is None or (isinstance(wafer_map, np.ndarray) and wafer_map.size == 0):
                # Return dummy data for corrupted samples
                wafer_map = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            else:
                wafer_map = np.array(wafer_map, dtype=np.float32).copy()  # Force copy to avoid memory issues

            label = int(row['label'])

            wafer_image = self._preprocess_wafer_map(wafer_map)

            if self.transform:
                wafer_image = self.transform(wafer_image)

            metadata = {
                'waferIndex': row.get('waferIndex', -1),
                'lotName': row.get('lotName', 'unknown'),
                'dieSize': row.get('dieSize', -1),
                'original_shape': wafer_map.shape
            }

            return wafer_image, label, metadata

        except Exception as e:
            # Return dummy data if any error occurs
            print(f"Warning: Error loading sample {idx}: {e}")
            dummy_image = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
            return dummy_image, 0, {'waferIndex': -1, 'lotName': 'error', 'dieSize': -1, 'original_shape': (0, 0)}

    def _preprocess_wafer_map(self, wafer_map: np.ndarray) -> torch.Tensor:
        """Preprocess wafer map to fixed size image."""
        height, width = wafer_map.shape

        if height == 0 or width == 0:
            wafer_map = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        else:
            aspect_ratio = height / width
            if abs(aspect_ratio - 1.0) < self.aspect_ratio_threshold + 1.0:
                wafer_map = cv2.resize(
                    wafer_map,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_CUBIC
                )
            else:
                max_dim = max(height, width)
                padded = np.zeros((max_dim, max_dim), dtype=np.float32)
                y_offset = (max_dim - height) // 2
                x_offset = (max_dim - width) // 2
                padded[y_offset:y_offset+height, x_offset:x_offset+width] = wafer_map

                wafer_map = cv2.resize(
                    padded,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_CUBIC
                )

        wafer_map = np.clip(wafer_map, 0, 2)
        wafer_map = wafer_map / 2.0

        wafer_tensor = torch.from_numpy(wafer_map).unsqueeze(0)
        wafer_tensor = wafer_tensor.repeat(3, 1, 1)

        return wafer_tensor


def get_transforms(config: Dict, training: bool = True) -> transforms.Compose:
    """Get data augmentation transforms."""
    if training and 'augmentation' in config:
        aug_config = config['augmentation']
        transform_list = [
            transforms.RandomRotation(aug_config.get('random_rotation', 15)),
        ]

        if aug_config.get('random_flip_horizontal', False):
            transform_list.append(transforms.RandomHorizontalFlip())

        if aug_config.get('random_flip_vertical', False):
            transform_list.append(transforms.RandomVerticalFlip())

        if aug_config.get('random_brightness', 0) > 0 or aug_config.get('random_contrast', 0) > 0:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=aug_config.get('random_brightness', 0),
                    contrast=aug_config.get('random_contrast', 0)
                )
            )

        transform_list.append(
            transforms.Normalize(
                mean=config['preprocessing']['mean'],
                std=config['preprocessing']['std']
            )
        )
    else:
        transform_list = [
            transforms.Normalize(
                mean=config['preprocessing']['mean'],
                std=config['preprocessing']['std']
            )
        ]

    return transforms.Compose(transform_list)


def create_dataloaders(
    config: Dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""

    train_transform = get_transforms(config, training=True)
    val_transform = get_transforms(config, training=False)

    train_dataset = WaferMapDataset(
        train_df,
        image_size=config['data']['image_size'],
        transform=train_transform
    )

    val_dataset = WaferMapDataset(
        val_df,
        image_size=config['data']['image_size'],
        transform=val_transform
    )

    test_dataset = WaferMapDataset(
        test_df,
        image_size=config['data']['image_size'],
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    processor = WaferDataProcessor(config)
    df = processor.load_data()

    print("\nClass distribution:")
    dist = processor.get_class_distribution(df)
    for name, count in dist.items():
        print(f"{name}: {count}")

    train_df, val_df, test_df = processor.split_data(df)
    class_weights = processor.compute_class_weights(train_df)
