import torch
import torch.nn as nn
import timm
from typing import Dict, Tuple, Optional
import numpy as np


class WaferViT(nn.Module):
    """Vision Transformer for wafer defect classification."""

    def __init__(
        self,
        model_name: str = 'vit_tiny_patch16_224',
        num_classes: int = 9,
        pretrained: bool = True,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1
    ):
        super(WaferViT, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
            drop_path_rate=drop_path_rate
        )

        if hasattr(self.backbone, 'head'):
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes)
            )

        print(f"Model initialized: {model_name}")
        print(f"Number of classes: {num_classes}")
        print(f"Pretrained: {pretrained}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)

    def get_attention_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """Extract attention maps from the model."""
        attention_maps = []

        def hook_fn(module, input, output):
            if hasattr(output, 'shape') and len(output.shape) == 4:
                attention_maps.append(output.detach())

        hooks = []
        for name, module in self.backbone.named_modules():
            if 'attn' in name and hasattr(module, 'forward'):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)

        logits = self.forward(x)

        for hook in hooks:
            hook.remove()

        return logits, attention_maps

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WaferViTWithGradCAM(nn.Module):
    """ViT model with Grad-CAM support for visualization."""

    def __init__(self, base_model: WaferViT):
        super(WaferViTWithGradCAM, self).__init__()
        self.model = base_model
        self.gradients = None
        self.activations = None

    def save_gradient(self, grad):
        """Save gradients during backward pass."""
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with activation capture."""
        return self.model(x)

    def get_activations_gradient(self):
        """Get saved gradients."""
        return self.gradients

    def get_activations(self, x: torch.Tensor):
        """Get activations from intermediate layer."""
        return self.activations

    def generate_cam(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate class activation map."""
        self.model.eval()
        x.requires_grad = True

        output = self.forward(x)

        if target_class is None:
            target_class = output.argmax(dim=1)

        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward(retain_graph=True)

        gradients = self.get_activations_gradient()
        activations = self.get_activations(x)

        if gradients is not None and activations is not None:
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

            for i in range(activations.size(1)):
                activations[:, i, :, :] *= pooled_gradients[i]

            heatmap = torch.mean(activations, dim=1).squeeze()
            heatmap = torch.relu(heatmap)
            heatmap /= torch.max(heatmap)

            return heatmap.cpu().detach().numpy()

        return None


def create_model(config: Dict, device: str = 'cuda') -> WaferViT:
    """Create model from configuration."""
    model_config = config['model']

    model = WaferViT(
        model_name=model_config['name'],
        num_classes=model_config['num_classes'],
        pretrained=model_config['pretrained'],
        dropout=model_config['dropout'],
        drop_path_rate=model_config['drop_path_rate']
    )

    model = model.to(device)

    num_params = model.get_num_params()
    trainable_params = model.get_trainable_params()

    print(f"\nModel Statistics:")
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {num_params * 4 / 1024 / 1024:.2f} MB")

    return model


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = 'cuda'
) -> Tuple[nn.Module, int, float]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)

    print(f"Loaded checkpoint from epoch {epoch} with accuracy {best_acc:.4f}")

    return model, epoch, best_acc


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_acc: float,
    checkpoint_path: str,
    is_best: bool = False
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }

    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = checkpoint_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")


if __name__ == "__main__":
    import yaml

    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = create_model(config, device)

    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}")
    print(f"Model created successfully!")
