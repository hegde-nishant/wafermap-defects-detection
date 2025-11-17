# Wafer Defect Detection using Vision Transformers

A machine learning project for semiconductor wafer defect pattern classification using Vision Transformers (ViT) and the WM-811K dataset.

![Class Distribution](outputs/visualizations/class_distribution.png)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Sample Wafer Defects](#sample-wafer-defects)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Model Architecture](#model-architecture)
- [Results](#results)

## Overview

This project implements a Vision Transformer-based classification system for detecting and categorizing defect patterns on semiconductor wafers. Using the WM-811K (LSWMD) dataset containing over 170,000 labeled wafer maps from real manufacturing processes, the model learns to identify 9 distinct defect patterns that occur during semiconductor fabrication.

### Key Highlights

- **Architecture:** Vision Transformer (ViT-tiny) with pretrained ImageNet weights
- **Comprehensive Visualizations:** 2D, 3D, and interactive plots for data exploration and model interpretability
- **Production-Ready Code:** Modular design with industry-standard practices
- **Advanced Analytics:** PCA, t-SNE clustering, attention maps, and GradCAM
- **Imbalance Handling:** Class-weighted loss and data augmentation strategies

### Visual Gallery

<p align="center">
  <img src="outputs/visualizations/sample_wafer_maps.png" width="45%" />
  <img src="outputs/visualizations/pca_clustering.png" width="45%" />
</p>

<p align="center">
  <img src="outputs/visualizations/3d_wafer_Donut.png" width="45%" />
  <img src="outputs/visualizations/spatial_defect_distribution.png" width="45%" />
</p>

*Examples of data visualizations: Sample wafer maps, PCA clustering, 3D surface plots, and spatial defect distributions*

## Features

### Machine Learning

- Vision Transformer (ViT-tiny) architecture via timm library
- Mixed precision training for faster convergence
- Early stopping and learning rate scheduling
- Comprehensive metrics tracking (accuracy, precision, recall, F1)
- Model checkpointing and TensorBoard logging
- Attention map extraction for model interpretability

### Visualizations

- **Data Exploration:**
  - Class distribution analysis
  - Wafer dimension and aspect ratio distributions
  - Defect density analysis
  - Sample visualizations per defect class

- **Advanced Analytics:**
  - 3D surface plots of wafer defect patterns
  - Interactive Plotly visualizations
  - PCA and t-SNE clustering
  - Spatial defect distribution heatmaps
  - Radial defect profile analysis
  - Contour plots

- **Model Interpretability:**
  - Attention map visualizations
  - Prediction confidence plots
  - Confusion matrices
  - Per-class performance analysis

## Dataset

### WM-811K (LSWMD) Dataset

The project uses the WM-811K wafer map dataset from MIR Lab, National Taiwan University.

- **Total Samples:** 811,457 wafer maps
- **Labeled Samples:** ~172,950 (21%)
- **Classes:** 9 defect pattern types
- **Format:** Pickle file (LSWMD.pkl, ~2GB)
- **Source:** Real semiconductor manufacturing data

### Defect Classes

1. **Center** - Defects concentrated in the wafer center
2. **Donut** - Ring-shaped defect patterns
3. **Edge-Loc** - Localized edge defects
4. **Edge-Ring** - Defects forming a ring around the edge
5. **Loc** - Localized defect clusters
6. **Near-full** - Nearly complete wafer coverage
7. **Random** - Randomly distributed defects
8. **Scratch** - Linear scratch patterns
9. **none** - Normal wafers without defects

### Dataset Statistics

- **Class Distribution:** Heavily imbalanced (85% "none" class)
- **Wafer Dimensions:** Variable (handled via resizing to 224x224)
- **Die Size:** Variable across wafers
- **Pre-split:** Training/Test labels provided in dataset

### Download Dataset

```bash
# Download from Kaggle or MIR Lab
# Place LSWMD.pkl in the data/ directory
wget [DATASET_URL] -O data/LSWMD.pkl
```

### Dataset Visualization

![Dataset Split](outputs/visualizations/train_test_split.png)
*Train/Validation/Test split distribution*

![Wafer Dimensions](outputs/visualizations/wafer_dimensions.png)
*Analysis of wafer dimensions across the dataset*

## Sample Wafer Defects

Visual examples of different defect patterns found in semiconductor wafers:

<table>
  <tr>
    <td align="center">
      <img src="outputs/visualizations/samples_Center.png" width="250px"/>
      <br/><b>Center Defects</b>
    </td>
    <td align="center">
      <img src="outputs/visualizations/samples_Donut.png" width="250px"/>
      <br/><b>Donut Defects</b>
    </td>
    <td align="center">
      <img src="outputs/visualizations/samples_Edge-Ring.png" width="250px"/>
      <br/><b>Edge-Ring Defects</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="outputs/visualizations/samples_Loc.png" width="250px"/>
      <br/><b>Localized Defects</b>
    </td>
    <td align="center">
      <img src="outputs/visualizations/samples_Scratch.png" width="250px"/>
      <br/><b>Scratch Defects</b>
    </td>
    <td align="center">
      <img src="outputs/visualizations/samples_Random.png" width="250px"/>
      <br/><b>Random Defects</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="outputs/visualizations/samples_Edge-Loc.png" width="250px"/>
      <br/><b>Edge-Loc Defects</b>
    </td>
    <td align="center">
      <img src="outputs/visualizations/samples_Near-full.png" width="250px"/>
      <br/><b>Near-full Defects</b>
    </td>
    <td align="center">
      <img src="outputs/visualizations/samples_none.png" width="250px"/>
      <br/><b>Normal Wafers</b>
    </td>
  </tr>
</table>

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- 10GB+ disk space (including dataset)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/wafer-defects-detection.git
cd wafer-defects-detection
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download and place the dataset:
```bash
# Place LSWMD.pkl in the data/ directory
mkdir -p data
# Download from your source and place in data/
```

## Project Structure

```
wafer-defects-detection/
├── configs/
│   └── config.yaml                 # Hyperparameters and settings
├── data/
│   └── LSWMD.pkl                   # WM-811K dataset (not in repo)
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Data analysis and visualization
│   └── 02_advanced_visualizations.ipynb  # 3D plots, clustering, interpretability
├── outputs/
│   ├── logs/                       # TensorBoard logs
│   ├── models/                     # Model checkpoints
│   ├── results/                    # Evaluation results
│   └── visualizations/             # Generated plots and figures
├── src/
│   ├── data.py                     # Data loading and preprocessing
│   ├── model.py                    # ViT model architecture
│   ├── train.py                    # Training pipeline
│   ├── inference.py                # Evaluation and inference
│   └── utils.py                    # Utility functions
├── .gitignore
├── README.md
├── progress.md                     # Development progress tracking
└── requirements.txt                # Python dependencies
```

## Usage

### 1. Data Exploration

Explore the dataset and generate visualizations:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This notebook provides:
- Dataset overview and statistics
- Class distribution analysis
- Wafer dimension analysis
- Sample visualizations
- Defect density analysis

### 2. Advanced Visualizations

Generate advanced analytics and 3D visualizations:

```bash
jupyter notebook notebooks/02_advanced_visualizations.ipynb
```

Features:
- 3D surface plots
- Interactive Plotly visualizations
- PCA and t-SNE clustering
- Spatial distribution heatmaps
- Radial defect profiles

### 3. Training

Train the ViT model:

```bash
cd src
python train.py --config ../configs/config.yaml
```

Training features:
- Automatic checkpointing every 5 epochs
- Early stopping (patience: 15 epochs)
- TensorBoard logging
- Mixed precision training
- Class-weighted loss

Monitor training with TensorBoard:
```bash
tensorboard --logdir outputs/logs
```

### 4. Inference and Evaluation

Evaluate the trained model:

```bash
cd src
python inference.py \
    --config ../configs/config.yaml \
    --checkpoint ../outputs/models/checkpoint_epoch_X_best.pth \
    --visualize-attention \
    --num-samples 20
```

Outputs:
- Confusion matrix
- Classification report
- Attention visualizations
- Prediction samples
- Evaluation metrics (YAML)

### 5. Custom Configuration

Modify `configs/config.yaml` to adjust:
- Model architecture parameters
- Training hyperparameters
- Data augmentation settings
- Logging and checkpoint intervals

## Visualizations

### Data Exploration

The project generates comprehensive visualizations to understand the dataset and defect patterns:

#### Defect Density Analysis
![Defect Density](outputs/visualizations/defect_density.png)
*Density analysis of defects across different wafer regions*

#### Spatial Defect Distribution
![Spatial Distribution](outputs/visualizations/spatial_defect_distribution.png)
*Heatmap showing spatial distribution of defects for each class*

### Advanced Analytics

#### 3D Visualizations

Surface plots revealing the three-dimensional structure of defect patterns:

<table>
  <tr>
    <td align="center">
      <img src="outputs/visualizations/3d_wafer_Center.png" width="300px"/>
      <br/><b>3D View - Center Defects</b>
    </td>
    <td align="center">
      <img src="outputs/visualizations/3d_wafer_Donut.png" width="300px"/>
      <br/><b>3D View - Donut Defects</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="outputs/visualizations/3d_wafer_Edge-Ring.png" width="300px"/>
      <br/><b>3D View - Edge-Ring Defects</b>
    </td>
    <td align="center">
      <img src="outputs/visualizations/3d_wafer_Near-full.png" width="300px"/>
      <br/><b>3D View - Near-full Defects</b>
    </td>
  </tr>
</table>

#### Clustering Analysis

Dimensionality reduction techniques reveal distinct clusters for different defect types:

![PCA Clustering](outputs/visualizations/pca_clustering.png)
*PCA visualization showing separation between defect classes*

![t-SNE Clustering](outputs/visualizations/tsne_clustering.png)
*t-SNE embedding revealing natural clusters in the feature space*

#### Spatial Analysis

![Radial Profiles](outputs/visualizations/radial_profiles.png)
*Radial defect density profiles from wafer center for each class*

![Contour Plots](outputs/visualizations/contour_plots.png)
*Contour plots showing defect distribution patterns*

### Model Interpretability

Model interpretability features including attention maps and prediction analysis will be generated after training. These visualizations help understand:

- ViT attention visualization showing which regions the model focuses on
- Layer-wise attention patterns across transformer blocks
- Class-specific attention focus for different defect types
- Prediction confidence distributions
- Confusion matrices and per-class performance metrics

## Model Architecture

### Vision Transformer (ViT-tiny)

```python
Model: vit_tiny_patch16_224
- Patch Size: 16x16
- Image Size: 224x224
- Embedding Dim: 192
- Depth: 12 layers
- Attention Heads: 3
- MLP Ratio: 4
- Parameters: ~5-6M
```

### Key Components

1. **Patch Embedding:** Divides 224x224 image into 196 patches (14x14 grid)
2. **Transformer Encoder:** 12 layers with multi-head self-attention
3. **Classification Head:** Custom head with LayerNorm and Dropout
4. **Pretrained Weights:** ImageNet-1K initialization

### Training Strategy

- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler:** Cosine Annealing
- **Loss:** CrossEntropyLoss with class weights
- **Batch Size:** 32
- **Mixed Precision:** Enabled (FP16)
- **Gradient Clipping:** 1.0
- **Augmentation:** Rotation, flips, color jitter

## Results

### Expected Performance

Based on similar work with WM-811K:

| Metric | Expected Range |
|--------|----------------|
| Overall Accuracy | 85-95% |
| Weighted F1 Score | 80-92% |
| Training Time (GPU) | 2-4 hours |
| Inference Speed | <10ms per image |

### Performance by Class

Results vary by defect class due to imbalance:
- High accuracy on majority classes (none, Edge-Ring)
- Lower but acceptable on minority classes (Near-full, Donut)
- Class weights help balance performance

### Training Visualizations

After training, visualizations will be automatically generated:

#### Training History
Loss and accuracy curves across epochs showing model convergence:
- Training vs Validation Loss
- Training vs Validation Accuracy
- Precision and Recall metrics
- Learning rate schedule

*Training curves will be saved to `outputs/visualizations/training_history.png`*

#### Confusion Matrix
Normalized confusion matrix showing per-class prediction accuracy:
- Diagonal elements show correct predictions
- Off-diagonal elements reveal common misclassifications
- Helps identify which defect types are confused

*Confusion matrix will be saved to `outputs/visualizations/confusion_matrix.png`*

#### Attention Maps
Visualization of what the Vision Transformer focuses on:
- Attention heatmaps overlaid on wafer maps
- Multiple samples per defect class
- Helps validate that model learns relevant patterns

*Attention maps will be saved to `outputs/visualizations/attention_maps/`*

#### Sample Predictions
Visual comparison of true labels vs predicted labels:
- Wafer map visualization
- Prediction probabilities for all classes
- Correct predictions (green) vs misclassifications (red)

*Sample predictions will be saved to `outputs/visualizations/predictions.png`*

### Model Outputs

After training, the following artifacts are generated:
- `outputs/models/checkpoint_epoch_X_best.pth` - Best model weights
- `outputs/visualizations/training_history.png` - Loss/accuracy curves
- `outputs/visualizations/confusion_matrix.png` - Confusion matrix visualization
- `outputs/visualizations/predictions.png` - Sample predictions
- `outputs/visualizations/attention_maps/` - Attention visualizations
- `outputs/logs/` - TensorBoard logs
- `outputs/results/evaluation_results.yaml` - Detailed metrics

## Configuration

### Key Configuration Parameters

Edit `configs/config.yaml`:

```yaml
data:
  data_path: "data/LSWMD.pkl"
  image_size: 224
  num_classes: 9

model:
  name: "vit_tiny_patch16_224"
  pretrained: true
  dropout: 0.1

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.0001
  optimizer: "adamw"
  scheduler: "cosine"
  early_stopping_patience: 15

class_weights:
  use_weights: true
  compute_from_data: true
```

### Customization

- **Model Variants:** Change to vit_small, vit_base for more capacity
- **Image Size:** Adjust to 96, 128, 256 (requires model change)
- **Batch Size:** Reduce if GPU memory limited
- **Augmentation:** Modify rotation angle, flip probability

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions and classes
- Include type hints where appropriate
- Write descriptive commit messages

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Dataset:** MIR Lab, National Taiwan University for the WM-811K dataset
- **Architecture:** Google Research for the Vision Transformer (ViT) architecture
- **Library:** Ross Wightman for the timm library
- **Inspiration:** Kaggle community and semiconductor manufacturing research

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{wafer-defect-detection,
  author = {Your Name},
  title = {Wafer Defect Detection using Vision Transformers},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/wafer-defects-detection}
}
```

## References

1. Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
2. WM-811K Dataset: http://mirlab.org/dataSet/public/
3. timm: PyTorch Image Models - https://github.com/rwightman/pytorch-image-models

## Contact

For questions, issues, or collaboration:
- GitHub Issues: [Project Issues](https://github.com/yourusername/wafer-defects-detection/issues)
- Email: your.email@example.com

---

**Built with PyTorch, timm, and passion for semiconductor manufacturing quality control.**
