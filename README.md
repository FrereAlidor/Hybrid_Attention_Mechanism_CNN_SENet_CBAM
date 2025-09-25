# Hybrid Attention Mechanism: CNN + SENet + CBAM

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Performance Results](#performance-results)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Visualizations](#visualizations)
- [Experimental Validation](#experimental-validation)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This repository presents a systematic integration of Squeeze-and-Excitation Networks (SENet) and Convolutional Block Attention Modules (CBAM) for enhanced CNN-based image classification. Our hybrid approach uses element-wise summation to combine channel-wise attention (SENet) with spatial-channel attention (CBAM), achieving significant performance improvements across multiple CNN architectures.

## 🔑 Key Contributions

- **Systematic Fusion Strategy**: Theoretically grounded element-wise summation of SENet and CBAM with residual connections
- **Cross-Architecture Validation**: Comprehensive evaluation across ResNet18, VGG16, AlexNet, and SqueezeNet
- **Statistical Rigor**: 5-fold cross-validation with statistical significance testing (p < 0.001)
- **Computational Efficiency**: Minimal parameter overhead (1.5-5.8%) with substantial performance gains
- **Enhanced Interpretability**: Integrated Grad-CAM analysis for attention visualization

## 📊 Performance Results

### Classification Performance (CIFAR-10)

| Architecture | Baseline Accuracy | Hybrid Accuracy | Improvement | F1-Score Gain |
|--------------|-------------------|-----------------|-------------|---------------|
| ResNet18     | 77.93%           | **90.71%**      | **+12.78%** | +4.97%        |
| VGG16        | 55.78%           | **70.17%**      | **+14.39%** | +19.77%       |
| AlexNet      | 62.67%           | **71.82%**      | **+9.15%**  | +12.55%       |
| SqueezeNet   | 71.91%           | **78.29%**      | **+6.38%**  | +8.58%        |

### Computational Efficiency Analysis

| Model + Attention | Parameters (M) | Parameter Overhead | Inference Time (ms) | FLOPs (M) |
|-------------------|----------------|--------------------|--------------------|-----------|
| ResNet18 Baseline | 11.17          | -                  | 2.3 ± 0.1          | 556.3     |
| ResNet18 + Hybrid | 11.82          | **+5.8%**         | 2.5 ± 0.1          | 578.6     |
| VGG16 Baseline    | 138.36         | -                  | 8.7 ± 0.3          | 15,300.2  |
| VGG16 + Hybrid    | 142.45         | **+3.0%**         | 9.2 ± 0.3          | 15,489.7  |
| AlexNet Baseline  | 61.10          | -                  | 3.1 ± 0.2          | 714.8     |
| AlexNet + Hybrid  | 62.03          | **+1.5%**         | 3.3 ± 0.2          | 731.2     |
| SqueezeNet Baseline | 1.25         | -                  | 1.8 ± 0.1          | 351.9     |
| SqueezeNet + Hybrid | 1.32         | **+5.6%**         | 1.9 ± 0.1          | 364.8     |

### Training Efficiency

| Architecture | Epochs to Convergence | Training Acceleration | Validation Loss Reduction |
|--------------|----------------------|----------------------|---------------------------|
| ResNet18     | 40 (vs 47 baseline)  | +14.9%               | 0.545 → 0.533            |
| VGG16        | 42 (vs 50 baseline)  | +16.0%               | 1.803 → 1.633            |
| AlexNet      | 41 (vs 49 baseline)  | +16.3%               | 1.363 → 0.625            |
| SqueezeNet   | 39 (vs 48 baseline)  | +18.8%               | 0.796 → 0.603            |

*Average training convergence improvement: **+16.5%***

## 🏗️ Architecture

Our hybrid attention mechanism integrates SENet and CBAM through the following process:

```python
def hybrid_attention(x):
    # Apply attention mechanisms in parallel
    senet_output = apply_senet(x)
    cbam_output = apply_cbam(x)
    
    # Element-wise fusion
    hybrid_output = senet_output + cbam_output
    
    # Residual connection
    final_output = x + hybrid_output
    
    return final_output
```

## 🚀 Installation

### Requirements

```bash
pip install torch torchvision
pip install numpy matplotlib
pip install scikit-learn
pip install grad-cam
```

### Setup Instructions

1. **Clone Repository**
```bash
git clone https://github.com/your-username/hybrid-attention-cnn.git
cd hybrid-attention-cnn
```

2. **Create Virtual Environment** (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify Installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
```

## 💻 Usage

### Option 1: Quick Start (Single Model)

Train a single model configuration:

```bash
# Train ResNet18 with Hybrid Attention (recommended for best results)
python quick_start.py --backbone resnet18 --attention hybrid --epochs 50

# Train VGG16 with CBAM only
python quick_start.py --backbone vgg16 --attention cbam --epochs 50

# Train AlexNet without attention (baseline)
python quick_start.py --backbone alexnet --attention none --epochs 50

# Train SqueezeNet with SENet only
python quick_start.py --backbone squeezenet --attention senet --epochs 50
```

**Available Arguments:**
- `--backbone`: resnet18, vgg16, alexnet, squeezenet
- `--attention`: none, senet, cbam, hybrid
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.003)
- `--device`: cuda or cpu (default: cuda)

### Option 2: Complete Experimental Pipeline

Reproduce all paper results (trains 16 models):

```bash
python experiments/run_all_experiments.py
```

This will:
- Train all 4 architectures with 4 attention configurations (16 models total)
- Generate comprehensive result tables
- Save all metrics to CSV files
- Create performance comparison plots

*Expected runtime: 8-12 hours on GPU, 40+ hours on CPU*

### Option 3: Custom Training Script

```python
from data.cifar10_loader import get_cifar10_loaders
from models.hybrid_attention import create_model
from utils.trainer import Trainer

# Load data
train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=128)

# Create model
model = create_model(backbone='resnet18', attention_type='hybrid')

# Train
trainer = Trainer(model, device='cuda')
history = trainer.train(train_loader, val_loader, epochs=50)

# Evaluate
results = trainer.evaluate_metrics(test_loader)
print(f"Test Accuracy: {results['accuracy']:.2f}%")
```

### Attention Visualization

```python
from utils.visualization import generate_gradcam

# Generate Grad-CAM visualizations
model.eval()
gradcam = generate_gradcam(model, input_image, target_class)
visualize_attention_maps(gradcam)
```

## 📁 Project Structure

```
├── train_hybrid_attention.py
├── evaluate_model.py
├── quick_start.py
├── experiments/
│   └── run_all_experiments.py
├── models/
│   ├── resnet_hybrid.py
│   ├── alexnet_hybrid.py
│   ├── vgg_hybrid.py
│   ├── squeezenet_hybrid.py
│   └── hybrid_attention.py
├── data/
│   └── cifar10_loader.py
├── utils/
│   ├── trainer.py
│   └── visualization.py
├── results/
│   ├── all_experiments.csv
│   ├── classification_performance.csv
│   ├── validation_performance.csv
│   └── *.png
├── images/
│   └── [visualization images]
├── requirements.txt
└── README.md
```

### Output Files

After running experiments, the following files are generated in `results/`:

```
results/
├── all_experiments.csv              # Complete results table
├── classification_performance.csv   # Classification metrics
├── validation_performance.csv       # Validation metrics
├── [model]_[attention]_curves.png  # Training curves per model
├── [model]_[attention]_model.pth   # Saved model weights
├── comparative_performance.png      # Performance comparison plots
├── performance_heatmap.png         # Results heatmap
├── parameter_efficiency.png        # Efficiency analysis
└── summary_report.txt              # Text summary report
```

## 🔥 Visualizations

**The following visualizations show how the hybrid attention model better focuses on discriminative regions compared to baseline models:**

![Map_Hybrid](https://github.com/user-attachments/assets/828458d0-99f1-44da-a532-f3ba402790b1)

![CNN+SENet+BAM](https://github.com/user-attachments/assets/c04682c7-e44f-4e01-8c35-73384a7da6a0)

![Loss_Acc_ResNetHybrid](https://github.com/user-attachments/assets/31387409-0697-4e6d-9803-454512ef9de1)

![Loss_Acc_AlexNetBase](https://github.com/user-attachments/assets/a34c0373-5d98-4ff0-b971-263d08e44ce0)

![Loss_Acc_Hybrid](https://github.com/user-attachments/assets/c3f9dd59-d0c5-414d-bf77-3220cfb30744)

![output6](https://github.com/user-attachments/assets/fcaf8539-4047-4b5a-b7c4-eaad2a1b06fe)

## 🧪 Experimental Validation

### Statistical Significance
All reported improvements achieve statistical significance (p < 0.001) using paired t-tests with 5-fold cross-validation.

### Comparative Analysis
Our approach outperforms previous SENet+CBAM combinations by an average of 2.32% across architectures, addressing limitations in fusion strategies identified in prior work.

### Reproducibility
The code includes fixed random seeds for reproducibility:

```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
```

## 🛠️ Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python quick_start.py --batch_size 64

# Or use CPU (slower)
python quick_start.py --device cpu
```

### Missing Dependencies
```bash
# Install specific package
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For FLOPs calculation
pip install thop
```

### Data Download Issues
```bash
# Manually download CIFAR-10
mkdir -p data/cifar-10-batches-py
# Download from: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# Extract to data/ directory
```

### Performance Optimization

**GPU Training:**
```bash
# Check GPU availability
nvidia-smi

# Set specific GPU
export CUDA_VISIBLE_DEVICES=0
python quick_start.py --device cuda
```

**Mixed Precision Training** (Optional):
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SENet implementation inspired by Hu et al. (2018)
- CBAM implementation based on Woo et al. (2018)
- Statistical validation methodology following best practices for attention mechanism evaluation

## 📧 Contact

For questions or collaboration opportunities:
- **Email**: alidor.mbayandjambe@unikin.ac.cd

---

**Note**: This implementation focuses on systematic attention mechanism integration with rigorous experimental validation. The code prioritizes reproducibility and statistical rigor over performance optimization.
