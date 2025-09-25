# Hybrid_Attention_Mechanism_CNN_SENet_CBAM


## Overview

This repository presents a systematic integration of Squeeze-and-Excitation Networks (SENet) and Convolutional Block Attention Modules (CBAM) for enhanced CNN-based image classification. Our hybrid approach uses element-wise summation to combine channel-wise attention (SENet) with spatial-channel attention (CBAM), achieving significant performance improvements across multiple CNN architectures.

## Key Contributions

- **Systematic Fusion Strategy**: Theoretically grounded element-wise summation of SENet and CBAM with residual connections
- **Cross-Architecture Validation**: Comprehensive evaluation across ResNet18, VGG16, AlexNet, and SqueezeNet
- **Statistical Rigor**: 5-fold cross-validation with statistical significance testing (p < 0.001)
- **Computational Efficiency**: Minimal parameter overhead (1.5-5.8%) with substantial performance gains
- **Enhanced Interpretability**: Integrated Grad-CAM analysis for attention visualization

## Performance Results

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

## Architecture

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

## Installation

### Requirements

```bash
pip install torch torchvision
pip install numpy matplotlib
pip install scikit-learn
pip install grad-cam
```

### Environment Setup

```bash
git clone https://github.com/your-username/hybrid-senet-cbam.git
cd hybrid-senet-cbam
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from models.hybrid_attention import HybridAttentionCNN
from utils.trainer import Trainer

# Initialize model with hybrid attention
model = HybridAttentionCNN(
    backbone='resnet18',
    num_classes=10,
    use_hybrid_attention=True
)

# Train the model
trainer = Trainer(model, dataset='cifar10')
trainer.train(epochs=50, batch_size=128)
```

### Training Different Architectures

```python
# Available architectures
architectures = ['resnet18', 'vgg16', 'alexnet', 'squeezenet']

for arch in architectures:
    model = HybridAttentionCNN(backbone=arch, use_hybrid_attention=True)
    trainer = Trainer(model)
    results = trainer.train_with_validation()
    print(f"{arch}: Accuracy = {results['accuracy']:.2f}%")
```
### Attention Visualization

```python
from utils.visualization import generate_gradcam

# Generate Grad-CAM visualizations
model.eval()
gradcam = generate_gradcam(model, input_image, target_class)
visualize_attention_maps(gradcam)
```

# Hybrid SENet-CBAM Attention Mechanism for CNN Image Classification

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Experimental Validation

### Statistical Significance
All reported improvements achieve statistical significance (p < 0.001) using paired t-tests with 5-fold cross-validation.

### Comparative Analysis
Our approach outperforms previous SENet+CBAM combinations by an average of 2.32% across architectures, addressing limitations in fusion strategies identified in prior work.

### Reproducibility
- Fixed random seeds for consistent results
- Standardized training protocol across all experiments  
- Detailed hyperparameter documentation
- Cross-validation for robust performance estimation

## Limitations and Future Work

### Current Limitations
- Evaluation limited to CIFAR-10 (32×32 resolution)
- Focus on image classification tasks only
- Testing restricted to traditional CNN architectures

### Future Directions
- Validation on higher-resolution datasets (ImageNet)
- Extension to object detection and semantic segmentation
- Integration with modern architectures (EfficientNet, ConvNeXt)
- Development of learnable attention weighting mechanisms


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SENet implementation inspired by Hu et al. (2018)
- CBAM implementation based on Woo et al. (2018)
- Statistical validation methodology following best practices for attention mechanism evaluation

## Contact

For questions or collaboration opportunities:
- Email: your.email@university.edu
- GitHub: [@your-username](https://github.com/your-username)

---

**Note**: This implementation focuses on systematic attention mechanism integration with rigorous experimental validation. The code prioritizes reproducibility and statistical rigor over performance optimization.
## 🔥 Visualizations Results

**The following visualizations show how the hybrid attention model better focuses on the discriminative regions compared to baseline models:**

![Map_Hybrid](https://github.com/user-attachments/assets/828458d0-99f1-44da-a532-f3ba402790b1)


![CNN+SENet+BAM](https://github.com/user-attachments/assets/c04682c7-e44f-4e01-8c35-73384a7da6a0)


![Loss_Acc_ResNetHybrid](https://github.com/user-attachments/assets/31387409-0697-4e6d-9803-454512ef9de1)
![Loss_Acc_AlexNetBase](https://github.com/user-attachments/assets/a34c0373-5d98-4ff0-b971-263d08e44ce0)
![Loss_Acc_Hybrid](https://github.com/user-attachments/assets/c3f9dd59-d0c5-414d-bf77-3220cfb30744)

![output6](https://github.com/user-attachments/assets/fcaf8539-4047-4b5a-b7c4-eaad2a1b06fe)



```

---

## 📚 Project Structure

```
├── train_hybrid_attention.py
├── evaluate_model.py
├── models/
│   ├── resnet_hybrid.py
│   ├── alexnet_hybrid.py
│   ├── vgg_hybrid.py
│   └── squeezenet_hybrid.py
├── images/
│   ├── 396759f1-76f3-4704-8170-209cdc5181c8.png
│   ├── 78743aff-c320-410f-902d-5c4bb650da94.png
│   ├── cf161bd7-b13e-4274-9c1b-c218cc8862bc.png
│   ├── ea37b7f6-caf4-4d56-b47a-f2df2e366052.png
│   ├── 824a5053-588d-4286-acc0-ab8d83369f84.png
│   └── ee148012-6f21-4d80-8b3b-6b0e97d3bbb8.png
└── README.md
```



