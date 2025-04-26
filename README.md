# Hybrid_Attention_Mechanism_CNN_SENet_CBAM


## 📄 Overview

Convolutional Neural Networks (CNNs) have shown remarkable success in image classification. However, they often lack the capability to dynamically prioritize the most informative features.  
While **SENet** (channel attention) and **CBAM** (spatial-channel attention) individually address this limitation, their **synergistic fusion** has remained largely unexplored.

We propose a **hybrid architecture** that integrates SENet's squeeze-excitation blocks with CBAM’s parallel spatial-channel attention into popular CNN backbones (**ResNet18**, **VGG16**, **AlexNet**, and **SqueezeNet**), significantly enhancing their performance on the **CIFAR-10** dataset.

---

## 🚀 Key Contributions

- **Hybrid Attention Fusion**: Combining SENet and CBAM for simultaneous channel and spatial attention.
- **Enhanced Architectures**: Hybrid modules integrated into ResNet18, VGG16, AlexNet, and SqueezeNet.
- **Significant Performance Gains**:
  - **ResNet18**: 81.22% → **90.71%**
  - **SqueezeNet**: 71.91% → **78.29%** (+6.3%)
  - **AlexNet**: 62.67% → **71.82%** (+12.5%)
- **Faster Convergence**: 15% fewer epochs needed (40 vs. 50 epochs).
- **Reduced Validation Loss**: 0.545 → **0.533** (ResNet18).
- **Improved Interpretability**: 23% sharper Grad-CAM focus compared to individual attention mechanisms.

---

## 📈 Performance Improvements

| Model      | Baseline Accuracy | Hybrid Accuracy | Improvement |
|------------|--------------------|-----------------|-------------|
| ResNet18   | 81.22%              | **90.71%**       | +9.49%      |
| SqueezeNet | 71.91%              | **78.29%**       | +6.38%      |
| AlexNet    | 62.67%              | **71.82%**       | +12.5%      |

---

## 🔥 Visualizations (Grad-CAM)

**The following visualizations show how the hybrid attention model better focuses on the discriminative regions compared to baseline models:**

![Map_Hybrid](https://github.com/user-attachments/assets/828458d0-99f1-44da-a532-f3ba402790b1)

![image](https://github.com/user-attachments/assets/bee7596d-e60b-46fa-9a3e-bc6654ce084d)

![CNN+SENet+BAM](https://github.com/user-attachments/assets/c04682c7-e44f-4e01-8c35-73384a7da6a0)


![Loss_Acc_ResNetHybrid](https://github.com/user-attachments/assets/31387409-0697-4e6d-9803-454512ef9de1)
![Loss_Acc_AlexNetBase](https://github.com/user-attachments/assets/a34c0373-5d98-4ff0-b971-263d08e44ce0)
![Loss_Acc_Hybrid](https://github.com/user-attachments/assets/c3f9dd59-d0c5-414d-bf77-3220cfb30744)

![output6](https://github.com/user-attachments/assets/fcaf8539-4047-4b5a-b7c4-eaad2a1b06fe)



```

---

## 🛠️ How to Use

### 1. Install Dependencies
```bash
pip install torch torchvision matplotlib
```

### 2. Train the Model
Example for ResNet18 with Hybrid Attention:
```bash
python train_hybrid_attention.py --model resnet18 --dataset cifar10
```

(Adapt `--model` for `alexnet`, `vgg16`, or `squeezenet`.)

### 3. Evaluate and Visualize Attention Maps
```bash
python evaluate_model.py --visualize-gradcam
```
The Grad-CAM outputs will be saved in the `/images` folder.

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

---

## 🔮 Future Work

- Extend the hybrid attention modules to **larger datasets** (e.g., ImageNet, Pascal VOC).
- Explore **lightweight designs** for mobile deployment.
- Apply the hybrid model to **medical image analysis** and **autonomous driving** datasets.

---



## 📜 License

This project is released under the **MIT License**.

