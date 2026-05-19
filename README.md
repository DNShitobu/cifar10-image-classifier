# 🖼️ CIFAR-10 Image Classifier

PyTorch residual CNN classifying 32×32 colour images into 10 categories with data augmentation and cosine LR scheduling.

## Overview
| Detail | Value |
|--------|-------|
| Type | Multi-class Image Classification |
| Dataset | CIFAR-10 (60,000 colour images, auto-downloaded) |
| Framework | PyTorch |
| Architecture | Stem + 4 residual ConvBlocks + AdaptiveAvgPool |

## Getting Started
```bash
git clone https://github.com/Dnshitobu/cifar10-image-classifier.git
cd cifar10-image-classifier
pip install -r requirements.txt
python cifar10_cnn.py
```

## Architecture
```
Input (3×32×32)
  Stem Conv(3→32) + BN + ReLU
  ConvBlock(32→64) with skip connection
  ConvBlock(64→128, stride=2)
  ConvBlock(128→256, stride=2)
  ConvBlock(256→512, stride=2)
  AdaptiveAvgPool → Dropout → Linear(512→10)
```

## Results
| Epochs | Accuracy |
|--------|:---:|
| 10 | ~78% |
| 20 | ~83% |

## Concepts Covered
Data augmentation · Residual connections · AdamW · Cosine annealing · Label smoothing · Per-class accuracy
