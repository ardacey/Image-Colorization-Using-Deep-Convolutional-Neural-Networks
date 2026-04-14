# Image Colorization Using Deep Convolutional Neural Networks

A PyTorch implementation of automatic image colorization. Given a grayscale image, the model predicts plausible color information by learning from a large set of natural scene images. The project compares multiple encoder-decoder architectures, pretrained backbones, attention mechanisms, and loss functions.

---

## Overview

Image colorization is an inherently ambiguous problem — a single grayscale image can correspond to many valid colorizations. This project tackles that challenge using the **CIELAB color space**:

- **Input:** L\* channel (lightness / grayscale)
- **Output:** a\* and b\* channels (color)

Decoupling luminance from chrominance makes the prediction task simpler and avoids artifacts caused by predicting all three RGB channels jointly.

---

## Architectures

| Model | Description |
|---|---|
| Baseline | Custom encoder-decoder with low-level + global feature extractors |
| VGG16 (fine-tuned) | VGG16 backbone for global features, end-to-end training |
| VGG16 (frozen) | VGG16 backbone with frozen weights |
| ResNet50 | ResNet50 backbone for global features |
| Improved (CBAM) | U-Net-style skip connections + CBAM attention module |
| PatchGAN | Improved model trained with adversarial PatchGAN discriminator |

---

## Loss Functions

- **L1 Loss** — pixel-wise reconstruction
- **Perceptual Loss** — VGG16 feature-space similarity
- **Combined Loss** — L1 + Perceptual
- **Adversarial Loss** — GAN-based with PatchGAN discriminator

---

## Dataset

[Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) preprocessed to 128×128 RGB images.

| Split | Images |
|---|---|
| Training | 6,400 |
| Validation | 1,600 |
| **Total** | **8,000** |

---

## Results

Best model (Improved + CBAM, Combined Loss) after 50 epochs:

| Metric | Value |
|---|---|
| Val Loss | ~0.0498 |
| PSNR | ~27.0 dB |
| SSIM | ~0.962 |

---

## Requirements

```
torch
torchvision
numpy
Pillow
scikit-image
matplotlib
tqdm
```

The notebook is designed to run on **Google Colab** with GPU acceleration. Update the dataset paths in the `Config` class to match your environment.

---

## Usage

1. Upload your dataset to Google Drive with the following structure:
   ```
   dataset/
   ├── train/   # 6,400 RGB images
   └── val/     # 1,600 RGB images
   ```

2. Open `image_colorization.ipynb` in Google Colab.

3. Mount your Drive and update the paths in the `Config` cell.

4. Run all cells to train, evaluate, and visualize results.

---

## Project Structure

```
.
├── image_colorization.ipynb   # Main notebook
└── README.md
```

---

## Key Findings

- **Pretrained backbones** (VGG16 fine-tuned) outperform the custom baseline in semantic color accuracy.
- **Attention mechanisms** (CBAM) improve color boundary sharpness.
- **Combined Loss** (L1 + Perceptual) yields better perceptual quality than L1 alone.
- **PatchGAN** produces sharper and more saturated colors but can introduce artifacts.
- Common failure cases: ambiguous object colors (e.g., cars, clothing), unusual lighting, and rare/saturated hues.
