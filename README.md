# SOLID 

[LogoFarmer Studio](https://dribbble.com/shots/25590129-S-Eye-Logo-Design)

![solid-vision-logo](https://github.com/MosesTheRedSea/SOLID/blob/main/solid-vision-design.jpg)

## Introduction

**Spatial Object Learning & Integrated Dimensions Vision (SOLID Vision)** is a multi-modal object classification framework that fuses RGB images, depth maps, and 3D geometry to achieve robust recognition in complex real-world environments.

Traditional models relying solely on RGB data lack spatial awareness, limiting their applicability in robotics and autonomy. They face key limitations such as:

- **No 3D Spatial Reasoning:** RGB-only models lack depth and geometric context.
- **Environmental Sensitivity:** Performance degrades under occlusions, poor lighting, or partial views.
- **Incomplete Representations:** Single modalities fail to capture real-world object complexity.

SOLID Vision bridges these gaps with transformer-based cross-modal architectures, enabling accurate spatial understanding, robustness under challenging conditions, and high performance across robotics and autonomous perception use cases.

## Key Highlights

- Multi-modal fusion of RGB, depth, and 3D geometric data
- Transformer-based architecture with cross-modal attention
- Built with PyTorch, PointNet++, ResNet, and Vision Transformers (ViT)
- Optimized for real-time robotics, AR/VR, and autonomous perception

## System Overview

### Data Modalities

- **RGB Images:** Capture color, texture, and fine-grained features (ResNet/ViT backbones).
- **Depth Maps:** Provide spatial cues via normalized depth channels processed with CNN-based encoders.
- **3D Geometry:** Point clouds or voxel grids processed using PointNet++, volumetric CNNs, or graph-based models.

### Fusion Strategies

- **Early Fusion:** Combine raw inputs before feature extraction.
- **Late Fusion:** Merge modality features after independent encoding.
- **Cross-Attention Fusion:** Transformer-based modules to learn shared multi-modal representations.

## Features

- Multi-modal integration: RGB, depth, and geometry fusion
- Flexible architectures: Early, late, and attention-based fusion options
- Modular design: Easy extension to new modalities or tasks
- Benchmark compatibility: Supports ModelNet40, SUN RGB-D, and more
- Visualization tools: t-SNE plots, attention maps, and prediction visualizations
- Robotics deployment: Integrated with Unitree Robotics for real-world testing
- PyTorch implementation: Fully open-source and customizable

For questions or inquiries, feel free to open an issue or contact:
Email: [mosesoluwatobiadewolu@gmail.com](mailto:mosesoluwatobiadewolu@gmail.com)  
