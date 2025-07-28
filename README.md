# SOLID 

[LogoFarmer Studio](https://dribbble.com/shots/25590129-S-Eye-Logo-Design)

![solid-vision-logo](https://github.com/MosesTheRedSea/SOLID/blob/main/solid-vision-design.jpg)

<br />
<p align="center">
  <h3 align="center">Spatial Object Learning Integrated Dimensions</h3>
  <p align="center">
    <a href="https://github.com/catiaspsilva/README-template/blob/main/images/docs.txt"><strong>Documentaton</strong></a>
    <a href="https://github.com/catiaspsilva/README-template/issues">Report Bug</a>
    <a href="https://github.com/catiaspsilva/README-template/issues">Add New Feature</a><br><br>
    SOLID is a modular deep learning framework for multimodal 3D object classification using RGB, depth, and point cloud geometry. It leverages pretrained transformers and point-based networks to integrate spatial cues across modalities. The goal is to improve object recognition in complex scenes using complementary representations.
  </p>
</p>


<!-- GETTING STARTED -->
## Getting Started

In this section you should provide instructions on how to use this repository to recreate your project locally.

```
SOLID/
├── configs
│   ├── baseline
│   └── multimodal
├── data
│   ├── dataset.py
│   ├── dataset2.py
│   └── depthdata.py
├── models
│   ├── baseline
│   └── multimodal
│       ├── classifier.py
│       ├── depth_encoder.py
│       ├── fusion.py
│       ├── geometry_encoder.py
│       ├── rgb_encoder.py
│       └── solid_pipeline.py
├── scripts
│   ├── baseline
│   │   ├── train_cloud_baseline.py
│   │   ├── train_depth_baseline.py
│   │   └── train_rgb_baseline.py
│   ├── constants.py
│   └── multimodal
│       ├── evaluate.py
│       └── train_solid.py
├── utils
│   └── visualization.py
```

### Installation

Clone the repo
   ```bash
   git clone https://github.com/MosesTheRedSea/SOLID.git
   ```
Setup (and activate) your environment
  ```bash
 uv sync or uv run pyproject.toml
  ```
Activate Python Virtual Environment

```bash
source .venv/bin/activate
```

## Configuration Setup

### constants.py
- Modify paths within this file to point to your SUNRGBD dataset and project directory

### train_config.yaml
- This file controls training parameters and logging paths for each modality. Make sure the paths are valid on your machine
- 
### Module Import Paths
- Some training or evaluation scripts use internal modules. Add these paths at the top of your script to allow local imports
  ```bash
  import sys
  sys.path.extend([
      "/path/to/SOLID/models/multimodal/",
      "/path/to/SOLID/data/"
  ])
  ```
<!-- USAGE EXAMPLES -->

## Data Set 
The SUN RGB-D dataset is a large-scale benchmark dataset designed for scene understanding from RGB-D images, combining both color (RGB) and depth modalities. It includes over 10,000 indoor scene images captured using four different RGB-D sensors: Kinect v1, Kinect v2, Intel RealSense, and Structure Sensor

- Download the SUNRGBD RGB-D dataset
  
```bash
cd data

mkdir -p sunrgbd && cd sunrgbd

curl -O http://rgbd.cs.princeton.edu/data/SUNRGBD.zip

unzip SUNRGBD.zip -d SUNRGBD
```

- Download the SUNRGBDtoolbox

```bash
curl -O http://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip

unzip SUNRGBDtoolbox.zip -d SUNRGBDtoolbox

curl -O http://rgbd.cs.princeton.edu/data/SUNRGBDMeta2DBB_v2.mat

curl -O http://rgbd.cs.princeton.edu/data/SUNRGBDMeta3DBB_v2.mat

mv SUNRGBDMeta2DBB_v2.mat SUNRGBD/

mv SUNRGBDMeta3DBB_v2.mat SUNRGBD/
```

## Usage

Train RGB Baseline (ResNet34)

```bash
python scripts/baseline/train_rgb_baseline.py
```

Train Depth Baseline  (ResNet34)

```bash
python scripts/baseline/train_depth_baseline.py
```

Train Point Cloud Baseline (DGCNN)

```bash
python scripts/baseline/train_cloud_baseline.py
```

### Training the SOLID Multimodal Fusion Model
- To train the full fusion pipeline integrating RGB, depth, and geometry:

```bash
python scripts/multimodal/train_solid.py
```

### Visualizing Results

- Open the visualization script

```bash
  utils/visualization.py
```

```bash
if __name__ == "__main__":
    # To visualize baseline model performance
    graph_baseline()

    # To visualize SOLID multimodal model performance
    graph_multimodal()
```

Then run

```bash
python utils/visualization.py
```

<!-- ROADMAP -->
## SOLID Architecture

<p align="center">
  <img src="https://github.com/MosesTheRedSea/SOLID/blob/main/solid-pipeline.jpg" alt="solid-pipeline-roadmap" />
</p><br>

  - RGB Images are passed through a CLIP Vision Transformer to extract high-level semantic features.
  - Depth Images are processed using a Swin Transformer tailored for single-channel input, capturing spatial cues from scene geometry.
  - 3D Point Clouds are encoded using DGCNN (Dynamic Graph CNN) to represent local geometric structures.

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

