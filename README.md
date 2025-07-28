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
├── main.py
├── README.md
├── pyproject.toml
├── solid_model_proposal.pdf
├── solid_team_project_checkin.pdf
├── notebooks/
│   └── analysis.ipynb
├── data/
│   └── dataset.py
├── scripts/
│   └── multimodal/
│       └── train_solid.py
├── models/
│   ├── multimodal/
│   │   ├── rgb_encoder.py
│   │   ├── depth_encoder.py
│   │   ├── geometry_encoder.py
│   │   ├── fusion.py
│   │   ├── classifier.py
│   │   └── solid_pipeline.py
├── utils/
│   ├── data_utils.py
│   ├── training_utils.py
│   ├── visualization.py
│   └── model_utils.py
├── outputs/
│   ├── checkpoints/
│   │   └── multimodal/
│   ├── results/
│   │   └── multimodal/
│   └── logs/
├── configs/
│   └── multimodal/train_config.yaml
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
## Usage

Use this space to show useful examples of how a project can be used. For course projects, include which file to execute and the format of any input variables.

Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/catiaspsilva/README-template/issues) for a list of proposed features (and known issues).

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- Authors -->
## Authors
Moses Adewolu - [@MosesTheRedSea](https://twitter.com/MosesTheRedSea) [mosesoluwatobiadewolu@gmail.com](mosesoluwatobiadewolu@gmail.com)

Project Link: [https://github.com/MosesTheRedSea/MNISTique.git](https://github.com/MosesTheRedSea/MNISTique.git)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

You can acknowledge any individual, group, institution or service.
* [Catia Silva](https://faculty.eng.ufl.edu/catia-silva/)
