# Villiam: Vision-Integrated Large Language Adaptive Model

Welcome to the official GitHub repository of Villiam, a groundbreaking project that aims to revolutionize the field of Large Language Models (LLMs) with a focus on enhancing language technology for low-resource languages, particularly Thai. Villiam integrates advanced data augmentation techniques with robust quality control measures and introduces vision capabilities to LLMs, thereby democratizing access to state-of-the-art language technologies.

---

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model and Datasets](#model-and-datasets)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Introduction
[↑ Back to top](#table-of-contents)

Villiam is designed to address the significant disparity in language technology between high-resource and low-resource languages. By providing a scalable, efficient, and cost-effective solution, Villiam opens new possibilities in AI for languages that have traditionally been underrepresented in this field.

## Features
[↑ Back to top](#table-of-contents)

- **Cross-Lingual Data Augmentation**: Expands dataset size significantly using advanced translation techniques and synthetic data generation.
- **Quality Control Measures**: Utilizes clustering and semantic embeddings to ensure data diversity and reduce noise.
- **Vision-Integrated Capabilities**: Enables the model to perform tasks that involve understanding and conversing about image content.
- **Scalable and Adaptable Architecture**: Designed for ease of adaptation to other low-resource languages.

## Usage
[↑ Back to top](#table-of-contents)

```bash
# Clone the repository
git clone https://github.com/parinzee/villiam-scb10x.git

# Install the required packages
mamba env create -f environment.yml

# Activate the environment
mamba activate villiam

# Run the training script
cd villiam-model

# Download our data
...

# Run the pretraining script
./scripts/pretrain.sh

# Run the finetuning script
./scripts/finetune.sh
```

## Model and Datasets
[↑ Back to top](#table-of-contents)

- **Villiam Model (Image + Text —  Conversational)**: [Download Link](#) - Here, you'll find the pre-trained Villiam model ready for use.
- **Villiam Model (Text Only —  Trained on QA)**: [Download Link](https://huggingface.co/parinzee/villiam-qa-100-beta-7b) - Here, you'll find the pre-trained Villiam text-only model trained on QA datasets.
- **Datasets**:
    - [Villiam QA Dataset](https://huggingface.co/datasets/parinzee/claq-qa-thai-dataset)
    - [Villiam Image Instruction-Following Dataset Finetune](https://huggingface.co/datasets/senmeetechin/LLaVA-TH)

## Contributing
[↑ Back to top](#table-of-contents)

We welcome contributions from the community. Please read our [contributing guidelines](CONTRIBUTING.md) before submitting your contributions.

## License
[↑ Back to top](#table-of-contents)

Villiam is released under [MIT License](LICENSE). Please review the license terms before using or contributing to the project.

## Citation
[↑ Back to top](#table-of-contents)

If you use Villiam in your research, please cite it using this format.

```bibtex
@article{villiam2023,
  title={Villiam: Vision-Integrated Large Language Adaptive Model},
  journal={GitHub Repository},
  year={2023},
}
```