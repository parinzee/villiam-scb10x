# Villiam: Vision-Integrated Large Language Adaptive Model

<div align="center">
  <img src="https://raw.githubusercontent.com/parinzee/villiam-scb10x/main/logo.png" alt="Villiam Logo">
</div>

Welcome to the official GitHub repository of Villiam, a groundbreaking project that aims to revolutionize the field of Large Language Models (LLMs) with a focus on enhancing language technology for low-resource languages, particularly Thai. Villiam integrates advanced data augmentation techniques with robust quality control measures and introduces vision capabilities to LLMs, thereby democratizing access to state-of-the-art language technologies.

---

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model and Datasets](#model-and-datasets)
- [License](#license)
- [Evaluation](#evaluation)

---

## Introduction
[↑ Back to top](#table-of-contents)

Villiam is designed to address the significant disparity in language technology between high-resource and low-resource languages. By providing a scalable, efficient, and cost-effective solution, Villiam opens new possibilities in AI for languages that have traditionally been underrepresented in this field.

## Features
[↑ Back to top](#table-of-contents)

- **Cross-Lingual Data Augmentation**: Expands dataset size significantly using advanced translation techniques and synthetic data generation. This builds on our previous work on [CLAQ](https://github.com/parinzee/cross-lingual-data-augmentation-for-thai-qa/tree/807e40bf80b38b5759d4e29dbd1f6ae004b6d250)
- **Quality Control Measures**: Utilizes clustering and semantic embeddings to ensure data diversity and reduce noise.
- **Vision-Integrated Capabilities**: Enables the model to perform tasks that involve understanding and conversing about image content. This is built upon [OpenThaiGPT](https://openthaigpt.aieat.or.th/) and [LLaVa](https://github.com/haotian-liu/LLaVA)
- **Scalable and Adaptable Architecture**: Designed for ease of adaptation to other low-resource languages. Although our model here is in Thai, we can simply use [NLLB](https://ai.meta.com/research/no-language-left-behind/) or Google Translate to translate the data to other languages.

## Usage
[↑ Back to top](#table-of-contents)

```bash
# Clone the repository
git clone https://github.com/parinzee/villiam-scb10x.git --recurse-submodules --remote-submodules

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
 
## Evaluation
[↑ Back to top](#table-of-contents)
- **Villiam Model using Evaluate with Exactly Match**
Our GPT-assisted evaluation pipeline for multimodal modeling is provided for a comprehensive understanding of the capabilities of vision-language models.  Please see  paper for more details.

1. Generate LLaVA responses

```Shell
python model_vqa.py \
    --model-path ./checkpoints/LLaVA-13B-v0 \
    --question-file \
    playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder \
    /path/to/coco2014_val \
    --answers-file \
    /path/to/answer-file-our.jsonl
```

2. Evaluate the generated responses.  In our case, the response was generated with Thai language by text-only GPT-4 (0314), with the context captions/boxes provided.

```Shell
OPENAI_API_KEY="sk-***********************************" python llava/eval/eval_gpt_review_visual.py \
    --question playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --context llava/eval/table/caps_boxes_coco2014_val_80.jsonl \
    --answer-list \
    /path/to/answer-file-ref.jsonl \
    /path/to/answer-file-our.jsonl \
    --rule llava/eval/table/rule.json \
    --output /path/to/review.json
```

3. Summarize the evaluation results

```Shell
python summarize_gpt_review.py
```


## License
[↑ Back to top](#table-of-contents)

Villiam is released under [MIT License](LICENSE). Please review the license terms before using or contributing to the project.
