# Multilingual Fact-Checking Framework

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

An automated framework that utilizes large language models (LLMs) to evaluate and verify factual claims in text, with comprehensive support for multiple languages. This project integrates evidence retrieval, cross-lingual evaluation, and fine-tuning capabilities using LLaMA-Factory.

## Overview

This project develops an end-to-end multilingual fact-checking system that:
- Retrieves and processes evidence from web sources
- Fine-tunes LLMs for fact-checking tasks using LoRA adapters
- Evaluates claims across multiple languages
- Supports cross-lingual evidence evaluation
- Handles multiple fact-checking datasets with different label schemes

## Features

- **Multilingual Support**: Handles 20+ languages including Turkish, Portuguese, Indonesian, Serbian, Italian, German, Romanian, Tamil, Polish, Hindi, Arabic, Spanish, Bengali, Persian, Gujarati, Marathi, Punjabi, Norwegian, Sinhala, Albanian, Russian, Azerbaijani, Dutch, and French
- **Evidence Retrieval**: Multiple evidence retrieval strategies including:
  - Semantic chunking with embedding-based similarity search
  - Sentence-level chunking with token-based grouping
  - Web search snippet integration
  - LLM-generated evidence
- **Model Fine-tuning**: Integration with LLaMA-Factory for efficient LoRA-based fine-tuning
- **Cross-lingual Evaluation**: Evaluate claims in one language using evidence translated to other languages
- **Multiple Datasets**: Support for X-FACT and Ru22Fact datasets with different label schemes
- **Flexible Prompting**: Multiple prompting strategies (vanilla, chain-of-thought, role-based)
- **Comprehensive Evaluation**: Detailed classification reports and prediction outputs

## Supported Languages

The framework supports the following languages:

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| `tr` | Turkish | `ka` | Georgian | `pt` | Portuguese |
| `id` | Indonesian | `sr` | Serbian | `it` | Italian |
| `de` | German | `ro` | Romanian | `ta` | Tamil |
| `pl` | Polish | `hi` | Hindi | `ar` | Arabic |
| `es` | Spanish | `bn` | Bengali | `fa` | Persian |
| `gu` | Gujarati | `mr` | Marathi | `pa` | Punjabi |
| `no` | Norwegian | `si` | Sinhala | `sq` | Albanian |
| `ru` | Russian | `az` | Azerbaijani | `nl` | Dutch |
| `fr` | French | | | | |

## Supported Datasets

### X-FACT Dataset
- **Labels**: `true`, `mostly true`, `partly true/misleading`, `false`, `mostly false`, `complicated/hard-to-categorise`, `other`
- **Format**: TSV files with claims, labels, languages, and evidence snippets
- **Evidence Types**: Search snippets, retrieved web evidence, LLM-generated evidence

### Ru22Fact Dataset
- **Labels**: `supported`, `refuted`, `nei` (not enough information)
- **Format**: CSV files with claims, labels, and evidence
- **Evidence Types**: LLM-generated evidence

## Installation

### Prerequisites

- Python 3.11
- CUDA-capable GPU (recommended for training and inference)
- Git

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/Viratsingh527/Multilingual_factchecking.git
cd Multilingual_factchecking
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
make requirements
# or manually:
pip install -r requirements.txt
```

4. **Install the package**:
```bash
pip install -e .
```

### Additional Dependencies

For evidence retrieval, you may need:
- `langchain` and `langchain-community` for FAISS vector stores
- `langchain-experimental` for semantic chunking
- `deep-translator` for evidence translation
- `nltk` for sentence tokenization

## Project Organization

```
├── LICENSE                    <- MIT License
├── Makefile                   <- Convenience commands (make data, make train, make lint, etc.)
├── README.md                  <- This file
├── pyproject.toml             <- Project configuration and metadata
├── requirements.txt           <- Python dependencies
├── run_all.sh                 <- Batch inference script for multiple languages
│
├── data/                      <- Data directory
│   ├── external/              <- Data from third party sources
│   ├── interim/               <- Intermediate data (e.g., with web data)
│   ├── processed/             <- Final processed datasets for modeling
│   ├── raw/                   <- Original, immutable data dumps
│   └── translated/            <- Translated evidence datasets
│
├── dataset_preprocessing/     <- Data preprocessing scripts
│   ├── Evidence_retreiver.py  <- Evidence retrieval with semantic/sentence chunking
│   ├── formating_dataset.py   <- Dataset formatting for LLaMA-Factory
│   ├── translate.py           <- Evidence translation utilities
│   └── WebCrawling.py         <- Web crawling utilities
│
├── inferencefile/             <- Inference and evaluation scripts
│   ├── test.py                <- Main evaluation script with LoRA adapters
│   ├── prompting.py           <- Prompting strategies and evaluation
│   └── classification_report.py <- Classification report generation
│
├── multilingual_factchecking/ <- Source code package
│   ├── __init__.py
│   ├── config.py              <- Configuration paths and settings
│   ├── dataset.py             <- Dataset processing utilities
│   ├── features.py            <- Feature engineering
│   ├── modeling/
│   │   ├── train.py           <- Model training scripts
│   │   └── predict.py         <- Prediction utilities
│   └── plots.py               <- Visualization utilities
│
├── LLaMA-Factory/             <- LLaMA-Factory submodule for fine-tuning
│
├── models/                    <- Trained models and LoRA adapters
│
├── outputs/                   <- Inference outputs and predictions
│
├── notebooks/                 <- Jupyter notebooks for exploration
│
├── docs/                      <- Documentation (mkdocs)
│
├── references/                <- Data dictionaries and manuals
│
├── reports/                   <- Generated analysis reports
│   └── figures/               <- Generated graphics
│
├── tests/                     <- Unit tests
│
└── utils/                     <- Utility scripts
    ├── check_missing_points.py
    └── checking_duplicate_claims_similarity.py
```

## Usage

### 1. Data Preprocessing

#### Evidence Retrieval

Retrieve evidence from web sources using semantic or sentence-level chunking:

```bash
python dataset_preprocessing/Evidence_retreiver.py \
    --dataset xfact \
    --input train \
    --chunker semantic \
    --semantic-embed-model intfloat/multilingual-e5-large-instruct \
    --k 1
```

Options:
- `--chunker`: Choose `sentence` or `semantic` chunking
- `--semantic-embed-model`: Embedding model for semantic chunking
- `--sent-max-tokens`: Max tokens per sentence chunk (default: 128)
- `--k`: Number of top-k evidence chunks to retrieve per source
- `--no-retriever`: Only convert CSV/TSV to JSONL format

#### Dataset Formatting

Format datasets for LLaMA-Factory training:

```bash
python dataset_preprocessing/formating_dataset.py \
    --dataset xfact \
    --input train \
    --output data/processed/xfact/train_formatted.json
```

#### Evidence Translation

Translate evidence to multiple languages for cross-lingual evaluation:

```bash
python dataset_preprocessing/translate.py \
    --input data/processed/xfact/train_with_evidences.jsonl \
    --output data/translated/xfact/
```

### 2. Model Training

Use LLaMA-Factory for fine-tuning. Configure training parameters in YAML files and run:

```bash
cd LLaMA-Factory
llamafactory-cli train <config_file>.yaml
```

Example training configuration should include:
- Base model (e.g., `mistralai/Mistral-7B-Instruct-v0.3`)
- Dataset path
- LoRA parameters
- Training hyperparameters

### 3. Inference and Evaluation

#### Single Language Evaluation

Evaluate a fine-tuned model on a test set:

```bash
python inferencefile/test.py \
    --base_model mistralai/Mistral-7B-Instruct-v0.3 \
    --adapter LLaMA-Factory/saves/mistral_7b_xfact_sentence_chunking_evidences/lora/sft \
    --data data/processed/xfact/xfact_test_with_sentence_level_chunked_retrieved_evidence.jsonl \
    --out_dir outputs/xfact_predictions \
    --language all \
    --max_length 10000
```

#### Cross-lingual Evaluation

Evaluate claims in one language using evidence translated to other languages:

```bash
python inferencefile/test.py \
    --base_model mistralai/Mistral-7B-Instruct-v0.3 \
    --adapter LLaMA-Factory/saves/mistral_7b_xfact_sentence_chunking_evidences/lora/sft \
    --data data/processed/xfact/xfact_test_with_sentence_level_chunked_retrieved_evidence.jsonl \
    --out_dir outputs/xfact_predictions \
    --language pt \
    --max_length 10000
```

This will evaluate Portuguese claims using evidence translated to all supported languages.

#### Batch Inference

Run inference for multiple languages sequentially:

```bash
bash run_all.sh
```

The script automatically waits for GPU availability and processes languages: `pt`, `id`, `sr`, `it`, `de`, `ro`, `ta`, `pl`, `es`.

### 4. Prompting Strategies

Evaluate with different prompting strategies:

```bash
python inferencefile/prompting.py \
    --base_model mistralai/Mistral-7B-Instruct-v0.3 \
    --data data/processed/xfact/test_formatted.json \
    --out_dir outputs/prompting_experiments \
    --prompting cot  # Options: vanilla, cot, role-based
```

## Key Components

### Evidence Retrieval (`Evidence_retreiver.py`)

- **Semantic Chunking**: Uses embedding models to split documents into semantically coherent chunks
- **Sentence Chunking**: Groups sentences based on token limits
- **FAISS Vector Store**: Efficient similarity search for evidence retrieval
- **Multi-source Support**: Handles multiple web sources per claim

### Evaluation Script (`test.py`)

- **Canonical Label Normalization**: Handles label variations and spelling differences
- **Cross-lingual Mode**: Evaluates claims with translated evidence
- **Comprehensive Metrics**: Generates classification reports with precision, recall, F1
- **Output Management**: Saves predictions and metrics in organized directory structure

### Dataset Formatting (`formating_dataset.py`)

- **Multi-dataset Support**: Handles X-FACT and Ru22Fact formats
- **Instruction Generation**: Creates dataset-specific instructions
- **Label Standardization**: Normalizes labels across datasets

## Evidence Retrieval Methods

1. **Semantic Chunking**: 
   - Uses embedding models (e.g., `intfloat/multilingual-e5-large-instruct`)
   - Splits documents at semantic boundaries
   - Configurable threshold types (percentile, standard deviation)

2. **Sentence Chunking**:
   - Token-based sentence grouping
   - Configurable max tokens per chunk
   - Uses NLTK for sentence tokenization

3. **Search Snippets**: 
   - Direct use of search engine snippets (X-FACT dataset)

4. **LLM-generated**: 
   - Evidence generated by language models (Ru22Fact dataset)

## Makefile Commands

```bash
make requirements    # Install Python dependencies
make data           # Process datasets
make train          # Train models
make lint           # Lint code with ruff
make format         # Format code with ruff
make test           # Run tests
make clean          # Remove compiled Python files
make help           # Show all available commands
```

## Output Structure

Inference outputs are organized as:
```
outputs/
└── {dataset}/
    └── {testset}/
        └── {model_shortname}/
            └── {technique}/
                ├── predictions_{seed}.json
                ├── metrics_{seed}.txt
                └── {claim_lang}/          # For cross-lingual evaluation
                    ├── predictions_{claim_lang}_{evidence_lang}.jsonl
                    └── metrics_{claim_lang}_{evidence_lang}.txt
```

## Requirements

- Python 3.11
- PyTorch (with CUDA support recommended)
- Transformers
- PEFT (for LoRA adapters)
- LangChain (for evidence retrieval)
- scikit-learn (for evaluation metrics)
- See `requirements.txt` for full list

## Citation

If you use this project in your research, please cite:

```bibtex
@software{multilingual_factchecking,
  title = {Multilingual Fact-Checking Framework},
  author = {Babu Kumar},
  year = {2025},
  url = {https://github.com/Viratsingh527/Multilingual_factchecking.git}
}
```

## Acknowledgement

This project makes use of [LLaMA-Factory](https://arxiv.org/abs/2403.13372).  
If you use this repository, please also cite their work:

> Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma.  
> *LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models*.  
> Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), 2024.  
> [https://arxiv.org/abs/2403.13372](https://arxiv.org/abs/2403.13372)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on the repository or contact the maintainer.
