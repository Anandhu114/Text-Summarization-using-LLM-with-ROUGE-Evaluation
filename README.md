# Text Summarization Using Large Language Models (LLMs)

## 📌 Project Overview

Text Summarization Using LLM is a Natural Language Processing (NLP) project that generates concise summaries from longer text conversations using a pretrained Transformer model.

This project uses the **T5-small (Text-to-Text Transfer Transformer)** model from Hugging Face and the SAMSum dialogue summarization dataset. The model is trained to transform conversations into shorter summaries while preserving their main information.

The project demonstrates an end-to-end text summarization workflow, including data preprocessing, tokenization, model training, model saving, and summary generation.

## 🎯 Project Objectives

- Understand the application of Transformer models in NLP.
- Load and preprocess dialogue data.
- Handle missing values and duplicate records.
- Prepare training, validation, and testing datasets.
- Tokenize dialogues and their corresponding summaries.
- Fine-tune the pretrained T5-small model for summarization.
- Generate summaries from sample conversations.
- Save the trained model and tokenizer for future use.

## 🛠️ Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Hugging Face Transformers | Pretrained Transformer models and training |
| Hugging Face Datasets | Dataset preparation and processing |
| PyTorch | Deep learning framework |
| NLTK | Natural Language Processing utilities |
| Evaluate | Model evaluation framework |
| Scikit-learn | Dataset splitting |
| SciPy | Statistical functions |
| Statsmodels | Statistical analysis |
| Plotly | Interactive data visualization |
| Jupyter Notebook | Interactive development environment |

## 📂 Dataset

This project uses the **SAMSum dataset**, which contains conversational dialogues and human-written summaries.

Dataset source: [SAMSum Dataset on Hugging Face](https://huggingface.co/datasets/Samsung/samsum)

The notebook reads three CSV files:

- `samsum-train.csv`
- `samsum-validation.csv`
- `samsum-test.csv`

### Dataset Columns

| Column | Description |
|---|---|
| `dialogue` | The original conversation that needs summarization |
| `summary` | The reference summary of the conversation |
| `id` | An identifier for a dataset record, if present |

The notebook combines the three CSV files into one DataFrame, removes missing values and duplicates, and then creates new training, validation, and testing splits.

## 🔄 Project Workflow

### 1. Importing Libraries

The project imports the required libraries for data manipulation, NLP, Transformer models, deep learning, and evaluation.

### 2. Loading the Dataset

The three SAMSum CSV files are loaded using
