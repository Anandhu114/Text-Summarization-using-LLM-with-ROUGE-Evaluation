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

The three SAMSum CSV files are loaded using Pandas and combined into a single DataFrame.

```python
df1 = pd.read_csv("samsum-test.csv")
df2 = pd.read_csv("samsum-validation.csv")
df3 = pd.read_csv("samsum-train.csv")

df = pd.concat([df1, df2, df3], ignore_index=True)
```

### 3. Data Cleaning

The notebook checks for missing values and removes rows containing missing data and duplicate records.

```python
df = df.dropna()
df = df.drop_duplicates()
```

This helps prepare the data before tokenization and model training.

### 4. Splitting the Dataset

The notebook creates training, validation, and testing subsets using `train_test_split`.

```python
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42
)
```

The resulting proportions are approximately:

- Training: 80%
- Validation: 10%
- Testing: 10%

**Note:** Because the notebook combines the original SAMSum splits before creating new splits, these proportions apply to the combined and cleaned dataset, not necessarily to the original dataset partitions.

### 5. Tokenization Using T5

The project loads the pretrained T5 tokenizer.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-small")
```

The tokenizer converts dialogue text into token IDs that the Transformer model can process.

The notebook adds the `summarize:` task prefix to each dialogue because T5 uses a text-to-text approach.

The input dialogue is truncated or padded to a maximum length of 512 tokens, while the target summary is truncated or padded to 128 tokens.

### 6. Loading the Pretrained T5 Model

The notebook loads T5-small as a sequence-to-sequence language model.

```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
```

A sequence-to-sequence model takes an input sequence, such as a conversation, and generates an output sequence, such as a summary.

### 7. Model Training

The notebook configures training for five epochs and uses the Hugging Face Trainer API.

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    num_train_epochs=5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer
)

trainer.train()
```

During training, the model learns to generate summaries that resemble the reference summaries in the dataset.

**Implementation note:** In the current notebook, the preprocessing function is mapped to the training dataset only. The validation dataset is passed to the Trainer without the same tokenization step. This should be fixed before treating the training and validation workflow as complete.

### 8. Saving the Model

After training, the notebook saves the model and tokenizer locally.

```python
trainer.save_model("./summarization_model")
tokenizer.save_pretrained("./summarization_model")
```

The saved files can be used to reload the trained model later without starting the training process again.

### 9. Generating a Summary

The notebook selects a sample dialogue from the testing DataFrame and passes it to the model.

```python
inputs = tokenizer(
    "summarize: " + sample_text,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

summary_ids = model.generate(
    inputs["input_ids"],
    max_length=100,
    num_beams=4
)

summary = tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)
```

The generated token IDs are decoded into readable text. The notebook then prints the original dialogue and its generated summary.

The `num_beams=4` parameter enables beam search with four beams during generation.

## ✨ Key Features

- Dialogue summarization using a pretrained Transformer model.
- SAMSum conversational dataset processing.
- Missing-value and duplicate removal.
- Text tokenization and sequence-length management.
- Fine-tuning using the Hugging Face Trainer.
- Summary generation from a sample dialogue.
- Local saving of the trained model and tokenizer.
- Support for further evaluation and experimentation.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or another Python version compatible with the selected library versions.
- Jupyter Notebook or JupyterLab.
- An internet connection for downloading the pretrained model and tokenizer.
- Sufficient memory and processing resources for model training.

### 1. Clone the Repository

Replace the example URL with the actual URL of your GitHub repository.

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

### 2. Navigate to the Project Directory

```bash
cd YOUR-REPOSITORY
```

### 3. Install the Required Libraries

```bash
pip install pandas numpy torch transformers datasets scikit-learn evaluate nltk scipy statsmodels plotly
```

### 4. Add the Dataset Files

Place these files in the project directory:

```text
samsum-train.csv
samsum-validation.csv
samsum-test.csv
```

Make sure the filenames and paths match the notebook.

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `Text Summarization using LLM.ipynb` and execute the cells in order.

**Important:** The notebook's training arguments and dataset preprocessing may need adjustments depending on the installed Transformers version and available hardware.

## 📁 Suggested Repository Structure

```text
text-summarization-llm/
│
├── Text Summarization using LLM.ipynb
├── samsum-train.csv
├── samsum-validation.csv
├── samsum-test.csv
├── summarization_model/
│
├── images/
│   ├── dialogue-example.png
│   └── summary-output.png
│
├── requirements.txt
└── README.md
```

This is a suggested structure, not a verified listing of the files currently in your repository. Large datasets and trained model files may be better linked externally rather than committed directly to GitHub.

## 📈 Evaluation

The notebook imports the Hugging Face `evaluate` library and loads the ROUGE evaluation metric.

ROUGE is commonly used to compare generated summaries with reference summaries. However, the current notebook does not calculate and report final ROUGE scores.

Future evaluation can include:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- Comparison of generated summaries with reference summaries
- Qualitative review of summary relevance, readability, and information retention

No numerical performance or accuracy result is claimed in this README because the notebook does not provide a verified final evaluation result.

## 💡 Applications

Text summarization can be useful for:

- Summarizing chat conversations.
- Creating short meeting notes.
- Condensing customer support conversations.
- Summarizing lengthy text documents.
- Helping users quickly understand long discussions.
- Supporting information extraction and text-processing workflows.

These are potential applications of the technique; the notebook itself demonstrates dialogue summarization rather than a deployed application.

## 🔮 Future Improvements

- Apply the preprocessing function to training, validation, and testing datasets consistently.
- Evaluate the model using ROUGE scores.
- Compare T5-small with other summarization models.
- Display the original dialogue alongside the generated and reference summaries.
- Add a simple web interface using Streamlit or Flask.
- Experiment with different generation settings.
- Improve training efficiency and reproducibility.
- Add qualitative examples and screenshots to the README.
- Keep dataset partitions separate where appropriate to avoid evaluation leakage.

## 👨‍💻 Author

**Anandhu P S**

- GitHub: https://github.com/Anandhu114

## 📚 Key Concepts Demonstrated

- Natural Language Processing (NLP)
- Large Language Models (LLMs)
- Transformer Architecture
- T5 Sequence-to-Sequence Model
- Text Tokenization
- Fine-Tuning Pretrained Models
- Dialogue Summarization
- Hugging Face Transformers
- PyTorch
- ROUGE Evaluation

## 📄 License

No license is specified in the notebook. Add an appropriate license file if you intend to allow others to reuse, modify, or distribute your code.

---

If you find this project useful, feel free to star the repository!
