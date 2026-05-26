# Legal Contract Risk Classifier

Fine-tuned BERT model for binary risk classification of legal contract clauses.

## Overview
This project fine-tunes a pre-trained BERT model to classify legal contract clauses as high-risk or low-risk, building an end-to-end NLP pipeline from raw text to prediction.

## Tech Stack
- Python, HuggingFace Transformers, PyTorch
- Datasets: CUAD (Contract Understanding Atticus Dataset)

## Pipeline
1. Data loading and preprocessing
2. Tokenization (BERT tokenizer)
3. Fine-tuning BERT for sequence classification
4. Evaluation on held-out test set

## Results
- Accuracy: TBD (updating after training completes)

## How to Run
```bash
pip install transformers torch datasets
jupyter notebook "BERT MVP.ipynb"
```
