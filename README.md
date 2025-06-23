# Financial Tweet Sentiment Analysis

This project implements and compares multiple approaches for sentiment classification of financial tweets, from traditional machine learning to state-of-the-art transformer models.

## Project Overview

We classify financial tweets into three sentiment categories:
- **Bearish** (Negative outlook)
- **Neutral** (No clear direction)
- **Bullish** (Positive outlook)

## Model Performance Comparison

| Method                                      | Precision | Recall  | F1-Score | Accuracy |
|--------------------------------------------|-----------|---------|----------|----------|
| Logistic Regression                        | 74.26%    | 73.79%  | 73.73%   | 73.79%   |
| Linear SVM                                 | 74.04%    | 73.59%  | 73.54%   | 73.59%   |
| FinBERT (No Fine-Tuning)                   | 57.96%    | 48.18%  | 45.94%   | 48.18%   |
| FinBERT + LoRA                             | 75.00%    | 74.00%  | 74.00%   | 74.25%   |
| FinBERT + LoRA (minimal preprocessing)     | 77.73%    | 77.41%  | 77.43%   | 77.41%   |
| FinBERT + LoRA (+ !? counts)               | 77.74%    | 77.39%  | 77.42%   | 77.39%   |
| FinBERT + QLoRA                            | 74.83%    | 74.55%  | 74.56%   | 74.55%   |
| FinBERT + QLoRA (minimal preprocessing)    | 77.21%    | 76.83%  | 76.85%   | 76.83%   |
| FinBERT + QLoRA (+ !? counts)              | 77.60%    | 77.35%  | 77.37%   | 77.35%   |

## Quick Start

### Prerequisites
```bash
python -m venv sentimentAnalysis-venv
source sentimentAnalysis-venv/bin/activate  # On Windows: sentimentAnalysis-venv\Scripts\activate
pip install -r requirements.txt
```

### Run Baseline Models
```bash
# Traditional ML
jupyter notebook notebooks/baseline_src/logistic_reg_baseline.ipynb
jupyter notebook notebooks/baseline_src/svm_baseline.ipynb

# Transformer-based
jupyter notebook notebooks/baseline_src/FinBERT+LoRA_baseline.ipynb
```

## Project Structure
```
financial_sentimentAnalysis/
├── notebooks/
│   └── baseline_src/         # Baseline experiments
├── src/                      # Helper functions
│   ├── preprocessing.py      # Text preprocessing
│   └── evaluation_*.py       # Evaluation metrics
├── dataset/                  # Data files (not included)
├── models/                   # Trained models (see below)
└── evaluation/               # Results and metrics
```

## Trained Models

Pre-trained model weights are available on Google Drive:
- [Download Models](https://huggingface.co/Financial-Sentiment-Analysis/model-weights/tree/main/models/v1-1/baseline)

## Key Findings

1. **Minimal preprocessing works better** for transformer models (77.73% vs 75.00%)
2. **FinBERT requires fine-tuning** - without it, performance drops to 48.18%
3. **QLoRA maintains performance** while reducing memory usage by ~75%
4. **Traditional ML is competitive** with proper feature engineering

## Technical Details

### Model Configurations
- **LoRA**: rank=16, alpha=32, dropout=0.1
- **QLoRA**: 4-bit quantization with bfloat16
- **Training**: 5 epochs, learning rate=2e-5

## Coming Soon

- Semi-supervised learning experiments (230k+ tweets)
- Web application for real-time sentiment analysis
- Integration with live market data

## Authors

- Tecson Gacrama
- Timothy Tong

## License

This project is part of academic research. Please cite if used.
