# Credit Score Classification with Deep Learning

Comparing MLP, LSTM, GRU, and Random Forest models to automate credit score prediction — a core use case in Banking and Financial Services AI.

## Overview

Traditional credit scoring is manual, slow, and error-prone. This project automates classification of customers into **Poor**, **Standard**, or **Good** credit score categories using machine learning and deep learning on 8 months of real customer financial behavior data.

**Dataset:** [Credit Score Classification — Kaggle](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)

- 50,000 time-stamped records across 12,500 unique customers
- 27 features: demographics, financial attributes, behavioral data

-----

## Models Compared

|Model        |Test Accuracy|Macro F1-Score|
|-------------|-------------|--------------|
|Random Forest|**72%**      |**0.72**      |
|LSTM         |69%          |0.67          |
|MLP          |62%          |0.45          |
|GRU          |52%          |0.51          |

**Key finding:** Random Forest outperformed all neural networks, demonstrating that tree-based ensemble methods can be more robust than deep learning on structured, tabular financial data — especially under class imbalance.

-----

## Technical Highlights

- **Data pipeline:** Cleaned 27 features including median imputation, Z-score outlier capping (5th–95th percentile), ordinal encoding, and one-hot encoding of occupations
- **Feature engineering:** Debt-to-Income ratio, spending level extraction from payment behavior, month ordinal encoding
- **Class imbalance handling:** Class weights, stratified train/val/test splits (70/15/15)
- **Sequence modeling:** LSTM and GRU used zero-padded customer time-series (8 months per customer)
- **Hyperparameter tuning:** RandomSearch on LSTM and GRU architectures
- **Evaluation:** Classification reports, confusion matrices, and training/validation curves

-----

## Model Architectures

### MLP

- Customer-level aggregated features (mean across months)
- 3 layers: Dense → Dropout → Dense → Dropout → Output
- Sparse categorical cross-entropy, 20 epochs

### LSTM

- Padded time-series sequences per customer
- Input masking → 2 LSTM layers → Dense (ReLU) → Softmax output
- Early stopping (patience=10), 50 epochs, Adam optimizer

### GRU

- Padded time-series sequences per customer
- Input masking → 3 GRU layers → Dense → Softmax output
- Adam optimizer, RandomSearch tuning

### Random Forest

- Mean-aggregated features per customer
- 200 trees, max depth=3, min samples per leaf=2
- Best precision on **Poor** class (0.80)

-----

## Results & Key Insights

- **Random Forest** was the strongest model overall — interpretable, robust to imbalance
- **LSTM** best captured temporal patterns, improved recall on minority classes vs MLP
- **Class imbalance** (especially the “Good” class) was the main challenge across all models
- MLP completely failed to predict the “Good” class (0% F1), motivating the use of sequential models

-----

## Tech Stack

`Python` · `TensorFlow/Keras` · `scikit-learn` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Google Colab`

-----

## Project Structure

```
credit-score-classification/
│
├── data/                  # Dataset (download from Kaggle link above)
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory data analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_mlp.ipynb
│   ├── 04_lstm.ipynb
│   ├── 05_gru.ipynb
│   └── 06_random_forest.ipynb
├── results/               # Confusion matrices, training curves
├── requirements.txt
└── README.md
```

-----

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/credit-score-classification.git
cd credit-score-classification
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) and place it in the `data/` folder.

-----

## Authors

Omnia Dafalla · Haley Burger · Zayne Maughan  
Utah State University — M.S. Data Analytics, 2025