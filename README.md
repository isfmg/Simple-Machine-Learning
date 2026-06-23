# Text Score Regression — Kaggle Competition

Predicting a continuous score from raw text using TF-IDF features, with two model approaches compared: Ridge Regression and a Fully Connected Neural Network.

## Overview

Task: regress a numeric `score` from a `text` column (train/val/test CSVs). Built for a Kaggle competition as part of a Practical Machine Learning course project. First time working with text-input models — prior background was CNNs for image tasks.

Both models share the same feature pipeline (TF-IDF) but use different vectorization sizes and architectures.

## Feature Engineering

**TF-IDF (Term Frequency – Inverse Document Frequency)**

```
TF-IDF(t, d) = TF(t, d) * IDF(t)

TF(t, d) = (occurrences of term t in document d) / (total terms in d)
IDF(t)   = log(total documents / documents containing t)
```

Each text sample is a "document"; each word/n-gram is a "term". Output is a sparse vector where each dimension is a unique unigram/bigram weighted by its TF-IDF score.

## Models

### 1. Ridge Regression (`modelridge.py`)

- Vectorizer: `TfidfVectorizer(max_features=10000, ngram_range=(1, 2))` — top 10k unigrams + bigrams
- Model: `sklearn.linear_model.Ridge`, L2-regularized linear regression

```
Error = Σ(prediction - value)² + α·Σ(coefficient)²
```

Alpha sweep results (validation set):

| Alpha | MAE | MSE | Spearman | Kendall |
|---|---|---|---|---|
| 1 | 0.51607 | 0.41305 | 0.59204 | 0.41550 |
| 1.5 | 0.52207 | 0.42582 | 0.60505 | 0.42272 |
| 2 | 0.52241 | 0.42292 | 0.60968 | 0.42604 |
| 2.5 | 0.52397 | 0.42239 | 0.61195 | 0.42816 |
| 3 | 0.52607 | 0.42304 | 0.61246 | 0.42836 |
| 3.5 | 0.52859 | 0.42433 | 0.61358 | 0.42950 |
| 4 | 0.53103 | 0.42597 | 0.61342 | 0.42945 |
| 5 | 0.53569 | 0.42981 | 0.61253 | 0.42845 |
| 50 | 0.62340 | 0.52301 | 0.58418 | 0.40713 |

Diminishing returns past alpha = 3.5–4; underfitting beyond alpha = 5.

### 2. Fully Connected Neural Network (`nn.py`)

- Vectorizer: `TfidfVectorizer(max_features=5000)` — top 5k unigrams
- Targets standardized with `StandardScaler` before training
- Architecture: `Input → Dense(128, ReLU) → Dropout → Dense(64, ReLU) → Dropout → Dense(1, linear)`
- Optimizer: Adam, loss: MSE, `EarlyStopping` on `val_loss` (patience 5, restore best weights)

Hyperparameter search (validation set):

**First layer size**

| Neurons | MAE | MSE | Spearman | Kendall |
|---|---|---|---|---|
| 128 | 0.50174 | 0.40399 | 0.62591 | 0.44550 |
| 256 | 0.50318 | 0.40727 | 0.62394 | 0.44486 |
| 64 | 0.51072 | 0.41899 | 0.61101 | 0.43605 |

**Second layer size**

| Neurons | MAE | MSE | Spearman | Kendall |
|---|---|---|---|---|
| 64 | 0.50174 | 0.40399 | 0.62591 | 0.44550 |
| 128 | 0.51114 | 0.41035 | 0.62339 | 0.44002 |
| 32 | 0.50440 | 0.41196 | 0.62007 | 0.44089 |

**Dropout rate**

| Rate | MAE | MSE | Spearman | Kendall |
|---|---|---|---|---|
| 0.3 | 0.50174 | 0.40399 | 0.62591 | 0.44550 |
| 0.6 | 0.50681 | 0.40772 | 0.62467 | 0.44359 |
| 0.1 | 0.50421 | 0.41319 | 0.61611 | 0.43927 |

**Adam learning rate**

| Rate | MAE | MSE | Spearman | Kendall |
|---|---|---|---|---|
| 0.0005 | 0.50174 | 0.40399 | 0.62591 | 0.44550 |
| 0.0010 | 0.49910 | 0.41103 | 0.62043 | 0.44486 |
| 0.00005 | 0.50700 | 0.41371 | 0.61680 | 0.43793 |

Best configuration outperforms Ridge across MAE, MSE, Spearman, and Kendall correlation.

## Results Summary

| Model | MAE | MSE | Spearman | Kendall |
|---|---|---|---|---|
| Ridge Regression (best, α≈1) | 0.51607 | 0.41305 | 0.59204 | 0.41550 |
| Neural Network (best) | 0.50174 | 0.40399 | 0.62591 | 0.44550 |

The neural network outperforms the linear Ridge baseline on every metric, confirming that the added non-linear capacity captures patterns the TF-IDF + linear model misses.

## Project Structure

```
modelridge.py   # TF-IDF + Ridge Regression baseline
nn.py           # TF-IDF + Fully Connected Neural Network
```

## Requirements

```
pandas
scikit-learn
scipy
tensorflow
```

## Usage

Place `train.csv`, `val.csv`, `test.csv` (with `id`, `text`, `score` columns) in the working directory, then:

```bash
python modelridge.py   # writes IncarcareFinalv1.csv
python nn.py           # writes PredictiiFinaleV3.csv
```

## Future Work

- Tune training parameters (epochs, batch size) not explored in this iteration
- Try richer embeddings (word2vec, transformer-based) beyond TF-IDF
- Ensemble Ridge + NN predictions
