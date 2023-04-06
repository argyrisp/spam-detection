# spam-detection

This is a short project regarding spam detection on emails by utilizing Random Forest, k-Nearest Neighbors & Gaussian Naive Bayes.

Quickjump:

[Clone Repository](#clone-repository) | [Environment](#environment) | [Run](#run) | [Notes](#notes) | [Results](#results)

## Clone Repository
Grab the current repository by:
```
git clone git@github.com:argyrisp/spam-detection.git
cd spam-detection
```
## Environment
Import & enable environment:
```
conda env create -f environment.yml
conda activate sentence_transformer
```
## Run
Simply run the script:
```commandline
python3 main.py
```
TODO: add flags for preprocessor mode, training mode and prediction mode

## Notes
- This script utilizes Sentence Transformer to vectorize the text.
- The text is "cleaned" before it is fed to the transformer (no capital letters, ignore stopwords and special characters).
- Dataset preprocessing takes a lot of time due to the Sentence Transformer, it is not recommended to preprocess, as the vectorized dataset is provided in this repository.
- Preprocessing can be sped up if gpu torch is utilized.
- Requires Python version 3.9.

## Results
Dataset info:

|             | spam       | benign     | total       |
|-------------|------------|------------|-------------|
| samples     | 1499 (29%) | 3672 (71%) | 5171 (100%) |
| train_split | 1196       | 2940       | 4136 (80%)  |
| test_split  | 303        | 732        | 1035 (20%)  |


Model Evaluations:

| Gaussian NB       | precision  | recall | f1-score | n samples |
|-------------------|------------|--------|----------|-----------|
| benign            | 0.99       | 0.89   | 0.94     | 732       |
| spam              | 0.78       | 0.98   | 0.87     | 303       |
| -                 | -          | -      | -        | -         |
| accuracy          | 0.91       |        |          | 1035      |
| **ROC AUC score** | **0.9318** |        |          | 1035      |



| k-NN              | precision  | recall | f1-score | n samples |
|-------------------|------------|--------|----------|-----------|
| benign            | 0.99       | 1.00   | 0.99     | 732       |
| spam              | 0.99       | 0.96   | 0.98     | 303       |
| -                 | -          | -      | -        | -         |
| accuracy          | 0.99       |        |          | 1035      |
| **ROC AUC score** | **0.9805** |        |          | 1035      |

| Random Forest     | precision  | recall | f1-score | n samples |
|-------------------|------------|--------|----------|-----------|
| benign            | 1.00       | 1.00   | 1.00     | 732       |
| spam              | 1.00       | 1.00   | 1.00     | 303       |
| -                 | -          | -      | -        | -         |
| accuracy          | 1.00       |        |          | 1035      |
| **ROC AUC score** | **0.9983** |        |          | 1035      |

Based on the above tables, we observe that Random Forest with max_depth=10 is the best performing.

|                    | gNB    | kNN    | rf         |
|--------------------|--------|--------|------------|
| ROC AUC score      | 0.9318 | 0.9805 | **0.9983** |