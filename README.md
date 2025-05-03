# Legal NER Project

This repository contains all code, data preparation scripts, and training configurations for our Legal Named Entity Recognition experiments on the **InLegalNER** dataset. It is built upon and extends the work from the original EkStep/Legal_NER repository: https://github.com/Legal-NLP-EkStep/legal_NER

------

## Project Structure

```text
.
├── dataset/                    # spaCy files (train.spacy, dev.spacy, test.spacy)
├── backtranslate.py            # Back-translation data augmentation script
├── training/
│   ├── config_a.cfg            # A: Baseline (RoBERTa)
│   ├── config_b.cfg            # B: LegalBERT
│   ├── config_c.cfg            # C: Back-Translation
│   └── config_d.cfg            # D: Combo (LegalBERT + Back-Translation)
├── output/                     # Model outputs for experiments A–D
├── evaluate/                   # Evaluation JSON metrics
└── README.md                   # This file
```

------

## Setup

1. **Clone the repository**:

   ```bash
   git clone <https://github.com/YueranCao2001/ENLP_Project_Group3>
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

------

## 1. Data Augmentation: Back-Translation

Paraphrase 20% of the training data via English↔Chinese translation:

```bash
python backtranslate.py \
  --input    dataset/train.spacy \
  --output   dataset/train_bt.spacy \
  --ratio    0.2 \
  --max_length 400
```

- `--ratio`: fraction of docs to augment (0.2 = 20%)
- `--max_length`: max token length per translation batch

This produces `dataset/train_bt.spacy` alongside the original `train.spacy`.

------

## 2. Model Training

Train four experiments using spaCy:

```bash
# A: Baseline (RoBERTa)
python -m spacy train training/config_a.cfg \
  --output       ./output/output_a \
  --paths.train  dataset/train.spacy \
  --paths.dev    dataset/dev.spacy

# B: LegalBERT (domain-adaptive pretraining)
python -m spacy train training/config_b.cfg \
  --output       ./output/output_b \
  --paths.train  dataset/train.spacy \
  --paths.dev    dataset/dev.spacy

# C: Back-Translation Augmentation
python -m spacy train training/config_c.cfg \
  --output       ./output/output_c \
  --paths.train  dataset/train_bt.spacy \
  --paths.dev    dataset/dev.spacy

# D: Combo (LegalBERT + Back-Translation)
python -m spacy train training/config_d.cfg \
  --output       ./output/output_d \
  --paths.train  dataset/train_bt.spacy \
  --paths.dev    dataset/dev.spacy
```

Each command trains a `transformer`→`ner` pipeline and saves the best model to `output/output_*`.

------

## 3. Evaluation

Evaluate each best model on the held-out test set:

```bash
python -m spacy evaluate output/output_a/model-best \
  dataset/test.spacy \
  --output      evaluate/metrics_test_a.json

python -m spacy evaluate output/output_b/model-best \
  dataset/test.spacy \
  --output      evaluate/metrics_test_b.json

python -m spacy evaluate output/output_c/model-best \
  dataset/test.spacy \
  --output      evaluate/metrics_test_c.json

python -m spacy evaluate output/output_d/model-best \
  dataset/test.spacy \
  --output      evaluate/metrics_test_d.json
```

Each `metrics_test_*.json` includes overall `ents_p`/`ents_r`/`ents_f` and per‐type P/R/F.

------

## 4. Results & Visualization

Use the provided JSON files under `evaluate/` to plot precision, recall, and F1 comparisons.

------

## 5. Using Pretrained Models from Releases

We have published our trained `model-best` checkpoints as assets on GitHub Releases so you don’t need to retrain locally if you only want to run inference.

1. **Download the Release asset**  
    Go to [Releases](https://github.com/YueranCao2001/ENLP_Project_Group3/releases) and download the `model-best_*.zip` file for the experiment (A, B, C, or D).

2. **Extract and place it under `output/output_*/`**  