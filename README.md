# Zebrafish Model Development

This repository contains the full pipeline for processing, modeling, and analyzing zebrafish regulatory genomics data across developmental stages and cell types. It focuses on generating meaningful datasets and building deep learning models to study gene regulation through chromatin accessibility and gene expression.

## Projects

### 1. `embryo/`

- **Source**: [Single-nucleus chromatin landscapes during zebrafish early embryogenesis ](https://pubmed.ncbi.nlm.nih.gov/37468546/)
- These datasets focus on embryonic-stage zebrafish data, integrating chromatin accessibility and gene expression to explore cell differentiation mechanisms in early emvryogenesis.

### 2. `lifelong/`

- **Source**: ['Lifelong single-cell profiling of cranial neural crest diversification in zebrafish'](https://www.nature.com/articles/s41467-021-27594-w)
- These datasets focus on zebrafish lifelong single cell profiling, where similar multi-omics analysis is applied to understand mature regulatory landscapes.
---

## Goals

- Prepare clean, biologically meaningful datasets from these two projects for downstream S2F modeling.
- Train **deep learning models** that predict **chromatin accessibility** and/or **gene expression** from DNA sequence.
- Study how gene regulation varies **across cell types** and **developmental stages** using interpretable models.
- Combine the two projects - enrich model's input (?)

---

# Training CLI

This repository provides a command-line interface to train deep learning models on prepared zebrafish ATAC-seq datasets.

## Usage

From the repository root, run:

```bash
python3 scripts/train.py \
  --split_dir data/embryo/training/train_atac_L2k_g11_embryo_cpm_l1_q1__seqs/80_20_chrom_split \
  --dataset embryo_cpm_l1_q1 \
  --model_name dilated_baseline_model \
  --loss_name mse \
  --batch_size 64 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --use_test_as_val \
  --num_epochs 300 \
  --early_stopping_patience 10
````

## Arguments

* `--split_dir` : Path to prepared training split folder (contains `X_train.npy`, `Y_train.npy`, `meta.json`, …).
* `--dataset`   : Dataset tag for organizing outputs (e.g. `embryo_cpm_l1_q1`).
* `--model_name`: Model architecture (`dilated_baseline_model`, `rna_onlySeq_model`, …).
* `--loss_name` : Loss function (`mse` for log-normalized data, `poisson_log` for raw counts).
* `--batch_size`: Training batch size.
* `--lr`        : Learning rate.
* `--weight_decay`: Weight decay for AdamW optimizer.
* `--use_test_as_val`: If set, use the test split as validation. Otherwise, carve `--val_frac` from training.
* `--val_frac`  : Fraction of training data used as validation (default 0.1).
* `--num_epochs`: Maximum training epochs.
* `--early_stopping_patience`: Stop if validation doesn’t improve for this many epochs.
* `--cuda`      : Pin to a specific GPU (e.g. `--cuda 0`).

## Outputs

Each run is saved under:

```
src/models/outputs/<dataset>/<model_name>/<run_id>/
```

Where `<run_id>` includes:

* split type (`80_20_chrom`, `80_20_random`),
* loss,
* batch size, lr, weight decay,
* validation source,
* timestamp and hash.

Inside each run folder:

* `*.pth` : model checkpoint with unique identifier.
* `training_config.json` : metadata (data split, hyperparameters, history).
* `history.csv` : per-epoch metrics.
* `losses.png` : training/validation loss curves.
* `correlations.png` : correlation metrics over epochs.
