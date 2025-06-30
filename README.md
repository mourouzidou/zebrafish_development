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
