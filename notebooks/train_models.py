import os, sys, json, torch
import numpy as np
sys.path.append(os.path.abspath("../src"))
from utils.training import make_loaders_from_arrays, train_model, load_split_folder, train_on_split

from models.dilated_baseline_model import ATACSeqCNN
from utils.training import make_loaders_from_arrays, train_model, save_training_metadata, load_split_folder

# train_on_split(
#     split_dir="../data/embryo/training/train_atac_L2k_g11_embryo_cpm_l1_q1__seqs/80_20_chrom_split",
#     dataset="embryo_cpm_l1_q1",
#     loss_name="mse",
#     model_name="dilated_baseline_model",
#     batch_size=64,
#     lr=1e-3,
#     weight_decay=1e-4,
#     use_test_as_val=True,   # or False to carve from train
#     val_frac=0.1,
#     num_epochs=300,
#     early_stopping_patience=10,
# )
train_on_split(
    split_dir="../data/lifelong/training/train_atac_L2k_g11_lifelong_cpm_l1_q1__seqs/80_20_chrom_split",
    dataset="lifelong_cpm_l1_q1",
    loss_name="mse",
    model_name="dilated_baseline_model",
    batch_size=96,         
    lr=5e-4,               
    weight_decay=5e-5,
    use_test_as_val=True,
    num_epochs=200,
    early_stopping_patience=4,
)

train_on_split(
    split_dir="../data/lifelong/training/train_atac_L2k_g11_lifelong_cpm_l1_q0__seqs/80_20_chrom_split",
    dataset="lifelong_cpm_l1_q0",
    loss_name="mse",
    model_name="dilated_baseline_model",
    batch_size=96,         
    lr=5e-4,               
    weight_decay=5e-5,
    use_test_as_val=True,
    num_epochs=200,
    early_stopping_patience=15,
)


