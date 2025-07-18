import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import quantile_transform

def one_hot_encode(sequence):
    seq_array = np.array(list(sequence))[:, None]
    bases = np.array(list('ACGT'))
    one_hot = (seq_array == bases).astype(np.float32)
    n_mask = seq_array.flatten() == 'N'
    one_hot[n_mask] = 0.25
    return one_hot

def reverse_complement_tensor(x):
    rev_x = torch.flip(x, dims=[2])
    rev_x = rev_x[:, [3, 2, 1, 0], :]
    return rev_x

def log_transform(arr):
    return np.log1p(arr)

def quantile_normalize(arr):
    # Apply quantile normalization to each column (cell type)
    return quantile_transform(arr, axis=0, output_distribution='normal', copy=True)


def plot_rawVStrans(arr_raw, arr_trans, column_names, output_dir=None):
    num_columns = arr_raw.shape[1]
    x = np.arange(num_columns)
    width = 0.4

    raw_means = np.mean(arr_raw, axis=0)
    raw_stds = np.std(arr_raw, axis=0)
    trans_means = np.mean(arr_trans, axis=0)
    trans_stds = np.std(arr_trans, axis=0)

    plt.figure(figsize=(15, 6))
    plt.bar(x - width / 2, raw_means, width, yerr=raw_stds, label='Raw', alpha=0.7, capsize=5)
    plt.bar(x + width / 2, trans_means, width, yerr=trans_stds, label='Transformed', alpha=0.7, capsize=5)

    plt.xticks(x, column_names, rotation=45, ha='right')
    plt.ylabel('Values')
    plt.title('Distribution of Target Values Before and After Transformation')
    plt.legend()

    if output_dir:
        plt.savefig(f"{output_dir}/distribution_comparison.png", bbox_inches='tight')
    else:
        plt.show()
def load_and_split_data(
    csv_path, sequence_length, 
    log_transform_flag=False,
    normalize=None,
    drop_n_last_cols=5,
    plot_norm=False,
    output_dir=None
):
    df = pd.read_csv(csv_path)
    df = df[df['sequence'].apply(lambda x: isinstance(x, str) and len(x) == sequence_length)].reset_index(drop=True)
    feature_cols = df.columns[:-drop_n_last_cols]
    y_raw = df[feature_cols].to_numpy(dtype=np.float32)  # Save raw copy for plotting

    y = y_raw.copy()
    transformed = False

    if log_transform_flag:
        y = log_transform(y)
        transformed = True

    if normalize == "quantile":
        y = quantile_normalize(y)
        transformed = True
    if plot_norm and transformed:
        plot_rawVStrans(y_raw, y, feature_cols, output_dir=output_dir)

    X = np.stack([one_hot_encode(seq) for seq in df['sequence']])
    X_tensor = torch.tensor(X).permute(0, 2, 1)
    y_tensor = torch.tensor(y)
    chroms = df['chromosome'].unique()
    np.random.shuffle(chroms)
    split_idx = int(len(chroms) * 0.8)
    train_mask = df['chromosome'].isin(chroms[:split_idx]).values
    val_mask = df['chromosome'].isin(chroms[split_idx:]).values
    return X_tensor[train_mask], X_tensor[val_mask], y_tensor[train_mask], y_tensor[val_mask], df
