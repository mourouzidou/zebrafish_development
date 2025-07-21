import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import quantile_transform
import scipy.stats

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


def custom_quantile_transform(data):
    """Version 4: True uniform using unique values"""
    transformed = np.zeros_like(data)
    
    for i in range(data.shape[1]):
        column_data = data[:, i]
        
        # Get unique values and their positions
        unique_vals, inverse_indices = np.unique(column_data, return_inverse=True)
        n_unique = len(unique_vals)
        
        # Create uniform spacing for unique values
        uniform_vals = np.linspace(0, 1, n_unique)
        
        # Map back to original positions
        transformed[:, i] = uniform_vals[inverse_indices]
    
    return transformed


def quantile_normalize(arr, distribution='uniform'):
    return quantile_transform(arr, axis=0, output_distribution=distribution, copy=True)

def get_feature_cols(df, drop_n_last_cols=5):
    # Returns feature columns (targets), not metadata
    return df.columns[:-drop_n_last_cols]

def load_and_split_data(
    csv_path, 
    sequence_length, 
    log_transform_flag=False,
    normalize=None,
    drop_n_last_cols=5,
    train_frac=0.7,
    val_frac=0.15,
    test_frac=0.15
):
    df = pd.read_csv(csv_path)
    df = df[df['sequence'].apply(lambda x: isinstance(x, str) and len(x) == sequence_length)].reset_index(drop=True)
    feature_cols = get_feature_cols(df, drop_n_last_cols)
    y_raw = df[feature_cols].to_numpy(dtype=np.float32)
    y = y_raw.copy()
    if log_transform_flag:
        y = log_transform(y)
    if normalize == "quantile":
        y = custom_quantile_transform(y)
    X = np.stack([one_hot_encode(seq) for seq in df['sequence']])
    X_tensor = torch.tensor(X).permute(0, 2, 1)
    y_tensor = torch.tensor(y)
    chroms = df['chromosome'].unique()
    np.random.shuffle(chroms)
    n = len(chroms)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    train_chroms = chroms[:train_end]
    val_chroms = chroms[train_end:val_end]
    test_chroms = chroms[val_end:]
    train_mask = df['chromosome'].isin(train_chroms).values
    val_mask = df['chromosome'].isin(val_chroms).values
    test_mask = df['chromosome'].isin(test_chroms).values
    return (
        X_tensor[train_mask], X_tensor[val_mask], X_tensor[test_mask],
        y_tensor[train_mask], y_tensor[val_mask], y_tensor[test_mask],
        y_raw, y, 
    )
