import numpy as np
import torch
import pandas as pd

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


def load_and_split_data(csv_path, sequence_length):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    df = df[df['sequence'].apply(lambda x: isinstance(x, str) and len(x) == sequence_length)].reset_index(drop=True)
    print(f"Filtered to {len(df)} rows with sequence length {sequence_length}")
    # Get all columns except the last  (chromosome, start, end, sequence)
    feature_cols = df.columns[:-5]
    y = df[feature_cols].to_numpy(dtype=np.float32)
    print(f"Target shape: {y.shape}")
    X = np.stack([one_hot_encode(seq) for seq in df['sequence']])
    X_tensor = torch.tensor(X).permute(0, 2, 1)
    y_tensor = torch.tensor(y)
    chroms = df['chromosome'].unique()
    np.random.shuffle(chroms)
    split_idx = int(len(chroms) * 0.8)
    train_mask = df['chromosome'].isin(chroms[:split_idx]).values
    val_mask = df['chromosome'].isin(chroms[split_idx:]).values
    return X_tensor[train_mask], X_tensor[val_mask], y_tensor[train_mask], y_tensor[val_mask], df

