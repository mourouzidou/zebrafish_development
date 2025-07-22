import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import quantile_transform

def one_hot_encode(sequence):
    seq_array = np.array(list(sequence))[:, None]
    bases = np.array(list('ACGT'))
    one_hot = (seq_array == bases).astype(np.float32)
    n_mask = seq_array.flatten() == 'N'
    one_hot[n_mask] = 0.25
    return one_hot

def quantile_normalize(x, axis=0):
    sorted_x = np.sort(x, axis=axis)
    meanx = np.mean(sorted_x, axis=axis-1)
    xqtrl = meanx[np.argsort(np.argsort(x, axis=axis), axis=axis)]
    return xqtrl

def get_feature_cols(df, drop_n_last_cols=5):
    return df.columns[:-drop_n_last_cols]

def preprocess_data(csv_path, sequence_length=2000, drop_n_last_cols=5, scale=0, apply_log=True, apply_quantile=True):
    df = pd.read_csv(csv_path)
    df = df[df['sequence'].apply(lambda x: isinstance(x, str) and len(x) == sequence_length)].reset_index(drop=True)
    
    feature_cols = get_feature_cols(df, drop_n_last_cols)
    y_raw = df[feature_cols].to_numpy(dtype=np.float32)
    
    if apply_log and scale > 0:
        y_processed = np.log2(y_raw * scale + 1)
    else:
        y_processed = y_raw
    
    if apply_quantile:
        y_quant_norm = quantile_normalize(y_raw)
        if apply_log and scale > 0:
            y_quant_processed = np.log2(y_quant_norm * scale + 1)
        else:
            y_quant_processed = y_quant_norm
    else:
        y_quant_processed = y_processed
    
    df_raw = pd.DataFrame(y_processed, columns=feature_cols)
    df_quant = pd.DataFrame(y_quant_processed, columns=feature_cols)

    return df_raw, df_quant
def plot_distributions(
    df_raw, df_quant, title_prefix="Data", cell_type_filter=None, pseudobulk_counts=None
):
    cols = list(df_raw.columns)

    # Always sort by cell count (descending), ignore stage/time entirely
    if pseudobulk_counts is not None:
        cols_sorted = sorted(cols, key=lambda x: -pseudobulk_counts.get(x, 0))
    else:
        cols_sorted = cols  # No sorting if counts are missing

    # Cell type for each column (still for coloring)
    cell_types = [col.split('_')[1] if len(col.split('_')) > 1 else col for col in cols_sorted]

    # Filter if specified
    if cell_type_filter is not None:
        filtered_indices = [i for i, ct in enumerate(cell_types) if ct in cell_type_filter]
        cols_filtered = [cols_sorted[i] for i in filtered_indices]
        cell_types_filtered = [cell_types[i] for i in filtered_indices]
    else:
        cols_filtered = cols_sorted
        cell_types_filtered = cell_types

    unique_cell_types = list(dict.fromkeys(cell_types_filtered))
    import seaborn as sns
    palette = dict(zip(unique_cell_types, sns.color_palette("tab20c", len(unique_cell_types))))
    box_colors = [palette[ct] for ct in cell_types_filtered]

    if pseudobulk_counts is not None:
        x_labels = [f"{col}\n(n={pseudobulk_counts.get(col, 0)})" for col in cols_filtered]
    else:
        x_labels = cols_filtered

    def _plot(df, subtitle, use_single_color=False):
        plt.figure(figsize=(32, 12))
        if use_single_color:
            ax = sns.boxplot(data=df[cols_filtered], color='lightgray')
        else:
            ax = sns.boxplot(data=df[cols_filtered], palette=box_colors)
        ax.set_title(f"{title_prefix}: {subtitle}")
        ax.set_xlabel("Pseudobulk")
        ax.set_ylabel("log2(Accessibility + 1)")
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=6)
        plt.tight_layout()
        plt.show()

    _plot(df_raw, "Raw", use_single_color=False)
    _plot(df_quant, "Quantile Normalized", use_single_color=True)
