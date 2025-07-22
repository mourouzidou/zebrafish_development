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

def log2_transform(arr, scale=1000):
    return np.log2(arr * scale + 1)


def quantile_normalize(x, axis=0):
    sorted_x = np.sort(x, axis=axis)
    meanx = np.mean(sorted_x, axis=axis-1)
    xqtrl = meanx[np.argsort(np.argsort(x, axis=axis), axis=axis)]
    return xqtrl

def get_feature_cols(df, drop_n_last_cols=5):
    return df.columns[:-drop_n_last_cols]

def load_and_preprocess_data(csv_path, sequence_length=2000, drop_n_last_cols=5):
    df = pd.read_csv(csv_path)
    df = df[df['sequence'].apply(lambda x: isinstance(x, str) and len(x) == sequence_length)].reset_index(drop=True)
    
    # Get features (accessibility)
    feature_cols = get_feature_cols(df, drop_n_last_cols)
    y_raw = df[feature_cols].to_numpy(dtype=np.float32)
    
    # Normalize and log transform
    y_quant_norm = quantile_normalize(y_raw)
    y_raw_log2 = log2_transform(y_raw)
    y_quant_log2 = log2_transform(y_quant_norm)
    
    # Prepare for plotting
    df_raw_log2 = pd.DataFrame(y_raw_log2, columns=feature_cols)
    df_quant_log2 = pd.DataFrame(y_quant_log2, columns=feature_cols)

    return df_raw_log2, df_quant_log2


def plot_distributions(df_raw_log2, df_quant_log2, title_prefix="Data", cell_type_filter=None, pseudobulk_counts=None):

    cols = df_raw_log2.columns
    
    # Sort columns by time point (first part before '_')
    def get_time_point(col_name):
        parts = col_name.split('_')
        if len(parts) > 1:
            try:
                return int(parts[0])
            except ValueError:
                return 0  # fallback for non-numeric time points
        return 0
    
    cols_sorted = sorted(cols, key=lambda x: (get_time_point(x), x))
    
    cell_types = [col.split('_')[1] if len(col.split('_')) > 1 else col for col in cols_sorted]
    
    # Apply cell type filtering if specified
    if cell_type_filter is not None:
        # Find columns that match the filtered cell types
        filtered_indices = [i for i, ct in enumerate(cell_types) if ct in cell_type_filter]
        cols_filtered = [cols_sorted[i] for i in filtered_indices]
        cell_types_filtered = [cell_types[i] for i in filtered_indices]
    else:
        cols_filtered = cols_sorted
        cell_types_filtered = cell_types
    
    # Get unique cell types and assign colors
    unique_cell_types = list(dict.fromkeys(cell_types_filtered))  # preserve order
    palette = dict(zip(unique_cell_types, sns.color_palette("tab20c", len(unique_cell_types))))
    box_colors = [palette[ct] for ct in cell_types_filtered]
    
    # Create labels with cell counts if provided
    if pseudobulk_counts is not None:
        x_labels = [f"{col}\n(n={pseudobulk_counts.get(col, 0)})" for col in cols_filtered]
    else:
        x_labels = cols_filtered
    
    def _plot(df, subtitle, use_single_color=False):
        plt.figure(figsize=(32, 12))  # wider
        if use_single_color:
            ax = sns.boxplot(data=df[cols_filtered], color='lightgray')
        else:
            ax = sns.boxplot(data=df[cols_filtered], palette=box_colors)
        ax.set_title(f"{title_prefix}: {subtitle}")
        ax.set_xlabel("Pseudobulk")
        ax.set_ylabel("log2(Accessibility + 1)")
        ax.set_xticklabels(
            x_labels, 
            rotation=45,    # more rotation!
            ha='right', 
            fontsize=6      # smaller font size
        )
        plt.tight_layout()
        plt.show()

    # Plot both versions with consistent filtering
    _plot(df_raw_log2, "Raw ×1000 then log2(x+1)", use_single_color=False)
    _plot(df_quant_log2, "Quantile Normalized then ×1000 and log2(x+1)", use_single_color=True)