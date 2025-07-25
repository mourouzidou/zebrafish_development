import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import seaborn as sns

from pathlib import Path
sys.path.append(os.path.abspath("../../src"))

def plot_rawVStrans(arr_raw, arr_trans, column_names, output_dir=None, name=""):
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
    plt.title(f'Distribution: Raw vs. {name} Transformed')
    plt.legend()
    plt.tight_layout()
    if output_dir:
        plt.savefig(f"{output_dir}/distribution_comparison_{name}.png", bbox_inches='tight')
    else:
        plt.show()

def plot_pseudobulk_cell_read_distributions(
    atac_data_df, atac_metadata_df, save_path=None, max_pseudobulks=None, show=True
):
    cell_map = atac_metadata_df.set_index('atac_cell')[['pseudobulk', 'atac_cell_type']]
    total_per_cell = atac_data_df.groupby('Cell')['Accessibility'].sum()
    df_reads = (
        total_per_cell.rename_axis('Cell').reset_index()
        .merge(cell_map, left_on='Cell', right_index=True, how='left')
        .dropna(subset=['pseudobulk', 'atac_cell_type'])
    )
    # Optionally restrict to top N pseudobulks by cell count
    cell_counts = df_reads['pseudobulk'].value_counts()
    order = cell_counts.index.tolist()
    if max_pseudobulks:
        order = order[:max_pseudobulks]
        df_reads = df_reads[df_reads['pseudobulk'].isin(order)]
        cell_counts = cell_counts.loc[order]
    x_labels = [f"{pb}\n(n={cell_counts[pb]})" for pb in order]
    plt.figure(figsize=(32, 12))
    ax = sns.boxplot(
        data=df_reads,
        x='pseudobulk',
        y='Accessibility',
        hue='atac_cell_type',
        order=order,
        showfliers=False,
        palette='tab20'
    )
    ax.set_xticklabels(
        x_labels,
        rotation=45,
        ha='right',
        fontsize=6,
        linespacing=1.7
    )
    plt.xlabel("Pseudobulk (sorted by #cells)", fontsize=12, fontweight='bold')
    plt.ylabel("Total Reads per Cell", fontsize=12)
    plt.title("ATAC Reads per Cell by Pseudobulk and Cell Type", fontsize=12)
    plt.legend(title="Cell Type", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=250)
    if show:
        plt.show()
    else:
        plt.close()


def plot_pseudobulk_agg_stats(
    atac_data_df, atac_metadata_df, save_path=None, show=True
):

    cell_map = atac_metadata_df.set_index('atac_cell')[['pseudobulk', 'atac_cell_type']]
    total_per_cell = atac_data_df.groupby('Cell')['Accessibility'].sum()
    df_reads = (
        total_per_cell.rename_axis('Cell').reset_index()
        .merge(cell_map, left_on='Cell', right_index=True, how='left')
        .dropna(subset=['pseudobulk', 'atac_cell_type'])
    )
    agg = df_reads.groupby(['pseudobulk', 'atac_cell_type']).agg(
        n_cells=('Accessibility', 'count'),
        median=('Accessibility', 'median'),
        std=('Accessibility', 'std'),
        max=('Accessibility', 'max'),
        min=('Accessibility', 'min')
    ).reset_index()
    agg['range'] = agg['max'] - agg['min']

    def scatter_and_save(x, y, ylabel, title, fname):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=agg,
            x=x, y=y,
            hue='atac_cell_type',
            palette='tab20',
            s=80,
            edgecolor='k'
        )
        plt.title(title, fontsize=13)
        plt.xlabel("Number of Cells in Pseudobulk")
        plt.ylabel(ylabel)
        plt.legend(title="Cell Type", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, fname), dpi=250)
        if show:
            plt.show()
        else:
            plt.close()

    scatter_and_save(
        x='n_cells',
        y='median',
        ylabel='Median Reads per Cell',
        title='Median Reads per Pseudobulk vs Number of Cells',
        fname="scatter_median_reads_vs_num_cells.png"
    )
    scatter_and_save(
        x='n_cells',
        y='std',
        ylabel='Std (Reads per Cell)',
        title='Std of Reads per Pseudobulk vs Number of Cells',
        fname="scatter_std_reads_vs_num_cells.png"
    )
    scatter_and_save(
        x='n_cells',
        y='range',
        ylabel='Max-Min Range (Reads per Cell)',
        title='Max-Min Range per Pseudobulk vs Number of Cells',
        fname="scatter_range_reads_vs_num_cells.png"
    )
def plot_reads_per_cell_by_celltype_and_stage(
    total_per_cell,
    atac_metadata_df,
    min_cells_per_group=10,
    figsize=(18, 8),
    save_path=None,
    show=True
):
    df_reads = (
        total_per_cell.rename_axis('Cell').reset_index()
        .merge(atac_metadata_df, left_on='Cell', right_on='atac_cell', how='left')
        .dropna(subset=['atac_cell_type', 'atac_stage'])
    )
    
    # Convert stage to integer for proper sorting
    df_reads['atac_stage_int'] = df_reads['atac_stage'].astype(int)
    
    # Filter to groups with enough cells
    ct_stage_counts = df_reads.groupby(['atac_cell_type', 'atac_stage_int']).size().reset_index(name='count')
    good = ct_stage_counts[ct_stage_counts['count'] >= min_cells_per_group]
    df_reads = df_reads.merge(
        good[['atac_cell_type', 'atac_stage_int']],
        on=['atac_cell_type', 'atac_stage_int'],
        how='inner'
    )
    
    # Get sorted stages for proper ordering
    sorted_stages = sorted(df_reads['atac_stage_int'].unique())
    
    # Create color palette with proper ordering
    n_stages = len(sorted_stages)
    colors = sns.color_palette('Spectral', n_stages)
    stage_colors = dict(zip(sorted_stages, colors))
    
    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df_reads,
        x='atac_cell_type',
        y='Accessibility',
        hue='atac_stage_int',
        hue_order=sorted_stages,  # Ensure proper ordering
        showfliers=False,
        palette=stage_colors  # Use ordered color mapping
    )
    
    plt.xlabel('Cell Type')
    plt.ylabel('Total Reads per Cell')
    plt.title('Reads per Cell by Cell Type and Stage')
    plt.legend(title='Stage', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()

    
def plot_distributions(
    df_raw, df_quant, title_prefix="Data", cell_type_filter=None, 
    pseudobulk_counts=None, save_dir=None, show=True):
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cols = list(df_raw.columns)
    if pseudobulk_counts is not None:
        cols_sorted = sorted(cols, key=lambda x: -pseudobulk_counts.get(x, 0))
    else:
        cols_sorted = cols
    
    # Cell type for each column (for coloring)
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
    palette = dict(zip(unique_cell_types, sns.color_palette("tab20c", len(unique_cell_types))))
    box_colors = [palette[ct] for ct in cell_types_filtered]
    
    if pseudobulk_counts is not None:
        x_labels = [f"{col}\n(n={pseudobulk_counts.get(col, 0)})" for col in cols_filtered]
    else:
        x_labels = cols_filtered
    
    def _plot(df, subtitle, use_single_color=False, fname=None):
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
        
        if fname:
            plt.savefig(fname, dpi=150)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    if save_dir:
        import os
        prefix = title_prefix.lower().replace(" ", "_")
        fname_raw = f"{save_dir}/{prefix}_raw_log2_boxplot.png"
        fname_quant = f"{save_dir}/{prefix}_quantile_log2_boxplot.png"
    else:
        fname_raw = fname_quant = None
    
    _plot(df_raw, "Raw", use_single_color=False, fname=fname_raw)
    _plot(df_quant, "Quantile Normalized", use_single_color=True, fname=fname_quant)