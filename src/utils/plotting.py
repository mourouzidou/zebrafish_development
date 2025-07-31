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




def plot_grouped_boxplot(
    long_df,
    value_for_plot,
    region_col='region_type',
    groupby='pseudobulk',
    figsize=(20,7),
    show=True,
    save_path=None,
    distance_cutoff=None,
    min_cells=1,
    region_palette=None,
):
    
    group_counts = long_df.groupby(groupby)['Cell'].nunique().sort_values(ascending=False)
    ordered_groups = group_counts.index.tolist()
    x_labels = [f"{grp}\n(n={group_counts[grp]})" for grp in ordered_groups]
    label_map = dict(zip(ordered_groups, x_labels))

    if region_palette is None:
        palette = dict(zip(long_df[region_col].unique(), sns.color_palette("tab20", n_colors=long_df[region_col].nunique())))
    else:
        palette = region_palette

    plt.figure(figsize=figsize)
    ax = sns.boxplot(
        data=long_df[long_df[groupby].isin(ordered_groups)],
        x=groupby,
        y=value_for_plot,
        hue=region_col,
        showfliers=False,
        order=ordered_groups,
        palette=palette
    )
    ax.set_xticklabels([label_map[tick.get_text()] for tick in ax.get_xticklabels()],
                       rotation=45, ha='right')
    cutoff_str = f" | Distance cutoff: {distance_cutoff} bp" if distance_cutoff is not None else ""
    y_lab = ("Mean" if 'mean' in value_for_plot else "Total") + " Accessibility per Cell"
    ax.set_title(
        f'{y_lab} by {groupby.replace("_", " ").title()} ({region_col}){cutoff_str}',
        fontsize=15
    )
    ax.set_xlabel(groupby.replace('_', ' ').title())
    ax.set_ylabel(y_lab)
    plt.legend(title=region_col.replace("_", " ").title())
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
def plot_summary_accessibility(
    summary_df, 
    gen_info="region_type",         # "region_type" or "genomic_context"
    group_by="atac_cell_type",      # "atac_cell_type" or "pseudobulk"
    dist=1000,                     # distance cutoff for region_type/promoter annotation
    show=True, 
    save_path=None
):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Set color palette based on combination of parameters
    palette_map = {
        ("region_type", "atac_cell_type"): "pastel",
        ("region_type", "pseudobulk"): "muted",
        ("genomic_context", "atac_cell_type"): "Set2",
        ("genomic_context", "pseudobulk"): "deep",
    }
    # Default palette if combination not specified
    palette = palette_map.get((gen_info, group_by), "pastel")

    # Column selection and labels
    if gen_info == "region_type":
        col1, col2 = "total_enhancer_accessibility", "total_promoter_accessibility"
        plot_labels = ["enhancer", "promoter"]
        legend_title = "Region Type"
    elif gen_info == "genomic_context":
        col1, col2 = "total_intergenic_accessibility", "total_intragenic_accessibility"
        plot_labels = ["intergenic", "intragenic"]
        legend_title = "Genomic Context"
    else:
        raise ValueError("gen_info must be 'region_type' or 'genomic_context'")

    # Prepare dataframe in long format for seaborn
    plot_df = summary_df[[group_by, col1, col2]].copy()
    plot_df = plot_df.melt(id_vars=group_by, value_vars=[col1, col2], 
                           var_name='Context', value_name='Accessibility')
    plot_df['Context'] = plot_df['Context'].map({
        col1: plot_labels[0],
        col2: plot_labels[1]
    })

    # Order groups by number of cells
    group_counts = summary_df[group_by].value_counts()
    order = group_counts.index.tolist()
    x_labels = [f"{grp}\n(n={group_counts[grp]})" for grp in order]

    # Plot
    plt.figure(figsize=(22, 8))
    ax = sns.boxplot(
        data=plot_df,
        x=group_by,
        y='Accessibility',
        hue='Context',
        order=order,
        palette=palette
    )
    plt.title(
        f"Mean Accessibility per Cell by {group_by.replace('_', ' ').title()} "
        f"({legend_title}) | Distance cutoff: {dist} bp"
    )
    plt.xlabel(group_by.replace('_', ' ').title())
    plt.ylabel("Mean Accessibility per Cell")
    ax.set_xticklabels(x_labels, rotation=25, ha='right', fontsize=9)
    plt.legend(title=legend_title, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180)
    if show:
        plt.show()
    plt.close()



## _____________RNA-seq Plotting Functions_____________ ##
def plot_marker_gene_fold_change(marker_genes_all, marker_genes_filtered, title="Fold Change in Marker Gene Counts"):
    def extract_stage(group_name):
        try:
            return int(group_name.split('_')[0])
        except:
            return float('inf')

    all_groups = set(marker_genes_all.keys()) | set(marker_genes_filtered.keys())
    sorted_groups = sorted(all_groups, key=extract_stage)

    fold_changes = []
    for group in sorted_groups:
        count_all = len(marker_genes_all.get(group, []))
        count_filtered = len(marker_genes_filtered.get(group, []))
        if count_all > 0:
            fc = count_filtered / count_all
        else:
            fc = float('inf') if count_filtered > 0 else 1
        fold_changes.append(fc)

    capped_fc = [min(fc, 2.0) if fc != float('inf') else 2.0 for fc in fold_changes]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(sorted_groups, capped_fc, alpha=0.8, color='slategrey')
    plt.axhline(1.0, color='red', linestyle='--', label='No change')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Pseudobulk")
    plt.ylabel("Fold change (Filtered / All)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    for bar, fc in zip(bars, fold_changes):
        label = f"{fc:.1f}" if fc != float('inf') else "inf"
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05, 
                 label, ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_marker_gene_counts(marker_genes_dict, title="Number of Marker Genes per Pseudobulk"):
    def extract_stage(group_name):
        try:
            return int(group_name.split('_')[0])
        except:
            return float('inf')

    sorted_groups = sorted(marker_genes_dict.keys(), key=extract_stage)
    counts = [len(marker_genes_dict[group]) for group in sorted_groups]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(sorted_groups, counts, alpha=0.8, color='steelblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Pseudobulk")
    plt.ylabel("Number of Marker Genes")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, 
                 str(count), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

def plot_marker_gene_fold_change(marker_genes_all, marker_genes_filtered, title="Fold Change in Marker Gene Counts"):
    def extract_stage(group_name):
        try:
            return int(group_name.split('_')[0])
        except:
            return float('inf')

    all_groups = set(marker_genes_all.keys()) | set(marker_genes_filtered.keys())
    sorted_groups = sorted(all_groups, key=extract_stage)

    fold_changes = []
    for group in sorted_groups:
        count_all = len(marker_genes_all.get(group, []))
        count_filtered = len(marker_genes_filtered.get(group, []))
        if count_all > 0:
            fc = count_filtered / count_all
        else:
            fc = float('inf') if count_filtered > 0 else 1
        fold_changes.append(fc)

    capped_fc = [min(fc, 2.0) if fc != float('inf') else 2.0 for fc in fold_changes]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(sorted_groups, capped_fc, alpha=0.8, color='slategrey')
    plt.axhline(1.0, color='red', linestyle='--', label='No change')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Pseudobulk")
    plt.ylabel("Fold change (Filtered / All)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    for bar, fc in zip(bars, fold_changes):
        label = f"{fc:.1f}" if fc != float('inf') else "inf"
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05, 
                 label, ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()



def plot_marker_gene_counts(marker_genes_dict, title="Number of Marker Genes per Pseudobulk"):
    def extract_stage(group_name):
        try:
            return int(group_name.split('_')[0])
        except:
            return float('inf')

    sorted_groups = sorted(marker_genes_dict.keys(), key=extract_stage)
    counts = [len(marker_genes_dict[group]) for group in sorted_groups]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(sorted_groups, counts, alpha=0.8, color='steelblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Pseudobulk")
    plt.ylabel("Number of Marker Genes")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, 
                 str(count), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()
def plot_distance_distributions_by_celltype(cell_to_psd_with_markers,
                                          cell_to_psd_dist_with_markers,
                                          cutoffs_df,
                                          output_dir="distance_violin_plots"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame({
        "cell": list(cell_to_psd_with_markers.keys()),
        "pseudobulk": [cell_to_psd_with_markers[c] for c in cell_to_psd_with_markers],
        "distance": [cell_to_psd_dist_with_markers[c] for c in cell_to_psd_with_markers]
    })

    df['celltype'] = df['pseudobulk'].str.split('_').str[1:]  # Everything after first underscore
    df['celltype'] = df['celltype'].apply(lambda x: '_'.join(x) if isinstance(x, list) else x)
    df['stage'] = df['pseudobulk'].str.split('_').str[0]  # First part before underscore
    
    unique_celltypes = sorted(df['celltype'].unique())
    all_stages = sorted(df['stage'].unique(), key=lambda x: int(x) if x.isdigit() else 999)
    
    fig, axes = plt.subplots(len(unique_celltypes), 1, 
                            figsize=(max(8, len(all_stages) * 1.5), 5 * len(unique_celltypes)),
                            sharex=True)
    
    if len(unique_celltypes) == 1:
        axes = [axes]
    
    #  create a subplot pewr cell type
    for idx, celltype in enumerate(unique_celltypes):
        celltype_df = df[df['celltype'] == celltype]
        
        # Sort stages numerically 
        stages = sorted(celltype_df['stage'].unique(), 
                       key=lambda x: int(x))
    
        sns.violinplot(data=celltype_df, x='stage', y='distance', 
                      inner='quartile', order=stages, palette='Set2', ax=axes[idx])
        
        # cutoff lines for each stage
        for i, stage in enumerate(stages):
            stage_pseudobulk = f"{stage}_{celltype}"
            cutoff = cutoffs_df[cutoffs_df['pseudobulk'] == stage_pseudobulk]['cutoff'].values
            if len(cutoff) > 0:
                axes[idx].axhline(y=cutoff[0], color='red', linestyle='--', alpha=0.7,
                                 xmin=(i)/(len(stages)), xmax=(i+1)/(len(stages)))
        
        axes[idx].set_title(f"Distance Distribution by Stage - {celltype}")
        axes[idx].set_ylabel("Euclidean Distance")
        
        # Only show x-axis label on the bottom subplot
        if idx == len(unique_celltypes) - 1:
            axes[idx].set_xlabel("Stage")
        else:
            axes[idx].set_xlabel("")
    
    axes[0].legend()
    plt.tight_layout()
    plt.suptitle("Pseudobulk Enrichment: \n Distance to Closest Pseudobulk ", fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(output_dir, "all_celltypes_stages_violin.png"), 
               dpi=300, bbox_inches='tight')
    plt.close()






def plot_pseudobulk_distributions(df, count_col, max_pseudobulks=None, save_path=None, show=True):
    cell_counts = df['pseudobulk'].value_counts()
    order = cell_counts.index.tolist()
    
    if max_pseudobulks:
        order = order[:max_pseudobulks]
        df = df[df['pseudobulk'].isin(order)]
        cell_counts = cell_counts.loc[order]
    
    x_labels = [f"{pb}\n(n={cell_counts[pb]})" for pb in order]
    
    plt.figure(figsize=(32, 16))
    ax = sns.boxplot(
        data=df,
        x='pseudobulk',
        y=count_col,
        hue='annotation',
        order=order,
        showfliers=False,
        palette='tab20'
    )
    
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=6, linespacing=1.7)
    plt.xlabel("Pseudobulk (sorted by #cells)", fontsize=12, fontweight='bold')
    plt.ylabel(f"{count_col} per Cell", fontsize=12)
    plt.title(f"{count_col} per Cell by Pseudobulk and Annotation", fontsize=12)
    plt.legend(title="Annotation", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=6)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=250)
    if show:
        plt.show()
    else:
        plt.close()


def plot_reads_per_cell_by_celltype_and_stage(
    df,
    count_col,
    min_cells_per_group=10,
    figsize=(18, 12),
    save_path=None,
    show=True
):
    df_reads = df.dropna(subset=['annotation', 'stage_dpf']).copy()
    df_reads['stage_dpf_num'] = df_reads['stage_dpf'].astype(float)

    ct_stage_counts = df_reads.groupby(['annotation', 'stage_dpf_num']).size().reset_index(name='count')
    good = ct_stage_counts[ct_stage_counts['count'] >= min_cells_per_group]
    df_reads = df_reads.merge(
        good[['annotation', 'stage_dpf_num']],
        on=['annotation', 'stage_dpf_num'],
        how='inner'
    )

    # Sort cell types alphabetically
    sorted_cell_types = sorted(df_reads['annotation'].unique())
    sorted_stages = sorted(df_reads['stage_dpf_num'].unique())

    n_stages = len(sorted_stages)
    colors = sns.color_palette('Spectral', n_stages)
    stage_colors = dict(zip(sorted_stages, colors))

    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df_reads,
        x='annotation',
        y=count_col,
        hue='stage_dpf_num',
        hue_order=sorted_stages,
        order=sorted_cell_types,
        showfliers=False,
        palette=stage_colors
    )

    plt.xlabel('Cell Type (alphabetical)')
    plt.ylabel(count_col)
    plt.title(f'{count_col} per Cell by Cell Type and Stage')
    plt.legend(title='Stage (dpf)', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()
