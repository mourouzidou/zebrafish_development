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







def plot_pseudobulk_distributions(df, count_col, dataset_name, max_pseudobulks=None, save_path=None, show=True):
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
    
    ax.set_xticklabels(x_labels, rotation=65, ha='right', fontsize=6, linespacing=1.7)
    plt.xlabel("Pseudobulk (sorted by #cells)", fontsize=14, fontweight='bold')
    plt.ylabel(f"{count_col} per Cell", fontsize=16)
    plt.title(f"{dataset_name}:  {count_col} per Cell by Pseudobulk", fontsize=20)
    plt.legend(title="Annotation", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=6)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=250)
    if show:
        plt.show()
    else:
        plt.close()
import os, numpy as np, matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

def plot_mean_accessibility_distributions_per_stage(
    df: pd.DataFrame,
    *,
    group_by: str = "stage",           # "stage" -> boxes per celltype at each stage
                                       # "celltype" -> boxes per stage at each celltype
    title: str = "Pseudobulk distributions",
    y_label: str = "Signal",
    save_path: str | None = None,
    show: bool = True,
    figsize=(18, 10),
    palette: str = "tab20",
    showfliers: bool = False,
    stage_filter=None,
    celltype_filter=None,
):
    # parse "<stage>_<celltype>"
    meta = []
    for c in df.columns:
        if "_" in c:
            s, ct = c.split("_", 1)
            meta.append((float(s), ct, c))
    if not meta:
        return None

    # filters
    if stage_filter is not None:
        sf = set(map(float, stage_filter))
        meta = [t for t in meta if t[0] in sf]
    if celltype_filter is not None:
        cf = set(celltype_filter)
        meta = [t for t in meta if t[1] in cf]

    stages = sorted({s for s, _, _ in meta}, key=float)
    celltypes = []
    for _, ct, _ in meta:
        if ct not in celltypes:
            celltypes.append(ct)

    # mapping per x-axis key
    if group_by == "stage":
        x_keys = stages
        sub_keys = celltypes
        by = {s: {} for s in stages}
        for s, ct, col in meta: by[s][ct] = col
    else:
        x_keys = celltypes
        sub_keys = [str(s) for s in stages]
        by = {ct: {} for ct in celltypes}
        for s, ct, col in meta: by[ct][str(s)] = col

    # colors for sub-groups (consistent across x)
    cmap = plt.get_cmap(palette)(np.linspace(0, 1, len(sub_keys)))
    sub_color = {k: cmap[i] for i, k in enumerate(sub_keys)}

    fig, ax = plt.subplots(figsize=figsize)
    width_total = 0.8
    box_w = width_total / max(len(sub_keys), 1)
    offsets = np.linspace(-width_total/2 + box_w/2, width_total/2 - box_w/2, len(sub_keys))

    for i, xk in enumerate(x_keys):
        data, pos, cols = [], [], []
        for j, sk in enumerate(sub_keys):
            col = by[xk].get(sk) if group_by == "stage" else by[xk].get(sk)
            if col is None: 
                continue
            v = df[col].to_numpy()             # all peaks
            data.append(v)
            pos.append(i + offsets[j])
            cols.append(sub_color[sk])

        bp = ax.boxplot(data, positions=pos, widths=box_w*0.9,
                        showfliers=showfliers, patch_artist=True)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_edgecolor("black"); patch.set_linewidth(0.6)
        for k in ("whiskers","caps","medians"):
            for line in bp[k]: line.set_color("black"); line.set_linewidth(0.6)

    ax.set_xlabel("Stage (dpf)" if group_by=="stage" else "Cell type")
    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(x_keys)))
    ax.set_xticklabels([str(x) for x in x_keys], rotation=0 if group_by=="stage" else 25, ha="right")
    ax.set_title(title)

    handles = [Patch(facecolor=sub_color[k], edgecolor="black", label=k) for k in sub_keys]
    ax.legend(handles=handles, title=("Cell type" if group_by=="stage" else "Stage (dpf)"),
              bbox_to_anchor=(1.02,1), loc="upper left")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show: plt.show()
    else: plt.close(fig)
    return save_path

def plot_reads_per_cell_by_celltype_and_stage_lifelong(
    df,
    count_col,
    min_cells_per_group=10,
    figsize=(18, 12),
    save_path=None,
    show=True,
    group_by='celltype'  # 'celltype' or 'stage'
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
    
    sorted_cell_types = sorted(df_reads['annotation'].unique())
    sorted_stages = sorted(df_reads['stage_dpf_num'].unique())
    
    plt.figure(figsize=figsize)
    
    if group_by == 'celltype':
        n_stages = len(sorted_stages)
        colors = sns.color_palette('Spectral', n_stages)
        stage_colors = dict(zip(sorted_stages, colors))
        
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
        plt.legend(title='Stage (dpf)', bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.xticks(rotation=25, ha='right')
        
    else:  # group_by == 'stage'
        n_celltypes = len(sorted_cell_types)
        colors = sns.color_palette('tab20', n_celltypes)
        celltype_colors = dict(zip(sorted_cell_types, colors))
        
        sns.boxplot(
            data=df_reads,
            x='stage_dpf_num',
            y=count_col,
            hue='annotation',
            hue_order=sorted_cell_types,
            order=sorted_stages,
            showfliers=False,
            palette=celltype_colors
        )
        plt.xlabel('Stage (dpf)')
        plt.legend(title='Cell Type', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.ylabel(count_col)
    plt.title(f'{count_col} per Cell by {"Cell Type and Stage" if group_by == "celltype" else "Stage and Cell Type"} at least {min_cells_per_group} cells')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()


def plot_pseudobulk_cell_counts(atac_cells_to_psd, rna_cells_to_psd):
    def normalize(pseudobulk_name):
        parts = pseudobulk_name.split('_')
        if len(parts) > 1 and parts[0] in ['150', '210']:
            return 'adult_' + '_'.join(parts[1:])
        return pseudobulk_name

    def sort_key(name):
        parts = name.split('_')
        if len(parts) > 1:
            try:
                if parts[0] in ['1.5']:
                    return (1.5, parts[0])
                elif parts[0] == 'adult':
                    return (999, parts[0])
                else:
                    return (float(parts[0]), parts[0])
            except ValueError:
                return (1000, parts[0])
        return (1001, name)

    # Normalize pseudobulk names
    atac_norm = atac_cells_to_psd.copy()
    atac_norm['pseudobulk'] = atac_norm['pseudobulk'].apply(normalize)

    rna_norm = rna_cells_to_psd.copy()
    rna_norm['pseudobulk'] = rna_norm['pseudobulk'].apply(normalize)

    # Count cells per pseudobulk
    atac_counts = atac_norm['pseudobulk'].value_counts().rename('ATAC').reset_index()
    atac_counts.columns = ['pseudobulk', 'ATAC']

    rna_counts = rna_norm['pseudobulk'].value_counts().rename('RNA').reset_index()
    rna_counts.columns = ['pseudobulk', 'RNA']

    # Merge and prepare plot data
    merged = pd.merge(atac_counts, rna_counts, on='pseudobulk', how='outer').fillna(0)
    sorted_names = sorted(merged['pseudobulk'].unique(), key=sort_key)

    plot_df = merged.melt(id_vars='pseudobulk', value_vars=['ATAC', 'RNA'],
                          var_name='Modality', value_name='Cell count')
    plot_df['pseudobulk'] = pd.Categorical(plot_df['pseudobulk'], categories=sorted_names, ordered=True)

    # Plotting
    plt.figure(figsize=(min(40, len(sorted_names) * 0.7), 8))
    sns.barplot(data=plot_df, x='pseudobulk', y='Cell count', hue='Modality')
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.xlabel('Pseudobulk')
    plt.ylabel('Cell count')
    plt.title('Cell Population per Pseudobulk: ATAC vs RNA (150_ and 210_ merged as adult_)')
    plt.legend(title='Modality')
    plt.tight_layout()
    plt.show()



def lineplot_mean_std_by_stage(df, signal_col='peak_region_fragments', stage_col='stage_dpf', celltype_col='annotation'):
    
    # Group and summarize
    summary_df = df.groupby([celltype_col, stage_col]).agg(
        mean_signal=(signal_col, 'mean'),
        std_signal=(signal_col, 'std'),
        n=(signal_col, 'count')
    ).reset_index()

    summary_df['sem'] = summary_df['std_signal'] / summary_df['n']**0.5

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True)

    # Plot mean
    sns.lineplot(
        data=summary_df,
        x=stage_col, y='mean_signal',
        hue=celltype_col, marker='o', ax=axes[0]
    )
    axes[0].set_title("Mean Signal per Cell Type Across Stages")
    axes[0].set_xlabel("Stage (dpf)")
    axes[0].set_ylabel("Mean " + signal_col)
    axes[0].tick_params(axis='x', rotation=45)

    # Plot std
    sns.lineplot(
        data=summary_df,
        x=stage_col, y='std_signal',
        hue=celltype_col, marker='o', ax=axes[1]
    )
    axes[1].set_title("STD of Signal per Cell Type Across Stages")
    axes[1].set_xlabel("Stage (dpf)")
    axes[1].set_ylabel("Standard Deviation of " + signal_col)
    axes[1].tick_params(axis='x', rotation=45)

    # Adjust legend
    axes[1].legend_.remove()
    axes[0].legend(title='Cell type', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_mean_vs_std_by_celltype_and_stage(
    df, 
    signal_col, 
    celltype_col='annotation', 
    stage_col='stage_dpf', 
    modality_label='ATAC'
):
    def stage_sort_key(stage):
        return 1.5 if stage == 1.5 else float(stage) if isinstance(stage, (int, float)) else 1000

    stats_df = df.groupby([celltype_col, stage_col])[signal_col].agg(['mean', 'std', 'count']).reset_index()
    stats_df.columns = ['cell_type', 'stage_dpf', 'mean_signal', 'std_signal', 'n_cells']
    stats_df['std_signal'] = stats_df['std_signal'].fillna(0)
    stats_df['stage_dpf_sorted'] = stats_df['stage_dpf'].apply(stage_sort_key)
    stats_df = stats_df.sort_values('stage_dpf_sorted').drop('stage_dpf_sorted', axis=1)

    cell_types = sorted(stats_df['cell_type'].unique())
    stages = sorted(stats_df['stage_dpf'].unique(), key=stage_sort_key)
    n_cell_types = len(cell_types)
    n_cols = min(4, n_cell_types)
    n_rows = (n_cell_types + n_cols - 1) // n_cols
    colors = plt.cm.tab10(np.linspace(0, 1, len(stages)))
    stage_color_map = dict(zip(stages, colors))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_cell_types == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, cell_type in enumerate(cell_types):
        ax = axes[idx]
        cell_data = stats_df[stats_df['cell_type'] == cell_type]
        for _, row in cell_data.iterrows():
            stage = row['stage_dpf']
            mean_val = row['mean_signal']
            std_val = row['std_signal']
            ax.scatter(mean_val, std_val,
                       color=stage_color_map[stage],
                       s=100, alpha=0.7, edgecolors='black', linewidth=1,
                       label=f'{stage} dpf (n={row["n_cells"]})')
            ax.annotate(f'{stage}', (mean_val, std_val),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left')

        ax.set_xlabel(f'Mean {modality_label} Signal')
        ax.set_ylabel('Standard Deviation')
        ax.set_title(f'{cell_type}', fontsize=11, weight='bold')
        ax.grid(True, alpha=0.3)

        if len(stages) <= 8:
            ax.legend(fontsize=8, loc='best', framealpha=0.8)

        if len(cell_data) > 0:
            x_margin = (cell_data['mean_signal'].max() - cell_data['mean_signal'].min()) * 0.1
            y_margin = (cell_data['std_signal'].max() - cell_data['std_signal'].min()) * 0.1
            ax.set_xlim(cell_data['mean_signal'].min() - x_margin, cell_data['mean_signal'].max() + x_margin)
            ax.set_ylim(max(0, cell_data['std_signal'].min() - y_margin), cell_data['std_signal'].max() + y_margin)

    for idx in range(n_cell_types, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'{modality_label} | Mean vs Std of Signal by Cell Type and Stage',
                 fontsize=14, weight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

import os, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# --- small helpers for smoothed 2D hist contours ---
def _gaussian_kernel1d(sigma: float, radius: int | None = None):
    if sigma <= 0:
        return np.array([1.0], dtype=float)
    if radius is None:
        radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k

def _gaussian_smooth2d(arr: np.ndarray, sigma: float = 1.0):
    if sigma <= 0:
        return arr
    k = _gaussian_kernel1d(sigma)
    tmp = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), axis=1, arr=arr)
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), axis=0, arr=tmp)
    return out
# ---------------------------------------------------

def plot_pairwise_stage_scatter_per_celltype(
    df: pd.DataFrame,
    outdir: str = "plots/atac_pairwise",
    title_prefix: str = "ATAC",
    sample: int | None = 200_000,       # downsample points per panel (None = all)
    robust_limit: float = 0.999,         # cap axes at this upper quantile (common per celltype)
    dpi: int = 200,
    save: bool = True,
    show: bool = False,
    *,
    label: str = "",                     # e.g. "Quantile normalized"
    filename_prefix: str = "",           # e.g. "qn_"
    subdir: str | None = None,           # e.g. "quantile_normalized"
    # density controls
    density: bool = True,
    density_method: str = "hist",        # "hist" (smoothed 2D hist) or "kde"
    density_bins: int = 80,              # for "hist": number of bins per axis
    density_sigma: float = 2.0,          # for "hist": Gaussian smooth (in bins)
    density_levels: tuple[float, ...] | None = (0.60, 0.80, 0.90, 0.97),  # quantiles of density
    kde_bandwidth: float = 0.25,         # for "kde": Gaussian bandwidth (in data units)
    kde_grid: int = 200                  # for "kde": grid resolution per axis
):
    """
    df columns must be "<stage>_<celltype>" (e.g., "14_teeth", "60_gill progenitor 1").
    For each celltype, create a figure with all pairwise stage comparisons (scatter + y=x).
    Subplots are ordered by descending |stage2 - stage1|.
    Optional density contours via smoothed 2D histogram or KDE.
    """
    # Resolve output directory
    save_dir = os.path.join(outdir, subdir) if (save and subdir) else outdir
    if save:
        os.makedirs(save_dir, exist_ok=True)

    # parse columns -> (stage, celltype)
    parsed = []
    for col in df.columns:
        if "_" in col:
            stage, celltype = col.split("_", 1)   # first "_" separates stage from celltype
            parsed.append((stage, celltype, col))
    if not parsed:
        raise ValueError("No columns matched the '<stage>_<celltype>' pattern.")

    # group columns by celltype
    by_celltype: dict[str, dict[str, str]] = {}
    for stage, celltype, col in parsed:
        by_celltype.setdefault(celltype, {})[stage] = col

    # label pieces
    label_clean = label.strip()
    label_for_title = f" — {label_clean}" if label_clean else ""
    label_for_file  = ("_" + re.sub(r"[^A-Za-z0-9._-]+", "_", label_clean)) if label_clean else ""

    # helpers
    def _stage_key(s):
        try:    return float(s)
        except: return s

    def _pair_distance(p):
        a, b = p
        try:
            return abs(float(a) - float(b))
        except:
            return -1.0  # non-numeric stages go last

    for celltype, stage2col in by_celltype.items():
        stages = list(stage2col.keys())
        if len(stages) < 2:
            continue  # nothing to compare

        stages = sorted(stages, key=_stage_key)
        pairs = sorted(combinations(stages, 2), key=_pair_distance, reverse=True)

        # robust, common axis limits per celltype
        vals = df[[stage2col[s] for s in stages]].to_numpy()
        upper = np.quantile(vals, robust_limit)
        lower = 0.0  # log2(CPM+1) should be >= 0
        lims = (float(lower), float(upper))

        n = len(pairs)
        ncols = min(4, max(1, math.ceil(math.sqrt(n))))
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(4.4*ncols, 4.2*nrows), squeeze=False)
        axes = axes.ravel()

        for ax, (s1, s2) in zip(axes, pairs):
            x = df[stage2col[s1]].to_numpy()
            y = df[stage2col[s2]].to_numpy()

            if sample is not None and sample < len(x):
                idx = np.random.choice(len(x), size=sample, replace=False)
                x = x[idx]; y = y[idx]

            # scatter
            ax.scatter(x, y, s=2, alpha=0.25, linewidths=0)

            # y = x reference
            ax.plot(lims, lims, linestyle="--", linewidth=1.2, color="red")

            # --- density overlay ---
            if density:
                if density_method == "hist":
                    H, xedges, yedges = np.histogram2d(x, y, bins=density_bins, range=[lims, lims])
                    H = _gaussian_smooth2d(H, sigma=density_sigma)
                    pos = H[H > 0]
                    levels = [np.quantile(pos, q) for q in density_levels] if (density_levels and pos.size) else None
                    if levels:
                        X = 0.5 * (xedges[:-1] + xedges[1:])
                        Y = 0.5 * (yedges[:-1] + yedges[1:])
                        ax.contour(X, Y, H.T, levels=levels, linewidths=1.0, colors="black")
                elif density_method == "kde":
                    try:
                        from sklearn.neighbors import KernelDensity
                        gx = np.linspace(lims[0], lims[1], kde_grid)
                        gy = np.linspace(lims[0], lims[1], kde_grid)
                        XX, YY = np.meshgrid(gx, gy)
                        grid = np.c_[XX.ravel(), YY.ravel()]
                        kde = KernelDensity(bandwidth=kde_bandwidth, kernel="gaussian").fit(np.c_[x, y])
                        ZZ = np.exp(kde.score_samples(grid)).reshape(XX.shape)
                        pos = ZZ[ZZ > 0]
                        levels = [np.quantile(pos, q) for q in density_levels] if (density_levels and pos.size) else None
                        if levels:
                            ax.contour(XX, YY, ZZ, levels=levels, linewidths=1.0, colors="black")
                    except Exception as e:
                        # fallback to histogram if sklearn not available or KDE fails
                        H, xedges, yedges = np.histogram2d(x, y, bins=density_bins, range=[lims, lims])
                        H = _gaussian_smooth2d(H, sigma=density_sigma)
                        pos = H[H > 0]
                        levels = [np.quantile(pos, q) for q in density_levels] if (density_levels and pos.size) else None
                        if levels:
                            X = 0.5 * (xedges[:-1] + xedges[1:])
                            Y = 0.5 * (yedges[:-1] + yedges[1:])
                            ax.contour(X, Y, H.T, levels=levels, linewidths=1.0, colors="black")
            # -----------------------

            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_xlabel(f"{s1}")
            ax.set_ylabel(f"{s2}")

            if x.size > 1 and y.size > 1:
                r = np.corrcoef(x, y)[0, 1]
                ax.text(0.04, 0.96, f"r={r:.2f}", transform=ax.transAxes, va="top")

            ax.set_title(f"{s1} vs {s2}", fontsize=9)

        # turn off extra axes
        for ax in axes[len(pairs):]:
            ax.axis("off")

        fig.suptitle(f"{title_prefix}: {celltype}{label_for_title}", y=0.995)
        fig.tight_layout()

        if save:
            safe_cell = re.sub(r"[^A-Za-z0-9._-]+", "_", celltype)
            fname = f"{filename_prefix}{safe_cell}{label_for_file}_pairwise_scatter.png"
            out_path = os.path.join(save_dir, fname)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved {out_path}")

        if show:
            plt.show()

        plt.close(fig)


# per stage scatterplot 
def plot_pairwise_celltypes_per_stage(
    df: pd.DataFrame,
    outdir: str = "plots/atac_pairwise_by_stage",
    title_prefix: str = "ATAC",
    sample: int | None = 200_000,
    robust_limit: float = 0.999,
    dpi: int = 200,
    save: bool = True,
    show: bool = False,
    *,
    label: str = "",
    filename_prefix: str = "",
    subdir: str | None = None,
    # density controls
    density: bool = True,
    density_method: str = "hist",
    density_bins: int = 80,
    density_sigma: float = 2.0,
    density_levels: tuple[float, ...] | None = (0.60, 0.80, 0.90, 0.97),
    kde_bandwidth: float = 0.25,
    kde_grid: int = 200,
    # layout controls
    figsize_scale: float = 1.0,   # multiplies base size
    hspace: float = 0.45,         # vertical gap between subplots
    wspace: float = 0.35,         # horizontal gap between subplots
    suptitle_size: int = 16,
    suptitle_y: float = 0.985,    # suptitle vertical position
    top: float = 0.90             # leave room for suptitle
):
    save_dir = os.path.join(outdir, subdir) if (save and subdir) else outdir
    if save:
        os.makedirs(save_dir, exist_ok=True)

    parsed = []
    for col in df.columns:
        if "_" in col:
            stage, celltype = col.split("_", 1)
            parsed.append((stage, celltype, col))
    if not parsed:
        raise ValueError("No columns matched '<stage>_<celltype>'.")

    by_stage: dict[str, dict[str, str]] = {}
    for stage, celltype, col in parsed:
        by_stage.setdefault(stage, {})[celltype] = col

    label_clean = label.strip()
    label_for_title = f" — {label_clean}" if label_clean else ""
    label_for_file  = ("_" + re.sub(r"[^A-Za-z0-9._-]+", "_", label_clean)) if label_clean else ""

    def _stage_key(s):
        try: return float(s)
        except: return s

    for stage in sorted(by_stage.keys(), key=_stage_key):
        ct2col = by_stage[stage]
        celltypes = sorted(ct2col.keys())
        if len(celltypes) < 2:
            continue

        vals = df[[ct2col[ct] for ct in celltypes]].to_numpy()
        upper = np.quantile(vals, robust_limit)
        lims = (0.0, float(upper))

        pairs = list(combinations(celltypes, 2))
        n = len(pairs)
        ncols = min(4, max(1, math.ceil(math.sqrt(n))))
        nrows = math.ceil(n / ncols)

        # --- bigger canvas + explicit gaps to avoid overlap ---
        fig_w = (4.4 * ncols) * figsize_scale
        fig_h = (4.0 * nrows) * figsize_scale + 0.8  # extra headroom for suptitle
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
        axes = axes.ravel()

        for ax, (ct1, ct2) in zip(axes, pairs):
            x = df[ct2col[ct1]].to_numpy()
            y = df[ct2col[ct2]].to_numpy()

            if sample is not None and sample < len(x):
                idx = np.random.choice(len(x), size=sample, replace=False)
                x = x[idx]; y = y[idx]

            ax.scatter(x, y, s=2, alpha=0.25, linewidths=0)
            ax.plot(lims, lims, linestyle="--", linewidth=1.2, color="red")

            if density:
                if density_method == "hist":
                    H, xedges, yedges = np.histogram2d(x, y, bins=density_bins, range=[lims, lims])
                    H = _gaussian_smooth2d(H, sigma=density_sigma)
                    pos = H[H > 0]
                    levels = [np.quantile(pos, q) for q in density_levels] if (density_levels and pos.size) else None
                    if levels:
                        X = 0.5 * (xedges[:-1] + xedges[1:])
                        Y = 0.5 * (yedges[:-1] + yedges[1:])
                        ax.contour(X, Y, H.T, levels=levels, linewidths=1.0, colors="black")
                elif density_method == "kde":
                    try:
                        from sklearn.neighbors import KernelDensity
                        gx = np.linspace(lims[0], lims[1], kde_grid)
                        gy = np.linspace(lims[0], lims[1], kde_grid)
                        XX, YY = np.meshgrid(gx, gy)
                        grid = np.c_[XX.ravel(), YY.ravel()]
                        kde = KernelDensity(bandwidth=kde_bandwidth, kernel="gaussian").fit(np.c_[x, y])
                        ZZ = np.exp(kde.score_samples(grid)).reshape(XX.shape)
                        pos = ZZ[ZZ > 0]
                        levels = [np.quantile(pos, q) for q in density_levels] if (density_levels and pos.size) else None
                        if levels:
                            ax.contour(XX, YY, ZZ, levels=levels, linewidths=1.0, colors="black")
                    except Exception:
                        H, xedges, yedges = np.histogram2d(x, y, bins=density_bins, range=[lims, lims])
                        H = _gaussian_smooth2d(H, sigma=density_sigma)
                        pos = H[H > 0]
                        levels = [np.quantile(pos, q) for q in density_levels] if (density_levels and pos.size) else None
                        if levels:
                            X = 0.5 * (xedges[:-1] + xedges[1:])
                            Y = 0.5 * (yedges[:-1] + yedges[1:])
                            ax.contour(X, Y, H.T, levels=levels, linewidths=1.0, colors="black")

            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_xlabel(ct1); ax.set_ylabel(ct2)

            if x.size > 1 and y.size > 1:
                r = np.corrcoef(x, y)[0, 1]
                ax.text(0.04, 0.96, f"r={r:.2f}", transform=ax.transAxes, va="top")

            ax.set_title(f"{ct1} vs {ct2}", fontsize=10)

        for ax in axes[len(pairs):]:
            ax.axis("off")

        # layout & title (no overlap)
        fig.tight_layout(pad=0.6)
        fig.subplots_adjust(top=top, hspace=hspace, wspace=wspace)
        fig.suptitle(f"{title_prefix}: stage {stage}{label_for_title}",
                    y=suptitle_y, fontsize=suptitle_size)

        if save:
            safe_stage = re.sub(r"[^A-Za-z0-9._-]+", "_", str(stage))
            fname = f"{filename_prefix}stage_{safe_stage}{label_for_file}_pairwise_celltypes.png"
            out_path = os.path.join(save_dir, fname)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved {out_path}")

        if show:
            plt.show()

        plt.close(fig)
