import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import quantile_transform
from Bio import SeqIO
from pathlib import Path
import seaborn as sns
import re
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import norm
import os



def one_hot_encode(sequence):
    seq_array = np.array(list(sequence))[:, None]
    bases = np.array(list('ACGT'))
    one_hot = (seq_array == bases).astype(np.float32)
    n_mask = seq_array.flatten() == 'N'
    one_hot[n_mask] = 0.25
    return one_hot

##__________________ATAC data preprocessing____________________

def quantile_normalize(x, axis=0):
    sorted_x = np.sort(x, axis=axis)
    meanx = np.mean(sorted_x, axis=axis-1)
    xqtrl = meanx[np.argsort(np.argsort(x, axis=axis), axis=axis)]
    return xqtrl



def extract_stage_number(stage_str):
    match = re.search(r'(\d+)', str(stage_str))
    return int(match.group(1)) if match else 0

def process_trajectory_data(df, cell_type_col, stage_col):
    stage_celltype_counts = df.groupby([cell_type_col, stage_col]).size().reset_index(name='count')
    stage_celltype_counts['stage_numeric'] = stage_celltype_counts[stage_col].apply(extract_stage_number)
    heatmap_data = stage_celltype_counts.pivot(index=cell_type_col, columns='stage_numeric', values='count')
    heatmap_data = heatmap_data.fillna(0)
    heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)
    return heatmap_data

def get_earliest_stage_unified(cell_type, atac_data, rna_data):
    earliest = float('inf')
    if atac_data is not None and cell_type in atac_data.index:
        for stage in sorted(atac_data.columns):
            if atac_data.loc[cell_type, stage] > 0:
                earliest = min(earliest, int(stage))
                break
    if rna_data is not None and cell_type in rna_data.index:
        for stage in sorted(rna_data.columns):
            if rna_data.loc[cell_type, stage] > 0:
                earliest = min(earliest, int(stage))
                break
    return earliest if earliest != float('inf') else 999

def plot_trajectories(ax, cell_types, atac_data, rna_data, all_stages, title, plot_type):
    if len(cell_types) == 0:
        ax.text(0.5, 0.5, 'No cell types in this category', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        return
    
    if plot_type == 'shared':
        cell_types_sorted = sorted(cell_types, 
                                  key=lambda x: get_earliest_stage_unified(x, atac_data, rna_data))
    elif plot_type == 'atac_only':
        cell_types_sorted = sorted(cell_types, 
                                  key=lambda x: get_earliest_stage_unified(x, atac_data, None))
    else:
        cell_types_sorted = sorted(cell_types, 
                                  key=lambda x: get_earliest_stage_unified(x, None, rna_data))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(cell_types_sorted)))
    color_map = {cell_type: colors[i] for i, cell_type in enumerate(cell_types_sorted)}
    
    y_offset = 0
    max_counts = []
    if atac_data is not None:
        max_counts.append(atac_data.values.max())
    if rna_data is not None:
        max_counts.append(rna_data.values.max())
    max_count = max(max_counts) if max_counts else 1000
    
    y_spacing = max(500, max_count * 0.12)
    cell_type_positions = {}
    
    for cell_type in cell_types_sorted:
        cell_color = color_map[cell_type]
        max_y_for_cell = 0
    
        if atac_data is not None and cell_type in atac_data.index:
            atac_y_values = []
            for stage in all_stages:
                if stage in atac_data.columns:
                    atac_y_values.append(atac_data.loc[cell_type, stage])
                else:
                    atac_y_values.append(0)
            
            atac_y_values_offset = [val + y_offset for val in atac_y_values]
            max_y_for_cell = max(max_y_for_cell, max(atac_y_values))
            
            ax.plot(all_stages, atac_y_values_offset, marker='o', linewidth=2.5, 
                   markersize=5, color=cell_color, alpha=0.8, linestyle='-',
                   label=f'{cell_type} (ATAC)' if plot_type == 'shared' else cell_type)
            
            non_zero_stages = [stage for stage, count in zip(all_stages, atac_y_values) if count > 0]
            non_zero_counts = [count + y_offset for count in atac_y_values if count > 0]
            if non_zero_stages:
                ax.scatter(non_zero_stages, non_zero_counts, color=cell_color, 
                          s=50, zorder=5, alpha=0.9, marker='o', edgecolor='white', linewidth=1)
        
        if rna_data is not None and cell_type in rna_data.index:
            rna_y_values = []
            for stage in all_stages:
                if stage in rna_data.columns:
                    rna_y_values.append(rna_data.loc[cell_type, stage])
                else:
                    rna_y_values.append(0)
            
            rna_y_values_offset = [val + y_offset for val in rna_y_values]
            max_y_for_cell = max(max_y_for_cell, max(rna_y_values))
            
            ax.plot(all_stages, rna_y_values_offset, marker='s', linewidth=2.5, 
                   markersize=5, color=cell_color, alpha=0.8, linestyle='--',
                   label=f'{cell_type} (RNA)' if plot_type == 'shared' else cell_type)
            
            non_zero_stages = [stage for stage, count in zip(all_stages, rna_y_values) if count > 0]
            non_zero_counts = [count + y_offset for count in rna_y_values if count > 0]
            if non_zero_stages:
                ax.scatter(non_zero_stages, non_zero_counts, color=cell_color, 
                          s=50, zorder=5, alpha=0.9, marker='s', edgecolor='white', linewidth=1)
        
        cell_type_positions[cell_type] = y_offset
        y_offset += max_y_for_cell + y_spacing
    
    for cell_type, y_pos in cell_type_positions.items():
        ax.text(max(all_stages) + 0.3, y_pos + y_spacing*0.1, cell_type, 
                va='bottom', ha='left', fontsize=9, fontweight='bold',
                color=color_map[cell_type])
    
    ax.set_xlabel('Developmental Stage (hpf)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cell Count', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(all_stages)
    ax.set_xticklabels([f'{stage}hpf' for stage in all_stages], rotation=45, ha='right')
    ax.set_xlim(min(all_stages)-0.5, max(all_stages)+1.5)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    for i, y_pos in enumerate(cell_type_positions.values()):
        if i > 0:
            ax.axhline(y=y_pos - y_spacing/2, color='gray', linestyle=':', alpha=0.3)

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
    df_raw, df_quant, title_prefix="Data", cell_type_filter=None, pseudobulk_counts=None, save_dir=None):
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
            plt.close()
        else:
            plt.show()

    if save_dir:
        import os
        prefix = title_prefix.lower().replace(" ", "_")
        fname_raw = f"{save_dir}/{prefix}_raw_log2_boxplot.png"
        fname_quant = f"{save_dir}/{prefix}_quantile_log2_boxplot.png"
    else:
        fname_raw = fname_quant = None

    _plot(df_raw, "Raw", use_single_color=False, fname=fname_raw)
    _plot(df_quant, "Quantile Normalized", use_single_color=True, fname=fname_quant)


from Bio import SeqIO
from pathlib import Path

def extract_centered_sequences(
    df, fasta_dir, expansion_length=None, save_dir="../../data/embryo/processed", df_name="atac"
):
    chr_seqs = {}
    sequences = []
    chrom_ints = []

    for idx, row in df.iterrows():
        chrom_raw = str(row['chromosome'])
        chrom_str = chrom_raw.replace("chr", "")
        chrom = int(chrom_str) if chrom_str.isdigit() else chrom_str

        start, end = int(row['start']), int(row['end'])

        fasta_chrom_key = str(chrom)
        if fasta_chrom_key not in chr_seqs:
            fasta_path = Path(fasta_dir) / f"Danio_rerio.GRCz11.dna.chromosome.{fasta_chrom_key}.fa"
            record = next(SeqIO.parse(fasta_path, "fasta"))
            chr_seqs[fasta_chrom_key] = record.seq

        if expansion_length:
            center = (start + end) // 2
            half_len = expansion_length // 2
            start = max(0, center - half_len)
            end = center + half_len

        seq = chr_seqs[fasta_chrom_key][start:end]
        seq_str = str(seq)

        if expansion_length and len(seq_str) < expansion_length:
            pad_len = expansion_length - len(seq_str)
            seq_str += "N" * pad_len

        sequences.append(seq_str)
        chrom_ints.append(chrom)

    df = df.copy()
    df['sequence'] = sequences
    df['chromosome'] = chrom_ints  

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{df_name}_with_sequences_{expansion_length if expansion_length else 'original'}.csv"
        df.to_csv(save_dir / fname, index=False)

    return df



def cpm_normalize_sparse(df, value_col='Accessibility', cell_col='Cell', scale=1e6):
    total_per_cell = df.groupby(cell_col)[value_col].transform('sum')
    df = df.copy()
    df['CPM'] = df[value_col] / total_per_cell * scale
    return df


def aggregate_atac_to_pseudobulk(
    atac_data_df, 
    value_col='CPM', 
    pseudobulk_col='pseudobulk',
    peak_col='Peak', 
    aggfunc='mean'
):
    pseudobulk_matrix = (
        atac_data_df.groupby([peak_col, pseudobulk_col])[value_col]
        .agg(aggfunc)
        .unstack(fill_value=0)
    )
    pseudobulk_matrix.index.name = 'Peak'
    return pseudobulk_matrix

def extract_gene_info_from_gtf(gtf_path):
    records = []

    with open(gtf_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'gene':
                continue

            chrom = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            attributes = fields[8]

            attrs = {
                k: v.strip('"') for k, v in 
                [item.strip().split(' ')[:2] for item in attributes.strip(';').split(';') if item]
            }

            gene_id = attrs.get("gene_id")
            gene_name = attrs.get("gene_name")
            if not gene_id or not gene_name:
                continue

            tss = start if strand == '+' else end

            records.append({
                'chrom': chrom,
                'start': start,
                'end': end,
                'tss_position': tss,
                'gene_name': gene_name,
                'gene_id': gene_id
            })

    return pd.DataFrame(records)


def fix_chromosome_types(peaks_df, tss_df):
    peaks_fixed = peaks_df.copy()
    tss_fixed = tss_df.copy()
    
    peaks_fixed['chromosome'] = peaks_fixed['chromosome'].astype(str)
    tss_fixed['chrom'] = tss_fixed['chrom'].astype(str)
    
    return peaks_fixed, tss_fixed

from scipy.spatial import cKDTree

def annotate_peaks_with_genomic_info(
    peaks_df, gene_df, distance_threshold=1000,
    meta_cols=['Peak', 'chromosome', 'start', 'end', 'sequence', 'region_type', 'genomic_context']
):
    peaks_df, gene_df = fix_chromosome_types(peaks_df, gene_df)

    promoter_enhancer_labels = []
    intra_intergenic_labels = []

    for chrom in peaks_df['chromosome'].unique():
        peaks_chr = peaks_df[peaks_df['chromosome'] == chrom]
        genes_chr = gene_df[gene_df['chrom'] == chrom]

        if genes_chr.empty:
            promoter_enhancer_labels.extend(['enhancer'] * len(peaks_chr))
            intra_intergenic_labels.extend(['intergenic'] * len(peaks_chr))
            continue

        peak_centers = ((peaks_chr['start'] + peaks_chr['end']) // 2).values.reshape(-1, 1)
        tss_positions = genes_chr['tss_position'].values.reshape(-1, 1)

        # Annotate promoter/enhancer
        tree = cKDTree(tss_positions)
        distances, _ = tree.query(peak_centers, k=1)
        promoter_enhancer = np.where(distances <= distance_threshold, 'promoter', 'enhancer')
        promoter_enhancer_labels.extend(promoter_enhancer.tolist())

        # Annotate intragenic/intergenic (overlap)
        intragenic = []
        for _, peak in peaks_chr.iterrows():
            peak_start, peak_end = peak['start'], peak['end']
            overlapping = genes_chr[
                (genes_chr['start'] <= peak_end) & (genes_chr['end'] >= peak_start)
            ]
            if not overlapping.empty:
                intragenic.append('intragenic')
            else:
                intragenic.append('intergenic')
        intra_intergenic_labels.extend(intragenic)

    annotated = peaks_df.copy()
    annotated['region_type'] = promoter_enhancer_labels
    annotated['genomic_context'] = intra_intergenic_labels

    remaining_cols = [col for col in annotated.columns if col not in meta_cols]
    new_order = meta_cols + remaining_cols
    annotated = annotated[new_order]

    return annotated
def compute_mean_accessibility_by_region(
    atac_data_df, peak_region_df, region_col='region_type', value_col='Accessibility', cell_col='Cell', peak_col='Peak'
):
    
    # Map region type to atac_data_df
    peak_region = peak_region_df.set_index(peak_col)[region_col].to_dict()
    atac_data_df = atac_data_df.copy()
    atac_data_df[region_col] = atac_data_df[peak_col].map(peak_region)

    # Promoter counts
    promoter_counts = (
        atac_data_df[atac_data_df[region_col] == 'promoter']
        .groupby(cell_col)[value_col].mean()
        .rename('total_promoter_accessibility')
        .reset_index()
    )
    # Enhancer counts
    enhancer_counts = (
        atac_data_df[atac_data_df[region_col] == 'enhancer']
        .groupby(cell_col)[value_col].mean()
        .rename('total_enhancer_accessibility')
        .reset_index()
    )

    return promoter_counts, enhancer_counts


def summarize_peak_accessibility(
    atac_data_df, ann_df, atac_metadata_df
):
    # Map each peak to its region_type and genomic_context
    peak_region = ann_df.set_index('Peak')['region_type'].to_dict()
    peak_context = ann_df.set_index('Peak')['genomic_context'].to_dict()
    atac_data_df = atac_data_df.copy()
    atac_data_df['region_type'] = atac_data_df['Peak'].map(peak_region)
    atac_data_df['genomic_context'] = atac_data_df['Peak'].map(peak_context)
    
    # Pivot: Region type (promoter/enhancer)
    region_summary = (
        atac_data_df.pivot_table(
            index='Cell',
            columns='region_type',
            values='Accessibility',
            aggfunc='mean',
            fill_value=0
        )
        .rename_axis(None, axis=1)
        .add_prefix('total_')
        .add_suffix('_accessibility')
        .reset_index()
    )
    
    # Pivot: Genomic context (intragenic/intergenic)
    context_summary = (
        atac_data_df.pivot_table(
            index='Cell',
            columns='genomic_context',
            values='Accessibility',
            aggfunc='mean',
            fill_value=0
        )
        .rename_axis(None, axis=1)
        .add_prefix('total_')
        .add_suffix('_accessibility')
        .reset_index()
    )
    
    # Merge both summaries
    summary = region_summary.merge(context_summary, on='Cell', how='outer')
    
    # Add metadata (cell type and pseudobulk)
    cell_meta_cols = ['Cell', 'atac_cell_type', 'pseudobulk']
    cell_meta = atac_metadata_df.rename(columns={'atac_cell': 'Cell'})[cell_meta_cols]
    summary = summary.merge(cell_meta, on='Cell', how='left')
    
    return summary


def get_cell_metadata(atac_metadata_df):
    cell_meta = atac_metadata_df.rename(columns={'atac_cell': 'Cell'})[['Cell', 'atac_cell_type', 'pseudobulk']]
    return cell_meta

def summarize_accessibility(
    atac_data_df,
    atac_metadata_df,
    region_col='region_type',   # Or 'genomic_context'
    value_col='Accessibility',
    groupby='pseudobulk',       # Or 'atac_cell_type'
    normalize_by_num_peaks=True,
    min_cells=1,
    region_include=None,        
    region_exclude=None        
):
    if region_include is not None:
        atac_data_df = atac_data_df[atac_data_df[region_col].isin(region_include)]
    if region_exclude is not None:
        atac_data_df = atac_data_df[~atac_data_df[region_col].isin(region_exclude)]

    df_sum = (
        atac_data_df
        .groupby(['Cell', region_col])[value_col]
        .sum()
        .reset_index()
        .rename(columns={value_col: 'total_accessibility'})
    )

    if normalize_by_num_peaks:
        region_sizes = atac_data_df.groupby(region_col)['Peak'].nunique().to_dict()
        df_sum['mean_accessibility'] = df_sum.apply(
            lambda r: r['total_accessibility'] / region_sizes.get(r[region_col], 1), axis=1
        )
        value_for_plot = 'mean_accessibility'
    else:
        value_for_plot = 'total_accessibility'

    cell_meta = get_cell_metadata(atac_metadata_df)
    df_sum = df_sum.merge(cell_meta, on='Cell', how='left')

    group_counts = df_sum[groupby].value_counts()
    valid_groups = group_counts[group_counts >= min_cells].index.tolist()
    df_sum = df_sum[df_sum[groupby].isin(valid_groups)]

    return df_sum, value_for_plot

def create_pseudobulk_stats(
    atac_df, 
    atac_metadata_df, 
    value_column="Accessibility",
    norm_reads=False,
    norm_target_reads=1e6,
    min_cells=None,
):
    atac_df = atac_df.copy()
    celltype_lookup = dict(zip(atac_metadata_df['atac_cell'], atac_metadata_df['atac_cell_type']))
    # Extract stage, celltype, and pseudobulk columns
    atac_df['stage'] = atac_df['Cell'].str.extract(r'(\d{1,2})[a-zA-Z]', expand=False)
    atac_df['celltype'] = atac_df['Cell'].map(celltype_lookup)
    atac_df['stage_celltype'] = atac_df['stage'] + '_' + atac_df['celltype']

    if norm_reads:
        # Normalize values within each cell
        cell_sum = atac_df.groupby('Cell')[value_column].transform('sum')
        # Avoid division by zero
        cell_sum = cell_sum.replace(0, np.nan)
        atac_df[value_column + "_norm"] = (atac_df[value_column] / cell_sum) * norm_target_reads
        value_column_use = value_column + "_norm"
    else:
        value_column_use = value_column

    if min_cells is not None:
        pseudobulk_counts = atac_df.groupby('stage_celltype')['Cell'].nunique()
        valid_pseudobulks = pseudobulk_counts.index[pseudobulk_counts >= min_cells]
        atac_df = atac_df[atac_df['stage_celltype'].isin(valid_pseudobulks)]

    grouped = atac_df.groupby(['Peak', 'stage_celltype'])[value_column_use]
    mean_df = grouped.mean().unstack(fill_value=0)
    std_df = grouped.std().unstack(fill_value=0).fillna(0)
    return mean_df, std_df


##_______________RNA data preprocessing____________________

def extract_stage(cell_id, delimiter='h', pos=0):
    return str(cell_id).split(delimiter)[pos].replace('h', '')



def extract_stage_number(stage_str):
    match = re.search(r'(\d+)', str(stage_str))
    return int(match.group(1)) if match else 0

def process_trajectory_data(df, cell_type_col, stage_col):
    stage_celltype_counts = df.groupby([cell_type_col, stage_col]).size().reset_index(name='count')
    stage_celltype_counts['stage_numeric'] = stage_celltype_counts[stage_col].apply(extract_stage_number)
    heatmap_data = stage_celltype_counts.pivot(index=cell_type_col, columns='stage_numeric', values='count')
    heatmap_data = heatmap_data.fillna(0)
    heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)
    return heatmap_data

def get_earliest_stage_unified(cell_type, atac_data, rna_data):
    earliest = float('inf')
    if atac_data is not None and cell_type in atac_data.index:
        for stage in sorted(atac_data.columns):
            if atac_data.loc[cell_type, stage] > 0:
                earliest = min(earliest, int(stage))
                break
    if rna_data is not None and cell_type in rna_data.index:
        for stage in sorted(rna_data.columns):
            if rna_data.loc[cell_type, stage] > 0:
                earliest = min(earliest, int(stage))
                break
    return earliest if earliest != float('inf') else 999

def plot_trajectories(ax, cell_types, atac_data, rna_data, all_stages, title, plot_type):
    if len(cell_types) == 0:
        ax.text(0.5, 0.5, 'No cell types in this category', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        return
    
    if plot_type == 'shared':
        cell_types_sorted = sorted(cell_types, 
                                  key=lambda x: get_earliest_stage_unified(x, atac_data, rna_data))
    elif plot_type == 'atac_only':
        cell_types_sorted = sorted(cell_types, 
                                  key=lambda x: get_earliest_stage_unified(x, atac_data, None))
    else:
        cell_types_sorted = sorted(cell_types, 
                                  key=lambda x: get_earliest_stage_unified(x, None, rna_data))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(cell_types_sorted)))
    color_map = {cell_type: colors[i] for i, cell_type in enumerate(cell_types_sorted)}
    
    y_offset = 0
    max_counts = []
    if atac_data is not None:
        max_counts.append(atac_data.values.max())
    if rna_data is not None:
        max_counts.append(rna_data.values.max())
    max_count = max(max_counts) if max_counts else 1000
    
    y_spacing = max(500, max_count * 0.12)
    cell_type_positions = {}
    
    for cell_type in cell_types_sorted:
        cell_color = color_map[cell_type]
        max_y_for_cell = 0
    
        if atac_data is not None and cell_type in atac_data.index:
            atac_y_values = []
            for stage in all_stages:
                if stage in atac_data.columns:
                    atac_y_values.append(atac_data.loc[cell_type, stage])
                else:
                    atac_y_values.append(0)
            
            atac_y_values_offset = [val + y_offset for val in atac_y_values]
            max_y_for_cell = max(max_y_for_cell, max(atac_y_values))
            
            ax.plot(all_stages, atac_y_values_offset, marker='o', linewidth=2.5, 
                   markersize=5, color=cell_color, alpha=0.8, linestyle='-',
                   label=f'{cell_type} (ATAC)' if plot_type == 'shared' else cell_type)
            
            non_zero_stages = [stage for stage, count in zip(all_stages, atac_y_values) if count > 0]
            non_zero_counts = [count + y_offset for count in atac_y_values if count > 0]
            if non_zero_stages:
                ax.scatter(non_zero_stages, non_zero_counts, color=cell_color, 
                          s=50, zorder=5, alpha=0.9, marker='o', edgecolor='white', linewidth=1)
        
        if rna_data is not None and cell_type in rna_data.index:
            rna_y_values = []
            for stage in all_stages:
                if stage in rna_data.columns:
                    rna_y_values.append(rna_data.loc[cell_type, stage])
                else:
                    rna_y_values.append(0)
            
            rna_y_values_offset = [val + y_offset for val in rna_y_values]
            max_y_for_cell = max(max_y_for_cell, max(rna_y_values))
            
            ax.plot(all_stages, rna_y_values_offset, marker='s', linewidth=2.5, 
                   markersize=5, color=cell_color, alpha=0.8, linestyle='--',
                   label=f'{cell_type} (RNA)' if plot_type == 'shared' else cell_type)
            
            non_zero_stages = [stage for stage, count in zip(all_stages, rna_y_values) if count > 0]
            non_zero_counts = [count + y_offset for count in rna_y_values if count > 0]
            if non_zero_stages:
                ax.scatter(non_zero_stages, non_zero_counts, color=cell_color, 
                          s=50, zorder=5, alpha=0.9, marker='s', edgecolor='white', linewidth=1)
        
        cell_type_positions[cell_type] = y_offset
        y_offset += max_y_for_cell + y_spacing
    
    for cell_type, y_pos in cell_type_positions.items():
        ax.text(max(all_stages) + 0.3, y_pos + y_spacing*0.1, cell_type, 
                va='bottom', ha='left', fontsize=9, fontweight='bold',
                color=color_map[cell_type])
    
    ax.set_xlabel('Developmental Stage (hpf)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cell Count', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(all_stages)
    ax.set_xticklabels([f'{stage}hpf' for stage in all_stages], rotation=45, ha='right')
    ax.set_xlim(min(all_stages)-0.5, max(all_stages)+1.5)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    for i, y_pos in enumerate(cell_type_positions.values()):
        if i > 0:
            ax.axhline(y=y_pos - y_spacing/2, color='gray', linestyle=':', alpha=0.3)

def compute_pseudobulk_cutoffs_df(rna_data_matched, atac_metadata, marker_genes_dict=None, cutoff_mode='celltype'):
    metadata = atac_metadata.set_index('rna_matching_cell').loc[rna_data_matched.columns].reset_index(names='rna_matching_cell')
    pseudobulk_names = metadata['pseudobulk'].unique()
    results = []
    
    for pb in pseudobulk_names:
        celltype = metadata[metadata['pseudobulk'] == pb]['atac_cell_type'].iloc[0]
        cells_in_pb = metadata[metadata['pseudobulk'] == pb]['rna_matching_cell']
        same_type_cells = metadata[metadata['atac_cell_type'] == celltype]['rna_matching_cell']
        same_type_pseudobulks = metadata[metadata['atac_cell_type'] == celltype]['pseudobulk'].unique()

        if len(same_type_cells) == 0:
            continue

        if marker_genes_dict is not None:
            marker_genes = marker_genes_dict.get(pb, [])
            marker_genes = [g for g in marker_genes if g in rna_data_matched.index]
            genes_to_use = marker_genes
        else:
            genes_to_use = rna_data_matched.index
        
        mean_expression = rna_data_matched.loc[genes_to_use, cells_in_pb].mean(axis=1)
        mean_vec_reshaped = mean_expression.values.reshape(1, -1)

        if cutoff_mode == 'pseudobulk':
            # Compute distances from cells of the same type to the pseudobulk mean expression vector
            same_type_matrix = rna_data_matched.loc[genes_to_use, same_type_cells].T.values
            distances = cdist(same_type_matrix, mean_vec_reshaped, metric='euclidean').flatten()
            cutoff = distances.max()
        elif cutoff_mode == 'celltype':
            # Compute the mean expression vector of all pseudobulks with the same cell type
            mean_same_type_expression = rna_data_matched.loc[genes_to_use, metadata[metadata['pseudobulk'].isin(same_type_pseudobulks)]['rna_matching_cell']].mean(axis=1)
            mean_same_type_vec_reshaped = mean_same_type_expression.values.reshape(1, -1)
            # Compute distances from all same cell type cells to this mean vector
            same_type_matrix = rna_data_matched.loc[genes_to_use, same_type_cells].T.values
            distances = cdist(same_type_matrix, mean_same_type_vec_reshaped, metric='euclidean').flatten()
            cutoff = distances.max()
        else:
            raise ValueError("Invalid cutoff_mode. Use 'max_distance' or 'mean_distance'.")
        
        results.append({
            'pseudobulk': pb,
            'atac_cell_type': celltype,
            'cutoff': cutoff,
            'n_cells_in_pseudobulk': len(cells_in_pb),
            'n_cells_same_type': len(same_type_cells)
        })
    
    return pd.DataFrame(results)

def create_rna_mean(rna_data, atac_metadata_df, valid_pseudobulks):
    cell_to_pseudobulk = atac_metadata_df.set_index('rna_matching_cell')['pseudobulk'].to_dict()
    rna_data_psd = rna_data.rename(columns=cell_to_pseudobulk)
    rna_data_filtered = rna_data_psd.loc[:, rna_data_psd.columns.intersection(valid_pseudobulks)]
    return rna_data_filtered.groupby(level=0, axis=1).mean()

def create_rna_std(rna_data, atac_metadata_df, valid_pseudobulks):
    cell_to_pseudobulk = atac_metadata_df.set_index('rna_matching_cell')['pseudobulk'].to_dict()
    rna_data_psd = rna_data.rename(columns=cell_to_pseudobulk)
    rna_data_filtered = rna_data_psd.loc[:, rna_data_psd.columns.intersection(valid_pseudobulks)]
    return rna_data_filtered.groupby(level=0, axis=1).std()

def find_marker_genes(rna_data, metadata_df, grouping_column, 
                     cell_id_column='rna_matching_cell',
                     pvalue_threshold=0.05, log_fc_threshold=0.5):
    
    # Create cell-to-group mapping and rename columns
    cell_to_group = metadata_df.set_index(cell_id_column)[grouping_column].to_dict()
    rna_data_grouped = rna_data.rename(columns=cell_to_group)
    
    # Compute group-level statistics
    group_counts = rna_data_grouped.columns.value_counts()
    rna_data_mean_group = rna_data_grouped.groupby(by=rna_data_grouped.columns, axis=1).mean()
    rna_data_std_group = rna_data_grouped.groupby(by=rna_data_grouped.columns, axis=1).std()
    
    # Perform t-tests for each group
    genes = rna_data_mean_group.index
    groups = rna_data_mean_group.columns
    marker_genes = {}
    
    for target_group in groups:        
        other_groups = [group for group in groups if group != target_group]
        
        # Extract statistics for target group
        mu1 = rna_data_mean_group[target_group]
        sd1 = rna_data_std_group[target_group]
        n1 = group_counts[target_group]
    
        mu2 = rna_data_mean_group[other_groups].mean(axis=1)
        sd2 = rna_data_std_group[other_groups].mean(axis=1)
        n2 = group_counts[other_groups].mean()
        
        # Welch's t-test with normal approximation
        numerator = mu1 - mu2
        denominator = np.sqrt((sd1 ** 2) / n1 + (sd2 ** 2) / n2)
        t_stats = numerator / denominator
        
        t_stats = t_stats.replace([np.inf, -np.inf], np.nan).fillna(0)

        p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))
        
        _, pvals_corrected, _, _ = multipletests(p_values, method="fdr_bh")
        
        log_fc = mu1 - mu2
        
        mask = (pvals_corrected < pvalue_threshold) & (log_fc > log_fc_threshold)
        marker_genes[target_group] = list(genes[mask])

    
    return marker_genes

def assign_cells_to_pseudobulks(rna_data_unmatched, rna_data_mean_pseudobulk):
    unmatched_matrix = rna_data_unmatched.T.values 
    pseudobulk_matrix = rna_data_mean_pseudobulk.T.values 
    distance_matrix = cdist(unmatched_matrix, pseudobulk_matrix, metric='euclidean')
    pseudobulk_names = list(rna_data_mean_pseudobulk.columns)
    cell_names = list(rna_data_unmatched.columns)
    closest_indices = np.argmin(distance_matrix, axis=1)
    closest_pseudobulks = [pseudobulk_names[i] for i in closest_indices]
    closest_distances = distance_matrix[np.arange(len(cell_names)), closest_indices]
    cell_to_pseudobulk = dict(zip(cell_names, closest_pseudobulks))
    cell_to_distance = dict(zip(cell_names, closest_distances))

    return cell_to_pseudobulk, cell_to_distance
def aggregate_and_merge_rna(rna_unmatched, rna_mean_all, cell_to_pseudobulk):
    common_cells = [cell for cell in rna_unmatched.columns if cell in cell_to_pseudobulk]
    
    pseudobulk_labels = [cell_to_pseudobulk[cell] for cell in common_cells]
    
    rna_unmatched_subset = rna_unmatched[common_cells].copy()
    rna_unmatched_subset.columns = pseudobulk_labels
    rna_unmatched_agg = rna_unmatched_subset.groupby(rna_unmatched_subset.columns, axis=1).mean()
    
    all_pseudobulks = set(rna_mean_all.columns) | set(rna_unmatched_agg.columns)
    combined_data = pd.DataFrame(index=rna_mean_all.index)
    
    for pseudobulk in all_pseudobulks:
        vals = []
        if pseudobulk in rna_mean_all.columns:
            vals.append(rna_mean_all[pseudobulk])
        if pseudobulk in rna_unmatched_agg.columns:
            vals.append(rna_unmatched_agg.reindex(rna_mean_all.index, fill_value=0)[pseudobulk])
        
        combined_data[pseudobulk] = pd.concat(vals, axis=1).mean(axis=1)
    
    return combined_data




# ___________________________________________________________________________
#              Lifelong
#____________________________________________________________________________\


def merge_cluster_metadata(modality, counts_col):
    stages = [1.5, 2, 3, 5, 14, 60, 150, 210]
    data_dir = "../../data/lifelong/raw/metadata_clusters"
    dfs = []
    
    for stage in stages:
        filepath = os.path.join(data_dir, f"{modality}_dpf{stage}_metadata.tsv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, sep='\t')
            df['stage_dpf'] = stage
            df['pseudobulk'] = df['stage_dpf'].astype(str) + '_' + df['annotation'].astype(str)
            dfs.append(df[['cell', counts_col, 'stage_dpf', 'annotation', 'pseudobulk']])
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def plot_mean_vs_mean(
    df,
    x_col='mean_atac',
    y_col='mean_rna',
    x_label='ATAC mean',
    y_label='RNA mean',
    title='ATAC vs RNA Mean Signal',
    hue='stage_dpf',
    style='cell_type',
    figsize=(8, 6)
):
    df = df.copy()
    
    # Harmonize adult stages
    if hue == 'stage_dpf' and df[hue].dtype in [float, int, 'float64', 'int64']:
        df[hue] = df[hue].replace({150.0: 'adult', 210.0: 'adult'})
    
    plt.figure(figsize=figsize)
    ax = sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue,
        style=style,
        s=100,
        edgecolor='black',
        linewidth=0.5
    )
    max_val = max(df[x_col].max(), df[y_col].max())
    ax.plot([0, max_val], [0, max_val], ls='--', color='gray', label='y=x')  # reference line
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()