import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import quantile_transform
from Bio import SeqIO
from pathlib import Path
import seaborn as sns
from scipy.spatial import cKDTree


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

def annotate_peaks_with_region_types(peaks_df, gene_df, distance_threshold=1000):
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

    return annotated

