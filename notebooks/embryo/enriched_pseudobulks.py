# %% [markdown]
# ### Embryo zebrafish development data: https://pubmed.ncbi.nlm.nih.gov/37468546/

# %%
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances
import warnings
from statsmodels.stats.multitest import multipletests



# %% [markdown]
# Load ATAC data

# %%
atac_data_df = pd.read_csv('../../data/embryo/raw/atac_peak_matrix_complete_sparse.csv', index_col=0)



# %% [markdown]
# Load RNA data

# %%
rna_data = pl.read_csv("../../data/embryo/raw/scRNA_countMatrix.csv", ignore_errors=True)
# Convert to dictionary first, then to pandas
data_dict = rna_data.to_dict(as_series=False)
rna_data_df = pd.DataFrame(data_dict)
# make 1st colun the index 
rna_data_df.set_index(rna_data_df.columns[0], inplace=True)
# remove empty rows and columns and set gene names as index
rna_data_df[rna_data_df.isna().any(axis=1)]
rna_data_df.dropna(axis=0, how='any', inplace=True)
rna_data_df 



# %% [markdown]
# Log transform RNA data

# %%
# transform rna data - log2(x+1) for downstream steps
rna_data_log = rna_data_df.copy()
rna_data_log.iloc[:, 0:] = np.log2(rna_data_log.iloc[:, 0:].astype(float) + 1)
rna_data_log

# %% [markdown]
# Load metadata

# %%
rna_metadata_df = pd.read_csv('../../data/embryo/raw/rna_all.metaData.txt')
rna_metadata_df.rename(columns = {'Unnamed: 0': 'rna_cell'}, inplace=True)
rna_metadata_df


# %%
atac_metadata_df = pd.read_csv('../../data/embryo/raw/atac_all.metaData.txt', index_col=0)
atac_metadata_df = atac_metadata_df[['celltype', 'predictedCell', 'predictedGroup'] ]
atac_metadata_df.rename(columns = {'celltype': 'atac_cell_type', 'predictedCell': 'rna_matching_cell', 'predictedGroup': 'rna_cell_type'}, inplace=True)
atac_metadata_df.reset_index(names = "atac_cell", inplace=True)
atac_metadata_df.head()


# %% [markdown]
# Create ATAC **mean** and **std** dataframes - group cells by same pseudobulk

# %%
celltype_lookup = dict(zip(atac_metadata_df['atac_cell'], atac_metadata_df['atac_cell_type'])) # map the cell types to the atac_cell names
atac_data_df['stage'] = atac_data_df['Cell'].str.extract(r'(\d{1,2})[a-zA-Z]', expand=False)
atac_data_df['celltype'] = atac_data_df['Cell'].map(celltype_lookup)
atac_data_df
atac_data_df['stage_celltype'] = atac_data_df['stage'] + '_' + atac_data_df['celltype']
atac_stage_counts = atac_data_df.groupby('stage')['Cell'].nunique()
atac_celltype_counts = atac_data_df.groupby('celltype')['Cell'].nunique()
print("shape of atac_data_df:", atac_data_df.shape)
print(atac_data_df.head())
# create mean dataframe
atac_mean_psd  = atac_data_df.pivot_table(index='Peak', 
    columns='stage_celltype', values='Accessibility', aggfunc='mean')

# Create std dataframe
atac_std_psd = atac_data_df.pivot_table(index='Peak', 
    columns='stage_celltype', values='Accessibility', aggfunc='std')

# replace nan with 0
atac_mean_psd.fillna(0, inplace=True)
atac_mean_psd


# %% [markdown]
# Check how many cells are in each pseudobulk

# %%

def extract_stage_from_cell_id(cell_id, stage_delimiter='hpf', pos=0):
    stage_parts = str(cell_id).split(stage_delimiter)
    stage = stage_parts[pos]
    stage = stage.replace('h', '')
    return stage

atac_metadata_df['rna_stage'] = atac_metadata_df['rna_matching_cell'].apply(lambda x: extract_stage_from_cell_id(x, 'h', 0))
atac_metadata_df['atac_stage'] = atac_metadata_df['atac_cell'].apply(lambda x: extract_stage_from_cell_id(x, 'hpf', 0))
atac_metadata_df['pseudobulk'] = atac_metadata_df['atac_stage'].astype(str) + '_' + atac_metadata_df['atac_cell_type']
atac_metadata_df.head( )

# %%
atac_counts = atac_metadata_df.groupby('pseudobulk')['atac_cell'].nunique().reset_index(name='ATAC')
rna_counts = atac_metadata_df.groupby('pseudobulk')['rna_matching_cell'].nunique().reset_index(name='RNA')

stacked_counts = pd.merge(atac_counts, rna_counts, on='pseudobulk', how='outer').fillna(0)
stacked_counts[['ATAC', 'RNA']] = stacked_counts[['ATAC', 'RNA']].astype(int)

stacked_counts.set_index('pseudobulk')[['ATAC', 'RNA']].plot(
    kind='bar',
    stacked=False,
    figsize=(16, 6),
    color=['#96ba99', '#ad8098']
)

plt.xticks(rotation=60, ha='right')
plt.xlabel('Pseudobulk')
plt.ylabel('Number of Cells')
plt.title('RNA and ATAC Cell Counts per Pseudobulk (Before RNA Cells Enrichment)')
plt.legend(title='Source')
plt.tight_layout()
plt.show()



# %%
atac_counts_per_psd = atac_metadata_df.groupby('pseudobulk')['atac_cell'].nunique().reset_index(name='ATAC')
# filter out pseudobulks with less than 30 cells
atac_counts_per_psd = atac_counts_per_psd[atac_counts_per_psd['ATAC'] >= 30]
rna_counts_per_psd = atac_metadata_df.groupby('pseudobulk')['rna_matching_cell'].nunique().reset_index(name='RNA')
stages_per_celltype = atac_metadata_df.groupby('atac_cell_type')['atac_stage'].unique().reset_index()
stages_per_celltype.rename(columns={'atac_stage': 'stages'}, inplace=True)

stages_per_celltype.sort_values(by='stages', key=lambda x: x.str.len(), ascending=False, inplace=True)

# add another column that will be a list that will have the number of cells per each cell type of the list in the second column respective to the stages
def get_cell_counts_per_stage(row):
    cell_type = row['atac_cell_type']
    stages = row['stages']
    
    cell_counts = []
    for stage in stages:
        # Create pseudobulk identifier (stage_celltype format)
        pseudobulk_id = f"{stage}_{cell_type}"
        # Count cells for this specific pseudobulk
        count = atac_metadata_df[atac_metadata_df['pseudobulk'] == pseudobulk_id]['atac_cell'].nunique()
        cell_counts.append(count)
    
    return cell_counts

stages_per_celltype['cell_counts_per_stage'] = stages_per_celltype.apply(get_cell_counts_per_stage, axis=1)

print("\nStages per cell type with cell counts:")
print(stages_per_celltype)

# %%
# Three-Panel ATAC vs RNA Developmental Trajectory Comparison

def extract_stage_number(stage_str):
    """Extract numeric value from stage string (e.g., '10hpf' -> 10)"""
    import re
    match = re.search(r'(\d+)', str(stage_str))
    return int(match.group(1)) if match else 0

def process_trajectory_data(df, cell_type_col, stage_col):
    """Process dataframe to create trajectory data"""
    # Group by celltype and stage to get counts
    stage_celltype_counts = df.groupby([cell_type_col, stage_col]).size().reset_index(name='count')
    
    # Add numeric stage column for sorting
    stage_celltype_counts['stage_numeric'] = stage_celltype_counts[stage_col].apply(extract_stage_number)
    
    # Create pivot table
    heatmap_data = stage_celltype_counts.pivot(index=cell_type_col, columns='stage_numeric', values='count')
    heatmap_data = heatmap_data.fillna(0)
    
    # Sort columns by stage number
    heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)
    
    return heatmap_data

def get_earliest_stage_unified(cell_type, atac_data, rna_data):
    """Get earliest stage for a cell type across both datasets"""
    earliest = float('inf')
    
    # Check ATAC data
    if atac_data is not None and cell_type in atac_data.index:
        for stage in sorted(atac_data.columns):
            if atac_data.loc[cell_type, stage] > 0:
                earliest = min(earliest, int(stage))
                break
    
    # Check RNA data
    if rna_data is not None and cell_type in rna_data.index:
        for stage in sorted(rna_data.columns):
            if rna_data.loc[cell_type, stage] > 0:
                earliest = min(earliest, int(stage))
                break
    
    return earliest if earliest != float('inf') else 999

def get_earliest_stage_single(row):
    """Get earliest stage for a single dataset"""
    for stage in sorted(row.index):
        if row[stage] > 0:
            return int(stage)
    return float('inf')

def plot_trajectories(ax, cell_types, atac_data, rna_data, all_stages, title, plot_type):
    """Plot trajectories for given cell types"""
    
    if len(cell_types) == 0:
        ax.text(0.5, 0.5, 'No cell types in this category', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        return
    
    # Sort cell types by earliest appearance
    if plot_type == 'shared':
        cell_types_sorted = sorted(cell_types, 
                                  key=lambda x: get_earliest_stage_unified(x, atac_data, rna_data))
    elif plot_type == 'atac_only':
        # Sort by earliest appearance in ATAC data
        cell_types_sorted = sorted(cell_types, 
                                  key=lambda x: get_earliest_stage_unified(x, atac_data, None))
    else:  # rna_only
        # Sort by earliest appearance in RNA data
        cell_types_sorted = sorted(cell_types, 
                                  key=lambda x: get_earliest_stage_unified(x, None, rna_data))
    
    # Create color mapping
    colors = plt.cm.tab20(np.linspace(0, 1, len(cell_types_sorted)))
    color_map = {cell_type: colors[i] for i, cell_type in enumerate(cell_types_sorted)}
    
    y_offset = 0
    # Calculate appropriate spacing
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
        
        # Plot ATAC trajectory (solid line) if available
        if atac_data is not None and cell_type in atac_data.index:
            atac_y_values = []
            for stage in all_stages:
                if stage in atac_data.columns:
                    atac_y_values.append(atac_data.loc[cell_type, stage])
                else:
                    atac_y_values.append(0)
            
            atac_y_values_offset = [val + y_offset for val in atac_y_values]
            max_y_for_cell = max(max_y_for_cell, max(atac_y_values))
            
            # Plot solid line for ATAC
            ax.plot(all_stages, atac_y_values_offset, marker='o', linewidth=2.5, 
                   markersize=5, color=cell_color, alpha=0.8, linestyle='-',
                   label=f'{cell_type} (ATAC)' if plot_type == 'shared' else cell_type)
            
            # Highlight non-zero points
            non_zero_stages = [stage for stage, count in zip(all_stages, atac_y_values) if count > 0]
            non_zero_counts = [count + y_offset for count in atac_y_values if count > 0]
            if non_zero_stages:
                ax.scatter(non_zero_stages, non_zero_counts, color=cell_color, 
                          s=50, zorder=5, alpha=0.9, marker='o', edgecolor='white', linewidth=1)
        
        # Plot RNA trajectory (dashed line) if available
        if rna_data is not None and cell_type in rna_data.index:
            rna_y_values = []
            for stage in all_stages:
                if stage in rna_data.columns:
                    rna_y_values.append(rna_data.loc[cell_type, stage])
                else:
                    rna_y_values.append(0)
            
            rna_y_values_offset = [val + y_offset for val in rna_y_values]
            max_y_for_cell = max(max_y_for_cell, max(rna_y_values))
            
            # Plot dashed line for RNA
            ax.plot(all_stages, rna_y_values_offset, marker='s', linewidth=2.5, 
                   markersize=5, color=cell_color, alpha=0.8, linestyle='--',
                   label=f'{cell_type} (RNA)' if plot_type == 'shared' else cell_type)
            
            # Highlight non-zero points
            non_zero_stages = [stage for stage, count in zip(all_stages, rna_y_values) if count > 0]
            non_zero_counts = [count + y_offset for count in rna_y_values if count > 0]
            if non_zero_stages:
                ax.scatter(non_zero_stages, non_zero_counts, color=cell_color, 
                          s=50, zorder=5, alpha=0.9, marker='s', edgecolor='white', linewidth=1)
        
        # Store position for labeling
        cell_type_positions[cell_type] = y_offset
        
        # Update offset for next cell type
        y_offset += max_y_for_cell + y_spacing
    
    # Add cell type labels on the right side
    for cell_type, y_pos in cell_type_positions.items():
        ax.text(max(all_stages) + 0.3, y_pos + y_spacing*0.1, cell_type, 
                va='bottom', ha='left', fontsize=9, fontweight='bold',
                color=color_map[cell_type])
    
    # Customize the plot
    ax.set_xlabel('Developmental Stage (hpf)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cell Count', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    # Set x-axis
    ax.set_xticks(all_stages)
    ax.set_xticklabels([f'{stage}hpf' for stage in all_stages], rotation=45, ha='right')
    ax.set_xlim(min(all_stages)-0.5, max(all_stages)+1.5)
    
    # Remove y-axis ticks
    ax.set_yticks([])
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add horizontal separator lines
    for i, y_pos in enumerate(cell_type_positions.values()):
        if i > 0:
            ax.axhline(y=y_pos - y_spacing/2, color='gray', linestyle=':', alpha=0.3)

# Process both datasets
print("Processing ATAC-seq data...")
try:
    atac_heatmap = process_trajectory_data(atac_metadata_df, 'atac_cell_type', 'atac_stage')
    atac_available = True
    print(f"ATAC data processed: {len(atac_heatmap)} cell types")
except Exception as e:
    print(f"ATAC data not available: {e}")
    atac_available = False
    atac_heatmap = None

print("Processing RNA-seq data...")
rna_heatmap = process_trajectory_data(rna_metadata_df, 'celltype', 'stage')
print(f"RNA data processed: {len(rna_heatmap)} cell types")

# Determine cell type categories
if atac_available:
    all_cell_types = set(atac_heatmap.index) | set(rna_heatmap.index)
    common_cell_types = set(atac_heatmap.index) & set(rna_heatmap.index)
    atac_only = set(atac_heatmap.index) - set(rna_heatmap.index)
    rna_only = set(rna_heatmap.index) - set(atac_heatmap.index)
    
    # Get all stages (union of both datasets)
    all_stages = sorted(set(atac_heatmap.columns) | set(rna_heatmap.columns))
    
    print(f"\nDataset overlap:")
    print(f"Common cell types: {len(common_cell_types)}")
    print(f"ATAC-only: {len(atac_only)}")
    print(f"RNA-only: {len(rna_only)}")
    print(f"Total unique: {len(all_cell_types)}")
    
else:
    # If no ATAC data, all are RNA-only
    common_cell_types = set()
    atac_only = set()
    rna_only = set(rna_heatmap.index)
    all_stages = sorted(rna_heatmap.columns)

# Create three-panel figure
fig, axes = plt.subplots(1, 3, figsize=(24, 16))

# Panel 1: Common cell types (both ATAC and RNA)
plot_trajectories(axes[0], common_cell_types, 
                 atac_heatmap if atac_available else None, 
                 rna_heatmap, all_stages, 
                 f'Shared Cell Types (n={len(common_cell_types)})\nATAC (solid, ○) vs RNA (dashed, □)', 
                 'shared')

# Panel 2: ATAC-only cell types
plot_trajectories(axes[1], atac_only, 
                 atac_heatmap if atac_available else None, 
                 None, all_stages, 
                 f'ATAC-seq Only (n={len(atac_only)})', 
                 'atac_only')

# Panel 3: RNA-only cell types
plot_trajectories(axes[2], rna_only, 
                 None, rna_heatmap, all_stages, 
                 f'RNA-seq Only (n={len(rna_only)})', 
                 'rna_only')

# Add overall title
fig.suptitle('Developmental Trajectory Comparison: ATAC-seq vs RNA-seq\n(Sorted by Earliest Appearance)', 
             fontsize=16, fontweight='bold', y=0.98)

# Add legend for shared panel
if len(common_cell_types) > 0 and atac_available:
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=2, linestyle='-', marker='o', 
                   markersize=6, label='ATAC-seq'),
        plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', marker='s', 
                   markersize=6, label='RNA-seq')
    ]
    axes[0].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()

# Print detailed summary statistics
print("\n" + "="*80)
print("THREE-PANEL TRAJECTORY ANALYSIS SUMMARY")
print("="*80)

if atac_available:
    print(f"Total unique cell types across both datasets: {len(all_cell_types)}")
    print(f"Shared cell types (Panel 1): {len(common_cell_types)}")
    print(f"ATAC-only cell types (Panel 2): {len(atac_only)}")
    print(f"RNA-only cell types (Panel 3): {len(rna_only)}")
    
    print(f"\nCell counts:")
    print(f"ATAC-seq total cells: {atac_heatmap.values.sum():,.0f}")
    print(f"RNA-seq total cells: {rna_heatmap.values.sum():,.0f}")
    
    if len(common_cell_types) > 0:
        print(f"\nShared cell types:")
        for ct in sorted(common_cell_types):
            atac_total = atac_heatmap.loc[ct].sum()
            rna_total = rna_heatmap.loc[ct].sum()
            print(f"  {ct}: ATAC={atac_total:,.0f}, RNA={rna_total:,.0f}")
    
    if len(atac_only) > 0:
        print(f"\nATAC-only cell types: {sorted(list(atac_only))}")
    
    if len(rna_only) > 0:
        print(f"\nRNA-only cell types: {sorted(list(rna_only))}")
        
else:
    print(f"RNA-seq only analysis:")
    print(f"Total cell types: {len(rna_only)}")
    print(f"Total cells: {rna_heatmap.values.sum():,.0f}")

print(f"\nDevelopmental stages analyzed: {[f'{stage}hpf' for stage in all_stages]}")

# Analysis insights
print(f"\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
if atac_available:
    overlap_percentage = len(common_cell_types) / len(all_cell_types) * 100
    print(f"• Dataset overlap: {overlap_percentage:.1f}% of cell types are shared")
    print(f"• ATAC-seq captures {len(atac_only)} unique cell types not seen in RNA-seq")
    print(f"• RNA-seq captures {len(rna_only)} unique cell types not seen in ATAC-seq")
    print(f"• Panel 1 shows direct technology comparison for {len(common_cell_types)} cell types")
    print(f"• Panels 2 & 3 show technology-specific cell type detection")

# %% [markdown]
# #### Only 2258 RNA cells are assigned to ATAC cells

# %%
# filter rna_data_log to only keep the rows with max  value >=4
rna_data_log_filter_low_genes = rna_data_log[rna_data_log.max(axis=1) >= 4]
rna_data_log_filter_low_genes

# %%

# continue with log values
matching_cells = atac_metadata_df['rna_matching_cell'].str.strip().unique().tolist()
rna_data_matched_all_genes = rna_data_log.loc[:, rna_data_log.columns.intersection(matching_cells)]
rna_data_unmatched_all_genes =rna_data_log.loc[:, ~rna_data_log.columns.isin(matching_cells)]




# %%
# continue with log values
matching_cells = atac_metadata_df['rna_matching_cell'].str.strip().unique().tolist()
rna_data_matched_filter_low_genes = rna_data_log_filter_low_genes.loc[:, rna_data_log_filter_low_genes.columns.intersection(matching_cells)]
rna_data_unmatched_filter_low_genes = rna_data_log_filter_low_genes.loc[:, ~rna_data_log_filter_low_genes.columns.isin(matching_cells)]
rna_data_unmatched_filter_low_genes

# %%
rna_data_matched_filter_low_genes

# %% [markdown]
# ##### Define a cutoff for each pseudobulk as the distance between this (mean vector of same psd cells) and the most distant cell of the same celltype (from any time point).
# ##### Compute the distance from same cell type cells to each pseudobulk considering only genes with max log expression >=4
#     - To enrich the pseudobulks with unmatched RNA cells we will use this cutoff
#     - If a new cell has a distance > cutoff, it is more different than any cell of that type used to define the pseudobulk 

# %%
from scipy.spatial.distance import cdist

def compute_pseudobulk_cutoffs_df(rna_data_matched, atac_metadata):
    metadata = atac_metadata.set_index('rna_matching_cell').loc[rna_data_matched.columns].reset_index(names='rna_matching_cell')
    pseudobulk_means = {
        pb: rna_data_matched[metadata[metadata['pseudobulk'] == pb]['rna_matching_cell']].mean(axis=1)
        for pb in metadata['pseudobulk'].unique()
    }

    results = []
    for pb, mean_vec in pseudobulk_means.items():
        celltype = metadata[metadata['pseudobulk'] == pb]['atac_cell_type'].iloc[0]
        same_type_cells = metadata[metadata['atac_cell_type'] == celltype]['rna_matching_cell']
        same_type_matrix = rna_data_matched[same_type_cells].T.values  # shape: (n_cells, n_genes)
        
        mean_vec_reshaped = mean_vec.values.reshape(1, -1)  # shape: (1, n_genes)
        distances = cdist(same_type_matrix, mean_vec_reshaped, metric='euclidean').flatten()  # shape: (n_cells,)
        
        results.append({
            'pseudobulk': pb,
            'atac_cell_type': celltype,
            'cutoff': distances.max(),
            'n_cells_in_pseudobulk': (metadata['pseudobulk'] == pb).sum(),
            'n_cells_same_type': (metadata['atac_cell_type'] == celltype).sum()
        })
    
    return pd.DataFrame(results)

cutoffs_high_exp_genes = compute_pseudobulk_cutoffs_df(rna_data_matched_filter_low_genes, atac_metadata_df)
cutoffs_high_exp_genes


# %%
# create rna_data_pseudobulk mean and std dataframes for the log transformed data. 
# Aggregate the RNA data by pseudobulk using the matched cells "og"
cell_to_pseudobulk = atac_metadata_df.set_index('rna_matching_cell')['pseudobulk'].to_dict()
rna_data_psd_names = rna_data_matched_all_genes.rename(columns=cell_to_pseudobulk)
rna_data_std_pseudobulk_og_all_genes= rna_data_psd_names.groupby(by=rna_data_psd_names.columns, axis=1).std()
rna_data_mean_pseudobulk_og_all_genes= rna_data_psd_names.groupby(by=rna_data_psd_names.columns, axis=1).mean()


# %%
pseudobulk_counts_og = rna_data_psd_names.columns.value_counts()


# %%
rna_data_psd_names

# %%
# create rna_data_pseudobulk mean and std dataframes for the log transformed data ONLY for HIGHLY EXPRESSED GENES.
# Aggregate the RNA data by pseudobulk using the matched cells "og"
cell_to_pseudobulk = atac_metadata_df.set_index('rna_matching_cell')['pseudobulk'].to_dict()
rna_data_psd_names = rna_data_matched_filter_low_genes.rename(columns=cell_to_pseudobulk)
rna_data_std_pseudobulk_og_filtered_genes= rna_data_psd_names.groupby(by=rna_data_psd_names.columns, axis=1).std()
rna_data_mean_pseudobulk_og_filtered_genes= rna_data_psd_names.groupby(by=rna_data_psd_names.columns, axis=1).mean()

rna_data_mean_pseudobulk_og_filtered_genes


# %% [markdown]
# #### assign pseudobulk labels on unmatched cells 
#         - considering all genes
#         - considering only the genees with argmax log2(x+1) >4

# %%
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

cell_to_pseudobulk_filtered_genes, cell_to_distance_filtered_genes = assign_cells_to_pseudobulks(rna_data_unmatched_filter_low_genes,rna_data_mean_pseudobulk_og_filtered_genes)
cell_to_pseudobulk_all_genes,cell_to_distance_all_genes= assign_cells_to_pseudobulks(rna_data_unmatched_all_genes, rna_data_mean_pseudobulk_og_all_genes)
print(cell_to_distance_filtered_genes)
print(cell_to_distance_all_genes)

# %%
from collections import Counter

def analyze_distance_changes(cell_to_distance_filtered, cell_to_distance_all, 
                                     cell_to_pseudobulk_filtered, cell_to_pseudobulk_all,
                                     title=""):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f'Distance Analysis: Filtered vs All Genes\n{title}', fontsize=16, fontweight='bold')

    common_cells = set(cell_to_distance_filtered) & set(cell_to_distance_all)
    distances_filtered = [cell_to_distance_filtered[cell] for cell in common_cells]
    distances_all = [cell_to_distance_all[cell] for cell in common_cells]

    axes[0].hist(distances_filtered, bins=30, alpha=0.7, label='Filtered genes', color='red', density=True)
    axes[0].hist(distances_all, bins=30, alpha=0.7, label='All genes', color='blue', density=True)
    axes[0].set_xlabel('Distance to closest pseudobulk')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distance Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(distances_all, distances_filtered, alpha=0.6, s=20)
    axes[1].plot([min(distances_all), max(distances_all)],
                 [min(distances_all), max(distances_all)], 'r--', label='y = x')
    axes[1].set_xlabel('Distance (All genes)')
    axes[1].set_ylabel('Distance (Filtered genes)')
    axes[1].set_title('Distance Correlation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    correlation = np.corrcoef(distances_all, distances_filtered)[0, 1]
    axes[1].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                 transform=axes[1].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))

    pseudobulk_counts_filtered = Counter(cell_to_pseudobulk_filtered.values())
    pseudobulk_counts_all = Counter(cell_to_pseudobulk_all.values())
    all_pseudobulks = sorted(set(pseudobulk_counts_filtered) | set(pseudobulk_counts_all))

    counts_filtered = [pseudobulk_counts_filtered.get(pb, 0) for pb in all_pseudobulks]
    counts_all = [pseudobulk_counts_all.get(pb, 0) for pb in all_pseudobulks]

    x_pos = np.arange(len(all_pseudobulks))
    width = 0.35

    axes[2].bar(x_pos - width / 2, counts_filtered, width, label='Filtered genes', color='red', alpha=0.7)
    axes[2].bar(x_pos + width / 2, counts_all, width, label='All genes', color='blue', alpha=0.7)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(all_pseudobulks, rotation=45, ha='right')
    axes[2].set_xlabel('Pseudobulk')
    axes[2].set_ylabel('Assigned cells')
    axes[2].set_title('Pseudobulk Assignment Counts')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return {
        'distances_filtered': distances_filtered,
        'distances_all': distances_all,
        'correlation': correlation
    }
results = analyze_distance_changes(
    cell_to_distance_filtered_genes,
    cell_to_distance_all_genes,
    cell_to_pseudobulk_filtered_genes,
    cell_to_pseudobulk_all_genes,
    title="Unmatched RNA cells assigned to pseudobulks: Filtered: High Expression Genes vs All Genes \n No Marker genes"
)


# %% [markdown]
# #### Some pseudobulks have 0 marker genes - they also have few cells assigned from the original assignment

# %%
cell_to_celltype = atac_metadata_df.set_index('rna_matching_cell')['atac_cell_type'].to_dict()

rna_data_celltype_named = rna_data_matched_all_genes.rename(columns=cell_to_celltype)

rna_data_mean_celltype = rna_data_celltype_named.groupby(by=rna_data_celltype_named.columns, axis=1).mean()
rna_data_std_celltype = rna_data_celltype_named.groupby(by=rna_data_celltype_named.columns, axis=1).std()
rna_data_mean_celltype


# %%
rna_data_std_celltype

# %%

from scipy.stats import norm


def find_marker_genes(rna_data, metadata_df, grouping_column, 
                     cell_id_column='rna_matching_cell',
                     pvalue_threshold=0.05, log_fc_threshold=0.5):
    
    
    print(f"Finding marker genes by {grouping_column}...")
    
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
        print(f"\nTesting for {target_group}...")
        
        other_groups = [group for group in groups if group != target_group]
        
        # Extract statistics for target group
        mu1 = rna_data_mean_group[target_group]
        sd1 = rna_data_std_group[target_group]
        n1 = group_counts[target_group]
        
        # Extract statistics for other groups (averaged)
        mu2 = rna_data_mean_group[other_groups].mean(axis=1)
        sd2 = rna_data_std_group[other_groups].mean(axis=1)
        n2 = group_counts[other_groups].mean()
        
        # Welch's t-test with normal approximation
        numerator = mu1 - mu2
        denominator = np.sqrt((sd1 ** 2) / n1 + (sd2 ** 2) / n2)
        t_stats = numerator / denominator
        
        # Handle division by zero and infinite values
        t_stats = t_stats.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Calculate two-tailed p-values
        p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))
        
        # FDR correction
        _, pvals_corrected, _, _ = multipletests(p_values, method="fdr_bh")
        
        # Calculate log fold-change
        log_fc = mu1 - mu2
        
        # Filter significant marker genes
        mask = (pvals_corrected < pvalue_threshold) & (log_fc > log_fc_threshold)
        marker_genes[target_group] = list(genes[mask])
        
        print(f"  Found {mask.sum()} marker genes for {target_group}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY - Marker genes by {grouping_column}")
    print(f"{'='*60}")
    total_markers = sum(len(genes) for genes in marker_genes.values())
    print(f"Total groups analyzed: {len(groups)}")
    print(f"Total marker genes found: {total_markers}")
    print(f"Average marker genes per group: {total_markers/len(groups):.1f}")
    
    for group, genes_list in marker_genes.items():
        print(f"  {group}: {len(genes_list)} genes")
    
    return marker_genes



marker_genes_celltype = find_marker_genes(
    rna_data=rna_data_matched_all_genes,
    metadata_df=atac_metadata_df,
    grouping_column='atac_cell_type',
    pvalue_threshold=0.05,
    log_fc_threshold=0.0
)

marker_genes_pseudobulk = find_marker_genes(
    rna_data=rna_data_matched_all_genes,
    metadata_df=atac_metadata_df,
    grouping_column='pseudobulk',
    pvalue_threshold=0.05,
    log_fc_threshold=0
)


marker_genes_pseudobulk_high_exp_genes= find_marker_genes(
    rna_data=rna_data_matched_filter_low_genes,
    metadata_df=atac_metadata_df,
    grouping_column='pseudobulk',
    pvalue_threshold=0.05,
    log_fc_threshold=0
)


# %%
# # print(number of marker genes per pseudobulk
# print("\nNumber of marker genes per pseudobulk:")
# for pb, genes in marker_genes_pseudobulk.items():
#     print(f"{pb}: {len(genes)} genes")  

# print("_____________")
# # print number of marker genes per pseudobulk for low expressed genes
# print("\nNumber of marker genes per pseudobulk (high exp genes only):")
# for pb, genes in marker_genes_pseudobulk_high_exp_genes.items():
#     print(f"{pb}: {len(genes)} genes")

# %%
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


results = plot_marker_gene_fold_change(
    marker_genes_pseudobulk, 
    marker_genes_pseudobulk_high_exp_genes,
)


# %%

plot_marker_gene_counts(marker_genes_pseudobulk, title="Marker Genes per Pseudobulk (All Genes)")
plot_marker_gene_counts(marker_genes_pseudobulk_high_exp_genes, title="Marker Genes per Pseudobulk (High Expression Genes Only)")
plot_marker_gene_counts(marker_genes_celltype, title="Marker Genes per Celltype")





# %%
rna_data_unmatched_all_genes
rna_data_mean_pseudobulk_og_all_genes

rna_data_unmatched_filter_low_genes
rna_data_mean_pseudobulk_og_filtered_genes

# %%
def assign_cells_to_pseudobulk_markers_only(rna_data_unmatched, rna_data_mean_pseudobulk_og, 
                                          marker_genes_dict):
    """
    Assign cells to pseudobulks using only marker genes for each pseudobulk
    """
    cell_names = list(rna_data_unmatched.columns)
    pseudobulk_names = list(rna_data_mean_pseudobulk_og.columns)
    
    # For each cell, compute distance to each pseudobulk using only that pseudobulk's markers
    cell_assignments = {}
    cell_distances = {}
    
    for cell in cell_names:
        min_distance = float('inf')
        best_pseudobulk = None
        distances_to_pseudobulks = {}
        
        for pseudobulk in pseudobulk_names:
            marker_genes = marker_genes_dict[pseudobulk]
            
            if len(marker_genes) == 0:
                # If no markers, use very high distance
                distance = float('inf')
            else:
                # Get expression values for marker genes only
                cell_expression = rna_data_unmatched.loc[marker_genes, cell].values
                pseudobulk_expression = rna_data_mean_pseudobulk_og.loc[marker_genes, pseudobulk].values
                
                # Compute euclidean distance
                distance = np.sqrt(np.sum((cell_expression - pseudobulk_expression)**2))
            
            distances_to_pseudobulks[pseudobulk] = distance
            
            if distance < min_distance:
                min_distance = distance
                best_pseudobulk = pseudobulk
        
        cell_assignments[cell] = best_pseudobulk
        cell_distances[cell] = min_distance
    
    return cell_assignments, cell_distances




# %%

def compute_marker_based_pseudobulk_cutoffs_df(rna_data_matched, atac_metadata, marker_genes_dict):
    metadata = atac_metadata.set_index('rna_matching_cell').loc[rna_data_matched.columns].reset_index(names='rna_matching_cell')  
    pseudobulk_names = metadata['pseudobulk'].unique()
    results = []

    for pb in pseudobulk_names:
        celltype = metadata[metadata['pseudobulk'] == pb]['atac_cell_type'].iloc[0]
        marker_genes = marker_genes_dict.get(pb, [])
        marker_genes = [g for g in marker_genes if g in rna_data_matched.index]
        cells_in_pb = metadata[metadata['pseudobulk'] == pb]['rna_matching_cell']
        mean_expression = rna_data_matched.loc[marker_genes, cells_in_pb].mean(axis=1)
        same_type_cells = metadata[metadata['atac_cell_type'] == celltype]['rna_matching_cell']
        if len(same_type_cells) == 0:
            continue
        same_type_matrix = rna_data_matched.loc[marker_genes, same_type_cells].T.values
        mean_vec_reshaped = mean_expression.values.reshape(1, -1)
        distances = cdist(same_type_matrix, mean_vec_reshaped, metric='euclidean').flatten()
        results.append({
            'pseudobulk': pb,
            'atac_cell_type': celltype,
            'cutoff': distances.max(),
            'n_cells_in_pseudobulk': len(cells_in_pb),
            'n_cells_same_type': len(same_type_cells)
        })
        
    return pd.DataFrame(results)

cutoffs_marker_based = compute_marker_based_pseudobulk_cutoffs_df(
    rna_data_matched_filter_low_genes, 
    atac_metadata_df, 
    marker_genes_pseudobulk_high_exp_genes
)



# %% [markdown]
# #### Assign each unmatched RNA cell to the pseudobulk with the minimum euyclidean distance 
# - use only genes with log2(x+1) expression >= 4 for at least one pseudobulk
# 
# - use only pseudobulk specific marker genes for each iteration

# %%
cell_to_psd_with_markers, cell_to_psd_Dist_with_markers = assign_cells_to_pseudobulk_markers_only(rna_data_unmatched_filter_low_genes, rna_data_mean_pseudobulk_og_filtered_genes,marker_genes_pseudobulk_high_exp_genes )

# %%
cell_to_psd_with_markers

# %%
pseudobulks  = {v for k, v in cell_to_psd_with_markers.items()}
pseudobulks



# %%
cells = {k for k,v in cell_to_psd_with_markers.items()}
len(cells)

# %%
psd_to_cells = {}
for k, v in cell_to_psd_with_markers.items():
    psd_to_cells[v] = psd_to_cells.get(v, []) + [k]

# %%
cells_per_psd_enriched = {k: len(v) for k, v in psd_to_cells.items()}
cells_per_psd_enriched

# %%
cell_to_psd_Dist_with_markers 

# %%
cutoffs_marker_based

# %%
cutoffs_high_exp_genes

# %%
# discard cells that are assigned to a pseudobulk with a distance higher that the pseudobulk's specific cutoff distance
for cell, distance in cell_to_psd_Dist_with_markers.items():
    print([cell, distance, cutoffs_high_exp_genes['cutoff']])


# %% [markdown]
# #### replace cell column names in rna_data_log with the values of the dictionaries 
#     - cell_to_psd_with_markers for unmatched cells 
#     - cell_to_psd_with_markers for matched_cells

# %% [markdown]
# #### group by pseudobulk name and the mean values 

# %% [markdown]
# #### plot the number of cells in unmatched and matched pseudobulks

# %%
import os

def plot_distance_distributions_separately(cell_to_psd_with_markers, 
                                           cell_to_psd_dist_with_markers,
                                           cutoffs_df,
                                           output_dir="distance_violin_plots"):
    os.makedirs(output_dir, exist_ok=True)

    # Create a dataframe with all distances and pseudobulk assignments
    df = pd.DataFrame({
        "cell": list(cell_to_psd_with_markers.keys()),
        "pseudobulk": [cell_to_psd_with_markers[c] for c in cell_to_psd_with_markers],
        "distance": [cell_to_psd_dist_with_markers[c] for c in cell_to_psd_with_markers]
    })

    # For each pseudobulk, plot a violin plot of distances
    unique_pseudobulks = sorted(df['pseudobulk'].unique(), key=lambda x: int(x.split("_")[0]) if x.split("_")[0].isdigit() else 999)

    for pb in unique_pseudobulks:
        sub_df = df[df['pseudobulk'] == pb]
        cutoff = cutoffs_df[cutoffs_df['pseudobulk'] == pb]['cutoff'].values[0] if pb in cutoffs_df['pseudobulk'].values else None

        plt.figure(figsize=(6, 5))
        sns.violinplot(data=sub_df, y='distance', inner='quartile', color='skyblue')
        plt.title(f"Distance Distribution to Pseudobulk: {pb}")
        plt.ylabel("Euclidean Distance")
        if cutoff:
            plt.axhline(y=cutoff, color='red', linestyle='--', label=f'Cutoff: {cutoff:.2f}')
            plt.legend()

        plt.tight_layout()
        # Sanitize filename by replacing "/" and spaces
        filename_safe_pb = pb.replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(output_dir, f"{filename_safe_pb}_distance_violin.png"))


plot_distance_distributions_separately(
    cell_to_psd_with_markers,
    cell_to_psd_Dist_with_markers,
    cutoffs_marker_based,
    output_dir="distance_violin_plots"
)



# %%

def plot_volcano_plots(mean_df, std_df, sample_sizes, marker_dict, title_prefix="Pseudobulk"):
    genes = mean_df.index
    pseudobulks = mean_df.columns

    for target_pb in pseudobulks:
        other_pbs = [pb for pb in pseudobulks if pb != target_pb]

        mu1 = mean_df[target_pb]
        sd1 = std_df[target_pb]
        n1 = sample_sizes[target_pb]

        mu2 = mean_df[other_pbs].mean(axis=1)
        sd2 = std_df[other_pbs].mean(axis=1)
        n2 = sample_sizes[other_pbs].mean()

        # Welch's t-test with normal approximation
        numerator = mu1 - mu2
        denominator = np.sqrt((sd1 ** 2) / n1 + (sd2 ** 2) / n2)
        t_stats = numerator / denominator
        t_stats = t_stats.replace([np.inf, -np.inf], np.nan).fillna(0)

        p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))
        _, pvals_corrected, _, _ = multipletests(p_values, method="fdr_bh")

        log_fc = mu1 - mu2

        plt.figure(figsize=(8, 6))
        plt.scatter(log_fc, -np.log10(pvals_corrected), s=5, alpha=0.5, label="All genes")
        
        # Highlight marker genes
        marker_genes = marker_dict[target_pb]
        marker_mask = mean_df.index.isin(marker_genes)
        plt.scatter(log_fc[marker_mask], -np.log10(pvals_corrected[marker_mask]), color='red', s=10, label='Marker genes')

        plt.title(f"{title_prefix}: {target_pb}")
        plt.xlabel("Log2 Fold Change")
        plt.ylabel("-log10(FDR-corrected p-value)")
        plt.axhline(-np.log10(0.05), color='grey', linestyle='--', linewidth=1)
        plt.axvline(0.5, color='grey', linestyle='--', linewidth=1)
        plt.legend()
        plt.tight_layout()
        plt.show()

# Call the volcano plot function for pseudobulks
plot_volcano_plots(rna_data_mean_pseudobulk_og_filtered_genes, rna_data_std_pseudobulk_og_filtered_genes, pseudobulk_counts_og, marker_genes_pseudobulk_high_exp_genes, title_prefix="Pseudobulk")



# %% [markdown]
# 


