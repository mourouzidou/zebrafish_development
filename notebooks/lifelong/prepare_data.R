# install.packages(c("openssl", "curl", "httr", "plotly", "Seurat"))
install.packages("hexpression")
setRepositories(ind=1:3)
install.packages("Signac")
library(Signac)
library(Seurat)
setwd("notebooks/lifelong")

# Load datasets
atac_rds <- "../../data/lifelong/raw/atac_datasets_list.rds"
atac_datasets <- readRDS(atac_rds)
rna_rds <- "../../data/lifelong/raw/rna_datasets_list.rds"
rna_datasets <- readRDS(rna_rds)


output_dir <- "../../data/lifelong/raw/"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

for (dataset_name in names(rna_datasets)) {
  rna_counts <- GetAssayData(rna_datasets[[dataset_name]], assay = "RNA", slot = "counts")
  filename <- paste0(output_dir, dataset_name, "_rna_counts.csv")
  write.csv(as.data.frame(rna_counts), file = filename, row.names = TRUE)
}

for (dataset_name in names(atac_datasets)) {
  atac_counts <- GetAssayData(atac_datasets[[dataset_name]], assay = "ATAC", slot = "counts")
  filename <- paste0(output_dir, dataset_name, "_atac_counts.csv")
  write.csv(as.data.frame(atac_counts), file = filename, row.names = TRUE)
}

# Display summary of what was created
print("\nSummary of exported files:")
rna_files <- paste0(names(rna_datasets), "_rna_counts.csv")
atac_files <- paste0(names(atac_datasets), "_atac_counts.csv")
