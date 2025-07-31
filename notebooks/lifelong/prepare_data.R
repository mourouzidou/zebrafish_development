
# install.packages(c("openssl", "curl", "httr", "plotly", "Seurat"))
install.packages("hexpression")


library(Seurat)




setwd("notebooks/lifelong")
atac_rds <- "../../data/lifelong/raw/atac_datasets_list.rds"
atac_datasets <- readRDS(atac_rds)

rna_rds <- "../../data/lifelong/raw/rna_datasets_list.rds"
rna_datasets <- readRDS(rna_rds)



annotation <- setNames(
  c(
    "stroma 1",
    "gill progenitor 1",
    "gill cartilage",
    "gill cartilage",
    "dermal fibroblast",
    "gill cartilage",
    "gill cartilage",
    "periosteum/tendon/ligament",
    "teeth",
    "perivascular",
    "pillar",
    "smooth muscle 2",
    "gill progenitor 2",
    "smooth muscle",
    "gill stroma",
    "cycling cells",
    "tunica media",
    "stroma 2",
    "hyaline cartilage",
    "bone"
  ),
  levels(droplevels(rna_datasets$dpf150$RNA_snn_res.0.8))
)




rna_datasets$dpf150$annotation <- as.factor(unname(
  annotation[as.character(rna_datasets$dpf150$RNA_snn_res.0.8)]
))

# print(rna_datasets$dpf150$annotation)
saveRDS(rna_datasets, rna_rds)



