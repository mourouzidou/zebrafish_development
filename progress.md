






I want to take advantage of both ATAC and RNA seq data and as we discussed. my goal is to build a model that will predict ATAC signals from sequences and gene expression inputs. This would enable us to get predictions of unseen cell types and stages and thus identify why the model predicted these signals, meaning which regions and combined with which TF abundance, are leading to a certain accessibility signal.


Currently:

I am currently using data from this study https://pubmed.ncbi.nlm.nih.gov/37468546/  that has ATAC and RNA single cell signals for 7 embryonic developmental stages of zebrafish from 3hpf to 24hpf.The data are single cell and the authors assigned some celltype labels separately to the RNA and the ATAC cells (they are different). However for me it would be beneficial to have as many pseudobulks (celltypes + time points) as possible while maintaining reliable information. 

One issue is that there are very few cell types that are common between the two modalities, and at the same time they did not map any RNA cell to neural crest, which we would be highly interested in as far as I have understood so far.
The authors also mappedeach RNA cell to the "closest" ATAC cell. I decided to continue with the original ATAC cell type labels and for the RNA cells, I used the cell type labels thta they inherited from the closest ATAC cell. The only downside with this is that many RNA cells might match with one ATAC cells and this means that I finally have annotations only for ~2000 RNA cells out of 50,000.
Our approach here: defined marker genes by running pairwise Welch's t-tests on the pseudobulks derived from the labeled 2000 RNA cells and extracted a superset of "marker" genes based on the low corrected p-value - the ones that are highly variable among pseudobulks. I used this superset to compute the euclidean distance from each unmapped RNA cell to each already defined pseudobulk and assigned the pseudobulk label with the minimum distance. 
I also defined a cutoff for each cell type based on how compact it is, and thus for the renrichment I didnt consider cells with euclidean distance from the closest pseudobulk that is higher than that threshold.
In this way I have an enriched RNA pseudobulk dataset with celltypes and developmental stages identical to the ATAC dataset.

ATAC Dataset:
I also attach a notebook to help have an idea and mention some of my key questions in the corresponding places of the analysis. Here is a more detailed idea and quesiton list:

* From the distribution of the total reads of each cell inside a pseudobulk we see that:
  some pseudobulks show a wide distribution meaning either that there is some technical noise so we should account for that by discarding the highly variable pseudobulks or by quantile normalize them so that all the distributions will be forced to be identical.

* Are there some celltype/ time points that are expected to be more or less involved in gene regulation (especially during embryonic development)? It seems that for the same cell type as the time passes, the overall read depth decreases. Is this something that we expect? Could this help us contextually filter or transform the data?

* Another question, there are two cell types with a similar name : one is YSL and the other is YSL/presumptive endoderm. Does this make sense to you in a sense that it is important to keep it as is?

* Scatterplots: I plotted the read statistics (median, std, and the range) of each pseudobulk against the size (number of cells in it) to check for any potential correlation that could imply missclassification due to inefficient sample size or any cell number related bias and it seems it is not the case. Could this indicate that the heterogeneity can be biological signal (for example that a pseudobulk can contain some subgroups)? If not, we come again to the same question, should I quantile normalize or filter to address this?

* Genomic position correlation:
- distance from TSS = 100: The cells in some cell types (e.g. blastomere) have a different (lower) mean read counts distribution for enhancers (>100bf from TSS) compared to candidate promoter regions (<100bp from TSS). The difefrences are smoothened as we increase the distance from TSS




In the attached notebook I have included the plots and some observations that I also summarize here.