import argparse
import numpy as np
import scanpy as sc
import os
import sys
import pandas as pd
from scipy.io import mmread


def process_werinreb(input_path="datasets_bio/original/",output_path="datasets_bio/processed/",filter=True):
    # preprocess
    matrix_file = f"{input_path}Weinreb_inVitro_normed_counts.mtx"
    print("matrix_file", matrix_file)
    genes_file = f"{input_path}Weinreb_inVitro_gene_names.txt"
    metadata_file = f"{input_path}Weinreb_inVitro_metadata.txt"
    mtx = mmread(matrix_file).tocsr()
    genes = pd.read_csv(genes_file, header=None, names=['genes'])
    adata = sc.AnnData(mtx, var=genes)
    
    if metadata_file:
        metadata = pd.read_csv(metadata_file, sep='\t')
        adata.obs = metadata.set_index(adata.obs.index)
    
    adata.write(f'{output_path}Weinreb.h5ad')

    # load data to model
    sc.pp.log1p(adata)
    adata.obs['celltype']=adata.obs['Cell type annotation']
    adata = adata[~adata.obs['celltype'].isna()]
    if filter:
        sc.pp.highly_variable_genes(adata, n_top_genes=500)
        adata = adata[:, adata.var['highly_variable']]
    
    if isinstance(adata.X, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        data = adata.X.toarray()
    else:
        data = adata.X
    data = np.array(data).astype(np.float32)
    
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean) / std  
    
    label_celltype = adata.obs['celltype']
    label_train_str = list(np.squeeze(label_celltype.values))
    label_train_str_set = sorted(list(set(label_train_str)))
    label = np.array([label_train_str_set.index(i) for i in label_train_str]).astype(np.int32)
    return data, label


process_werinreb()