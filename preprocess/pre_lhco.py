import argparse
import numpy as np
import scanpy as sc
import os
import sys
import pandas as pd
from scipy.io import mmread

def process_lhco(input_path="datasets_bio/original/",output_path="datasets_bio/processed/",filter=True):
    # preprocess
    adata = sc.read(f"{input_path}He_2022_NatureMethods_Day15.h5ad")
    adata.write(f'{output_path}LHCO.h5ad')
    # load data to model
    sc.pp.highly_variable_genes(adata, n_top_genes=500)
    adata = adata[:, adata.var['highly_variable']]
    data = adata.X

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
    print('data.shape', data.shape)
    return data, label


process_lhco()
