import argparse
import numpy as np
import scanpy as sc
import os
import sys
import pandas as pd
from scipy.io import mmread


def process_werinreb(input_path="datasets_bio/original/",output_path="datasets_bio/processed/",filter=True):
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

    # return


process_werinreb()