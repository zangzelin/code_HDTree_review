import argparse
import numpy as np
import scanpy as sc
import os
import sys
import pandas as pd
from scipy.io import mmread

def process_ecl(input_path="datasets_bio/original/",output_path="datasets_bio/processed/",filter=True):
    adata = sc.read(f"{input_path}/EpitheliaCell.h5ad")
    adata.obs['celltype']=adata.obs['cell_type']
    label_celltype = adata.obs['celltype'].to_list()
    
    # copy a new adata
    adata_sub = adata.copy()
    sc.pp.subsample(adata_sub, fraction=0.1)
    data_all = adata_sub.X.toarray().astype(np.float32)
    vars = np.var(data_all, axis=0)
    mask_gene = np.argsort(vars)[-500:]
    adata = adata[:, mask_gene]

    data = adata.X.toarray().astype(np.float32)

    label_count = {}
    for i in list(set(label_celltype)):
        label_count[i] = label_celltype.count(i)

    label_count = sorted(label_count.items(), key=lambda x: x[1], reverse=True)
    label_count = label_count[:10]

    mask_top10 = np.zeros(len(label_celltype)).astype(np.bool_)
    for str_label in label_count:
        mask_top10[str_label[0] == np.array(label_celltype)] = 1

    data_n = np.array(data).astype(np.float32)[mask_top10]
    label_train_str = np.array(list(np.squeeze(label_celltype)))[mask_top10]

    # downsample the 10k data for every cell type
    mask = np.zeros(len(label_train_str)).astype(np.bool_)
    for i in range(10):
        # random select 10k data for each cell type
        random_index = np.random.choice(
            np.where(label_train_str == label_count[i][0])[0],
            10000, replace=False)
        mask[random_index] = 1

    data_n = data_n[mask]
    label_train_str = label_train_str[mask]
    
    mean = data_n.mean(axis=0)
    std = data_n.std(axis=0)
    data_n = (data_n - mean) / std  
    label_train_str_set = sorted(list(set(label_train_str)))
    label = np.array([label_train_str_set.index(i) for i in label_train_str]).astype(np.int32)

    np.save(f'{output_path}/EpitheliaCell_data_n.npy', data_n)
    np.save(f'{output_path}/EpitheliaCell_label.npy', label)

    print('data.shape', data_n.shape, 'num_classes', max(label)+1)
    return data_n, label


process_ecl()