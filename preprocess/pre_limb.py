import argparse
import numpy as np
import scanpy as sc
import os
import sys
import pandas as pd
from scipy.io import mmread

def process_limb(input_path="./original/",output_path="./processed/",filter=True):
    adata = sc.read(f"{input_path}/LimbFilter.h5ad")
    data_all = adata.X.toarray().astype(np.float32)
    label_celltype = adata.obs['celltype'].to_list()
    if filter:
        vars = np.var(data_all, axis=0)
        mask_gene = np.argsort(vars)[-500:]
        data = data_all[:, mask_gene]

    del data_all

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

    mean = data_n.mean(axis=0)
    std = data_n.std(axis=0)
    data_n = (data_n - mean) / std

    label_train_str_set = sorted(list(set(label_train_str)))
    label = np.array([label_train_str_set.index(i) for i in label_train_str]).astype(np.int32)

    np.save(f'{output_path}/LimbFilter_data_n.npy', data_n)
    np.save(f'{output_path}/LimbFilter_label.npy', label)
    return


process_limb()
