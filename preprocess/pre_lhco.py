import argparse
import numpy as np
import scanpy as sc
import os
import sys
import pandas as pd
from scipy.io import mmread

def process_lhco(input_path="datasets_bio/original/",output_path="datasets_bio/processed/",filter=True):
    adata = sc.read(f"{input_path}He_2022_NatureMethods_Day15.h5ad")
    adata.write(f'{output_path}LHCO.h5ad')
    return adata

process_lhco()
