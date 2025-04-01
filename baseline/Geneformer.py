import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from datasets import load_from_disk
import torch.nn as nn, torch.nn.functional as F
import torch, json
from baseline.utils import BertModel as MedBertModel
from baseline.utils import LangCellDataCollatorForCellClassification as DataCollatorForCellClassification
from tqdm import tqdm
from torch.utils.data import DataLoader
from baseline.utils import LangCellTranscriptomeTokenizer
import scanpy as sc
import numpy as np

from geneformer import EmbExtractor

def load_geneformer_model():
    model =  EmbExtractor(model_type="CellClassifier",
                     num_classes=3,
                     filter_data=None,
                     max_ncells=None,
                     emb_layer=0,
                     emb_label=["celltype"],
                     labels_to_plot=["disease"],
                     forward_batch_size=36,
                     nproc=16)
    return model



def preprocess_data(data, output_path):

    data.obs['n_counts'] = data.X.sum(axis=1)
    # import pdb; pdb.set_trace()
    data.var['ensembl_id'] = data.var['gene_ids']
    data.obs['filter_pass'] = (data.obs['n_counts'] > 0)

    tk = LangCellTranscriptomeTokenizer(dict([(k, k) for k in data.obs.keys()]), nproc=4)
    tokenized_cells, cell_metadata = tk.tokenize_anndata(data)
    tokenized_dataset = tk.create_dataset(tokenized_cells, cell_metadata)

    tokenized_dataset.save_to_disk(output_path)


def preprocess_dataset(dataset_path):
    dataset = load_from_disk(dataset_path)
    dataset_sub = dataset.shuffle(seed=42)#.select(range(1000))
    for label_name in ["celltype", "cell_type", "str_labels", "labels"]:
        if label_name in dataset_sub.column_names:
            break
    if label_name != "celltype":
        dataset_sub = dataset_sub.rename_column(label_name,"celltype")

    return dataset_sub

def cell_encode(model, cell_input_ids, cell_atts):
    cell = model(cell_input_ids.to("cuda"), cell_atts.to("cuda"))
    cell_last_h = cell.last_hidden_state
    cell_pooler = cell.pooler_output
    return cell_last_h, cell_pooler


def GeneformerEmb(
    h5ad_file,
    batchsize = 128,
    # h5ad_path = '/root/commusim/data/Limb/LimbFilter.h5ad',
    dataset_path = 'Geneformer.dataset',
):

    preprocess_data(h5ad_file, dataset_path)
    model = load_geneformer_model()
    out = model.extract_embs(
        "./Geneformer/fine_tuned_models/gf-6L-30M-i2048_CellClassifier_cardiomyopathies_220224",
        dataset_path,
        "./output",
        "output_prefix")
    out = out.to_numpy()
    label_str = out[:,256]
    emb = out[:,:256]
    label_str_sorted = sorted(list(set(label_str)))
    data_label = np.array([label_str_sorted.index(i) for i in label_str]).astype(np.int32)
    return emb, data_label


if __name__ == '__main__':
    adata = sc.read("data.h5ad")
    data_all = adata.X.toarray().astype(np.float32)
    vars = np.var(data_all, axis=0)
    mask_gene = np.argsort(vars)[-500:]
    adata = adata[:, mask_gene]   
    emb = GeneformerEmb(adata)
    print(emb.shape)
    np.save('emb.npy', emb)
