import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from datasets import load_from_disk
import torch.nn as nn, torch.nn.functional as F
import torch, json
from transformers import BertTokenizer, BertModel
from baseline.utils import BertModel as MedBertModel
from baseline.utils import LangCellDataCollatorForCellClassification as DataCollatorForCellClassification
from tqdm import tqdm
from torch.utils.data import DataLoader
from baseline.utils import LangCellTranscriptomeTokenizer
import scanpy as sc
import numpy as np


class Pooler(nn.Module):
    def __init__(self, config, pretrained_proj, proj_dim):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, proj_dim)
        self.proj.load_state_dict(torch.load(pretrained_proj))
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = hidden_states[:, 0]
        pooled_output = F.normalize(self.proj(pooled_output), dim=-1)
        return pooled_output

def load_langcell_model():
    model = BertModel.from_pretrained('/root/model/ckpt/cell_bert')
    model.pooler = Pooler(model.config, pretrained_proj='/root/model/ckpt/cell_proj.bin', proj_dim=256)
    proj = model.pooler.proj
    # model = model.module
    model = model.to("cuda")
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


def build_dataloader(dataset_sub, batchsize = 32):


    types = list(set(dataset_sub['celltype']))
    type2num = dict([(type, i) for i, type in enumerate(types)])

    def classes_to_ids(example):
        example["label"] = type2num[example["celltype"]]
        return example

    testdataset = dataset_sub.map(classes_to_ids, num_proc=16)
    remove_columns = testdataset.column_names
    remove_columns.remove('input_ids')
    remove_columns.remove('label')
    testdataset = testdataset.remove_columns(remove_columns)
    
    collator = DataCollatorForCellClassification()
    dataloader = DataLoader(testdataset, batch_size=batchsize, collate_fn=collator, shuffle=False)
    return dataloader


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

def Cal_emb(model, dataset_sub, dataloader, batchsize = 64):

    cell_embs = torch.zeros(len(dataset_sub), 256)
    model.eval()
    
    # preds = torch.zeros(len(dataset_sub))
    with torch.no_grad():
        for i, d in tqdm(enumerate(dataloader)):
            cell_last_h, cellemb = cell_encode(model, d['input_ids'], d['attention_mask']) # batchsize * 256
            cell_embs[i * batchsize: (i + 1) * batchsize] = cellemb.cpu()
    print(cell_embs.shape)
    return cell_embs


def LangeCellEmb(
    h5ad_file,
    batchsize = 128,
    # h5ad_path = '/root/commusim/data/Limb/LimbFilter.h5ad',
    dataset_path = 'Limb.dataset',
):

    preprocess_data(h5ad_file, dataset_path)
    model = load_langcell_model()
    dataset_sub = preprocess_dataset(dataset_path=dataset_path)
    dataloader = build_dataloader(dataset_sub, batchsize=batchsize)
    emb = Cal_emb(model, dataset_sub, dataloader, batchsize=batchsize)
    return emb


if __name__ == '__main__':
    adata = sc.read("data.h5ad")
    data_all = adata.X.toarray().astype(np.float32)
    vars = np.var(data_all, axis=0)
    mask_gene = np.argsort(vars)[-500:]
    adata = adata[:, mask_gene]
    emb = LangeCellEmb(adata)
    print(emb.shape)
    np.save('emb.npy', emb)
