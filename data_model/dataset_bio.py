from sklearn.datasets import load_digits

from torch.utils import data
from sklearn.datasets import load_digits
from torch import tensor
import torchvision.datasets as datasets
from pynndescent import NNDescent
import os
import joblib
import torch
import numpy as np
from PIL import Image
import scanpy as sc
import scipy
from sklearn.decomposition import PCA
import sklearn
import pandas as pd

from data_model.dataset_meta import DigitsDataset
from data_model.dataset_meta import DigitsSEQDataset
import torchvision.datasets as datasets

np.random.seed(42)
seed=42
class MCAD9119Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):

        data = np.load(data_path + '/mca_data/mca_data_dim_34947.npy')
        data = data[:, data.max(axis=0) > 4].astype(np.float32)
        label = np.load(data_path + '/mca_data/mca_label_dim_34947.npy')

        label_count = {}
        for i in label:
            if i in label_count:
                label_count[i] += 1
            else:
                label_count[i] = 1

        for i in list(label_count.keys()):
            if label_count[i] < 500:
                label[label == i] = -1 

        data = data[label != -1]
        label = label[label != -1]

        return data, label.astype(np.int32)

class ActivityDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        train_data = pd.read_csv(data_path+'/feature_select/Activity_train.csv')
        test_data = pd.read_csv(data_path+'/feature_select/Activity_test.csv')
        all_data = pd.concat([train_data, test_data])
        data = all_data.drop(['subject', 'Activity'], axis=1).to_numpy()
        label_str = all_data['Activity'].tolist()
        label_str_set = sorted(list(set(label_str)))
        label = np.array([label_str_set.index(i) for i in label_str])
        data = (data-data.min())/(data.max()-data.min())
        
        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label)
        print('data.shape', data.shape)
        print(label)
        return data, label.astype(np.int32)

class Gast10k1458Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        
        sadata = sc.read(data_path+"/gast10kwithcelltype.h5ad")
        sadata_pca = np.array(sadata.X)
        data = np.array(sadata_pca).astype(np.float32)

        # Append a column of zeros to the data array to ensure the number of features
        # is a factor required by the Transformer's multi-head attention mechanism
        zeros_column = np.zeros((data.shape[0], 1), dtype=np.float32)  # Create a column of zeros
        data = np.hstack((data, zeros_column))

        label_train_str = list(sadata.obs['celltype'])
        label_train_str_set = sorted(list(set(label_train_str)))
        label_train = torch.tensor(
            [label_train_str_set.index(i) for i in label_train_str])
        label = np.array(label_train).astype(np.int32)
        return data, label


class SAMUSIKDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        data = pd.read_csv(data_path+'/samusik_01.csv')
        label_str = pd.read_csv(data_path+'/samusik01_labelnr.csv')
        data.fillna(data.min(), inplace=True)
        label = np.array(label_str)[:,-1]
        data = np.array(data)[:,1:]
        data = (data-data.min())/(data.max()-data.min())
        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label).astype(np.int32)
        # import pdb; pdb.set_trace()
        return data, label


class HCL60KDataset(DigitsDataset):
    def load_data(self, data_path, train=True):

        sadata = sc.read(data_path+"/HCL60kafter-elis-all.h5ad")
        data = np.array(sadata.X).astype(np.float32)
        label = np.array(np.array([int(i) for i in list(sadata.obs.louvain)]))
        
        # import pdb; pdb.set_trace()

        # import MinMaxScaler
        scaler = sklearn.preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data)

        mask_label = (np.zeros(label.shape)+1).astype(np.bool_)
        for l in range(label.max()+1):
            num_l = (label==l).sum()
            if num_l < 500:
                mask_label[label==l] = False

        data = data[mask_label]
        label = label[mask_label]
        
        # Append a column of zeros to the data array to ensure the number of features
        # is a factor required by the Transformer's multi-head attention mechanism
        zeros_column = np.zeros((data.shape[0], 1), dtype=np.float32)  # Create a column of zeros
        data = np.hstack((data, zeros_column))
        tissue = list(sadata.obs['tissue'])
        
        
        # dict_label_tissue = 
        
        print(data.shape)
        return data, label


class HCL60KPLOTDataset(DigitsDataset):
    def load_data(self, data_path, train=True):

        sadata = sc.read(data_path+"/HCL60kafter-elis-all.h5ad")
        data = np.array(sadata.X).astype(np.float32)
        # label = np.array(np.array([int(i) for i in list(sadata.obs.louvain)]))
        label_str = list(sadata.obs.tissue)
        set_list = list(set(label_str))
        label = np.array([set_list.index(i) for i in label_str])
        import pickle
        pickle.dump(set_list, open('HCL_set_list_label.pkl', 'wb'))
        
        
        # import pdb; pdb.set_trace()

        # import MinMaxScaler
        scaler = sklearn.preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data)

        mask_label = (np.zeros(label.shape)+1).astype(np.bool_)
        for l in range(label.max()+1):
            num_l = (label==l).sum()
            if num_l < 500:
                mask_label[label==l] = False

        data = data[mask_label]
        label = label[mask_label]
        
        # Append a column of zeros to the data array to ensure the number of features
        # is a factor required by the Transformer's multi-head attention mechanism
        zeros_column = np.zeros((data.shape[0], 1), dtype=np.float32)  # Create a column of zeros
        data = np.hstack((data, zeros_column))
        tissue = list(sadata.obs['tissue'])
        
        print(data.shape)
        return data, label


class HCL600KDataset(DigitsDataset):
    def load_data(self, data_path, train=True):

        sadata = sc.read(data_path+"/HCL60kafter-elis-all.h5ad")
        data = np.array(sadata.X).astype(np.float32)
        label = np.array(np.array([int(i) for i in list(sadata.obs.louvain)]))

        # import MinMaxScaler
        scaler = sklearn.preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data)

        mask_label = (np.zeros(label.shape)+1).astype(np.bool_)
        for l in range(label.max()+1):
            num_l = (label==l).sum()
            if num_l < 500:
                mask_label[label==l] = False

        data = data[mask_label]
        label = label[mask_label]
        
        # Append a column of zeros to the data array to ensure the number of features
        # is a factor required by the Transformer's multi-head attention mechanism
        zeros_column = np.zeros((data.shape[0], 1), dtype=np.float32)  # Create a column of zeros
        data = np.hstack((data, zeros_column))

        # repeat the data 10 times
        data = np.repeat(data, 10, axis=0)
        label = np.repeat(label, 10, axis=0)

        print(data.shape)
        return data, label


class HCL60K1000Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):

        sadata = sc.read(data_path+"/HCL60kafter-elis-all.h5ad")
        data = np.array(sadata.X).astype(np.float32)
        label = np.array(np.array([int(i) for i in list(sadata.obs.louvain)]))

        # import MinMaxScaler
        scaler = sklearn.preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data)

        mask_label = (np.zeros(label.shape)+1).astype(np.bool_)
        for l in range(label.max()+1):
            num_l = (label==l).sum()
            if num_l < 500:
                mask_label[label==l] = False

        data = data[mask_label]
        label = label[mask_label]
        
        # Append a column of zeros to the data array to ensure the number of features
        # is a factor required by the Transformer's multi-head attention mechanism
        feature_variances = np.var(data, axis=0)
        top_features_indices = np.argsort(feature_variances)[-1000:]
        data = data[:, top_features_indices]

        print(data.shape)
        return data, label


class CeleganDataset(DigitsDataset):
    def load_data(self, data_path, train=True):

        adata = sc.read(data_path+"/celegan/celegan.h5ad")
        data = adata.X
        data = np.array(data).astype(np.float32)
        label_celltype = pd.read_csv(data_path+'/celegan/celegan_celltype_2.tsv', sep='\t', header=None)
        adata.obs['celltype'] = pd.Categorical(np.squeeze(label_celltype))
        label_train_str = list(np.squeeze(label_celltype.values))
        label_train_str_set = sorted(list(set(label_train_str)))
        label = np.array([label_train_str_set.index(i) for i in label_train_str]).astype(np.int32)
        print(data)
        return data, label


class LimbSampleDataset(DigitsDataset):
    def load_data(self, data_path, train=True,filter=True):
        # adata = sc.read(data_path+"/Limb/LimbFilter.h5ad")
        adata = sc.read("/any/data/difftreedata/data/LimbFilter.h5ad")
        
        if filter:
            sc.pp.subsample(adata, fraction=0.05)
            sc.pp.highly_variable_genes(adata, n_top_genes=500)
            adata = adata[:, adata.var['highly_variable']]
        data = adata.X

        if isinstance(adata.X, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
            data = adata.X.toarray()
        else:
            data = adata.X
        data = np.array(data).astype(np.float32)
        label_celltype = adata.obs['celltype']
        label_train_str = list(np.squeeze(label_celltype.values))
        label_train_str_set = sorted(list(set(label_train_str)))
        label = np.array([label_train_str_set.index(i) for i in label_train_str]).astype(np.int32)
        print(data)
        return data, label


class LimbDataset(DigitsDataset):
    def load_data(self, data_path, train=True,filter=True):
        # adata = sc.read(data_path+"/Limb/LimbFilter.h5ad")
        # adata = sc.read("/any/data/difftreedata/data/LimbFilter.h5ad")
        # data_all = adata.X.toarray().astype(np.float32)
        # label_celltype = adata.obs['celltype'].to_list()
        # if filter:
        #     vars = np.var(data_all, axis=0)
        #     mask_gene = np.argsort(vars)[-500:]
        #     data = data_all[:, mask_gene]

        # del data_all

        # label_count = {}
        # for i in list(set(label_celltype)):
        #     label_count[i] = label_celltype.count(i)
        

        # label_count = sorted(label_count.items(), key=lambda x: x[1], reverse=True)
        # label_count = label_count[:10]

        # mask_top10 = np.zeros(len(label_celltype)).astype(np.bool_)
        # for str_label in label_count:
        #     mask_top10[str_label[0] == np.array(label_celltype)] = 1

        # data_n = np.array(data).astype(np.float32)[mask_top10]
        # label_train_str = np.array(list(np.squeeze(label_celltype)))[mask_top10]

        # mean = data_n.mean(axis=0)
        # std = data_n.std(axis=0)
        # data_n = (data_n - mean) / std

        # label_train_str_set = sorted(list(set(label_train_str)))
        # label = np.array([label_train_str_set.index(i) for i in label_train_str]).astype(np.int32)
        
        # np.save('/any/data/difftreedata/data/LimbFilter_data_n.npy', data_n)
        # np.save('/any/data/difftreedata/data/LimbFilter_label.npy', label)
        
        data_n = np.load('/any/data/difftreedata/data/LimbFilter_data_n.npy')
        label = np.load('/any/data/difftreedata/data/LimbFilter_label.npy')

        print('data.shape', data_n.shape)
        print('label', label.shape)
        return data_n, label


class LHCODataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        adata = sc.read("/any/data/difftreedata/datasets_bio/processed/LHCO.h5ad")
        if filter:
            # sc.pp.subsample(adata, fraction=0.1)
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


class WeinrebAllDataset(DigitsDataset):
    def load_data(self, data_path, train=True,filter=True):
        adata = sc.read("/any/data/difftreedata/datasets_bio/processed/Weinreb.h5ad")
        
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
        print('data.shape', data.shape, 'num_classes', len(label_train_str_set))
        # import pdb; pdb.set_trace()
        return data, label


class EpitheliaCellDataset(DigitsDataset):
    def load_data(self, data_path, train=True,filter=True):
        adata = sc.read(data_path+"/any/data/difftreedaata/datasets_bio/processed//EpitheliaCell.h5ad")
        adata.obs['celltype']=adata.obs['cell_type']
        adata = adata[~adata.obs['celltype'].isna()]
        if filter:
            sc.pp.highly_variable_genes(adata, n_top_genes=500)
            adata = adata[:, adata.var['highly_variable']]
        data = adata.X
        
        if isinstance(adata.X, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
            data = adata.X.toarray()
        else:
            data = adata.X
        data = np.array(data).astype(np.float32)
        label_celltype = adata.obs['celltype']
        label_train_str = list(np.squeeze(label_celltype.values))
        label_train_str_set = sorted(list(set(label_train_str)))
        label = np.array([label_train_str_set.index(i) for i in label_train_str]).astype(np.int32)
        print('data.shape', data.shape)
        # import pdb; pdb.set_trace()
        return data, label


class EpitheliaCell1000GDataset(DigitsDataset):
    def load_data(self, data_path, train=True,filter=True):
        # adata = sc.read("/any/data/difftreedata/data/EpitheliaCell.h5ad")
        # adata.obs['celltype']=adata.obs['cell_type']
        # label_celltype = adata.obs['celltype'].to_list()
        
        # # copy a new adata
        # adata_sub = adata.copy()
        # sc.pp.subsample(adata_sub, fraction=0.1)
        # data_all = adata_sub.X.toarray().astype(np.float32)
        # vars = np.var(data_all, axis=0)
        # mask_gene = np.argsort(vars)[-500:]
        # adata = adata[:, mask_gene]

        # data = adata.X.toarray().astype(np.float32)

        # label_count = {}
        # for i in list(set(label_celltype)):
        #     label_count[i] = label_celltype.count(i)

        # label_count = sorted(label_count.items(), key=lambda x: x[1], reverse=True)
        # label_count = label_count[:10]

        # mask_top10 = np.zeros(len(label_celltype)).astype(np.bool_)
        # for str_label in label_count:
        #     mask_top10[str_label[0] == np.array(label_celltype)] = 1

        # data_n = np.array(data).astype(np.float32)[mask_top10]
        # label_train_str = np.array(list(np.squeeze(label_celltype)))[mask_top10]

        # # downsample the 10k data for every cell type
        # mask = np.zeros(len(label_train_str)).astype(np.bool_)
        # for i in range(10):
        #     # random select 10k data for each cell type
        #     random_index = np.random.choice(
        #         np.where(label_train_str == label_count[i][0])[0],
        #         10000, replace=False)
        #     mask[random_index] = 1

        # data_n = data_n[mask]
        # label_train_str = label_train_str[mask]
        
        # mean = data_n.mean(axis=0)
        # std = data_n.std(axis=0)
        # data_n = (data_n - mean) / std  
        # label_train_str_set = sorted(list(set(label_train_str)))
        # label = np.array([label_train_str_set.index(i) for i in label_train_str]).astype(np.int32)

        # np.save('/any/data/difftreedata/data/EpitheliaCell_data_n.npy', data_n)
        # np.save('/any/data/difftreedata/data/EpitheliaCell_label.npy', label)

        data_n = np.load('/any/data/difftreedata/datasets_bio/processed/EpitheliaCell_data_n.npy')
        label = np.load('/any/data/difftreedata/datasets_bio/processed/EpitheliaCell_label.npy')

        # import pdb; pdb.set_trace()

        print('data.shape', data_n.shape, 'num_classes', max(label)+1)
        return data_n, label


class WeinrebAll1000GDataset(DigitsDataset):
    def load_data(self, data_path, train=True,filter=True):
        adata = sc.read(data_path+"/datasets_bio/processed/Weinreb.h5ad")
        sc.pp.log1p(adata)
        adata.obs['celltype']=adata.obs['Cell type annotation']
        adata = adata[~adata.obs['celltype'].isna()]
        if filter:
            # sc.pp.subsample(adata, fraction=0.1)
            sc.pp.highly_variable_genes(adata, n_top_genes=1000)
            adata = adata[:, adata.var['highly_variable']]
        data = adata.X
        # import pdb; pdb.set_trace()
        
        if isinstance(adata.X, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
            data = adata.X.toarray()
        else:
            data = adata.X
        data = np.array(data).astype(np.float32)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        data = (data - mean) / std  
        
        label_celltype = adata.obs['celltype']
        # import pdb; pdb.set_trace()
        label_train_str = list(np.squeeze(label_celltype.values))
        label_train_str_set = sorted(list(set(label_train_str)))
        label = np.array([label_train_str_set.index(i) for i in label_train_str]).astype(np.int32)
        print('data.shape', data.shape)
        # import pdb; pdb.set_trace()
        return data, label


class CELEGANT7Dataset(DigitsDataset):
    
    def load_data(self, data_path, train=True,filter=True):
        num_top_celltype = 7
        adata = sc.read("/any/data/difftreedata/data/celegan/celegan.h5ad")
        
        # top 500 genes
        # import pdb; pdb.set_trace()
        if filter:
        
            data = adata.X    
            np.var(data, axis=0)
            top_genes = np.argsort(np.var(data, axis=0))[-500:]
            data = data[:, top_genes].astype(np.float32)
        
        std = data.std(axis=0)
        mean = data.mean(axis=0)
        data = (data - mean) / std
        
        # import pdb; pdb.set_trace()
        label_celltype = pd.read_csv('/any/data/difftreedata/data/celegan/celegan_celltype_2.tsv', sep='\t', header=None)
        label_embryo_time = pd.read_csv('/any/data/difftreedata/data/celegan/celegan_embryo_time.tsv', sep='\t', header=None)
        adata.obs['celltype'] = pd.Categorical(np.squeeze(label_celltype))
        adata.obs['embryo_time'] = pd.Categorical(np.squeeze(label_embryo_time))
        label_train_str = list(np.squeeze(label_celltype.values))
        label_train_str_set = list(set(label_train_str))
        label_train_str_set = sorted(label_train_str_set)
        label = tensor(
            np.array([label_train_str_set.index(i) for i in label_train_str]))
        
        dict_str_num_sample = {}
        for str_label in label_train_str_set:
            dict_str_num_sample[str_label] = (label==label_train_str_set.index(str_label)).sum()
            
        sorted_dict = sorted(dict_str_num_sample.items(), key=lambda x: x[1], reverse=True)
        for i in range(num_top_celltype):
            print(sorted_dict[i])
        
        
        time_list = []
        for str_time in label_embryo_time[0].to_list():
            if '-' in str_time:
                time_list.append(float(str_time.split('-')[1]))
            elif '<' in str_time:
                time_list.append(float(str_time.split('<')[1])-50)
            elif '>' in str_time:
                time_list.append(float(str_time.split('>')[1])+100)
            else:
                print('error', str_time)
                
        # # select the top num of sample 5 classes
        num_cell_type = len(label_train_str_set)
        num_sample_list = []
        for i in range(num_cell_type):
            num_sample_list.append((label==i).sum())
        
        # # select the top 5 cell types
        top5_cell_type = np.argsort(num_sample_list)[-num_top_celltype:]
        mask = np.zeros(len(label)).astype(np.bool_)
        for i in top5_cell_type:
            mask[label==i] = 1
        data = data[mask]
        label = label[mask]
        
        label_str_new = list(set([str(l) for l in label]))
        label_str_new = sorted(label_str_new)
        label = torch.tensor([label_str_new.index(str(l)) for l in label])
        
        time_list = torch.tensor(time_list)
        time_list = time_list[mask]
            
        print('data.shape', data.shape)
        # time_label = torch.tensor(label_embryo_time)
        label = torch.cat((label.reshape(-1,1), time_list.reshape(-1,1)), dim=1)
        
        return data, label
        


class CELEGANT7NoNormDataset(DigitsDataset):
    
    def load_data(self, data_path, train=True,filter=True):
        num_top_celltype = 7
        adata = sc.read("/any/data/difftreedata/data/celegan/celegan.h5ad")
        
        # top 500 genes
        # import pdb; pdb.set_trace()
        if filter:
        
            data = adata.X    
            np.var(data, axis=0)
            top_genes = np.argsort(np.var(data, axis=0))[-500:]
            data = data[:, top_genes].astype(np.float32)
        
        # std = data.std(axis=0)
        # mean = data.mean(axis=0)
        # data = (data - mean) / std
        
        # import pdb; pdb.set_trace()
        label_celltype = pd.read_csv('/any/data/difftreedata/data/celegan/celegan_celltype_2.tsv', sep='\t', header=None)
        label_embryo_time = pd.read_csv('/any/data/difftreedata/data/celegan/celegan_embryo_time.tsv', sep='\t', header=None)
        adata.obs['celltype'] = pd.Categorical(np.squeeze(label_celltype))
        adata.obs['embryo_time'] = pd.Categorical(np.squeeze(label_embryo_time))
        label_train_str = list(np.squeeze(label_celltype.values))
        label_train_str_set = list(set(label_train_str))
        label_train_str_set = sorted(label_train_str_set)
        label = tensor(
            np.array([label_train_str_set.index(i) for i in label_train_str]))
        
        dict_str_num_sample = {}
        for str_label in label_train_str_set:
            dict_str_num_sample[str_label] = (label==label_train_str_set.index(str_label)).sum()
            
        sorted_dict = sorted(dict_str_num_sample.items(), key=lambda x: x[1], reverse=True)
        for i in range(num_top_celltype):
            print(sorted_dict[i])
        
        
        time_list = []
        for str_time in label_embryo_time[0].to_list():
            if '-' in str_time:
                time_list.append(float(str_time.split('-')[1]))
            elif '<' in str_time:
                time_list.append(float(str_time.split('<')[1])-50)
            elif '>' in str_time:
                time_list.append(float(str_time.split('>')[1])+100)
            else:
                print('error', str_time)
                
        # # select the top num of sample 5 classes
        num_cell_type = len(label_train_str_set)
        num_sample_list = []
        for i in range(num_cell_type):
            num_sample_list.append((label==i).sum())
        
        # # select the top 5 cell types
        top5_cell_type = np.argsort(num_sample_list)[-num_top_celltype:]
        mask = np.zeros(len(label)).astype(np.bool_)
        for i in top5_cell_type:
            mask[label==i] = 1
        data = data[mask]
        label = label[mask]
        
        label_str_new = list(set([str(l) for l in label]))
        label_str_new = sorted(label_str_new)
        label = torch.tensor([label_str_new.index(str(l)) for l in label])
        
        time_list = torch.tensor(time_list)
        time_list = time_list[mask]
            
        print('data.shape', data.shape)
        # time_label = torch.tensor(label_embryo_time)
        label = torch.cat((label.reshape(-1,1), time_list.reshape(-1,1)), dim=1)
        
        return data, label


class CELEGANT7PCA100Dataset(DigitsDataset):
    
    def load_data(self, data_path, train=True,filter=True):
        num_top_celltype = 7
        adata = sc.read("/any/data/difftreedata/data/celegan/celegan.h5ad")
        
        # top 500 genes
        # import pdb; pdb.set_trace()
        if filter:
        
            data = adata.X    
            np.var(data, axis=0)
            top_genes = np.argsort(np.var(data, axis=0))[-500:]
            data = data[:, top_genes].astype(np.float32)
        
        std = data.std(axis=0)
        mean = data.mean(axis=0)
        data = (data - mean) / std
        
        # import pdb; pdb.set_trace()
        label_celltype = pd.read_csv('/any/data/difftreedata/data/celegan/celegan_celltype_2.tsv', sep='\t', header=None)
        label_embryo_time = pd.read_csv('/any/data/difftreedata/data/celegan/celegan_embryo_time.tsv', sep='\t', header=None)
        adata.obs['celltype'] = pd.Categorical(np.squeeze(label_celltype))
        adata.obs['embryo_time'] = pd.Categorical(np.squeeze(label_embryo_time))
        label_train_str = list(np.squeeze(label_celltype.values))
        label_train_str_set = list(set(label_train_str))
        label_train_str_set = sorted(label_train_str_set)
        label = tensor(
            np.array([label_train_str_set.index(i) for i in label_train_str]))
        
        dict_str_num_sample = {}
        for str_label in label_train_str_set:
            dict_str_num_sample[str_label] = (label==label_train_str_set.index(str_label)).sum()
            
        sorted_dict = sorted(dict_str_num_sample.items(), key=lambda x: x[1], reverse=True)
        for i in range(num_top_celltype):
            print(sorted_dict[i])
        
        
        time_list = []
        for str_time in label_embryo_time[0].to_list():
            if '-' in str_time:
                time_list.append(float(str_time.split('-')[1]))
            elif '<' in str_time:
                time_list.append(float(str_time.split('<')[1])-50)
            elif '>' in str_time:
                time_list.append(float(str_time.split('>')[1])+100)
            else:
                print('error', str_time)
                
        # # select the top num of sample 5 classes
        num_cell_type = len(label_train_str_set)
        num_sample_list = []
        for i in range(num_cell_type):
            num_sample_list.append((label==i).sum())
        
        # # select the top 5 cell types
        top5_cell_type = np.argsort(num_sample_list)[-num_top_celltype:]
        mask = np.zeros(len(label)).astype(np.bool_)
        for i in top5_cell_type:
            mask[label==i] = 1
        data = data[mask]
        label = label[mask]
        
        # PCA to 100
        pca = PCA(n_components=100)
        data = pca.fit_transform(data)
        
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0)
        
        data = (data - data_mean) / data_std
        
        label_str_new = list(set([str(l) for l in label]))
        label_str_new = sorted(label_str_new)
        label = torch.tensor([label_str_new.index(str(l)) for l in label])
        
        time_list = torch.tensor(time_list)
        time_list = time_list[mask]
            
        print('data.shape', data.shape)
        # time_label = torch.tensor(label_embryo_time)
        label = torch.cat((label.reshape(-1,1), time_list.reshape(-1,1)), dim=1)
        
        return data, label
    
    
    

# class CELEGANT7NoNormDataset(DigitsDataset):
    
#     def load_data(self, data_path, train=True,filter=True):
#         num_top_celltype = 7
#         adata = sc.read("/any/data/difftreedata/data/celegan/celegan.h5ad")
        
#         # top 500 genes
#         # import pdb; pdb.set_trace()
#         if filter:
        
#             data = adata.X    
#             np.var(data, axis=0)
#             top_genes = np.argsort(np.var(data, axis=0))[-500:]
#             data = data[:, top_genes].astype(np.float32)
        
#         std = data.std()
#         mean = data.mean()
#         data = (data - mean) / std
        
#         # import pdb; pdb.set_trace()
#         label_celltype = pd.read_csv('/any/data/difftreedata/data/celegan/celegan_celltype_2.tsv', sep='\t', header=None)
#         label_embryo_time = pd.read_csv('/any/data/difftreedata/data/celegan/celegan_embryo_time.tsv', sep='\t', header=None)
#         adata.obs['celltype'] = pd.Categorical(np.squeeze(label_celltype))
#         adata.obs['embryo_time'] = pd.Categorical(np.squeeze(label_embryo_time))
#         label_train_str = list(np.squeeze(label_celltype.values))
#         label_train_str_set = list(set(label_train_str))
#         label_train_str_set = sorted(label_train_str_set)
#         label = tensor(
#             np.array([label_train_str_set.index(i) for i in label_train_str]))
        
#         dict_str_num_sample = {}
#         for str_label in label_train_str_set:
#             dict_str_num_sample[str_label] = (label==label_train_str_set.index(str_label)).sum()
            
#         sorted_dict = sorted(dict_str_num_sample.items(), key=lambda x: x[1], reverse=True)
#         for i in range(num_top_celltype):
#             print(sorted_dict[i])
        
        
#         time_list = []
#         for str_time in label_embryo_time[0].to_list():
#             if '-' in str_time:
#                 time_list.append(float(str_time.split('-')[1]))
#             elif '<' in str_time:
#                 time_list.append(float(str_time.split('<')[1])-50)
#             elif '>' in str_time:
#                 time_list.append(float(str_time.split('>')[1])+100)
#             else:
#                 print('error', str_time)
                
#         # # select the top num of sample 5 classes
#         num_cell_type = len(label_train_str_set)
#         num_sample_list = []
#         for i in range(num_cell_type):
#             num_sample_list.append((label==i).sum())
        
#         # # select the top 5 cell types
#         top5_cell_type = np.argsort(num_sample_list)[-num_top_celltype:]
#         mask = np.zeros(len(label)).astype(np.bool_)
#         for i in top5_cell_type:
#             mask[label==i] = 1
#         data = data[mask]
#         label = label[mask]
        
#         # PCA to 100
#         pca = PCA(n_components=100)
#         data = pca.fit_transform(data)
        
#         data_mean = data.mean(axis=0)
#         data_std = data.std(axis=0)
        
#         data = (data - data_mean) / data_std
        
#         label_str_new = list(set([str(l) for l in label]))
#         label_str_new = sorted(label_str_new)
#         label = torch.tensor([label_str_new.index(str(l)) for l in label])
        
#         time_list = torch.tensor(time_list)
#         time_list = time_list[mask]
            
#         print('data.shape', data.shape)
#         # time_label = torch.tensor(label_embryo_time)
#         label = torch.cat((label.reshape(-1,1), time_list.reshape(-1,1)), dim=1)
        
#         return data, label