from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
import numpy as np
import scanpy as sc
# embedding
CellPLM = CellEmbeddingPipeline(
            pretrain_prefix='20231027_85M', 
            pretrain_directory='CellPLM/ckpt')  
if __name__ == '__main__':
    adata = sc.read("data.h5ad")
    data_all = adata.X.toarray().astype(np.float32)
    vars = np.var(data_all, axis=0)
    mask_gene = np.argsort(vars)[-500:]
    adata = adata[:, mask_gene]
    emb = CellPLM.predict(adata, device='cuda:0').cpu().numpy()  
    print(emb.shape)
    np.save('emb.npy', emb)