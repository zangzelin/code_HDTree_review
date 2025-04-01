# HDTree Documentation

## 1. Introduction

**HDTree** is a toolkit designed for obtaining high-dimensional data representations in a tree-structured latent space. It includes the following key components:

- **Datasets**: Tools for downloading, managing, and preprocessing datasets (e.g., MNIST, FMINST, 20News, Limb, Weinreb, ECL).
- **Training**: Implementation of the HDTree model's training process.
- **Evaluation**: Multiple evaluation metrics to assess the trained model.

This project is built for Python 3.9 and uses `conda` for environment management.

## 2. Environment Setup

It is recommended to use Anaconda (or Miniconda) to set up the Python environment. Follow these steps to install and manage dependencies:

```bash
conda create -n hdtree python=3.9
conda activate hdtree
bash install.sh
```

- `conda create -n hdtree python=3.9`: Creates a conda environment named **hdtree** with Python 3.9.
- `conda activate hdtree`: Activates the **hdtree** environment.
- `bash install.sh`: Runs the installation script to automatically install all required dependencies.

## 3. Datasets

This project supports the following datasets:

- MNIST

http://yann.lecun.com/exdb/mnist/

- Fashion-MINST

https://github.com/zalandoresearch/fashion-mnist

- 20News

http://qwone.com/~jason/20Newsgroups/

- CIFAR-10

https://www.cs.toronto.edu/~kriz/cifar.html

- LHCO

https://drctdb.cowtransfer.com/s/cc09cb54750

- Limb

https://cellgeni.cog.sanger.ac.uk/limb-dev/221024LimbCellranger3annotated_filtered_adjusted_20221124.minimal.h5ad

- **Weinreb:**

https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_gene_names.txt.gz

https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_metadata.txt.gz

https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_clone_matrix.mtx.gz

https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_normed_counts.mtx.gz

- **ECL:**

https://datasets.cellxgene.cziscience.com/1e466ab7-25b4-4540-8bed-b6aa63e28636.h5ad

### Dataset Organization

Ensure that datasets are placed in the appropriate directory or downloaded to the default directory, as configured in the project’s configuration files.

For example, the all dataset might be organized as follows:

Image Data

```
datasets/
├── MNIST/
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── FashionMNIST/
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── 20news/
│   ├── alt.atheism/
│   │   ├── 12345.txt
│   │   ├── 67890.txt
│   │   └── ...
│   ├── comp.graphics/
│   │   ├── 12346.txt
│   │   ├── 67891.txt
│   │   └── ...
│   └── ...
└── cifar-10-batches-py/
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    ├── test_batch
    └── batches.meta
```

Biology Data

```

datasets_bio/
├──  original/
|   ├── EpitheliaCell.h5ad
|   ├── LimbFilter.h5ad
|   ├── He_2022_NatureMethods_Day15.h5ad
|   ├── Weinreb_inVitro_clone_matrix.mtx
|   ├── Weinreb_inVitro_gene_names.txt
|   ├── Weinreb_inVitro_metadata.txt
|   └── Weinreb_inVitro_normed_counts.mtx
└── processed/  (it existes untill you run the process)
    ├── EpitheliaCell_data_n.npy
    ├── EpitheliaCell_label.npy
    ├── LimbFilter_data_n.npy
    ├── LimbFilter_label.npy
    ├── LHCO.h5ad
    └── Weinreb.h5ad

```

Other datasets should follow their respective official structures or be organized using the preprocessing scripts provided in this project.

## 4. Preprocessing

Before training, datasets need to be preprocessed. Preprocessing steps differ depending on the type of dataset. Below are the detailed guidelines:

### 4.1. Image Datasets

For image datasets (e.g., **MNIST**, **FMINST**), preprocessing is straightforward and can leverage the code from **TreeVAE**. The steps include:

1. **Downloading and organizing the data**: Ensure the dataset files (e.g., images and labels) are downloaded and placed in the appropriate directory.
2. **Converting raw data into NumPy arrays**: Use the provided scripts to load the raw image and label data and convert them into NumPy arrays.
3. **Normalization and formatting**: Normalize pixel values (e.g., scale to [0, 1]) and ensure the data format is compatible with the HDTree model.

You can directly use the preprocessing code from the TreeVAE project, located in the `datasets/` folder (e.g., `datasets/mnist.py`).

### 4.2. Biological Datasets

For biological datasets (**LHCO**, **Limb**, **Weinreb**, **ECL**), preprocessing is more complex and tailored to each dataset. Below are the detailed preprocessing steps for each:

#### **4.2.1 LHCO**

1. **Data cleaning**:
   - Remove duplicate entries and invalid samples.
   - Handle missing values by performing imputation or filtering rows with excessive missing data.
2. **Feature extraction**:
   - Extract relevant features from the raw data, such as particle kinematics or event-level feaThe detail code is shown in 'preprocess/pre_lhco.py'tures.
   - Apply feature scaling (e.g., standardization or normalization).
3. **Splitting**:
   - Split the dataset into training, validation, and test sets based on configurations.

The detail code is shown in 'preprocess/pre_lhco.py'

#### **4.2.2 Limb**

1. **Data loading**:
   - Read the dataset from the provided files (e.g., CSV or HDF5 formats).
2. **Filtering and cleaning**:
   - Remove noise or artifacts in the recorded data.
   - Apply deduplication and handle missing values appropriately.
3. **Feature engineering**:
   - Transform raw biological signals into meaningful features (e.g., limb motion patterns).
   - Normalize features to ensure consistent scales.
4. **Splitting**:
   - Stratify the dataset into training, validation, and test splits to maintain label distribution.

The detail code is shown in 'preprocess/pre_limb.py'

#### **4.2.3 Weinreb**

1. **Data transformation**:
   - Convert raw gene expression matrices into log-transformed or normalized values (e.g., CPM, TPM, or FPKM).
   - Filter out low-expression genes or cells with insufficient data.
2. **Dimensionality reduction**:
   - Apply PCA or other techniques to reduce the dimensionality of the dataset before input to the model.
3. **Batch correction**:
   - Perform batch effect correction if the data comes from multiple sources.
4. **Splitting**:
   - Divide the dataset into training, validation, and test sets, ensuring balanced cell-type distributions.

The detail code is shown in 'preprocess/pre_weinreb.py'

#### **4.2.4 ECL**

1. **Loading and parsing**:
   - Parse the raw files (e.g., sequencing or proteomics data) into tabular formats.
2. **Preprocessing**:
   - Normalize the data using z-scores or Min-Max scaling.
   - Handle missing values by imputing or removing incomplete samples.
3. **Feature selection**:
   - Select biologically relevant features based on domain knowledge (e.g., specific molecular signatures).
4. **Splitting**:
   - Ensure the dataset is split into training, validation, and test subsets while maintaining class balance.

The detail code is shown in 'preprocess/pre_ecl.py'

## 5 Baseline Methods

#### **5.1 TreeVAE**

The TreeVAE is installed as follow

```bash
git clone https://github.com/lauramanduchi/treevae.git
cd treevae
pip install -r minimal_requirements.txt
```

We run the TreeVAE with differnent configs, the details coinfigs refer to 'baseline/treevae_cfgs'

#### 5.2 CellPLM

The CellPLM is installed as follow:

```bash
git clone https://github.com/OmicsML/CellPLM.git
cd CellPLM
conda install cudatoolkit=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

We use the CellPLM like official [tutorials](https://github.com/OmicsML/CellPLM/blob/main/tutorials/cell_embedding.ipynb) in the [CelllPLM](https://github.com/OmicsML/CellPLM.git)

The details you can refer to 'baseline/CellPLM.py'

#### **5.3 LangCell**

The LangCell is installed as follow.

```bash
git clone https://github.com/PharMolix/LangCell.git
pip install -r requirements.txt
cd LangCell
cd geneformer_001
pip install .
```

We use the LangCell like offical [tutorials](https://github.com/PharMolix/LangCell/blob/main/LangCell-annotation-zeroshot/zero-shot.ipynb) in the [LangCell](https://github.com/PharMolix/LangCell.git)

The details you can refer to 'baseline/LangCell.py'

#### 5.4 Geneformer

After you install LangCell, the Geneformer is also installed.

We use the Geneformer like offical [tutorials](https://github.com/jkobject/geneformer/blob/main/examples/extract_and_plot_cell_embeddings.ipynb) in the [Geneformer](https://github.com/jkobject/geneformer.git)

The detials you can refer to 'baseline/Geneformer.py'

## 6. Running HDTree

### 6.1 Training Example

To train HDTree on the MNIST dataset in a single-GPU environment and obtain the final latent embedding, run:

```bash
python main.py fit -c conf/difftree/G_mnist_1gpu.yaml
```

### 6.2 Validate Example

For validation, you can use the following command:

```bash
python main.py validate -c conf/difftree/C_celegan_1gpu.yaml
python main.py validate -c conf/difftree/C_Weinreb_1gpu.yaml
python main.py validate -c conf/difftree/C_ECL_1gpu.yaml
python main.py validate -c conf/difftree/C_Limb_1gpu.yaml
python main.py validate -c conf/difftree/C_LHCO_1gpu.yaml
python main.py validate -c conf/difftree/G_fmnist_1gpu.yaml
python main.py validate -c conf/difftree/G_mnist_1gpu.yaml
python main.py validate -c conf/difftree/G_News20_1gpu.yaml
python main.py validate -c conf/difftree/G_omni_1gpu.yaml
```

- `main.py`: The main entry point for HDTree.
- `validate`: Specifies the training process.
- `-c ***.ymal`: Specifies the configuration file for the training process.

## 8. files of the project

The project is organized into the following directories:

- `aug`: Contains data augmentation methods.
- `call_backs`: Contains callback functions for training and validation.
- `conf`: Configuration files for different datasets and training settings.
- `conf/difftree`: Configuration files for the HDTree model.
- `data_model`: Contains the data model and data loading utilities.
- `eval`: Evaluation metrics and functions.
- `manifolds`: Contains scoft contrastive learning methods.
- `model`: Implementation of the HDTree model.
- `preprocess`: Preprocessing scripts for different datasets.
