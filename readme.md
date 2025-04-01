
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

<!-- 这里帮我把每个数据集的下载地址贴上 -->
- MNIST
- FMINST
- 20News
- LHCO
- Limb
- Weinreb
- ECL

### Dataset Organization

Ensure that datasets are placed in the appropriate directory or downloaded to the default directory, as configured in the project’s configuration files.

For example, the all dataset might be organized as follows:

<!-- 这里帮我把每个数据集的下载地址所下载的位置贴上 -->

```
datasets/
└── MNIST/
    ├── train-images-idx3-ubyte
    ├── train-labels-idx1-ubyte
    ├── t10k-images-idx3-ubyte
    └── t10k-labels-idx1-ubyte
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
2. **4.2.1 Feature extraction**:
   - Extract relevant features from the raw data, such as particle kinematics or event-level features.
   - Apply feature scaling (e.g., standardization or normalization).
3. **4.2.1 Splitting**:
   - Split the dataset into training, validation, and test sets based on configurations.

The detail code is shown in 'preprocess/pre_lhco.py'

#### **Limb**
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

#### **Weinreb**
1. **Data transformation**:
   - Convert raw gene expression matrices into log-transformed or normalized values (e.g., CPM, TPM, or FPKM).
   - Filter out low-expression genes or cells with insufficient data.
2. **Dimensionality reduction**:
   - Apply PCA or other techniques to reduce the dimensionality of the dataset before input to the model.
3. **Batch correction**:
   - Perform batch effect correction if the data comes from multiple sources.
4. **Splitting**:
   - Divide the dataset into training, validation, and test sets, ensuring balanced cell-type distributions.

#### **ECL**
1. **Loading and parsing**:
   - Parse the raw files (e.g., sequencing or proteomics data) into tabular formats.
2. **Preprocessing**:
   - Normalize the data using z-scores or Min-Max scaling.
   - Handle missing values by imputing or removing incomplete samples.
3. **Feature selection**:
   - Select biologically relevant features based on domain knowledge (e.g., specific molecular signatures).
4. **Splitting**:
   - Ensure the dataset is split into training, validation, and test subsets while maintaining class balance.



## Running HDTree

### Minimal Reproducible Example

To train HDTree on the MNIST dataset in a single-GPU environment and obtain the final latent embedding, run:

```bash
python main.py fit -c=conf/difftree/G_mnist_1gpu.yaml
```

- `main.py`: The main entry point for HDTree.
- `fit`: Specifies the training process.
- `-c=conf/difftree/G_mnist_1gpu.yaml`: Specifies the configuration file, which includes dataset paths, model hyperparameters, and training strategies.

### Training on Other Datasets or Multi-GPU Environments

To train on other datasets or in multi-GPU environments, update the configuration file (e.g., `conf/difftree/G_*_*.yaml`) accordingly.


## References

- For advanced features (e.g., custom datasets or multi-threaded training), refer to the configuration files and source code comments.
- If you encounter dependency or compatibility issues, ensure your local environment is correctly set up. Check the project’s issues or discussion section for potential solutions.
