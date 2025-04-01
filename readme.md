
# HDTree Documentation

## Introduction

**HDTree** is a toolkit designed for obtaining high-dimensional data representations in a tree-structured latent space. It includes the following key components:

- **Datasets**: Tools for downloading, managing, and preprocessing datasets (e.g., MNIST, FMINST, 20News, Limb, Weinreb, ECL).
- **Training**: Implementation of the HDTree model's training process.
- **Evaluation**: Multiple evaluation metrics to assess the trained model.

This project is built for Python 3.9 and uses `conda` for environment management.

---

## Environment Setup

It is recommended to use Anaconda (or Miniconda) to set up the Python environment. Follow these steps to install and manage dependencies:

```bash
conda create -n hdtree python=3.9
conda activate hdtree
bash install.sh
```

- `conda create -n hdtree python=3.9`: Creates a conda environment named **hdtree** with Python 3.9.
- `conda activate hdtree`: Activates the **hdtree** environment.
- `bash install.sh`: Runs the installation script to automatically install all required dependencies.

---

## Datasets

This project supports the following datasets:

- MNIST
- FMINST
- 20News
- LHCO
- Limb
- Weinreb
- ECL

### Dataset Organization

Ensure that datasets are placed in the appropriate directory or downloaded to the default directory, as configured in the project’s configuration files.

For example, the MNIST dataset might be organized as follows:

```
datasets/
└── MNIST/
    ├── train-images-idx3-ubyte
    ├── train-labels-idx1-ubyte
    ├── t10k-images-idx3-ubyte
    └── t10k-labels-idx1-ubyte
```

Other datasets should follow their respective official structures or be organized using the preprocessing scripts provided in this project.

---

## Preprocessing

Before training, datasets need to be preprocessed. Typical preprocessing steps include:

1. **Converting raw data** into directly loadable formats (e.g., NumPy arrays).
2. **Data cleaning and filtering**, such as deduplication and handling missing values.
3. **Splitting the data** into training, validation, and test sets.

Refer to the corresponding dataset modules (e.g., `datasets/mnist.py`, `datasets/20news.py`) and configuration files for preprocessing scripts and paths.

---

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

---

## References

- For advanced features (e.g., custom datasets or multi-threaded training), refer to the configuration files and source code comments.
- If you encounter dependency or compatibility issues, ensure your local environment is correctly set up. Check the project’s issues or discussion section for potential solutions.
