# HDTree 

The code includes the following modules:
* Datasets (Mnist, FMinst, 20News, Limb, Weinreb, ECL)
* Training for HDTree
* Evaluation metrics 


## Configurating python environment

We recommend using conda for configuration. You can refer to our `install.sh` to configure the environment.

```bash
conda create -n hdtree python=3.9
conda activate hdtree
bash install.sh
```

## Dataset

This project utilizes several datasets, including `20NG`, `HCL`, `MNIST`, and `CIFAR-10`. Please follow the instructions below to understand the dataset structure and usage.

### 1. 20NG Dataset
The `20NG` dataset is already included in this GitHub repository.

### 2. HCL Dataset
The `HCL` dataset must be manually downloaded from the following link: [Download HCL Dataset](https://gofile.me/7794C/rSolqImMJ). Once downloaded, please place the file `HCL60kafter-elis-all.h5ad` into the `data_path/` directory.

### 3. MNIST and CIFAR-10 Datasets
The `MNIST` and `CIFAR-10` datasets do not require manual download. These datasets will be automatically downloaded upon the first execution of the project.
Please ensure that you have a stable internet connection during the first run to automatically download these datasets.

## Run HDTree

You can run HDTree with a single line of code to get latent embedding.

### Minimun replication

Running minimal replication can be done with the following command:

```bash
python main.py fit -c=conf/difftree/G_mnist_1gpu.yaml
```

