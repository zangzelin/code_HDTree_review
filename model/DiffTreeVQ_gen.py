import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.cluster import KMeans
import scipy
from lightning import LightningModule
from model.diffmodel.model import AE_CNN_bottleneck_deep, AE, AE_layer2
from model.diffmodel.diffusion_gen_plot import GaussianDiffusion, make_beta_schedule
from joblib import parallel_backend
from collections import OrderedDict
import time
import uuid


def kmeans_numpy(X: np.ndarray,
                 n_clusters: int,
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 random_state: int = 0,
                 verbose: bool = False):
    """
    使用 NumPy 实现的简易 KMeans 算法

    参数:
    -------
    X: np.ndarray
        输入数据，形状为 [N, D]，N 为样本数，D 为特征维度。
    n_clusters: int
        聚类簇的数量。
    max_iter: int, default=100
        最大迭代次数。
    tol: float, default=1e-4
        用于判断中心变化量是否小于该阈值从而提前停止迭代。
    random_state: int, default=0
        随机种子，方便复现。
    verbose: bool, default=False
        是否打印每次迭代的信息。

    返回:
    -------
    labels: np.ndarray
        每个样本的聚类标签，形状为 [N]。
    centers: np.ndarray
        最终聚类中心，形状为 [n_clusters, D]。
    """
    # ---------------------------------------------------------
    # 1. 数据准备
    # ---------------------------------------------------------
    np.random.seed(random_state)
    X = X.astype(np.float32)  # 确保为 float 类型
    N, D = X.shape

    # ---------------------------------------------------------
    # 2. 初始化中心
    # ---------------------------------------------------------
    # 随机从 X 中选取 n_clusters 个样本作为初始中心
    initial_idxs = np.random.choice(N, n_clusters, replace=False)
    centers = X[initial_idxs].copy()

    # 用于存储上一轮迭代的中心
    old_centers = np.zeros_like(centers)

    # ---------------------------------------------------------
    # 3. 迭代更新
    # ---------------------------------------------------------
    for i in range(max_iter):
        # a) 计算每个样本到各中心的距离
        # distance: [N, n_clusters]
        # distances[n, k]: 第 n 个样本到第 k 个中心的距离
        # 这里可以使用广播技巧，也可以直接做循环
        # 为了简洁，这里直接用广播 (X[:, None, :] - centers[None, :, :])**2
        # 并在最后再对特征维度进行 sum 和 sqrt
        distances = np.sqrt(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))

        # b) 根据最小距离分配标签
        labels = np.argmin(distances, axis=1)

        # c) 更新聚类中心
        for k in range(n_clusters):
            # 找到分到第 k 簇的所有数据点
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centers[k] = np.mean(cluster_points, axis=0)
            else:
                # 如果某个簇为空，可根据需要重新随机或其他策略
                # 这里简单跳过不做特殊处理
                pass

        # d) 判断是否收敛（中心移动量）
        center_shift = np.sqrt(((centers - old_centers) ** 2).sum())
        
        if verbose:
            print(f"Iter {i+1}/{max_iter}, center shift: {center_shift:.6f}")

        if center_shift < tol:
            break

        old_centers = centers.copy()
        
    return labels, centers

def progressive_samples_fn_simple(
    model,
    diffusion,
    shape,
    device,
    cond,
    include_x0_pred_freq=50,
    img_init=None,
):

    samples, history = diffusion.p_sample_loop_progressive_simple(
        model=model,
        shape=shape,
        noise_fn=torch.randn,
        device=device,
        include_x0_pred_freq=include_x0_pred_freq,
        cond=cond,
        img_init=img_init,
    )
    return samples, history


def progressive_samples_fn_simple_zl_step2(
    model,
    diffusion,
    shape,
    device,
    cond,
    include_x0_pred_freq=50,
    img_init=None,
):
    if img_init is None:
        img_init = torch.randn(shape, dtype=torch.float32).to(device)

    samples, history = diffusion.p_sample_loop_progressive_zl_step2(
        model=model,
        shape=shape,
        noise_fn=torch.randn,
        device=device,
        include_x0_pred_freq=include_x0_pred_freq,
        cond=cond,
        img_init=img_init,
    )
    return samples, history


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


# Function to apply Gumbel softmax to multiple sets of logits and select top N indices
def muti_gumbel(logits, tau=1, hard=False, eps=1e-10, dim=-1, top_N=10, num_use_moe=10):
    """
    Applies Gumbel softmax to multiple sets of logits and selects top N indices for each.
    Returns hard and soft masks.

    Args:
        logits (Tensor): Input logits of shape (batch_size, num_use_moe, num_features).
        tau (float): Temperature parameter for Gumbel softmax.
        hard (bool): Whether to return hard one-hot samples.
        eps (float): Small value to avoid numerical issues (deprecated).
        dim (int): Dimension along which softmax is applied.
        top_N (int): Number of top indices to select.
        num_use_moe (int): Number of mixtures of experts.

    Returns:
        mask (Tensor): Hard masks of shape (batch_size, num_use_moe, num_features).
        mask_soft (Tensor): Soft masks of shape (batch_size, num_use_moe, num_features).
    """
    mask_list = []
    mask_soft_list = []
    for i in range(num_use_moe):
        # Apply Gumbel softmax to each set of logits
        mask_soft, mask = gumbel_softmax_topN(
            logits[:, i, :], tau=tau, hard=hard, eps=eps, dim=dim, top_N=top_N
        )
        mask_list.append(mask)
        mask_soft_list.append(mask_soft)
    # Stack masks along new dimension
    return torch.stack(mask_list, dim=1), torch.stack(mask_soft_list, dim=1)


# Function to perform Gumbel softmax sampling and select top N indices
def gumbel_softmax_topN(logits, tau=1, hard=False, eps=1e-10, dim=-1, top_N=10):
    """
    Performs Gumbel softmax sampling and selects top N indices.

    Args:
        logits (Tensor): Input logits of shape (batch_size, num_features).
        tau (float): Temperature parameter.
        hard (bool): Whether to return hard one-hot samples.
        eps (float): Small value to avoid numerical issues (deprecated).
        dim (int): Dimension along which softmax is applied.
        top_N (int): Number of top indices to select.

    Returns:
        y_soft (Tensor): Softmax probabilities after Gumbel noise is added.
        ret (Tensor): Hard or soft samples depending on 'hard' flag.
    """
    # Note: 'eps' parameter is deprecated and has no effect
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    # Sample Gumbel noise
    gumbels = -torch.empty_like(logits).exponential_().log()
    # Add Gumbel noise to logits and scale by temperature
    gumbels = (logits + gumbels) / tau
    # Apply softmax
    y_soft = gumbels.softmax(dim)

    if hard:
        # Get top N indices
        index = y_soft.topk(k=top_N, dim=dim)[1]
        # Create hard one-hot encoding
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        # Straight-through estimator
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Return soft probabilities
        ret = y_soft
    return y_soft, ret


# Cosine annealing learning rate scheduler with warmup
class CosineAnnealingSchedule(_LRScheduler):
    """Cosine annealing with warmup."""

    def __init__(self, opt, final_lr=0, n_epochs=1000, warmup_epochs=10, warmup_lr=0):
        """
        Initializes the scheduler.

        Args:
            opt (Optimizer): Optimizer.
            final_lr (float): Final learning rate after decay.
            n_epochs (int): Total number of epochs.
            warmup_epochs (int): Number of warmup epochs.
            warmup_lr (float): Initial learning rate for warmup.
        """
        self.opt = opt
        self.optimizer = self.opt
        self.base_lr = base_lr = opt.defaults["lr"]
        self.final_lr = final_lr
        self.n_epochs = n_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr

        # Compute number of decay epochs
        decay_epochs = 1 + n_epochs - warmup_epochs
        self.decay_epochs = decay_epochs

        # Warmup schedule: linearly increase lr from warmup_lr to base_lr
        warmup_schedule = np.linspace(warmup_lr, base_lr, warmup_epochs)
        # Decay schedule: cosine annealing from base_lr to final_lr
        decay_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_epochs) / decay_epochs)
        )
        # Concatenate warmup and decay schedules
        self.lr_schedule = np.hstack((warmup_schedule, decay_schedule))

        self._last_lr = self.lr_schedule[0]
        self.cur_epoch = 0

        self.init_opt()

    def init_opt(self):
        """Initializes the optimizer learning rate."""
        self.step()
        # self.set_epoch(0)

    def get_lr(self):
        """Gets the current learning rate."""
        return self.lr_schedule[self.cur_epoch]

    def step(self):
        """Updates the learning rate for the optimizer."""
        lr = self.get_lr()
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

        self.cur_epoch += 1
        self._last_lr = lr
        return lr

    def set_epoch(self, epoch):
        """Sets the current epoch (for resuming training)."""
        self.cur_epoch = epoch


# Define a neural network module with Linear, BatchNorm, and LeakyReLU layers
class NN_FCBNRL_MM(nn.Module):
    """
    Neural network module consisting of Linear, BatchNorm, Dropout, and LeakyReLU layers.
    """

    def __init__(
        self, in_dim, out_dim, channel=8, use_RL=True, use_BN=True, use_DO=True
    ):
        """
        Initializes the module.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            channel (int): Unused parameter.
            use_RL (bool): Whether to use LeakyReLU activation.
            use_BN (bool): Whether to use BatchNorm1d.
            use_DO (bool): Whether to use Dropout.
        """
        super(NN_FCBNRL_MM, self).__init__()
        layers = []
        # Linear layer
        layers.append(nn.Linear(in_dim, out_dim))
        # Optional Dropout
        # if use_DO:
        #     layers.append(nn.Dropout(p=0.02))
        # Optional BatchNorm
        if use_BN:
            layers.append(nn.BatchNorm1d(out_dim))
        # Optional LeakyReLU activation
        if use_RL:
            layers.append(nn.LeakyReLU(0.1))

        # Create the sequential block
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of the module."""
        return self.block(x)


# Transformer Encoder with optional Mixture of Experts (MoE)
class TransformerEncoder(nn.Module):
    """
    Transformer Encoder module with optional Mixture of Experts (MoE).
    """

    def __init__(
        self,
        num_layers=2,
        num_attention_heads=6,
        hidden_size=240,
        intermediate_size=300,
        max_position_embeddings=784,
        num_input_dim=784,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_use_moe=10,
        output_dim=512,
        use_moe=True,
    ):
        """
        Initializes the Transformer Encoder.

        Args:
            num_layers (int): Number of layers.
            num_attention_heads (int): Number of attention heads.
            hidden_size (int): Hidden size.
            intermediate_size (int): Intermediate size.
            max_position_embeddings (int): Maximum position embeddings.
            num_input_dim (int): Input dimension size.
            hidden_dropout_prob (float): Dropout probability for hidden layers.
            attention_probs_dropout_prob (float): Dropout probability for attention.
            num_use_moe (int): Number of experts in MoE.
            use_moe (bool): Whether to use Mixture of Experts.
        """
        super(TransformerEncoder, self).__init__()
        self.use_moe = use_moe

        # Determine the type of network to use based on input dimension
        if num_input_dim == 3072:
            nn_type = "resnet"
            print("Using ResNet")
        else:
            nn_type = "nn"
            print("Using fully connected network")

        self.enc = self.network_single(
            num_input_dim,
            hidden_size,
            num_layers,
            nn_type=nn_type,
        )

        self.fc = nn.Sequential(
            NN_FCBNRL_MM(hidden_size, output_dim, use_RL=False),
        )

    def network_single(self, num_input_dim, hidden_size, num_layers, nn_type="nn"):
        """
        Creates a single network (either ResNet or fully connected).

        Args:
            num_input_dim (int): Input dimension.
            hidden_size (int): Hidden size.
            num_layers (int): Number of layers.
            nn_type (str): Type of network ('nn' or 'resnet').

        Returns:
            enc (nn.Module): The network module.
        """
        if nn_type == "resnet":
            # Use ResNet architecture
            enc = ResNet(BasicBlock, [2, 2, 2, 2], 3)
        else:
            # Build fully connected network
            layers = []
            layers.append(NN_FCBNRL_MM(num_input_dim, hidden_size))
            for _ in range(num_layers):
                layers.append(NN_FCBNRL_MM(hidden_size, hidden_size))
            layers.append(NN_FCBNRL_MM(hidden_size, hidden_size, use_RL=False))
            enc = nn.Sequential(*layers)
        return enc

    def forward(self, input_x):
        """
        Forward pass of the Transformer Encoder.

        Args:
            input_x (Tensor): Input tensor of shape (batch_size, num_use_moe, ...).

        Returns:
            emb (Tensor): Output embeddings.
        """
        # if self.use_moe:
        #     # If using MoE, apply each expert to the input
        #     emb_all = [self.fc(enc(input_x[:, i, :])) for i, enc in enumerate(self.enc)]
        #     emb = torch.stack(emb_all, dim=1)
        # else:
        # Single encoder
        emb = self.fc(self.enc(input_x))
        return emb


# Main model class
class DMTEVT_model(LightningModule):
    """
    DMTEVT_model is a PyTorch Lightning module that implements the training and evaluation of the model.
    """

    def __init__(
        self,
        lr=0.005,
        sigma=0.05,
        sample_rate_feature=0.6,
        num_input_dim=64,
        num_train_data=60000,
        weight_decay=0.0001,
        exaggeration_lat=1,
        exaggeration_emb=1,
        weight_mse=2,
        weight_nepo=1,
        nu_lat=0.1,
        nu_emb=0.1,
        tau=1,
        T_num_layers=2,
        T_num_attention_heads=6,
        T_hidden_size=240,
        T_intermediate_size=300,
        t_output_dim=512,
        T_hidden_dropout_prob=0.1,
        T_attention_probs_dropout_prob=0.1,
        ckpt_path=None,
        use_orthogonal=False,
        num_use_moe=1,
        vis_dim=2,
        trans_out_dim=50,
        max_epochs=600,
        ec_ce_weight=10.0,
        n_neg_sample=4,
        test_noise=False,
        training_str=None,
        tree_depth=10,
        n_timestep=1000,
        epoch_num_base=0,
        validate_bool=False,
        weight_e_latent=0.25,
        step2_epoch=2000,
        step2_r_epoch=4000,
        use_tree_rout=False,
        gen_data_bool=False,
        weightrout=0.1,
        **kwargs,
    ):
        """
        Initializes the model with given hyperparameters.

        Args:
            lr (float): Learning rate.
            sigma (float): Sigma parameter for similarity function.
            sample_rate_feature (float): Sampling rate for features.
            num_input_dim (int): Input dimension size.
            num_train_data (int): Number of training data samples.
            weight_decay (float): Weight decay for optimizer.
            exaggeration_lat (float): Exaggeration parameter for latent space.
            exaggeration_emb (float): Exaggeration parameter for embedding space.
            weight_mse (float): Weight for MSE loss.
            weight_nepo (float): Weight for NEPO loss.
            nu_lat (float): Degrees of freedom for t-distribution in latent space.
            nu_emb (float): Degrees of freedom for t-distribution in embedding space.
            tau (float): Temperature parameter.
            T_num_layers (int): Number of layers in Transformer.
            T_num_attention_heads (int): Number of attention heads in Transformer.
            T_hidden_size (int): Hidden size in Transformer.
            T_intermediate_size (int): Intermediate size in Transformer.
            T_hidden_dropout_prob (float): Dropout probability in Transformer.
            T_attention_probs_dropout_prob (float): Dropout probability for attention in Transformer.
            ckpt_path (str): Path to checkpoint for loading model.
            use_orthogonal (bool): Whether to use orthogonal loss.
            num_use_moe (int): Number of experts in Mixture of Experts.
            vis_dim (int): Dimension of visualization space.
            trans_out_dim (int): Output dimension of Transformer.
            max_epochs (int): Maximum number of epochs.
            v_latent (float): Variance parameter in latent space.
            n_neg_sample (int): Number of negative samples.
            test_noise (bool): Whether to test with noise.
            **kwargs: Additional arguments.
        """
        super().__init__()

        self.setup_bool_zzl = False
        self.save_hyperparameters()

        num_input_dim = self.hparams.num_input_dim
        self.init_exp_bool = False
        self.lat_vis_mean = nn.Parameter(torch.zeros(2))
        self.lat_vis_std = nn.Parameter(torch.zeros(2))
        self.init_imge = None
        
        self.uuid_str = str(uuid.uuid4())[:10]

        if self.hparams.nu_emb < 0:
            self.hparams.nu_emb = self.hparams.nu_lat
        if self.hparams.exaggeration_emb < 0:
            self.hparams.exaggeration_emb = self.hparams.exaggeration_lat

        # Initialize the encoder
        self.enc = TransformerEncoder(
            num_layers=T_num_layers,
            num_attention_heads=T_num_attention_heads,
            hidden_size=T_hidden_size,
            intermediate_size=T_intermediate_size,
            max_position_embeddings=20,
            num_input_dim=num_input_dim,
            hidden_dropout_prob=T_hidden_dropout_prob,
            attention_probs_dropout_prob=T_attention_probs_dropout_prob,
            num_use_moe=num_use_moe,
            output_dim=t_output_dim,
        )

        self.UNet_model = AE_layer2(
            in_dim=self.hparams.num_input_dim,
            mid_dim=16000,
            cond_input_len=self.hparams.tree_depth * 2,
        )
        self.UNet_ema = AE_layer2(
            in_dim=self.hparams.num_input_dim,
            mid_dim=16000,
            cond_input_len=self.hparams.tree_depth * 2,
        )
        
        self.val_vis_list = []

        self.tree_node_embedding = nn.ModuleList(
            [nn.Embedding(2 ** (i + 1), 2) for i in range(self.hparams.tree_depth)]
        )

        self.vis = self.InitNetworkMLP(
            NS=[t_output_dim * num_use_moe, 500, vis_dim], last_relu=False
        )

        self.betas = make_beta_schedule(
            schedule="linear", start=1e-4, end=2e-2, n_timestep=n_timestep
        )
        self.diffusion = GaussianDiffusion(
            betas=self.betas,
            model_mean_type="eps",
            model_var_type="fixedlarge",
            loss_type="mse",
        )

        if training_str == None:
            self.training_str = "step1"
        else:
            self.training_str = training_str

        self.validate_bool = validate_bool

        # Load checkpoint if provided
        if ckpt_path is not None:

            state_dict = torch.load(ckpt_path)
            # import pdb; pdb.set_trace()
            if "module." in list(state_dict.keys())[0]:
                print("Loading checkpoint from multi gpu:", ckpt_path)
                # state_dict = torch.load('path_to_your_model')
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith("module.") else k  # remove `module.`
                    new_state_dict[name] = v
                self.load_state_dict(new_state_dict)
            else:
                print("Loading checkpoint from:", ckpt_path)
                self.load_state_dict(torch.load(ckpt_path))
            # self.current_epoch = 5000

    def InitNetworkMLP(self, NS, last_relu=True, use_DO=True, use_BN=True, use_RL=True):
        """
        Initializes a multi-layer perceptron (MLP) network.

        Args:
            NS (list): List of layer sizes.
            last_relu (bool): Whether to use ReLU activation on the last layer.
            use_DO (bool): Whether to use Dropout.
            use_BN (bool): Whether to use BatchNorm.
            use_RL (bool): Whether to use LeakyReLU activation.

        Returns:
            model_pat (nn.Sequential): The MLP network.
        """
        layers = []
        for i in range(len(NS) - 1):
            # Determine if last layer should have activation
            if i == len(NS) - 2 and not last_relu:
                layers.append(
                    NN_FCBNRL_MM(
                        NS[i], NS[i + 1], use_RL=False, use_DO=use_DO, use_BN=use_BN
                    )
                )
            else:
                layers.append(
                    NN_FCBNRL_MM(
                        NS[i], NS[i + 1], use_RL=use_RL, use_DO=use_DO, use_BN=use_BN
                    )
                )
        model_pat = nn.Sequential(*layers)
        return model_pat

    def align_loss(
        self,
        rooter_input,
        emb_level_item,
        distances,
    ):

        num_embeddings = emb_level_item.shape[0]
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (B*H*W, 1)

        encodings = torch.zeros(
            encoding_indices.size(0), num_embeddings, device=rooter_input.device
        )
        encodings.scatter_(1, encoding_indices, 1)  # One-hot encoding

        # Quantize and reshape
        quantized = torch.matmul(encodings.detach(), emb_level_item).view(
            rooter_input.shape
        )  # Reshape back
        quantized = quantized.contiguous()  # (B, C, H, W)

        # import pdb; pdb.set_trace()
        e_latent_loss = F.mse_loss(quantized.detach(), rooter_input)
        q_latent_loss = F.mse_loss(quantized, rooter_input.detach())
        loss = q_latent_loss + self.hparams.weight_e_latent * e_latent_loss

        quantized = (
            rooter_input + (quantized - rooter_input).detach()
        )  # Straight-through estimator

        return encoding_indices, quantized, loss

    def cal_distance_matrix_with_tree(
        self,
        rooter_input,
        emb_level_item,
        last_tree_node_idx=None,
        tree_rout_bool=False,
    ):

        batch_size = rooter_input.shape[0]
        distances = (
            (rooter_input**2).sum(dim=1, keepdim=True)
            + (emb_level_item**2).sum(dim=1)
            - 2 * torch.matmul(rooter_input, emb_level_item.t())
        )
        if last_tree_node_idx is not None and tree_rout_bool:
            distances_plus = torch.full_like(distances, float("inf"))

            row_indices = torch.arange(
                batch_size, device=rooter_input.device
            ).repeat_interleave(2)
            index_s = last_tree_node_idx * 2
            col_indices = torch.arange(2, device=rooter_input.device).repeat(
                batch_size
            ) + index_s.repeat_interleave(2)
            distances_plus[row_indices, col_indices] = 0
            distances_on_tree = distances + distances_plus
        else:
            distances_on_tree = distances

        return distances, distances_on_tree

    def router_forward(self, rooter_input, tree_rout_bool=False, ec_ce_weight=10):
        tree_rout_list = []
        vector_list = []
        loss_list = []

        for i in range(len(self.tree_node_embedding)):
            emb_level_item = self.tree_node_embedding[i].weight
            if i > 0:
                last_tree_node_idx = tree_rout_list[-1]
            else:
                last_tree_node_idx = None

            distances, distances_on_tree = self.cal_distance_matrix_with_tree(
                rooter_input, emb_level_item, last_tree_node_idx, tree_rout_bool
            )

            if last_tree_node_idx is not None:
                encoding_indices, quantized, loss_ec_tree = self.align_loss(
                    rooter_input, emb_level_item, distances_on_tree
                )
                _, _, loss_ce_tree = self.align_loss(
                    emb_level_item, rooter_input, distances_on_tree.t()
                )
                loss = loss_ec_tree  + loss_ce_tree * ec_ce_weight
            else:
                encoding_indices, quantized, loss_ec = self.align_loss(
                    rooter_input, emb_level_item, distances
                )
                _, _, loss_ce = self.align_loss(
                    emb_level_item, rooter_input, distances.t()
                )
                loss = loss_ec  + loss_ce * ec_ce_weight

            tree_rout_list.append(encoding_indices.reshape(-1))
            vector_list.append(quantized)
            loss_list.append(loss)

        tree_rout = torch.stack(tree_rout_list, axis=1)
        vector_rout = torch.stack(vector_list, axis=1)
        loss = torch.stack(loss_list).mean()
        return tree_rout, vector_rout, loss

    def _DistanceSquared(self, x, y=None, metric="euclidean"):
        """
        Computes squared Euclidean distance between samples.

        Args:
            x (Tensor): Input tensor of shape (n_samples, n_features).
            y (Tensor): Optional second input tensor.
            metric (str): Distance metric to use ('euclidean').

        Returns:
            dist (Tensor): Distance matrix.
        """
        if metric == "euclidean":
            if y is not None:
                m, n = x.size(0), y.size(0)
                xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
                yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
                dist = xx + yy
                dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
                dist = dist.clamp(min=1e-12)
            else:
                m, n = x.size(0), x.size(0)
                xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
                yy = xx.t()
                dist = xx + yy
                dist = torch.addmm(dist, mat1=x, mat2=x.t(), beta=1, alpha=-2)
                dist = dist.clamp(min=1e-12)
                dist[torch.eye(dist.shape[0], dtype=torch.bool)] = 1e-12
        return dist

    def _CalGamma(self, v):
        """
        Calculates the gamma function value.

        Args:
            v (float): Degrees of freedom.

        Returns:
            out (float): Gamma function value.
        """
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b
        return out

    def _Similarity(self, dist, sigma=0.3):
        """
        Computes similarity using Gaussian kernel.

        Args:
            dist (Tensor): Distance matrix.
            sigma (float): Standard deviation of the Gaussian kernel.

        Returns:
            Pij (Tensor): Similarity matrix.
        """
        dist = dist.clamp(min=0)
        Pij = torch.exp(-dist / (2 * sigma**2))
        return Pij

    def t_distribution_similarity(self, distance_matrix, df):
        """
        Computes similarity matrix using t-distribution kernel.

        Args:
            distance_matrix (Tensor): Distance matrix.
            df (float): Degrees of freedom for t-distribution.

        Returns:
            similarity_matrix (Tensor): Similarity matrix.
        """
        distance_matrix = distance_matrix + 1e-6
        numerator = (1 + distance_matrix**2 / df) ** (-(df + 1) / 2)
        denominator = torch.sum(numerator, dim=1, keepdim=True) - torch.diagonal(
            numerator, 0
        ).unsqueeze(1)
        similarity_matrix = numerator / denominator
        return similarity_matrix

    def LossManifold(self, latent_data, temperature=1, exaggeration=1, nu=0.1):
        """
        Computes the manifold loss between two views of the data.

        Args:
            latent_data (Tensor): Latent representations of shape (2 * batch_size, ...).
            temperature (float): Temperature scaling.
            exaggeration (float): Exaggeration factor.
            nu (float): Degrees of freedom for t-distribution.

        Returns:
            loss (Tensor): Computed loss.
        """
        batch_size = latent_data.shape[0] // 2
        features_a = latent_data[:batch_size]
        features_b = latent_data[batch_size:]

        # Compute pairwise distances
        dis_aa = torch.cdist(features_a, features_a) * temperature
        dis_bb = torch.cdist(features_b, features_b) * temperature
        dis_ab = torch.cdist(features_a, features_b) * temperature

        # Compute similarity matrices using t-distribution
        sim_aa = self.t_distribution_similarity(dis_aa, df=nu)
        sim_bb = self.t_distribution_similarity(dis_bb, df=nu)
        sim_ab = self.t_distribution_similarity(dis_ab, df=nu)

        # Compute alignment term
        tempered_alignment = (torch.diagonal(sim_ab).log()).mean()

        # Exclude self similarities
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, 0.0)
        sim_bb.masked_fill_(self_mask, 0.0)

        # Compute uniformity terms
        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).sum(1).log_().mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).sum(1).log_().mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        # Compute final loss
        loss = -(exaggeration * tempered_alignment - raw_uniformity / 2)

        return loss

    def batch_patten_loss(self, feature_tra, mask):
        """
        Computes orthogonal loss to encourage diversity among experts.

        Args:
            feature_tra (Tensor): Transformed features.
            mask (Tensor): Masks indicating selected features.

        Returns:
            loss (Tensor): Computed loss.
        """
        # Add small noise to features
        feature_tra = (
            feature_tra + torch.randn_like(feature_tra) * 0.001 * feature_tra.std()
        )
        batch_size = feature_tra.shape[0] // 8
        feature_tra = feature_tra[:batch_size]
        mask = mask[:batch_size]

        mean_value_list = []
        for i in range(feature_tra.shape[1]):
            fea_ins = feature_tra[:, i, :]
            mask_ins = mask[:, i, :] == 1
            fea_ins_umask = fea_ins[mask_ins == 1].reshape((feature_tra.shape[0], -1))
            # Compute cosine similarity
            cosine_similarity_matrix = torch.nn.functional.cosine_similarity(
                fea_ins_umask.unsqueeze(1), fea_ins_umask.unsqueeze(0), dim=2
            )
            upper_triangular_matrix_no_diag = torch.triu(
                cosine_similarity_matrix, diagonal=1
            )
            mean_value = upper_triangular_matrix_no_diag.mean()
            mean_value_list.append(mean_value)

        # Return the mean of the mean values
        return 1 + torch.stack(mean_value_list).mean()

    def forward(self, x, tau=100.0):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input data.
            tau (float): Temperature parameter for Gumbel softmax.

        Returns:
            x_masked (Tensor): Masked input data.
            lat_higt_dim_out (Tensor): High-dimensional latent outputs.
            lat_vis (Tensor): Low-dimensional visualization outputs.
            lat_high_dim (Tensor): High-dimensional latent representations.
        """
        batch_size = x.shape[0] // 2
        x_masked = x

        # Pass through encoder
        lat_higt_dim_out = self.enc(x_masked)
        lat_vis = self.vis(lat_higt_dim_out)

        # import pdb; pdb.set_trace()

        return x_masked, lat_higt_dim_out, lat_vis, lat_higt_dim_out

    def get_weight(self):
        """
        Retrieves and processes the expert weights.

        Returns:
            weight (Tensor): Processed weights.
        """
        w = self.exp(torch.arange(self.hparams.num_use_moe).to(self.device)).reshape(
            1, self.hparams.num_use_moe, -1
        )
        weight = F.tanh(w) * 10
        return weight

    def get_tau(self, epoch, total_epochs=900, tau_start=100, tau_end=1.001):
        """
        Computes the temperature parameter tau for Gumbel softmax.

        Args:
            epoch (int): Current epoch.
            total_epochs (int): Total number of epochs.
            tau_start (float): Initial tau value.
            tau_end (float): Final tau value.

        Returns:
            tau (float): Computed tau value.
        """
        if epoch >= total_epochs:
            return tau_end
        else:
            return tau_start * (tau_end / tau_start) ** (epoch / (total_epochs - 1))

    def forward_train_enc(self, x_masked, lat_high_dim, lat_vis):

        # Compute orthogonal loss if required
        if self.hparams.use_orthogonal:
            orthogonal_loss = self.batch_patten_loss(x_masked, self.mask)
        else:
            orthogonal_loss = 0

        # Compute manifold losses
        loss_lat = self.LossManifold(
            latent_data=lat_high_dim.reshape(lat_high_dim.shape[0], -1),
            temperature=1,
            exaggeration=self.hparams.exaggeration_lat,
            nu=self.hparams.nu_lat,
        )
        loss_emb = self.LossManifold(
            latent_data=lat_vis.reshape(lat_vis.shape[0], -1),
            temperature=1,
            exaggeration=self.hparams.exaggeration_emb,
            nu=self.hparams.nu_emb,
        )

        return loss_emb, loss_lat, orthogonal_loss

    def update_training_str(self, epoch):
        """
        Updates the training string based on the current epoch.

        If the current epoch is greater than 20, the training string is set to
        'step2', indicating that the model is in the second stage of training.
        """

        if epoch > self.hparams.step2_epoch:
            self.training_str = "step2_s"
        if epoch > self.hparams.step2_r_epoch:
            self.training_str = "step2_r"

        print(f"self.training_str {self.training_str}, epoch {epoch}")

    def align_the_node_embedding(self, emb):
        """
        Reinitialize the embeddings in `self.tree_node_embedding` such that
        their mean and variance match the provided embedding `emb`.

        Args:
            emb (Tensor): Input tensor of shape (batch_size, embedding_dim), used to
                        calculate the mean and variance for reinitialization.
        """
        # Ensure the input embedding has valid dimensions
        if emb.ndim != 2:
            raise ValueError(
                "Expected emb to have shape (batch_size, embedding_dim), got shape: {}".format(
                    emb.shape
                )
            )

        # Calculate mean and standard deviation from input embedding
        mean_emb = emb.mean(dim=0, keepdim=True)
        std_emb = emb.std(dim=0, keepdim=True)

        # Iterate through each embedding layer in `self.tree_node_embedding`
        for i, embedding_layer in enumerate(self.tree_node_embedding):
            # Get the shape of the embedding weight matrix
            num_embeddings, embedding_dim = embedding_layer.weight.shape

            # Reinitialize the weights using the calculated mean and std
            with torch.no_grad():  # Ensure no gradients are recorded
                # Generate random normal values with the same mean and std
                embedding_layer.weight.normal_(mean=mean_emb.item(), std=std_emb.item())

        print("Reinitialized all embeddings in `self.tree_node_embedding`.")

    def update_node_embedding(self, emb, device):
        print("Update the node embedding")
        # import pdb; pdb.set_trace()
        
        # label_kmeans_list = [torch.zeros(emb.shape[0], dtype=torch.int64)]
        # for i in range(len(self.tree_node_embedding)):
            
        #     label_last = label_kmeans_list[-1]
        #     label_new = torch.zeros(emb.shape[0], dtype=torch.int64, device=emb.device)-1 
        #     for j in range(max(2 ** (i), 1)):
        #         mask = label_last == j
        #         select_emb = emb[mask]
        #         print(
        #             f"The number of the node:{i+1}_{j}", "The number of the data:", 
        #             mask.sum()
        #         )
        #         # import pdb; pdb.set_trace()

        #         if mask.sum() > 1:
        #             with parallel_backend('loky', n_jobs=10):
        #                 kmeans = KMeans(n_clusters=2, random_state=0).fit(select_emb.cpu().detach().numpy())
        #             label_kmeans = kmeans.labels_ + 2*j
        #             label_new[mask] = torch.tensor(label_kmeans, dtype=torch.int64, device=emb.device)

        #             # update the embedding
        #             self.tree_node_embedding[i].weight.data[2*j] = torch.tensor(kmeans.cluster_centers_[0], device=emb.device)
        #             self.tree_node_embedding[i].weight.data[2*j+1] = torch.tensor(kmeans.cluster_centers_[1], device=emb.device)
        #         else:
        #             label_new[mask] = torch.tensor([2*j] * mask.sum() , dtype=torch.int64, device=emb.device)
        #             self.tree_node_embedding[i].weight.data[2*j] = self.tree_node_embedding[i-1].weight.data[j] + torch.randn_like(self.tree_node_embedding[i-1].weight.data[j]) * 0.005
        #             self.tree_node_embedding[i].weight.data[2*j+1] = self.tree_node_embedding[i-1].weight.data[j] + torch.randn_like(self.tree_node_embedding[i-1].weight.data[j]) * 0.005
                    
        # tree_node_embedding = [np.zeros((2 ** i, embedding_dim)) for i in range(tree_depth)]

        # 假设 emb 是输入的 NumPy 数组
        label_kmeans_list = [np.zeros(emb.shape[0], dtype=np.int64)]
        for i in range(len(self.tree_node_embedding)):
            label_last = label_kmeans_list[-1]
            label_new = np.full(emb.shape[0], fill_value=-1, dtype=np.int64)
            
            for j in range(max(2 ** i, 1)):
                mask = label_last == j
                select_emb = emb[mask]
                
                # print(
                #     f"The number of the node: {i+1}_{j}", "The number of the data:", 
                #     np.sum(mask)
                # )
                
                if np.sum(mask) > 1:
                    # with parallel_backend('loky', n_jobs=10):
                    # import pdb; pdb.set_trace()
                    # if select_emb.shape[0] > 10000:
                    #     select_emb_down = select_emb[np.random.choice(select_emb.shape[0], 10000, replace=False)]
                    # kmeans = KMeans(n_clusters=2, random_state=0).fit(select_emb)
                    kmeans_labels, kmeans_cluster_centers_ = kmeans_numpy(select_emb, 2)
                    
                    label_kmeans = kmeans_labels + 2 * j
                    label_new[mask] = label_kmeans

                    self.tree_node_embedding[i].weight.data[2 * j] = torch.tensor(kmeans_cluster_centers_[0], device=device)
                    self.tree_node_embedding[i].weight.data[2 * j + 1] = torch.tensor(kmeans_cluster_centers_[1], device=device)
                else:
                    label_new[mask] = [2 * j] * np.sum(mask)
                    noise1 = np.random.randn(*self.tree_node_embedding[i - 1].weight.data[j].shape) * 0.005
                    noise2 = np.random.randn(*self.tree_node_embedding[i - 1].weight.data[j].shape) * 0.005
                    self.tree_node_embedding[i].weight.data[2 * j] = self.tree_node_embedding[i - 1].weight.data[j] + torch.tensor(noise1, device=device)
                    self.tree_node_embedding[i].weight.data[2 * j + 1] = self.tree_node_embedding[i - 1].weight.data[j] + torch.tensor(noise2, device=device)

            label_kmeans_list.append(label_new)



    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            loss_all (Tensor): Computed loss.
        """
        data_input_item = batch["data_input_item"]
        data_input_aug = batch["data_input_aug"]
        index = batch["index"]

        log_dict = {}
        # Concatenate original and augmented data
        data_input = torch.cat([data_input_item, data_input_aug])
        # Forward pass
        x_masked, lat_high_dim, lat_vis, _ = self(
            data_input,
            tau=self.hparams.tau,
        )
        # Compute mean over experts
        # lat_high_dim = lat_high_dim_exp.mean(dim=1)

        if self.training_str == "step1":
            loss_emb, loss_lat, orthogonal_loss = self.forward_train_enc(
                x_masked, lat_high_dim, lat_vis
            )
            # Compute total loss

            # Log losses
            log_dict.update(
                {
                    "loss_emb": loss_emb,
                    "loss_lat": loss_lat,
                    "orthogonal_loss": orthogonal_loss,
                }
            )
            loss_all = (loss_emb + loss_lat) / 2 + orthogonal_loss * 10

        elif "step2" in self.training_str:

            lat_vis = (lat_vis - self.lat_vis_mean) / (self.lat_vis_std + 1e-8)
            cond = lat_vis.detach()

            batch_size = data_input_item.shape[0]
            tree_rout, vector_rout, loss_rout = self.router_forward(
                cond.float().detach(), tree_rout_bool=True,
                ec_ce_weight=self.hparams.ec_ce_weight
            )

            loss_diff = self.diffusion_loss(
                data_input_item, cond=vector_rout[:batch_size].detach()
            )

            log_dict.update(
                {
                    "loss_rute": loss_rout,
                    "loss_diff": loss_diff,
                    "epoch": self.current_epoch,
                }
            )
            if self.training_str == "step2_r":
                loss_all = 0.01 * (0.01 * loss_diff)
            else:
                loss_all = 0.01 * (0.01 * loss_diff + self.hparams.weightrout * loss_rout)

        log_dict.update(
            {
                "lr": float(self.trainer.optimizers[0].param_groups[0]["lr"]),
                "loss_all": loss_all,
            }
        )
        # self.log('lr', float(self.trainer.optimizers[0].param_groups[0]["lr"]), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # self.log('loss_all', loss_all, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.log_dict(log_dict)

        accumulate(
            self.UNet_ema,
            (
                self.UNet_model.module
                if isinstance(self.UNet_model, nn.DataParallel)
                else self.UNet_model
            ),
            0.9999,
        )

        return loss_all

    def diffusion_loss(self, data_after_tokened, cond):
        data_diff = data_after_tokened
        views = data_diff.reshape(data_diff.shape[0], -1)
        # import pdb; pdb.set_trace()
        time = (
            (torch.rand(data_diff.shape[0]) * self.hparams.n_timestep)
            .type(torch.int64)
            .to(data_diff.device)
        )
        loss_diff = self.diffusion.training_losses(
            model=self.UNet_model,
            x_0=views,
            t=time,
            cond=cond,
        ).mean()
        return loss_diff

    def augment_data_simple(self, cond_input_val, img_init=None):
        shape = (cond_input_val.shape[0], 1, self.hparams.num_input_dim)
        self.UNet_ema.eval()
        if img_init is None:
            img_init = torch.randn(shape, device=self.device)

        samples, history = progressive_samples_fn_simple(
            self.UNet_ema,
            self.diffusion,
            shape,
            device=self.device,
            cond=cond_input_val,
            include_x0_pred_freq=50,
            img_init=img_init,
        )
        self.UNet_ema.train()
        return samples, torch.stack(history).permute(1, 0, 2, 3)[:, :, 0, :]

    def validation_step(self, batch, batch_idx, test=False, dataloader_idx=0):
        """
        Performs a validation step.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Batch index.
            test (bool): Whether this is a test step.
            dataloader_idx (int): Index of the dataloader.

        Returns:
            None
        """
        data_input_item = batch["data_input_item"]
        data_input_aug = batch["data_input_aug"]
        index = batch["index"]
        
        x_masked, lat_high_dim_exp, lat_vis, lat_high_dim = self(
            data_input_item,
            tau=self.hparams.tau,
        )
        
        self.val_vis_list.append(lat_vis.detach().cpu())

        if self.lat_vis_mean.sum() != 0:
            lat_vis = (lat_vis - self.lat_vis_mean) / (self.lat_vis_std + 1e-8)
        
        cond = lat_vis.detach()
        batch_size = data_input_item.shape[0]
        tree_rout, vector_rout, loss_rout = self.router_forward(
            cond.float(), tree_rout_bool=True
        )

        # if self.hparams.gen_data_bool:
        #     self.gen_batch = True
        # if batch_idx < 1:
        init_imge = data_input_item[0].repeat(data_input_item.shape[0], 1, 1)
        # import pdb; pdb.set_trace()
        self.reconstruct_example, self.reconstruct_history = (
            self.augment_data_simple(
                cond_input_val=vector_rout.float(),
                img_init=init_imge,
            )
        )
        # else:
        #     self.gen_batch = False
        #     self.reconstruct_example = data_input_item
        #     self.reconstruct_history = data_input_item
        
        if self.hparams.test_noise:
            noist_test_result_dict = []
            for i in range(5):
                noist_test_result = self.noise_map(
                    lat_high_dim, noise_level=i * 0.1 + 0.1
                )
                noist_test_result_dict.append(noist_test_result)
            self.noist_test_result_dict = torch.stack(noist_test_result_dict).cpu()
        
        self.validation_origin_input = data_input_item.detach()
        self.validation_step_outputs_high = lat_high_dim.detach()
        self.validation_step_outputs_vis = lat_vis.detach()
        self.validation_step_lat_vis_exp = lat_vis.detach()
        self.validation_step_rute = tree_rout.detach()
        self.validation_step_vector_rout = vector_rout.detach()

    def on_validation_epoch_end(self):

        self.update_training_str(self.current_epoch)        
        if self.current_epoch == self.hparams.step2_epoch - 1:
            val_vis_all = torch.cat(self.val_vis_list, dim=0).detach().cpu().numpy()
            mean = np.mean(val_vis_all, axis=0)
            std = np.std(val_vis_all, axis=0)
            val_vis_all = (val_vis_all - mean) / (std + 1e-8)
            
            device = self.lat_vis_mean.device
            self.lat_vis_mean.data = torch.tensor(mean).to(device)
            self.lat_vis_std.data = torch.tensor(std).to(device)
                
        self.val_vis_list = []


    def test_step(self, batch, batch_idx):
        """
        Performs a test step.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            None
        """
        data_input_item = batch["data_input_item"]
        data_input_aug = batch["data_input_aug"]
        label = batch["label"]

        x_masked, lat_high_dim, lat_vis, _ = self(
            data_input_item,
        )

        # Store outputs for further processing
        self.test_step_outputs_high = lat_high_dim.cpu().detach()
        self.test_step_outputs_vis = lat_vis.cpu().detach()
        self.test_step_outputs_label = label.cpu().detach()

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            dict: Dictionary containing optimizer and scheduler.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=self.hparams.weight_decay,
            lr=self.hparams.lr,
        )
        lrsched = CosineAnnealingSchedule(
            optimizer, n_epochs=self.hparams.max_epochs, warmup_epochs=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lrsched,
                "interval": "epoch",
            },  # interval "step" for batch update
        }

    def noise_map(self, data, num_exp=10, noise_level=0.1):
        """
        Tests the robustness of the embeddings to noise.

        Args:
            data (Tensor): Input data.
            num_exp (int): Number of experiments.
            noise_level (float): Level of noise to add.

        Returns:
            distance_tensor (Tensor): Tensor containing distances.
        """
        exp_feature_num = int(data.shape[1] // num_exp)

        emb = self.vis(data)

        distance_list = []
        for i in range(num_exp):
            start_index = i * exp_feature_num
            end_index = (i + 1) * exp_feature_num
            noise_data_delta = torch.rand_like(data) * noise_level * data.std(dim=0)
            noise_data = torch.clone(data)
            noise_data[:, start_index:end_index] += noise_data_delta[
                :, start_index:end_index
            ]
            noise_emb = self.vis(noise_data)
            distance = torch.norm(noise_emb - emb, dim=1)
            distance_list.append(distance)

        distance_tensor = torch.stack(distance_list, dim=1)
        return distance_tensor
