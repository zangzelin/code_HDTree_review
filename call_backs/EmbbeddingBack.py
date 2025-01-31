import typing
from typing import Optional
import attr
import numpy
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.special import comb
# from sctree.utils.utils import count_values_in_sequence
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # 导入KNN分类器
from collections import Counter

from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
import torch
import wandb
import eval.eval_core_base as ecb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC
import numpy as np
import os
from sklearn.manifold import TSNE
import plotly.graph_objects as go

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import metrics
import eval.eval_core as ec
import eval.eval_core_base as ecb
from eval import evaluation

def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()

def euclidean_to_hyperbolic_matrix(u, c=0.5, min_norm = 1e-15):
    u = torch.tensor(u).float()
    
    u = 1.5 * ( u-u.mean(dim=0) )/u.std(dim=0)
    
    sqrt_c = c ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), min_norm)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1.detach().numpy()


class EmbbeddingBack(Callback):
    def __init__(self, inter=10, dirpath='', fully_eval=False, dataset='', only_val=False, *args, **kwargs):
        super().__init__()
        self.inter = inter
        self.plot_sample_train = 0
        self.plot_sample_test = 0
        self.only_val = only_val
        # self.train_len_list = []
        # self.train_acc_list = []

        self.val_input= {'val1': [], 'val2': [], 'val3': []}
        self.val_high = {'val1': [], 'val2': [], 'val3': []}
        self.val_vis = {'val1': [], 'val2': [], 'val3': []}
        self.val_vis_exp = {'val1': [], 'val2': [], 'val3': []}
        self.val_label = {'val1': [], 'val2': [], 'val3': []}
        # self.val_recon = []

        # self.val_high_v2 = []
        # self.val_vis_v2 = []
        # self.val_label_v2 = []


        self.test_high = []
        self.test_vis = []
        self.test_recon = []
        self.test_label = []
        
        self.dirpath = dirpath
        self.dataset = dataset
        self.fully_eval = fully_eval
        self.best_acc = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # preds = pl_module.step_last_outputs_val  # 假设模型的输出字典中包含预测结果
        # data_input_item, data_input_aug, label, index, = batch
        label = batch["label"]
        
        key_value = f"val{dataloader_idx+1}"
        
        self.val_input[key_value].append(pl_module.validation_origin_input)
        self.val_high[key_value].append(pl_module.validation_step_outputs_high)
        self.val_vis[key_value].append(pl_module.validation_step_outputs_vis)
        # self.val_vis_exp[key_value].append(pl_module.validation_step_lat_vis_exp)
        
        # self.validation_weight = pl_module.validation_weight
        # self.val_recon.append(pl_module.validation_step_outputs_recons)
        self.val_label[key_value].append(label)
        
        
        # self.val_high.append(pl_module.validation_step_outputs_high)
        # self.val_vis.append(pl_module.validation_step_outputs_vis)
        # # self.val_recon.append(pl_module.validation_step_outputs_recons)
        # self.val_label.append(label)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # preds = pl_module.step_last_outputs_val  # 假设模型的输出字典中包含预测结果
        # data_input_item, data_input_aug, label, index, = batch
        label = batch["label"]
        
        # self.test_high.append(pl_module.test_step_outputs_high)
        self.test_vis.append(pl_module.test_step_outputs_vis)
        # self.test_recon.append(pl_module.test_step_outputs_recons)
        self.test_label.append(label)  

    def plot_scatter(self, gathered_val_vis, gathered_val_label, trainer):
        fig = plt.figure(figsize=(10, 10))
        if gathered_val_vis.shape[0] >= 10000: 
            s=1
        else :
            s=3
        plt.scatter(
            gathered_val_vis[:, 0],
            gathered_val_vis[:, 1],
            c=gathered_val_label, 
            cmap='rainbow',
            s=s
        )
        return fig

    def plot_scatter_hyper(self, gathered_val_vis, gathered_val_label, trainer):
        
        gathered_val_vis_hy = euclidean_to_hyperbolic_matrix(gathered_val_vis)
        
        if gathered_val_vis.shape[0] >= 10000: 
            s=1
        else :
            s=3
        
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(
            gathered_val_vis_hy[:, 0],
            gathered_val_vis_hy[:, 1],
            c=gathered_val_label, 
            cmap='rainbow',
            s=s
        )
        return fig

    ### def the mertic acculate method
    def count_values_in_sequence(self,sequence):
        counts = {}
        for value in sequence:
            if value in counts:
                counts[value] += 1
            else:
                counts[value] = 1
        return counts

    def cluster_acc(self,y_true, y_pred, return_index=False):
        """
        Calculate clustering accuracy.
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.astype(int).max(), y_true.astype(int).max()) + 1
        w = np.zeros((int(D), (D)), dtype=np.int64)
        for i in range(y_pred.size):
            w[int(y_pred[i]), int(y_true[i])] += 1
        ind = np.array(linear_assignment(w.max() - w))
        if return_index:
            assert all(ind[0] == range(len(ind[0])))  # Assert rows don't change order
            cluster_acc = sum(w[ind[0], ind[1]]) * 1.0 / y_pred.size
            return cluster_acc
        else:
            return sum([w[ind[0,i], ind[1,i]] for i in range(len(ind[0]))]) * 1.0 / y_pred.size
        
    def leaf_purity(self,tree_root, ground_truth):
        values = []
        weights = []

        def get_leaf_purities(node):
            nonlocal values
            nonlocal weights

            if node is None:
                return

            if node.is_leaf:
                node_total_dp_count = len(node.dp_ids)
                node_per_label_counts = self.count_values_in_sequence(
                    [ground_truth[id] for id in node.dp_ids]
                )
                if node_total_dp_count > 0:
                    purity_rate = max(node_per_label_counts.values()) / node_total_dp_count
                else:
                    purity_rate = 1.0
                values.append(purity_rate)
                weights.append(node_total_dp_count)
            else:
                get_leaf_purities(node.left_child)
                get_leaf_purities(node.right_child)

        get_leaf_purities(tree_root)

        return numpy.average(values, weights=weights)

    def dendrogram_purity(self,tree_root, ground_truth):
        total_per_label_frequencies = self.count_values_in_sequence(ground_truth)
        total_per_label_pairs_count = {
            k: comb(v, 2, exact=True) for k, v in total_per_label_frequencies.items()
        }
        total_n_of_pairs = sum(total_per_label_pairs_count.values())

        one_div_total_n_of_pairs = 1.0 / total_n_of_pairs

        purity = 0.0

        def calculate_purity(node, level):
            nonlocal purity
            if node.is_leaf:
                node_total_dp_count = len(node.dp_ids)
                node_per_label_frequencies = self.count_values_in_sequence(
                    [ground_truth[id] for id in node.dp_ids]
                )
                node_per_label_pairs_count = {
                    k: comb(v, 2, exact=True) for k, v in node_per_label_frequencies.items()
                }

            elif node.left_child is None or node.right_child is None:
                # We are in an internal node with pruned leaves and thus only have one child. Therefore no prunity calculation here!
                node_left, node_right = node.left_child, node.right_child
                child = node_left if node_left is not None else node_right
                node_per_label_frequencies, node_total_dp_count = calculate_purity(
                    child, level + 1
                )
                return node_per_label_frequencies, node_total_dp_count

            else:  # it is an inner node
                left_child_per_label_freq, left_child_total_dp_count = calculate_purity(
                    node.left_child, level + 1
                )
                right_child_per_label_freq, right_child_total_dp_count = calculate_purity(
                    node.right_child, level + 1
                )
                node_total_dp_count = left_child_total_dp_count + right_child_total_dp_count
                node_per_label_frequencies = {
                    k: left_child_per_label_freq.get(k, 0)
                    + right_child_per_label_freq.get(k, 0)
                    for k in set(left_child_per_label_freq)
                    | set(right_child_per_label_freq)
                }

                node_per_label_pairs_count = {
                    k: left_child_per_label_freq.get(k) * right_child_per_label_freq.get(k)
                    for k in set(left_child_per_label_freq)
                    & set(right_child_per_label_freq)
                }

            for label, pair_count in node_per_label_pairs_count.items():
                label_freq = node_per_label_frequencies[label]
                label_pairs = node_per_label_pairs_count[label]
                purity += (
                    one_div_total_n_of_pairs
                    * label_freq
                    / node_total_dp_count
                    * label_pairs
                )
            return node_per_label_frequencies, node_total_dp_count

        calculate_purity(tree_root, 0)
        return purity

    def prune_dendrogram_purity_tree(self,tree, n_leaves):
        """
        This function collapses the tree such that it only has n_leaves.
        This makes it possible to compare different trees with different number of leaves.

        Important, it assumes that the node_id is equal to the split order, that means the tree root should have the smallest split number
        and the two leaf nodes that are splitted the last have the highest node id. And that  max(node_id) == #leaves - 2

        :param tree:
        :param n_levels:
        :return:
        """
        max_node_id = n_leaves - 2

        def recursive(node):
            if node.is_leaf:
                return node
            else:  # node is an inner node
                if node.node_id < max_node_id:
                    left_child = recursive(node.left_child)
                    right_child = recursive(node.right_child)
                    return self.DpNode(left_child, right_child, node.node_id)
                else:
                    work_list = [node.left_child, node.right_child]
                    dp_ids = []
                    while len(work_list) > 0:
                        nc = work_list.pop()
                        if nc.is_leaf:
                            dp_ids = dp_ids + nc.dp_ids
                        else:
                            work_list.append(nc.left_child)
                            work_list.append(nc.right_child)
                    return self.DpLeaf(dp_ids, node.node_id)

        return recursive(tree)

    def to_dendrogram_purity_tree(self,children_array):
        """
        Can convert the children_ matrix of a  sklearn.cluster.hierarchical.AgglomerativeClustering outcome to a dendrogram_purity tree
        :param children_array:  array-like, shape (n_samples-1, 2)
            The children of each non-leaf nodes. Values less than `n_samples`
                correspond to leaves of the tree which are the original samples.
                A node `i` greater than or equal to `n_samples` is a non-leaf
                node and has children `children_[i - n_samples]`. Alternatively
                at the i-th iteration, children[i][0] and children[i][1]
                are merged to form node `n_samples + i`
        :return:
        """
        n_samples = children_array.shape[0] + 1
        max_id = 2 * n_samples - 2
        node_map = {dp_id: self.DpLeaf([dp_id], max_id - dp_id) for dp_id in range(n_samples)}
        next_id = max_id - n_samples

        for idx in range(n_samples - 1):
            next_fusion = children_array[idx, :]
            child_a = node_map.pop(next_fusion[0])
            child_b = node_map.pop(next_fusion[1])
            node_map[n_samples + idx] = self.DpNode(child_a, child_b, next_id)
            next_id -= 1
        if len(node_map) != 1:
            raise RuntimeError(
                "tree must be fully developed! Use ompute_full_tree=True for AgglomerativeClustering"
            )
        root = node_map[n_samples + n_samples - 2]
        return root

    @attr.define()
    class DpNode(object):
        """
        node_id should be in such a way that a smaller number means split before a larger number in a top-down manner
        That is the root should have node_id = 0 and the children of the last split should have node id
        2*n_dps-2 and 2*n_dps-1

        """

        left_child: typing.Any = None
        right_child: typing.Any = None
        node_id: Optional[int] = None

        @property
        def children(self):
            return [self.left_child, self.right_child]

        @property
        def is_leaf(self):
            return False

    @attr.s(cmp=False)
    class DpLeaf(object):
        dp_ids = attr.ib()
        node_id = attr.ib()

        @property
        def children(self):
            return []

        @property
        def is_leaf(self):
            return True

    def modeltree_to_dptree(self, tree, y_predicted, n_leaves):
        i = 0
        root = self.DpNode(node_id=i)
        list_nodes = [{"node": tree, "id": 0, "parent": None, "dpNode": root}]
        labels_leaf = [i for i in range(n_leaves)]
        while len(list_nodes) != 0:
            current_node = list_nodes.pop(0)
            if current_node["node"].router is not None:
                node_left, node_right = (
                    current_node["node"].left,
                    current_node["node"].right,
                )
                i += 1
                if node_left.decoder is not None:
                    y_leaf = labels_leaf.pop(0)
                    ind = np.where(y_predicted == y_leaf)[0]
                    current_node["dpNode"].left_child = self.DpLeaf(node_id=i, dp_ids=ind)
                else:
                    current_node["dpNode"].left_child = self.DpNode(node_id=i)
                    list_nodes.append(
                        {
                            "node": node_left,
                            "id": i,
                            "parent": current_node["id"],
                            "dpNode": current_node["dpNode"].left_child,
                        }
                    )
                i += 1
                if node_right.decoder is not None:
                    y_leaf = labels_leaf.pop(0)
                    ind = np.where(y_predicted == y_leaf)[0]
                    current_node["dpNode"].right_child = self.DpLeaf(node_id=i, dp_ids=ind)
                else:
                    current_node["dpNode"].right_child = self.DpNode(node_id=i)
                    list_nodes.append(
                        {
                            "node": node_right,
                            "id": i,
                            "parent": current_node["id"],
                            "dpNode": current_node["dpNode"].right_child,
                        }
                    )

            else:
                # We are in an internal node with pruned leaves and will only add the non-pruned leaves
                node_left, node_right = (
                    current_node["node"].left,
                    current_node["node"].right,
                )
                child = node_left if node_left is not None else node_right
                i += 1

                if node_left is not None:
                    if node_left.decoder is not None:
                        y_leaf = labels_leaf.pop(0)
                        ind = np.where(y_predicted == y_leaf)[0]
                        current_node["dpNode"].left_child = self.DpLeaf(node_id=i, dp_ids=ind)
                    else:
                        current_node["dpNode"].left_child = self.DpNode(node_id=i)
                        list_nodes.append(
                            {
                                "node": node_left,
                                "id": i,
                                "parent": current_node["id"],
                                "dpNode": current_node["dpNode"].left_child,
                            }
                        )
                else:
                    if node_right.decoder is not None:
                        y_leaf = labels_leaf.pop(0)
                        ind = np.where(y_predicted == y_leaf)[0]
                        current_node["dpNode"].right_child = self.DpLeaf(node_id=i, dp_ids=ind)
                    else:
                        current_node["dpNode"].right_child = self.DpNode(node_id=i)
                        list_nodes.append(
                            {
                                "node": node_right,
                                "id": i,
                                "parent": current_node["id"],
                                "dpNode": current_node["dpNode"].right_child,
                            }
                        )

        return root

    def tree_to_dptree(self,tree, labels):
        """
        This function reformats the output from a scanpy.tl.dendrogram to a DPdendrogram.
        This makes it possible to compute Leaf and Dendrogram Purity.
        The trick is to create the dendrogram of only the clusters and afterwards, replace each cluster_id with the sample_ids that are in the respective cluster.

        :param tree:
        :param labels:
        :return:
        """

        dptree_clusters = self.to_dendrogram_purity_tree(tree["linkage"][:, :2])

        def recursive(node):
            if node.is_leaf:
                cluster_id = node.dp_ids[0]
                sample_ids = np.where(labels == cluster_id)[0]
                return self.DpLeaf(list(sample_ids), node.node_id)
            else:  # node is an inner node
                left_child = recursive(node.left_child)
                right_child = recursive(node.right_child)
                return self.DpNode(left_child, right_child, node.node_id)

        return recursive(dptree_clusters)

    def compute_metrics(adata, labels, pruned_tree, celltype_key, batch_key, run_time):
        results = {}
        label_encoder = LabelEncoder()
        if type(labels) is dict:
            for name, agg_labels, tree in zip(
                labels.keys(), labels.values(), pruned_tree.values()
            ):
                name = name.split("_")[0]
                results.update(
                    {
                        f"{name}/NMI": normalized_mutual_info_score(
                            agg_labels, adata.obs[celltype_key].values
                        ),
                        f"{name}/ARI": adjusted_rand_score(
                            agg_labels, adata.obs[celltype_key].values
                        ),
                        f"{name}/Dendrogram purity": dendrogram_purity(
                            tree, adata.obs[celltype_key].values
                        ),
                        f"{name}/Leaf purity": leaf_purity(
                            tree, adata.obs[celltype_key].values
                        ),
                        f"{name}/ACC": cluster_acc(
                            label_encoder.fit_transform(adata.obs[celltype_key].values), agg_labels, return_index=True
                        ),
                    }
                )
                if batch_key:
                    results[f"{name}/NMI_batch"] = normalized_mutual_info_score(
                        agg_labels, adata.obs[batch_key].values
                    )

        else:
            results.update(
                {
                    "NMI": normalized_mutual_info_score(
                        labels, adata.obs[celltype_key].values
                    ),
                    "ARI": adjusted_rand_score(labels, adata.obs[celltype_key].values),
                    "Dendrogram purity": dendrogram_purity(
                        pruned_tree, adata.obs[celltype_key].values
                    ),
                    "Leaf purity": leaf_purity(pruned_tree, adata.obs[celltype_key].values),
                    "ACC": cluster_acc(
                        label_encoder.fit_transform(adata.obs[celltype_key].values), labels, return_index=True
                    ),
                }
            )
            if batch_key:
                results["NMI_batch"] = normalized_mutual_info_score(
                    labels, adata.obs[batch_key].values
                )
        results["run_time"] = run_time

        return results

    def build_tree(self,feature, n_clusters):
        clustering = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=None, compute_full_tree=True)
        clustering.fit(feature)
        return clustering.children_, clustering.labels_

    def calculate_tree_leaf_purity_mean(self, children_array, true_labels):
        """
        Calculate the mean leaf purity across all nodes in the tree.
        
        :param children_array: Children array from sklearn Agglomerative Clustering
        :param true_labels: Ground truth labels (n_samples,)
        :return: Mean leaf purity across all nodes
        """
        n_samples = len(true_labels)  # Number of samples
        n_nodes = children_array.shape[0] + n_samples  # Total number of nodes

        # Initialize a dictionary to store the leaves covered by each node
        node_to_leaves = {i: [i] for i in range(n_samples)}  # Leaf nodes

        # Build node-to-leaves mapping from the children array
        for node_id, (child_a, child_b) in enumerate(children_array, start=n_samples):
            node_to_leaves[node_id] = node_to_leaves[child_a] + node_to_leaves[child_b]

        # Calculate purity for each node
        purities = []
        for node_id, leaves in node_to_leaves.items():
            leaf_labels = true_labels[leaves]
            most_common_label, count = Counter(leaf_labels).most_common(1)[0]
            purity = count / len(leaves)
            purities.append(purity)

        return np.mean(purities)

    def calculate_metrics_Embedding(self,children_array, labels, y_true):
        root = self.to_dendrogram_purity_tree(children_array)
        leaf_purity_value = self.calculate_tree_leaf_purity_mean(children_array, y_true)
        dendrogram_purity_value = self.dendrogram_purity(root, y_true) if root else None
        acc_value = self.cluster_acc(y_true, labels)
        nmi_value = normalized_mutual_info_score(y_true, labels)
        ari_value = adjusted_rand_score(y_true, labels)
        return {
            "Leaf Purity": leaf_purity_value,
            "Dendrogram Purity": dendrogram_purity_value,
            "Accuracy": acc_value,
            "NMI": nmi_value,
            "ARI": ari_value}

    def calculate_metrics_Annontation(self, labels, y_true):
        accuracy = accuracy_score(y_true, labels)
        f1 = f1_score(y_true, labels, average='weighted')
        precision = precision_score(y_true, labels, average='weighted')
        recall = recall_score(y_true, labels, average='weighted')
        return {
            "A-Accuracy": accuracy,
            "A-F1 Score": f1,
            "A-Precision": precision,
            "A-Recall": recall,
        }
    
    def get_rout_vis(self, tree_node_embedding, emb_vis):            
        # import pdb; pdb.set_trace()
        
        path_list = []
        for i in range(len(tree_node_embedding)):
            # token_emb = tree_node_embedding[i].weight.detach().cpu().numpy()
            token_emb = tree_node_embedding[i].weight.detach().cpu().numpy()
            print('token_emb.shape:', token_emb.shape)
            print('emb_vis.shape:', emb_vis.shape)
            # if token_emb.shape[0] > 10000:
            #     idx = np.random.choice(token_emb.shape[0], 10000, replace=False)
            #     token_emb = token_emb[idx]
            # if emb_vis.shape[0] > 10000:
            #     idx = np.random.choice(token_emb.shape[0], 10000, replace=False)
            #     emb_vis = emb_vis[idx]
             

            # rute_bool = rout_vector.copy()[:,:,0]
            # rute_bool[rute_bool>0.5] = 1
            # rute_bool[rute_bool<=0.5] = 0
            # rute_str = rute_bool[:, :(i+1)].astype(np.int32).astype(str)
            # rute_str_list = [''.join(rute_str[j]) for j in range(rute_str.shape[0])]
            # import pdb; pdb.set_trace()
            plotly_fig_rute = go.Figure()

            plotly_fig_rute.add_trace(
                go.Scatter(
                    x=emb_vis[:, 0],
                    y=emb_vis[:, 1],
                    mode='markers',
                    marker=dict(size=1, color='blue'),
                    name="emb_vis"
                )
            )

            # Add additional scatter points for `token_emb`
            plotly_fig_rute.add_trace(
                go.Scatter(
                    x=token_emb[:, 0],
                    y=token_emb[:, 1],
                    mode='markers',
                    marker=dict(size=5, color='red', symbol='star'),
                    name="token_emb"
                )
            )
            os.makedirs(self.dataset+"fig", exist_ok=True)
            plotly_fig_rute.write_image("./"+self.dataset+"fig/fig_emb_colored_with_rout_{}.png".format(i), scale=3)
            path_list.append("./"+self.dataset+"fig/fig_emb_colored_with_rout_{}.png".format(i))
        
        return path_list

    def on_validation_epoch_end(self, trainer, pl_module):
        
        num_val = len(self.val_vis)
        
        for val_index in range(num_val):
            
            val_name = f"val{val_index+1}"
            
            # val_input_current = self.val_input[val_name]
            val_vis_current = self.val_vis[val_name]
            # val_vis_exp_current = self.val_vis_exp[val_name]
            # hight_vis_current = self.val_high[val_name]
            val_label_current = self.val_label[val_name]
            # import pdb; pdb.set_trace()
            print(val_name, len(val_vis_current), len(val_label_current))
            if ( len(val_vis_current)!=0):
                feature = torch.cat(val_vis_current).cpu().numpy()
                label = torch.cat(val_label_current).cpu().numpy()

                children_array, preds = self.build_tree(feature, 67)
                # Compute evaluation metrics
                metrics = self.calculate_metrics_Embedding(children_array, preds, label)

                fig_scatter = self.plot_scatter(feature, label, trainer)
                fig_scatter_hyper = self.plot_scatter_hyper(feature, label, trainer)
                metrics.update({
                    f"scatter": wandb.Image(fig_scatter),
                    f"scatter_hyper": wandb.Image(fig_scatter_hyper),})
                acc_mean = metrics['Accuracy']
                if not os.path.exists(self.dirpath):
                    os.makedirs(self.dirpath, exist_ok=True)
                
                if acc_mean > self.best_acc:
                    self.best_acc = acc_mean
                    torch.save(pl_module.state_dict(), os.path.join(
                        self.dirpath, f"best_model_{self.dataset}_acc{acc_mean}.pth"))
                                
                    # if trainer.current_epoch > 400 and acc_mean < 0.8:
                    #     # stop the training
                    #     trainer.should_stop = True
                if trainer.is_global_zero:
                    vis_plot_path = self.get_rout_vis(pl_module.tree_node_embedding, feature)
                    for i, path in enumerate(vis_plot_path):
                        metrics[f"rout/emb_colored_with_rout_{i}"] = wandb.Image(path)
                trainer.logger.log_metrics(metrics)


        self.val_input = {'val1': [], 'val2': [], 'val3': []}    
        self.val_high = {'val1': [], 'val2': [], 'val3': []}
        self.val_vis = {'val1': [], 'val2': [], 'val3': []}
        self.val_label = {'val1': [], 'val2': [], 'val3': []}
        self.val_vis_exp = {'val1': [], 'val2': [], 'val3': []}


    def on_test_epoch_end(self, trainer, pl_module):
        
        test_vis = torch.cat(self.test_vis).cuda()
        test_label = torch.cat(self.test_label).cuda()
        gathered_val_vis = trainer.strategy.all_gather(test_vis).cpu().detach().numpy()
        gathered_val_label = trainer.strategy.all_gather(test_label).cpu().detach().numpy()
        feature = gathered_val_vis.squeeze()
        label = gathered_val_label.squeeze()
        X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.1, random_state=42)
        # clf = KNeighborsClassifier(n_neighbors=5)
        # clf.fit(X_train, y_train)
        # preds = clf.predict(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 创建 SVM 模型 (这里选择线性核函数作为例子)
        svm_model = SVC(kernel='linear', probability=True, random_state=42)

        # 训练模型
        svm_model.fit(X_train_scaled, y_train)

        # 预测测试集
        preds = svm_model.predict(X_test_scaled)

        metrics = self.calculate_metrics_Annontation(preds, y_test)
        print(metrics)
        import pdb; pdb.set_trace()
        # print(f"Metrics for {args.embedding_method}:")
        # for metric, value in metrics.items():
        #     print(f"{metric}: {value:.4f}")
        # acc_mean = self.get_svc_acc(
        #     gathered_val_vis, 
        #     gathered_val_label, 
        #     trainer
        #     )
        # print('gathered_val_vis', gathered_val_vis.shape)
        # print('gathered_val_label', gathered_val_label.shape)
        # print('acc_mean', acc_mean)      
