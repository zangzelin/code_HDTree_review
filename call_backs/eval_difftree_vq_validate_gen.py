from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
import torch
import wandb
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from collections import defaultdict
import eval.eval_core_base as ecb
from eval import evaluation
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances
import networkx as nx
from sklearn.neighbors import kneighbors_graph

def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()

def euclidean_to_hyperbolic_matrix(u, c=0.5, min_norm=1e-15):
    u = torch.tensor(u).float()
    u = 1.5 * (u - u.mean(dim=0)) / u.std(dim=0)
    sqrt_c = c ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), min_norm)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1.detach().numpy()

class EvalCallBack(Callback):
    def __init__(self, inter=10, dirpath='', fully_eval=False, dataset='', only_val=False, *args, **kwargs):
        super().__init__()
        self.inter = inter
        self.only_val = only_val

        # Use defaultdict to store data for multiple validation sets
        self.val_input = defaultdict(list)
        self.val_rout = defaultdict(list)
        self.val_high = defaultdict(list)
        self.val_vis = defaultdict(list)
        self.val_label = defaultdict(list)

        self.test_vis = []
        self.test_label = []

        self.dirpath = dirpath
        self.dataset = dataset
        self.fully_eval = fully_eval
        self.best_acc = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        label = batch["label"]
        key = dataloader_idx

        self.val_rout[key].append(pl_module.validation_step_rute)
        self.val_input[key].append(pl_module.validation_origin_input)
        self.val_high[key].append(pl_module.validation_step_outputs_high)
        self.val_vis[key].append(pl_module.validation_step_outputs_vis)
        self.val_label[key].append(label)
        try:
            self.val_sample[key].append(pl_module.validation_step_sample)
        except:
            # print('no sample in training process')
            pass

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        label = batch["label"]
        self.test_vis.append(pl_module.test_step_outputs_vis)
        self.test_label.append(label)

    def on_test_epoch_end(self, trainer, pl_module):
        test_vis = torch.cat(self.test_vis).cuda()
        test_label = torch.cat(self.test_label).cuda()
        gathered_vis = trainer.strategy.all_gather(test_vis).cpu().detach().numpy()
        gathered_label = trainer.strategy.all_gather(test_label).cpu().detach().numpy()

        gathered_vis = gathered_vis.reshape(-1, gathered_vis.shape[-1])
        gathered_label = gathered_label.reshape(-1)

        acc_mean = self.get_svc_acc(gathered_vis, gathered_label)
        print('Test SVC Accuracy:', acc_mean)

    def plot_scatter(self, vis_data, labels):
        fig = plt.figure(figsize=(10, 10))
        s = 1 if vis_data.shape[0] >= 10000 else 3
        plt.scatter(vis_data[:, 0], vis_data[:, 1], c=labels, cmap='rainbow', s=s)
        return fig

    def plot_scatter_hyper(self, vis_data, labels):
        vis_hyper = euclidean_to_hyperbolic_matrix(vis_data)
        s = 1 if vis_data.shape[0] >= 10000 else 3
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(vis_hyper[:, 0], vis_hyper[:, 1], c=labels, cmap='rainbow', s=s)
        return fig

    def get_svc_acc(self, vis_data, labels):
        method = SVC(kernel='linear', max_iter=900000)
        cv = RepeatedStratifiedKFold(n_splits=5, random_state=1)
        n_scores = cross_val_score(
            method,
            StandardScaler().fit_transform(vis_data),
            labels,
            scoring="accuracy",
            cv=cv,
            n_jobs=5,
        )
        return np.mean(n_scores)

    def get_svc_acc_rbf(self, vis_data, labels):
        method = SVC(kernel='rbf', max_iter=900000)
        cv = RepeatedStratifiedKFold(n_splits=5, random_state=1)
        n_scores = cross_val_score(
            method,
            StandardScaler().fit_transform(vis_data),
            labels,
            scoring="accuracy",
            cv=cv,
            n_jobs=5,
        )
        return np.mean(n_scores)

    def process_and_log_metrics(self, val_input, val_vis, val_label, trainer, val_idx):
        fig_scatter = self.plot_scatter(val_vis, val_label)
        fig_scatter_hyper = self.plot_scatter_hyper(val_vis, val_label)

        # Sample data if too large
        if val_vis.shape[0] > 10000:
            idx = np.random.choice(val_vis.shape[0], 10000, replace=False)
            val_input = val_input[idx]
            val_vis = val_vis[idx]
            val_label = val_label[idx]

        # Apply TSNE if latent dimension > 2
        if val_vis.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=0)
            val_vis = tsne.fit_transform(val_vis)

        # Flatten input if necessary
        if len(val_input.shape) > 2:
            val_input = val_input.reshape(val_input.shape[0], -1)

        acc_mean = self.get_svc_acc(val_vis, val_label)
        acc_mean_rbf = self.get_svc_acc_rbf(val_vis, val_label)

        dataset_names = ['train', 'validation', 'test']
        dataset_name = dataset_names[val_idx] if val_idx < len(dataset_names) else f'val{val_idx}'

        log_dict = {
            f"{dataset_name}_svc": acc_mean,
            f"{dataset_name}_svc_rbf": acc_mean_rbf,
            "epoch": trainer.current_epoch,
            f"{dataset_name}_scatter": wandb.Image(fig_scatter),
            f"{dataset_name}_scatter_hyper": wandb.Image(fig_scatter_hyper),
        }

        if self.fully_eval:
            for k in [120]:
                ecb_e_train = ecb.Eval(input=val_input, latent=val_vis, label=val_label, k=k)
                trust = ecb_e_train.E_trustworthiness()
                continuity = ecb_e_train.E_continuity()
                log_dict[f"{dataset_name}_trust{k}"] = trust
                log_dict[f"{dataset_name}_continuity{k}"] = continuity

            fknn = np.mean(evaluation.faster_knn_eval_series(val_vis, val_label))
            fct = evaluation.faster_centroid_triplet_eval(val_input, val_vis, val_label)
            log_dict[f"{dataset_name}_fknn"] = fknn
            log_dict[f"{dataset_name}_fct"] = fct

        trainer.logger.log_metrics(log_dict)
        plt.close()

        # Save best model
        # if acc_mean > self.best_acc:
        if True:
            self.best_acc = acc_mean
            os.makedirs(self.dirpath, exist_ok=True)
            model_path = os.path.join(
                self.dirpath, f"best_model_epoch{trainer.current_epoch}_{self.dataset}_acc{acc_mean}.pth")
            torch.save(trainer.model.state_dict(), model_path)

    def get_rout_svc_acc(self, rout_vector, label_vector):
        
        method = SVC(kernel='linear', max_iter=9000, )
        cv = RepeatedStratifiedKFold(n_splits=5, random_state=1)
        n_scores = cross_val_score(
            method,
            StandardScaler().fit_transform(rout_vector),
            label_vector,
            scoring="accuracy",
            cv=cv,
            n_jobs=1,
        )
        return np.mean(n_scores)


    def get_predict_label(self, tree_list, token_emb, emb_vis):
        
        distances = pairwise_distances(emb_vis, token_emb)  # (n_samples, k)
        if len(tree_list) > 0:
            last_element = tree_list[-1].reshape(-1, 1)
            mask = np.zeros((last_element.shape[0], token_emb.shape[0])) + 1e9
            for i_emb_vis in range(emb_vis.shape[0]):
                start = int(last_element[i_emb_vis]*2)
                end = int((last_element[i_emb_vis]+1)*2)
                mask[i_emb_vis, start:end] = 0

            distances += mask

        label_predict = np.argmin(distances, axis=1)
        
        list_count = []
        for label_i in range(label_predict.max()+1):
            list_count.append(np.sum(label_predict==label_i))
        return label_predict

    def plot_arrow(self, fig, x0, y0, x1, y1, arrow_size=0.05, color="rgba(0, 0, 255, 0.5)"):
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=1), 
                showlegend=False,
            )
        )

        
        dx, dy = x1 - x0, y1 - y0
        arrow_norm = (dx**2 + dy**2)**0.5
        dx /= arrow_norm
        dy /= arrow_norm
        # import pdb; pdb.set_trace()

        arrow_x = [x1, x1 - arrow_size * (dx + dy), x1 - arrow_size * (dx - dy)]
        arrow_y = [y1, y1 - arrow_size * (dy - dx), y1 - arrow_size * (dy + dx)]

        fig.add_trace(
            go.Scatter(
                x=arrow_x,
                y=arrow_y,
                fill="toself",
                fillcolor="blue",
                line=dict(color="blue", width=0),
                mode="lines",
                showlegend=False
            )
        )
        return fig

    def plot_path(self, G, tree_node_embedding, i, near_index, end_node, plotly_fig_rute):
        shortest_path = nx.shortest_path(
            G, 
            source=f"L{i}/{near_index}", 
            target=f'L{i}/{end_node}',
            weight='weight',
            )
        print(f'shortest_path: {shortest_path}')
        # import pdb; pdb.set_trace()

        for index_str_path in range(len(shortest_path)-1):
            s_node_str = shortest_path[index_str_path][1:]
            e_node_str = shortest_path[index_str_path+1][1:]
            s_node_level, s_node_index  = s_node_str.split('/')
            e_node_level, e_node_index  = e_node_str.split('/')

            s_node_index = int(s_node_index)
            s_node_level = int(s_node_level)
            e_node_index = int(e_node_index)
            e_node_level = int(e_node_level)

            if e_node_level < 0:
                shortest_path[index_str_path+1] = shortest_path[index_str_path]
            else:
                star_emb = tree_node_embedding[s_node_level].weight.detach().cpu().numpy()[s_node_index]
                end_emb = tree_node_embedding[e_node_level].weight.detach().cpu().numpy()[e_node_index]

                plotly_fig_rute = self.plot_arrow(
                    plotly_fig_rute,
                    star_emb[0], 
                    star_emb[1], 
                    end_emb[0],
                    end_emb[1],
                    color="rgba(0, 0, 255, 0.5)"
                )


    def get_rout_vis(self, tree_node_embedding, emb_vis):            
        # import pdb; pdb.set_trace()

        G = nx.Graph()
        G.add_node('L-1/0')

        path_list = []
        tree_list = []
        plotly_fig_rute_dict = {}
        for i in range(len(tree_node_embedding)):

            # token_emb = tree_node_embedding[i].weight.detach().cpu().numpy()
            token_emb = tree_node_embedding[i].weight.detach().cpu().numpy()
            print('token_emb.shape:', token_emb.shape)
            print('emb_vis.shape:', emb_vis.shape)
            
            if token_emb.shape[0] > 20000:
                idx = np.random.choice(token_emb.shape[0], 20000, replace=False)
                token_emb = token_emb[idx]
            if emb_vis.shape[0] > 20000:
                idx = np.random.choice(token_emb.shape[0], 20000, replace=False)
                emb_vis = emb_vis[idx]
            
            label_predict = self.get_predict_label(tree_list, token_emb, emb_vis)
            tree_list.append(label_predict)
            
            for label_i in range(label_predict.max()+1):
                print(f'level: {i}, label: {label_i}, num: {np.sum(label_predict==label_i)}')

             
            plotly_fig_rute = go.Figure()
            d3_colors = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
                "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
            ] * 1000

            plotly_fig_rute.add_trace(
                go.Scatter(
                    x=emb_vis[:, 0],
                    y=emb_vis[:, 1],
                    mode='markers',
                    marker=dict(
                        size=1, 
                        # color=label_predict, 
                        color=[d3_colors[c_i] for c_i in label_predict], 
                        ),
                    name="emb_vis",
                    
                )
            )

            # Add additional scatter points for `token_emb`
            plotly_fig_rute.add_trace(
                go.Scatter(
                    x=token_emb[:, 0],
                    y=token_emb[:, 1],
                    mode='markers',
                    text=[str(i) for i in range(token_emb.shape[0])],
                    marker=dict(
                        size=5, 
                        color='red', 
                        symbol='star',
                        ),
                    textposition="top center",     
                    textfont=dict(
                            size=12,
                            color="black"
                        ),
                    name="token_emb"
                )
            )


            num_nodes = token_emb.shape[0]
            adjacency_matrix = kneighbors_graph(token_emb, n_neighbors=min(3, num_nodes-1), mode='connectivity', metric="euclidean")
            G_ = nx.Graph(adjacency_matrix)
            mapping = {i_node: f'L{i}/{i_node}' for i_node in range(num_nodes)}
            G_ = nx.relabel_nodes(G_, mapping)
            
            edges_list = list(G_.edges)
            for i_edge in range(len(G_.edges)):
                edge = edges_list[i_edge]
                s_node = edge[0]
                e_node = edge[1]
                index_s = int(s_node.split('/')[1])
                index_e = int(e_node.split('/')[1])
                weight = 10**(len(tree_node_embedding)-i) + np.linalg.norm(token_emb[index_s] - token_emb[index_e])
                print(f'edge: {s_node} -> {e_node}, weight: {weight}')
                G.add_edge(s_node, e_node, weight=weight)
                # import pdb; pdb.set_trace()
                

            distance_to_zero = np.linalg.norm(token_emb, axis=1)
            near_index = np.argmin(distance_to_zero)

            for i_node in range(token_emb.shape[0]):
                G.add_edge(f'L{i-1}/{i_node//2}', f'L{i}/{i_node}', weight=10**(len(tree_node_embedding)-i))

            end_node_list = range(token_emb.shape[0])
            for end_node in end_node_list:
                self.plot_path(G, tree_node_embedding, i, near_index, end_node, plotly_fig_rute)

            plotly_fig_rute.update_layout(
                plot_bgcolor='white',   
                paper_bgcolor='white',  
                width=800,   
                height=600,   
                title="Tree Visualization",
                xaxis=dict(
                    visible=False 
                ),
                yaxis=dict(
                    visible=False 
                ),
                xaxis_scaleanchor="y",  
                yaxis_scaleanchor="x",
                legend=dict(
                    title="Legend",
                    x=0.01,
                    y=0.99
                )
            )

            # plotly_fig_rute_dict[f'tree/tree_{i}'] = plotly_fig_rute
            # fig_graph = plt.figure(figsize=(10, 10))
            # nx.draw(G, with_labels=True, node_size=700, node_color="lightblue")
            # fig_graph.savefig(f"fig_graph_{i}.png")
            # fig_graph.clear()

            # plotly_fig_rute_dict[f'graph/tree_{i}'] = fig_graph
            plotly_fig_rute.write_image(f"fig_emb_colored_with_rout_{i}.png", scale=3)
            path_list.append(f"fig_emb_colored_with_rout_{i}.png")
        
        return path_list, plotly_fig_rute_dict


    def rout_eval(self, trainer, pl_module, gathered_rout, gathered_label, emb_vis, gathered_sample):
        
        # import pdb; pdb.set_trace()
        print('Evaluating the routing vector, calculating SVC accuracy...')
        gathered_rout_flat = gathered_rout.reshape(gathered_rout.shape[0],-1)
        # acc = self.get_rout_svc_acc(gathered_rout_flat, gathered_label)
        acc = 1.00
        
        if gathered_rout_flat.shape[0] > 20000:
            rand_idx = np.random.choice(gathered_rout_flat.shape[0], 20000, replace=False)
            gathered_rout_flat = gathered_rout_flat[rand_idx]
            gathered_label = gathered_label[rand_idx]
            emb_vis = emb_vis[rand_idx]
        
        # tsne visualization of gathered_rout_flat
        # print('Evaluating the routing vector, tsne visualization of the routing vector...')
        # TSNE_vis = TSNE(n_components=2, random_state=0).fit_transform(gathered_rout_flat)
        # fig = plt.figure(figsize=(10, 10))
        # plt.scatter(TSNE_vis[:, 0], TSNE_vis[:, 1], c=gathered_label, cmap='rainbow', s=3)
        
        print('Evaluating the routing vector, uploading the results to wandb...')
        dict = {
            "rout/svc_acc": acc,
            # "rout/tsne": wandb.Image(fig),
            "rout/distribution_gathered_rout_flat": wandb.Histogram(gathered_rout_flat),
            "epoch": trainer.current_epoch,
        }        
        
        print('Evaluating the routing vector, visualizing the embedding space...')

        vis_plot_path, plotly_fig_rute_dict = self.get_rout_vis(pl_module.tree_node_embedding, emb_vis)
        dict.update(plotly_fig_rute_dict)
        for i, path in enumerate(vis_plot_path):
            dict[f"rout/emb_colored_with_rout_{i}"] = wandb.Image(path)

        if gathered_sample is not None:
            print('Evaluating the routing vector, plotting the sample images...')
            gathered_image = gathered_sample.reshape(-1, 28, 28)
            # show 10*10 images
            fig_sample = plt.figure(figsize=(30, 30))
            for i in range(100):
                plt.subplot(10, 10, i+1)
                plt.imshow(gathered_image[i], cmap='gray')
                plt.axis('off')

            dict.update({"rout/fig_sample_image": wandb.Image(fig_sample)})
            
            plt.close()
        
        wandb.log(dict)
        # import pdb; pdb.set_trace()

    def on_validation_epoch_end(self, trainer, pl_module):
        for val_idx in self.val_vis.keys():
            # val_sample_current = self.val_sample[val_idx]
            val_rout_current = self.val_rout[val_idx]
            val_input_current = self.val_input[val_idx]
            val_vis_current = self.val_vis[val_idx]
            val_label_current = self.val_label[val_idx]

            if True:
                # val_sample = torch.cat(val_sample_current).cuda()
                val_rout = torch.cat(val_rout_current[:1000]).cuda()
                val_input = torch.cat(val_input_current[:1000]).cuda()
                val_vis = torch.cat(val_vis_current[:1000]).cuda()
                val_label = torch.cat(val_label_current[:1000]).cuda()

                # gathered_sample = trainer.strategy.all_gather(val_sample).cpu().detach().numpy()
                gathered_rout = trainer.strategy.all_gather(val_rout).cpu().detach().numpy()
                gathered_input = trainer.strategy.all_gather(val_input).cpu().detach().numpy()
                gathered_vis = trainer.strategy.all_gather(val_vis).cpu().detach().numpy()
                gathered_label = trainer.strategy.all_gather(val_label).cpu().detach().numpy()

                # Reshape if necessary
                if len(gathered_vis.shape) == 3:
                    
                    gathered_rout = gathered_rout.reshape(-1, *gathered_rout.shape[2:])
                    gathered_input = gathered_input.reshape(-1, *gathered_input.shape[2:])
                    gathered_vis = gathered_vis.reshape(-1, *gathered_vis.shape[2:])
                    gathered_label = gathered_label.reshape(-1)

                if trainer.is_global_zero:
                    self.process_and_log_metrics(
                        gathered_input,
                        gathered_vis,
                        gathered_label,
                        trainer,
                        val_idx
                    )
                    self.rout_eval(
                        trainer, 
                        pl_module, 
                        gathered_rout, 
                        gathered_label, 
                        gathered_vis,
                        pl_module.reconstruct_example.cpu().detach().numpy(),
                        )

        # Clear stored data
        self.val_rout.clear()
        self.val_input.clear()
        self.val_high.clear()
        self.val_vis.clear()
        self.val_label.clear()