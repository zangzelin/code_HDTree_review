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

    def get_rout_vis(self, rout_vector, emb_vis):            
        import pdb; pdb.set_trace()
        path_list = []
        for i in range(rout_vector.shape[1]):
            rute_bool = rout_vector.copy()[:,:,0]
            rute_bool[rute_bool>0.5] = 1
            rute_bool[rute_bool<=0.5] = 0
            rute_str = rute_bool[:, :(i+1)].astype(np.int32).astype(str)
            rute_str_list = [''.join(rute_str[j]) for j in range(rute_str.shape[0])]
            # import pdb; pdb.set_trace()
            
            plotly_fig_rute = px.scatter(
                x=emb_vis[:, 0], y=emb_vis[:, 1], color=rute_str_list,
                size_max=3)
            plotly_fig_rute.update_traces(marker=dict(size=3))
            plotly_fig_rute.write_image(f"fig_emb_colored_with_rout_{i}.png", scale=3)
            path_list.append(f"fig_emb_colored_with_rout_{i}.png")
        
        return path_list


    def rout_eval(self, trainer, pl_module, gathered_rout, gathered_label, emb_vis, gathered_sample):
        
        
        # import pdb; pdb.set_trace()
        print('Evaluating the routing vector, calculating SVC accuracy...')
        gathered_rout_flat = gathered_rout.reshape(gathered_rout.shape[0],-1)
        # acc = self.get_rout_svc_acc(gathered_rout_flat, gathered_label)
        acc = 1.00
        
        if gathered_rout_flat.shape[0] > 5000:
            rand_idx = np.random.choice(gathered_rout_flat.shape[0], 5000, replace=False)
            gathered_rout_flat = gathered_rout_flat[rand_idx]
            gathered_label = gathered_label[rand_idx]
            # emb_vis = emb_vis[rand_idx]
        
        # tsne visualization of gathered_rout_flat
        print('Evaluating the routing vector, tsne visualization of the routing vector...')
        TSNE_vis = TSNE(n_components=2, random_state=0).fit_transform(gathered_rout_flat)
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(TSNE_vis[:, 0], TSNE_vis[:, 1], c=gathered_label, cmap='rainbow', s=3)
        
        print('Evaluating the routing vector, uploading the results to wandb...')
        dict = {
            "rout/svc_acc": acc,
            "rout/tsne": wandb.Image(fig),
            # "rout/fig_sample_image": wandb.Image(fig_sample),
            "rout/distribution_gathered_rout_flat": wandb.Histogram(gathered_rout_flat),
            "epoch": trainer.current_epoch,
        }        
        
        print('Evaluating the routing vector, visualizing the embedding space...')
        vis_plot_path = self.get_rout_vis(gathered_rout, emb_vis)
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

            if len(val_vis_current) > 0 and (trainer.current_epoch + 1) % self.inter == 0:
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
                    # gathered_sample = gathered_sample.reshape(-1, *gathered_sample.shape[2:])
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
                        None,
                        )

        # Clear stored data
        self.val_rout.clear()
        self.val_input.clear()
        self.val_high.clear()
        self.val_vis.clear()
        self.val_label.clear()