import numpy as np
import torch


class Metrics:
    def __init__(self, n_classes, top_k, knn_k: int = 200, knn_t: float = 0.1):
        self.n_classes = n_classes
        self.top_k = top_k
        self.knn_k = knn_k
        self.knn_t = knn_t

    def run(self, f_test: np.array, z_test: np.array, y_test: np.array,
            f_train: np.array = None, z_train: np.array = None, y_train: np.array = None):
        result = dict()
        result.update(self.knn_acc(f_test, y_test, f_train, y_train))
        result.update(self.emb_std(z_train, z_test))
        result.update(self.emb_corr(z_train, z_test))
        return result

    # knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
    # implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
    def calc_sim(self, x_q, x_g, knn_k):
        x_g_norm = torch.FloatTensor(x_g / np.linalg.norm(x_g, axis=1, keepdims=True))
        x_q_norm = torch.FloatTensor(x_q / np.linalg.norm(x_q, axis=1, keepdims=True))
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(x_q_norm, x_g_norm.permute(1, 0))
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
        return sim_weight.numpy(), sim_indices.numpy()

    def knn_predict(self, sim_weight, sim_indices, y_g):
        sim_weight, sim_indices, y_g = torch.FloatTensor(sim_weight), torch.LongTensor(sim_indices), torch.LongTensor(y_g)
        # [B, K]
        sim_labels = torch.gather(y_g[:, 0].expand(sim_weight.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / self.knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(sim_weight.size(0) * self.knn_k, self.n_classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(
            one_hot_label.view(sim_weight.size(0), -1, self.n_classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels.numpy()

    def knn_acc(self, x_test, y_test, x_train, y_train):
        sim_weight, sim_indices = self.calc_sim(x_test, x_test, self.knn_k+1)
        sim_weight, sim_indices = sim_weight[:, 1:], sim_indices[:, 1:]
        pred_labels = self.knn_predict(sim_weight, sim_indices, y_test)
        acc = dict()
        _acc = get_accuracy(pred_labels, y_test, self.top_k)
        for k, v in _acc.items():
            acc[f'knn_{k}'] = v
        if x_train is not None and y_train is not None:
            sim_weight, sim_indices = self.calc_sim(x_test, x_train, self.knn_k)
            pred_labels = self.knn_predict(sim_weight, sim_indices, y_train)
            _acc = get_accuracy(pred_labels, y_test, self.top_k)
            for k, v in _acc.items():
                acc[f'full_knn_{k}'] = v
        return acc

    def emb_std(self, x_train, x_test):
        result = dict()
        if x_train is not None:
            result['full_gallery_std'] = np.mean(np.std(x_train, axis=0))
        result['query_std'] = np.mean(np.std(x_test, axis=0))
        return result

    def emb_corr(self, x_train, x_test):
        result = dict()
        if x_train is not None:
            result['full_gallery_corr'] = np.mean(np.abs(np.corrcoef(x_train.T)))-1/x_train.shape[1]
        result['query_corr'] = np.mean(np.abs(np.corrcoef(x_test.T)))-1/x_test.shape[1]
        return result


def get_accuracy(pred_labels, y, top_k):
    top_k_max = np.max(top_k)
    acc, same = dict(), np.zeros((pred_labels.shape[0], top_k_max))
    for k in range(top_k_max):
        same[:, k] = (pred_labels[:, k] == y[:, 0])
        if (k + 1) in set(top_k):
            acc[f'top_{k + 1}_acc'] = np.mean(np.sum(same[:, :k+1], axis=1))
    return acc
