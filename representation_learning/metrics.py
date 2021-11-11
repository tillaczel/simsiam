import faiss
import numpy as np


def knn_acc(x, y, top_k):
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x)
    dist, i = index.search(x, int(np.max(top_k)+1))
    n_samples = dist.shape[0]
    acc, same, correct_ratio = dict(), np.zeros((n_samples, np.max(top_k))), np.zeros((n_samples, np.max(top_k)))
    for k in range(np.max(top_k)):
        same[:, k] = (y[i[:, k+1]] == y)[:, 0]
        correct_ratio[:, k] = np.mean(same[:, 0:k+1], axis=1)
        if (k + 1) in set(top_k):
            acc[f'top_{k+1}_knn_acc'] = np.mean(0.5 < correct_ratio[:, k])
    return acc


class Metrics:
    def __init__(self, top_k, emb_dim: int = 2048):
        self.top_k = top_k
        self.emb_dim = emb_dim

    def run(self, x_test: np.array, y_test: np.array, x_train: np.array = None, y_train: np.array = None):
        self.assert_dim(x_test, y_test, x_train, y_train)
        result = dict()
        result.update(self.knn_acc(x_test, y_test, x_train, y_train))
        result.update(self.emb_std(x_train, x_test))
        return result

    def calc_dist(self, x_train, x_test):
        x_train = x_train / np.linalg.norm(x_train, axis=1, keepdims=True)
        x_test = x_test / np.linalg.norm(x_test, axis=1, keepdims=True)
        index = faiss.IndexFlatL2(x_train.shape[1])
        index.add(x_train)
        dist, idx = index.search(x_test, int(np.max(self.top_k) + 1))
        return dist, idx

    def knn_acc(self, x_test, y_test, x_train, y_train):
        if x_train is None:
            dist, idx = self.calc_dist(x_test, x_test)
        else:
            dist, idx = self.calc_dist(x_train, x_test)
        if y_train is None:
            y = y_test
        else:
            y = y_train
        n_samples, top_k_max = dist.shape[0], np.max(self.top_k)
        acc, same, correct_ratio = dict(), np.zeros((n_samples, top_k_max)), np.zeros((n_samples, top_k_max))
        for k in range(top_k_max):
            same[:, k] = (y[idx[:, k + 1]] == y_test)[:, 0]
            correct_ratio[:, k] = np.mean(same[:, 0:k + 1], axis=1)
            if (k + 1) in set(self.top_k):
                acc[f'top_{k + 1}_knn_acc'] = np.mean(0.5 < correct_ratio[:, k])
        return acc

    def emb_std(self, x_train, x_test):
        result = dict()
        if x_train is not None:
            result['gallery_std'] = np.mean(np.std(x_train, axis=1))
        result['query_std'] = np.mean(np.std(x_test, axis=1))
        return result

    def assert_dim(self, x_test: np.array, y_test: np.array, x_train: np.array = None, y_train: np.array = None):
        assert x_test.shape[1] == self.emb_dim
        assert x_test.shape[0] == y_test.shape[0]
        if x_train is not None:
            assert x_train.shape[1] == self.emb_dim
            assert x_test.shape[0] == x_train.shape[0]
            if y_train is not None:
                assert x_test.shape[0] == y_train.shape[0]

