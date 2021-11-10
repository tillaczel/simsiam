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
