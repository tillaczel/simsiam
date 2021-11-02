import faiss
import numpy as np


def knn_acc(x, y, top_k):
    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x)
    dist, i = index.search(x, int(np.max(top_k)+1))
    acc, same, correct_ratio = dict(), np.zeros(dist.shape), np.zeros(dist.shape)
    for k in range(np.max(top_k)):
        same[:, k] = (y[i[:, k+1]] == y)
        correct_ratio[:, k] = np.mean(same[:, 0:k+1], axis=1)
        if (k + 1) in set(top_k):
            acc[f'top_{k+1}_knn_acc'] = np.mean(0.5 < correct_ratio)
    return acc
