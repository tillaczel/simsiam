from torch import nn


def symmetric_cos_dist(z1, z2, p1, p2, mean=True):
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-08)
    return -(cos_sim(p1, z2).mean()+cos_sim(p2, z1).mean())/2
