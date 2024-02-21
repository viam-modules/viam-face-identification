from torch.nn import CosineSimilarity
import torch.nn.functional as F
from torch import Tensor
import numpy as np

def euclidean_distance(t1,t2):
    return F.pairwise_distance(t1,t2, p=1).item()

def euclidean_l2_distance(t1, t2):
    if isinstance(t1, np.ndarray) and  isinstance(t2, np.ndarray):
        euclidean_distance = t2 - t1
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance
    else:
        return F.pairwise_distance(t1,t2).item()

def cosine_distance(t1, t2):
    cos_sim = CosineSimilarity(dim=0, eps=1e-6)
    return 1-cos_sim(t1, t2).item()