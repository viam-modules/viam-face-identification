import numpy as np
import torch.nn.functional as F
from torch.nn import CosineSimilarity


def distance_norm_l1(t1, t2):
    """
    Computes the L1 norm distance between two tensors.

    Args:
        t1 (torch.Tensor): The first tensor.
        t2 (torch.Tensor): The second tensor.

    Returns:
        float: The Euclidean distance between the two tensors.
    """
    return F.pairwise_distance(t1, t2, p=1).item()  # pylint: disable=not-callable


def distance_norm_l2(t1, t2):
    """
    Computes the Euclidean distance between two tensors or numpy array.

    Args:
        t1 (torch.Tensor or numpy.ndarray): The first tensor or numpy array.
        t2 (torch.Tensor or numpy.ndarray): The second tensor or numpy array.

    Returns:
        float: The Euclidean distance between the two inputs.
    """
    if isinstance(t1, np.ndarray) and isinstance(t2, np.ndarray):
        euc_distance = t2 - t1
        euc_distance = np.sum(np.multiply(euc_distance, euc_distance))
        euc_distance = np.sqrt(euc_distance)
        return euc_distance
    return F.pairwise_distance(t1, t2).item()  # pylint: disable=not-callable


def cosine_distance(t1, t2):
    """
    Computes the cosine distance between two tensors.

    Args:
        t1 (torch.Tensor): The first tensor.
        t2 (torch.Tensor): The second tensor.

    Returns:
        float: The cosine distance between the two tensors.
    """
    cos_sim = CosineSimilarity(dim=0, eps=1e-6)
    return 1 - cos_sim(t1, t2).item()
