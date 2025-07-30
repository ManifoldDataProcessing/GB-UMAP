import warnings

import numpy as np
import umap
from sklearn.metrics import pairwise_distances
from umap.umap_ import nearest_neighbors

import granular_ball_computing
import load_datasets

warnings.filterwarnings("ignore")


def neighbor_preservation_rate(dist_high, dist_low, local_k=15):
    """
    计算邻域保持率，衡量局部误差
    :param dist_high:
    :param dist_low:
    :param local_k:
    :return:
    """
    # 高维邻域矩阵
    high_neighbors = np.argsort(dist_high, axis=1)[:, :local_k]
    # 低维邻域矩阵
    low_neighbors = np.argsort(dist_low, axis=1)[:, :local_k]
    # 计算邻域保持率
    preservation = np.mean([
        len(set(high_neighbors[i]).intersection(set(low_neighbors[i]))) / local_k
        for i in range(dist_high.shape[0])
    ])
    return round(1 - preservation, 4)


# 全局误差
def global_structure_error(dist_high, dist_low):
    """
    基于F范数衡量全局降维误差
    :param dist_high:
    :param dist_low:
    :return:
    """
    size = dist_high.shape[0]
    norm_F = np.linalg.norm(dist_high - dist_low, ord="fro")
    average_F = norm_F / (size * (size - 1))

    return average_F


def adaptive_umap(X, alpha=0.4, n_neighbors=15, dim_range=(2, 10), metric='precomputed', local_k=22):
    dist_high = pairwise_distances(X)
    knn_high = nearest_neighbors(dist_high, n_neighbors=n_neighbors, metric=metric, metric_kwds=None,
                                 angular=False,
                                 random_state=None, )

    best_dim = dim_range[0]
    final_embedding = None
    error = float('inf')

    for dim in range(dim_range[0], dim_range[1] + 1):
        # reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric=metric, min_dist=0,
        #                     precomputed_knn=knn_high, random_state=42)
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric=metric, min_dist=0,
                            precomputed_knn=knn_high)
        Z = reducer.fit_transform(dist_high)
        dist_low = pairwise_distances(Z)

        local_error = neighbor_preservation_rate(dist_high, dist_low, local_k)
        global_error = global_structure_error(dist_high, dist_low)
        weighted_error = alpha * local_error + (1 - alpha) * global_error

        if weighted_error < error:
            error = weighted_error
            best_dim = dim
            final_embedding = Z
    return best_dim, error, final_embedding

