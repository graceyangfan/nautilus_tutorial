from itertools import combinations, chain

import numpy as np
from scipy.stats import gaussian_kde
import polars as pl 

from ..evaluation.metrics import sharpe_ratio 
from ..evaluation.cross_validation import apply_purging_and_embargo 
from ..utils.stats import KDERv 

def compute_pbo(perm_matrix: np.ndarray, eval_fn=sharpe_ratio, n_partitions=10):
    """
    Compute the probability of backtest overfitting for an optimization procedure used to select
    strategies or model configuration. See Bailey et al. [2017] for the original paper.

    :return: probability, rank logits, train sets metrics, test sets associated mtrics
    """
    assert n_partitions % 2 == 0

    n_obs, n_strats = perm_matrix.shape
    partition_size = n_obs // n_partitions
    rank_logits = []
    train_optimal_perm = []
    test_assoc_perm = []

    # Iterate over all combinations of partitions
    for partitions in combinations(range(n_partitions), n_partitions // 2):
        # Calculate the indices of the train and test sets
        train_indices = list(chain.from_iterable(
            [list(range(int(p*partition_size), int((p+1)*partition_size))) for p in partitions]
        ))
        test_indices = [i for i in range(n_obs) if i not in train_indices]

        # Split the data into train and test sets
        train_set = perm_matrix[train_indices, :]
        test_set = perm_matrix[test_indices, :]

        # Calculate the evaluation metric for the train and test sets
        train_perm = eval_fn(train_set, axis=0)
        test_perm = eval_fn(test_set, axis=0)

        # Find the best-performing strategy on the train set
        train_optimal_idx = np.argmax(train_perm)
        train_optimal_perm.append(train_perm[train_optimal_idx])

        # Find the rank of the best-performing strategy on the test set
        test_assoc_perm.append(test_perm[train_optimal_idx])
        test_rank = 0
        for r, i in enumerate(np.argsort(test_perm)):
            if i == train_optimal_idx:
                test_rank = r
        test_rank /= test_perm.shape[0]

        # Calculate the logit of the rank
        test_rank_logit = np.log(test_rank/(1 - test_rank) + 1e-3)
        rank_logits.append(test_rank_logit)

    # Fit a Gaussian KDE to the logits of the ranks
    kde = gaussian_kde(rank_logits)

    # Compute the probability of backtest overfitting
    kde_dist = KDERv(kde)
    pbo = kde_dist.cdf(0)
    return pbo, rank_logits, train_optimal_perm, test_assoc_perm

