from itertools import combinations, chain

import numpy as np
from scipy.stats import gaussian_kde
import polars as pl 

from ..modelling.metrics import sharpe_ratio 
from ..modelling.cross_validation import apply_purging_and_embargo 
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



class CombinatorialPurgedCV:
    def __init__(self, num_groups, num_test_groups, bar_times, event_times, embargo_pct=0.):
        self.num_groups = num_groups
        self.num_test_groups = num_test_groups
        self.bar_times = bar_times
        self.event_times = event_times
        self.embargo_pct = embargo_pct

        self.num_obs = event_times.shape[0]
        self.group_size = self.num_obs // self.num_groups
        self.splits_of_test_groups = [[] for _ in range(num_groups)]

    def split(self):
        for split_index, test_groups in enumerate(combinations(range(self.num_groups), self.num_test_groups)):
            test_indices = list(chain.from_iterable(
                list(range(int(g * self.group_size), int((g + 1) * self.group_size)))
                for g in test_groups
            ))
            test_times = self.event_times[test_indices,:]

            train_indices = apply_purging_and_embargo(
                self.event_times, test_times, self.bar_times, self.embargo_pct
            )

            for g in test_groups:
                self.splits_of_test_groups[g].append(split_index)

            yield train_indices, test_indices

    def get_backtest_paths(self):
        num_paths = min(len(splits) for splits in self.splits_of_test_groups)
        for i in range(num_paths):
            path_splits = [splits[i] for splits in self.splits_of_test_groups]
            unique_path_splits = sorted(np.unique(path_splits))
            test_indices = [
                list(range(int(g * self.group_size), int((g + 1) * self.group_size)))
                for g in unique_path_splits
            ]
            yield list(zip(test_indices, unique_path_splits))
