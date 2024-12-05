import numpy as np
import polars as pl 
from itertools import combinations, chain

def _get_purged_train_indices(event_times, test_times):
    event_times = event_times.select(
        [pl.all(), pl.arange(0, pl.len()).alias("count_index")]
    )
    train_times = event_times.clone()
    overlap_df = pl.DataFrame([])
    for test_start,test_end in test_times.iter_rows():
        overlap1 = event_times.filter((test_start<=pl.col("event_starts")) & (pl.col("event_starts") <= test_end))
        overlap2 = event_times.filter((test_start<=pl.col("event_ends")) & (pl.col("event_ends") <= test_end))
        overlap3 = event_times.filter((test_start>=pl.col("event_starts")) & (pl.col("event_ends") >= test_end))
        overlap_df = pl.concat([overlap_df,overlap1,overlap2,overlap3],how="vertical").unique()
    train_times = train_times.join(overlap_df,on ="event_ends",how ="anti")        
    return train_times.select("count_index").to_numpy().flatten().tolist()


def _get_embargo_times(bar_times, embargo_pct):
    if isinstance(bar_times,np.ndarray):
        bar_times = list(bar_times)
    elif isinstance(bar_times,pl.DataFrame):
        bar_times = bar_times.to_numpy().flatten()

    
    step = 0 if bar_times is None else int(len(bar_times) * embargo_pct)
    if step == 0:
        res = pl.DataFrame({"event_starts":bar_times,"event_ends":bar_times})
    else:
        res = pl.concat([
            pl.DataFrame({"event_starts":bar_times[:-step],"event_ends":bar_times[step:]}),
            pl.DataFrame(
                {"event_starts":bar_times[-step:],"event_ends":[bar_times[-1] for i in range(step)]}
            )
        ],how="vertical")
    return res


def apply_purging_and_embargo(event_times, test_times, bar_times=None, embargo_pct=0.):
    if bar_times is not None:
        embargo_times = _get_embargo_times(bar_times, embargo_pct)
        adj_test_times = embargo_times.join(test_times,left_on = "event_starts",right_on="event_ends",how="inner").select(
            [pl.col("event_starts_right").alias("event_starts"),pl.col("event_ends")]
        )
    else:
        adj_test_times = test_times
    train_indices = _get_purged_train_indices(event_times, adj_test_times)
    return train_indices



class PurgedKFold:

    def __init__(self, n_splits=3, embargo_pct=0.):
        
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, event_times):
        '''
        params: event_times: pl.DataFrame with two columns("event_starts" and "event_ends").
        '''
        num_obs = event_times.shape[0]
        indices = np.arange(num_obs)
        embargo = int(num_obs * self.embargo_pct)
        test_splits = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)]

        for i, j in test_splits:
            test_indices = indices[i:j]
            test_times = pl.DataFrame(
                {
                    "event_starts": event_times[i.item(),"event_starts"],
                    "event_ends": event_times[min(j.item() - 1 + embargo,num_obs-1), "event_ends"]
                }
            )
            train_indices = _get_purged_train_indices(event_times, test_times)
            yield train_indices, test_indices


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





