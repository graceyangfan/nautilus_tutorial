import polars as pl
import numpy as np 

def drop_rare_labels(events, min_pct=0.05, min_classes=2):
    """
    Recursively drop labels with insufficient samples
    """
    while True:
        counts = events["label"].value_counts()
        counts = counts.with_column((pl.col("label")/pl.col("label").sum()).alias("rate"))
        if counts["rate"].min() > min_pct or counts.shape[0] <= min_classes:
            break
        events = events.filter(
            pl.col("label")!= counts[counts["rate"].arg_min(),"label"]
        )
    return events


def count_events_per_bar(bar_times, event_times):


    event_times = event_times.with_column(
        pl.col("event_ends").fill_null(event_times[-1,"event_ends"])
    )
    event_times_iloc1 = bar_times.select(
        pl.col("datetime").search_sorted(event_times[0,"event_starts"])
    ).to_numpy().flatten()[0]
    event_times_iloc2 = bar_times.select(
        pl.col("datetime").search_sorted(event_times["event_ends"].max())
    ).to_numpy().flatten()[0]
    res = pl.DataFrame(
        {
            "index":bar_times[event_times_iloc1:event_times_iloc2 + 1].to_numpy().flatten(),
            "values": np.zeros(event_times_iloc2-event_times_iloc1 + 1)
        }
    )
    for event_starts,event_ends in event_times.iterrows():
        res = res.with_column(
            pl.when((pl.col("index") >= event_starts) & (pl.col("index") <= event_ends))
            .then(pl.col("values")+1)
            .otherwise(pl.col("values"))
            .alias("values")
        )
    return res


def label_avg_uniqueness(bars, events):

    events_counts = count_events_per_bar(bars.select("datetime"), events)
    events_counts = events_counts.filter(~pl.col("index").is_duplicated())
    events_counts = events_counts.with_column(pl.col("values").fill_null(0))
    res = pl.DataFrame(
        {
            "index":events.select("event_starts").to_numpy().flatten(),
            "values": np.zeros(events.shape[0])
        }
    )
    for event_starts,event_ends in events.iterrows():
        res = res.with_column(
            pl.when((pl.col("index") == event_starts))
            .then((1.0 / events_counts.filter((pl.col("index")>= event_starts)&(pl.col("index")<=event_ends))["values"]).mean())
            .otherwise(pl.col("values"))
            .alias("values")
            )
    return res


def get_event_indicators(bar_times, event_times):
    dict1 = {str(i):np.zeros(bar_times.shape[0]) for i in range(event_times.shape[0])}
    dict1["index"] = bar_times["datetime"]
    res = pl.DataFrame(dict1)
    for i in range(event_times.shape[0]):
        res = res.with_column(
            pl.when((pl.col("index")>= event_times[i,"event_starts"])&(pl.col("index")<=event_times[i,"event_ends"]))
            .then(1)
            .otherwise(pl.col(str(i)))
            .alias(str(i))
            
        )
    return res

def _get_avg_uniqueness(event_indicators):
    event_indicators = event_indicators.select(pl.all().exclude("index"))
    concurrency = event_indicators.sum(axis=1)
    uniqueness =  event_indicators/concurrency
    uniqueness = uniqueness.fill_nan(0.0)
    avg_uniqueness = uniqueness.select(
        [
            pl.col(item).filter(pl.col(item)>0.0).mean()  for item in uniqueness.columns 
        ]
    )
    return avg_uniqueness


def sample_sequential_bootstrap(event_indicators, size=None):
    event_indicators = event_indicators.select(pl.all().exclude("index"))
    if size is None:
        size = event_indicators.shape[1]
    samples = []
    while len(samples) < size:
        trial_avg_uniq = dict() 
        for event_id in event_indicators.columns:
            new_samples = samples+[event_id]
            trial_event_indicators = event_indicators.select(
                [pl.col(new_samples[i]).alias(str(i)) for i in range(len(new_samples))]
            )
            trial_avg_uniq[event_id]  = _get_avg_uniqueness(trial_event_indicators)[:,-1].to_numpy()[-1]
        trial_avg_uniq_sum = sum(trial_avg_uniq.values())
        probs = [item / trial_avg_uniq_sum for item in trial_avg_uniq.values()]
        samples += [np.random.choice(event_indicators.columns, p=probs)]
    return samples

def _get_return_attributions(event_times, events_counts, bars):
    returns = bars.select([pl.col("Close").log().diff().alias("values"),pl.col("datetime").alias("index")])
    weights = pl.DataFrame(
        {
            "index":event_times.select("event_starts").to_numpy().flatten(),
            "values": np.zeros(event_times.shape[0])
        }
    )
    for event_starts,event_ends in event_times.iterrows():
        return_attributed = returns.filter(
            (pl.col("index")>=event_starts) & (pl.col("index")<=event_ends) 
        )["values"] / events_counts.filter(
            (pl.col("index")>=event_starts) & (pl.col("index")<=event_ends) 
        )["values"]
        weights = weights.with_column(
            pl.when(pl.col("index") == event_starts)
            .then(return_attributed.sum()).otherwise(pl.col("values")).alias("values")
        )
    weights = weights.select(
        [pl.col("index"),pl.col("values").abs()]
    )
    return weights

def compute_weights_by_returns(event_times, events_counts, bars):
    raw_weights = _get_return_attributions(event_times, events_counts, bars)
    norm_weights = raw_weights.select(
        [pl.col("index"),pl.col("values")/pl.col("values").sum()*raw_weights.shape[0]]
    )
    return norm_weights


def apply_time_decay_to_weights(avg_uniqueness, oldest_weight=1.0):
    cum_uniqueness = avg_uniqueness.select(
        [pl.col("index").sort(),pl.col("values").sort_by("index").cumsum()]
    )
    cum_uniqueness_last =  cum_uniqueness[-1,"values"]
    print(cum_uniqueness_last)
    if oldest_weight >= 0:
        slope = (1. - oldest_weight) / cum_uniqueness_last
    else:
        slope = 1. / ((oldest_weight + 1) * cum_uniqueness_last)
    const = 1. - slope * cum_uniqueness_last
    weights = cum_uniqueness.select([pl.col("index"),pl.col("values")*slope + const])
    weights = weights.select([pl.col("index"),pl.when(pl.col("values")>0).then(pl.col("values")).otherwise(0.0)])
    return weights


