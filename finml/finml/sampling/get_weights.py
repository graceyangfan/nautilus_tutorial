import polars as pl
import numpy as np 
from joblib import Parallel, delayed
from tqdm import tqdm 


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
            "values": np.zeros(event_times_iloc2-event_times_iloc1)
        }
    )
    event_times = event_times.select([pl.col("event_starts"),pl.col("event_ends")])
    for event_starts,event_ends in event_times.iter_rows():
        res = res.with_column(
            pl.when((pl.col("index") >= event_starts) & (pl.col("index") <= event_ends))
            .then(pl.col("values")+1)
            .otherwise(pl.col("values"))
            .alias("values")
        )
    return res


def label_avg_uniqueness(bars, events):

    events_counts = count_events_per_bar(bars.select("datetime"), events)
    events_counts = events_counts.with_column(pl.col("values").fill_null(0))
    res = pl.DataFrame(
        {
            "index":events.select("event_starts").to_numpy().flatten(),
            "values": np.zeros(events.shape[0])
        }
    )
    events = events.select([pl.col("event_starts"),pl.col("event_ends")])
    for event_starts,event_ends in events.iter_rows():
        res = res.with_column(
            pl.when((pl.col("index") == event_starts))
            .then((1.0 / events_counts.filter((pl.col("index")>= event_starts)&(pl.col("index")<=event_ends))["values"]).mean())
            .otherwise(pl.col("values"))
            .alias("values")
            )
    return res


def get_event_indicators(bar_times, event_times, njobs = 1):
    res = bar_times.select(pl.col("datetime").alias("index"))
    params = [{"res":res,
               "event_starts":event_times[i,"event_starts"],
               "event_ends":event_times[i,"event_ends"],
              "count_index":event_times[i,"count_index"],} for i in range(event_times.shape[0])]
    indicators = Parallel(n_jobs=njobs)(delayed(_get_event_indicator)(param) for param in tqdm(params))
    return pl.concat(indicators, how="horizontal")
    
def _get_event_indicator(params):
    return  params["res"].select(
            pl.when((pl.col("index")>= params["event_starts"])&(pl.col("index")<= params["event_ends"]))
            .then(1)
            .otherwise(0)
            .alias(str(params["count_index"]))
        )

def sample_sequential_bootstrap(event_indicators, size=None):
    if size is None:
        size = event_indicators.shape[1]
    if size > event_indicators.shape[1]:
        size = event_indicators.shape[1]
        
    samples_sum = pl.DataFrame({"sum":np.zeros(event_indicators.shape[0])})
    sample_columns = [] 
    
    while len(sample_columns) < size:
        trial_uniqueness = pl.concat([event_indicators,samples_sum], how="horizontal")
        trial_uniqueness = trial_uniqueness.select([
            (pl.col(item)/(pl.col(item)+pl.col("sum"))).alias(item) for item in event_indicators.columns
        ])
        avg_uniqueness = trial_uniqueness.select(
            [
                pl.col(item).filter(pl.col(item)>0.0).mean()  for item in event_indicators.columns
            ]
        )
        del trial_uniqueness 
        probs_sum = avg_uniqueness.sum(axis=1)
        probs = avg_uniqueness.select([pl.col(item)/probs_sum for item in avg_uniqueness.columns]).to_numpy()[0]
        del avg_uniqueness
        idxs = [np.random.choice(event_indicators.columns, p=probs)]
        sample_columns.extend(idxs)
        samples_sum = pl.concat([samples_sum,event_indicators.select(idxs)], how="horizontal")
        samples_sum = samples_sum.sum(axis=1).to_frame(name="sum")
        ##不放回采样的独立性更高
        event_indicators = event_indicators.drop(idxs[0])
    del event_indicators
    return sorted([int(item) for item in sample_columns])
    

def _get_return_attributions(event_times, events_counts, bars):
    returns = bars.select([pl.col("close").log().diff().alias("values"),pl.col("datetime").alias("index")])
    weights = pl.DataFrame(
        {
            "index":event_times.select("event_starts").to_numpy().flatten(),
            "values": np.zeros(event_times.shape[0])
        }
    )
    for event_starts,event_ends in event_times.iter_rows():
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

def compute_weights_by_returns(bars_times,event_times):

    events_counts = count_events_per_bar(bars_times.select("datetime"), event_times)
    events_counts = events_counts.with_column(pl.col("values").fill_null(0))

    raw_weights = _get_return_attributions(event_times, events_counts, bars_times)
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


