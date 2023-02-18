import polars as pl
import numpy as np
from finml.models.xgboost.xgboost_train import CV_train_classifier
from finml.models.xgboost.xgboost_tools import save_model,load_model,preidct_prob 
from finml.sampling.get_weights import label_avg_uniqueness,apply_time_decay_to_weights

def train_meta(
    bar_type,
    X,
    y,
    bar_times,
    event_times,
    sample_weights,
    n_split = 5,
    base_dir = "../models",
    xgb_type="classifier"
):
    models = load_model(
            bar_type,
            n_split = n_split,
            base_dir = base_dir,
            xgb_type = xgb_type,
        )
    predict_yprob = [] 
    for model in models:
        predict_yprob.append(model.predict_proba(X.to_pandas()))
    predict_yprob = np.mean(predict_yprob,axis=0)  # (sample_length,num_class)
    predict_y = np.argmax(predict_yprob,axis=1)
    meta_y = (predict_y == y).astype(int)[0] ## (sample_length,)
    X = pl.concat([X,pl.DataFrame({"predict_prob":predict_yprob[:,predict_yprob.shape[1]-1]})],how="horizontal")
    y = pl.DataFrame({"label":meta_y})
    meta_models,eval_score = CV_train_classifier(    
        X = X,
        y = y,
        sample_weight = sample_weights.select("values"),
        num_class = 2,
        num_groups = 4, 
        num_test_groups = 2, 
        bar_times = bar_times, 
        event_times = event_times,
        num_actors = 10,
        cpus_per_actor = 1,
        embargo_pct=0.01,
        return_info = True,
    )

    save_model(
        meta_models,
        "meta_"+bar_type,
        base_dir,
    )


if __name__ == "__main__":
    bar_type = "ETHBUSD-PERP.BINANCE-250-VALUE_IMBALANCE-LAST-EXTERNAL"
    df = pl.read_parquet(bar_type+".parquet")
    if '__index_level_0__' in df.columns:
        df = df.drop(['__index_level_0__'])
        labeled_df = df.filter(pl.col("label").abs()>=0.005)
    event_times = labeled_df.select([pl.col("event_starts"),pl.col("event_ends")])
    bar_times = labeled_df.select([pl.col("datetime")])
    avg_uniqueness = label_avg_uniqueness(bar_times,event_times)
    sample_weights = apply_time_decay_to_weights(avg_uniqueness,0.5)
    X = labeled_df.drop(columns=[
         'datetime',
         'open',
        'high',
        'low',
        #'count_index',
        'event_ends',
        #'source',
        'event_starts',
        'label']
    )
    print(X.columns)
    y = labeled_df.select(pl.col("label"))
    y = y.select(pl.when(pl.col("label")>0).then(1).otherwise(0).alias("label"))
    train_meta(
        bar_type,
        X,
        y,
        bar_times,
        event_times,
        sample_weights,
        n_split = 5,
        base_dir = "../models",
        xgb_type="classifier"
    )