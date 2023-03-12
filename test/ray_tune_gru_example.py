from functools import partial
from ray import tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from finml.models.gru import GRU 
from finml.models.loss import Tilde_Q
from finml.evaluation.metrics import negtivate_MSE
from finml.data.handler import StandardNorm 
from finml.evaluation.cross_validation import CombinatorialPurgedCV
import polars as pl 
from finml.sampling.get_weights import label_avg_uniqueness,apply_time_decay_to_weights
import os 
def train_gru(
    config,        
    X_train,
    y_train,
    X_test,
    y_test,
    save_prefix=None,
):
    loss_fn = Tilde_Q(0.99,1)
    metric_fn = Tilde_Q(0.99,1).neg_loss 
    x_handler = StandardNorm() 
    #y_handler = StandardNorm() 
    d_feat = X_train.shape[-1]
    predict_len = config["predict_len"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]

    model = GRU(
        d_feat=d_feat,
        predict_len=predict_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        x_handler=x_handler,
        y_handler=None,
        loss_fn=loss_fn,
        metric_fn=metric_fn
    )
    best_loss,best_valid  = model.fit(
        X_train,
        y_train,
        X_test,
        y_test,
        session
    )
    print(f"the best valid is {best_valid}")
    if save_prefix:
        model.save(save_prefix)

def tune_gru(
    X_train,
    y_train,
    X_test,
    y_test,
    save_prefix,
    max_num_epochs,
    num_samples,
    cpus_per_trial,
    gpus_per_trial
):
    config = {
        "predict_len":10,
        "hidden_size": tune.choice([16, 32, 64, 128, 256]),
        "num_layers": tune.choice([1, 2, 3]),
    }
    scheduler = ASHAScheduler(
        metric="val_score",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )
    reporter = CLIReporter(
        metric_columns=[ "training_iteration","train_score","val_score"]
    )
    result = tune.run(
        partial(
            train_gru,
            X_train = X_train,
            y_train = y_train,
            X_test = X_test,
            y_test = y_test,
            save_prefix=None,
        ),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )
    best_trial = result.get_best_trial("val_score", "max", "last")
    print("Best trial config: {}".format(best_trial.config))

    # train and save model 
    train_gru(
        config = best_trial.config,
        X_train = X_train,
        y_train = y_train,
        X_test = X_test,
        y_test = y_test,
        save_prefix= save_prefix,
    )



def CPCV_train_gru(
    X, 
    y, 
    sample_weight,
    num_groups, 
    num_test_groups, 
    bar_times, 
    event_times, 
    embargo_pct=0.01,
    max_num_epochs=10,
    num_samples=30,
    cpus_per_trial=1,
    gpus_per_trial=1,
    save_prefix="models"
):
    cv = CombinatorialPurgedCV(
        num_groups=num_groups, 
        num_test_groups=num_test_groups, 
        bar_times=bar_times, 
        event_times=event_times, 
        embargo_pct=embargo_pct
    )
    models = []
    valid_score = 0
    idx = 0 
    for train_indices, test_indices in cv.split():
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        X_train = X_train.to_pandas() 
        X_test = X_test.to_pandas()
        y_train = y_train.to_pandas()
        y_test = y_test.to_pandas()
        if sample_weight is None:
            train_sample_weight = None 
            test_sample_weight = None 
        else:
            train_sample_weight = sample_weight[train_indices, :].to_pandas().values.reshape(-1,)
            test_sample_weight = [sample_weight[test_indices, :].to_pandas().values.reshape(-1,)]

        tune_gru(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            save_prefix=save_prefix+"_"+str(idx),
            max_num_epochs=max_num_epochs,
            num_samples=num_samples,
            cpus_per_trial=cpus_per_trial,
            gpus_per_trial=gpus_per_trial,
        )

        idx += 1
    
if __name__ == "__main__":
    bar_type = "ETHBUSD-PERP.BINANCE-250-VALUE_IMBALANCE-LAST-EXTERNAL"
    df = pl.read_parquet("ETHBUSD-PERP.BINANCE-250-VALUE_IMBALANCE-LAST-EXTERNAL.parquet")
    if '__index_level_0__' in df.columns:
        df = df.drop(['__index_level_0__'])
        labeled_df = df[:10000]
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
    #print(len(X.columns))
    #print(X.to_pandas().values.shape)
    y = labeled_df.select(pl.col("label"))
    CPCV_train_gru(
        X=X, 
        y=y, 
        sample_weight = sample_weights.select("values"),
        num_groups = 4, 
        num_test_groups = 2, 
        bar_times = bar_times, 
        event_times = event_times,
        embargo_pct=0.01,
        max_num_epochs=10,
        num_samples=30,
        cpus_per_trial=1,
        gpus_per_trial=0,
        save_prefix=os.path.join("../models",bar_type),
    )