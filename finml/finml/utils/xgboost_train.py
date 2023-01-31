import gc 
import numpy as np
import polars as pl 
from xgboost_ray import RayParams
from xgboost_ray.sklearn import RayXGBClassifier
import optuna 
from finml.backtest.backtest import CombinatorialPurgedCV

def train_classifier(
    params, 
    X_train,
    y_train,
    X_test,
    y_test,
    num_actors,
    cpus_per_actor,
    verbose = True,
    eval_metric= "auc",
    sample_weight = None,
    sample_weight_eval_set = None,
    early_stopping_rounds=None, 
    callbacks=None
):

        model = RayXGBClassifier(**params)

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            sample_weight_eval_set= sample_weight_eval_set,
            eval_set=[(X_test, y_test)],
            eval_metric = eval_metric,
            early_stopping_rounds = early_stopping_rounds,
            callbacks=callbacks,
            ray_params=RayParams(
                num_actors=num_actors,
                cpus_per_actor=cpus_per_actor,  # Divide evenly across actors per machine
            ),
            verbose = verbose,
        )

        return model

def hyper_opt_classifier(
    params, 
    X_train,
    y_train,
    X_test,
    y_test,
    num_actors,
    cpus_per_actor,
    num_class,
    object_func = None,
    eval_metric= "auc",
    direction='maximize',
    sample_weight = None,
    sample_weight_eval_set = None,
    early_stopping_rounds=20, 
    callbacks=None,
    verbose_eval = 100,
    max_minutes=10, 
    n_trials = None, 
    study_name = "XGBoostLSS-HyperOpt", 
    silence=False
):
    if object_func is None:
        object_func = "multi:softmax" if num_class >2 else "binary:logistic"
    def objective(trial):
        hyper_params = {
            "booster": "gbtree",
            "num_class": num_class,
            "objective": object_func,
            "eta": trial.suggest_float("eta", params["eta"][0], params["eta"][1],log = True),
            "max_depth": trial.suggest_int("max_depth", params["max_depth"][0], params["max_depth"][1]),
            "gamma": trial.suggest_float("gamma", params["gamma"][0], params["gamma"][1],log = True),
            "subsample": trial.suggest_float("subsample", params["subsample"][0], params["subsample"][1],log = True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", params["colsample_bytree"][0], params["colsample_bytree"][1],log = True),
            "min_child_weight": trial.suggest_int("min_child_weight", params["min_child_weight"][0], params["min_child_weight"][1])
        }
        # Add pruning
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-"+eval_metric)

        xgb_param_tuning = train_classifier(
                params = hyper_params,
                X_train = X_train,
                y_train = y_train,
                X_test = X_test,
                y_test = y_test,
                num_actors=num_actors,
                cpus_per_actor=cpus_per_actor,
                sample_weight=sample_weight,
                sample_weight_eval_set= sample_weight_eval_set,
                early_stopping_rounds=early_stopping_rounds,
                eval_metric= eval_metric,
                callbacks=[pruning_callback],
                verbose=verbose_eval,
            )

        if direction == 'maximize':
            best_score = np.max(xgb_param_tuning.evals_result()["validation_0"][eval_metric])
        else:
            best_score = np.min(xgb_param_tuning.evals_result()["validation_0"][eval_metric])

        return best_score


    if silence:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
    study = optuna.create_study(pruner=pruner, direction=direction, study_name=study_name)
    study.optimize(objective, n_trials=n_trials, timeout=60 * max_minutes, show_progress_bar=True)

    print("Hyper-Parameter Optimization successfully finished.")
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    opt_param = study.best_trial

    print("  Value: {}".format(opt_param.value))
    print("  Params: ")
    for key, value in opt_param.params.items():
        print("    {}: {}".format(key, value))

    return opt_param.params,study.best_trial.value


def fit_xgboost(
    X_train,
    y_train,
    X_test,
    y_test,
    num_class,
    num_actors,
    cpus_per_actor,
    early_stopping_rounds = 20,
    eval_metric= "auc",
    direction='maximize',
):
    np.random.seed(123)

    # Specifies the parameters and their value range. The structure is as follows: "hyper-parameter": [lower_bound, upper_bound]. Currently, only the following hyper-parameters can be optimized:
    params = {"eta": [1e-5, 1],                   
            "max_depth": [1, 5],
            "gamma": [1e-8, 40],
            "subsample": [0.2, 1.0],
            "colsample_bytree": [0.2, 1.0],
            "min_child_weight": [0, 500]
    }
    opt_params,best_score = hyper_opt_classifier(
        params=params,
        X_train = X_train,
        y_train = y_train,
        X_test = X_test,
        y_test = y_test,
        num_class=num_class,
        num_actors=num_actors,
        cpus_per_actor=cpus_per_actor,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric= eval_metric,
        direction=direction,
        verbose_eval = 100,
        max_minutes=120,           # Time budget in minutes, i.e., stop study after the given number of minutes.
        n_trials=30,             # The number of trials. If this argument is set to None, there is no limitation on the number of trials.
        silence=False             # Controls the verbosity of the trail, i.e., user can silence the outputs of the trail.
    )
    np.random.seed(123)



    # Train Model with optimized hyper-parameters
    xgboostlss_model = train_classifier(
        params=opt_params,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_actors=num_actors,
        cpus_per_actor=cpus_per_actor,
        verbose = 100,
        eval_metric = eval_metric,
        early_stopping_rounds=20,
    )

    return xgboostlss_model,best_score 


def CV_train_classifier(
    X, 
    y, 
    num_groups, 
    num_test_groups, 
    bar_times, 
    event_times, 
    num_class,
    num_actors = 10,
    cpus_per_actor = 1,
    embargo_pct=0.01,
    return_info = False,
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
    for train_indices, test_indices in cv.split():
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        X_train = X_train.to_pandas() 
        X_test = X_test.to_pandas()
        y_train = y_train.to_pandas()
        y_test = y_test.to_pandas()
        if len(np.unique(y_train)) < num_class:
            continue
        model, score = fit_xgboost(
            X_train = X_train,
            y_train = y_train,
            X_test = X_test,
            y_test = y_test,
            num_class = num_class,
            num_actors=num_actors,
            cpus_per_actor=cpus_per_actor,
        )
        models.append(model)
        gc.collect()
        valid_score += score 
    valid_score /= len(models)
    print(f'The valid_score for the models are {valid_score}')
    if  return_info:
        return models,valid_score
    else:
        return valid_score
