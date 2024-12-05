import gc 
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl 
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from typing import List, Tuple, Any 
from finml.evaluation.cross_validation import PurgedKFold
from finml.models.lightgbm.dual_train import dual_label_one
from sklearn.metrics import roc_auc_score


def lgb_hyperopt_with_ray(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dual_numbers: int = 0,
    n_trials: int = 50,
    time_budget_minutes: int = 10,
    num_boost_round: int = 5000,
    early_stopping_rounds: int = 100,
    cpus_per_trial: int = 1,
    verbose_eval: int = 100,
):
    """
    Run distributed hyperparameter tuning for LightGBM using Ray Tune and Optuna, and return the best model and params.

    Returns:
        best_params: Best hyperparameters found during optimization.
        best_model: Best LightGBM model trained with the best parameters.
    """

    def objective_ray(config):
        """
        Objective function to run LightGBM training for each configuration sampled by Ray Tune.
        """
        # Define LightGBM parameters
        params = {
            'objective': 'binary',        # Binary classification
            'metric': 'auc',              # Evaluation metric: AUC
            'verbosity': -1,              # Suppress logs
            'boosting_type': config['boosting_type'],
            'learning_rate': config['learning_rate'],
            'num_leaves': int(config['num_leaves']),
            'max_depth': int(config['max_depth']),
            'feature_fraction': config['feature_fraction'],
            'bagging_fraction': config['bagging_fraction'],
            'bagging_freq': int(config['bagging_freq']),
            'lambda_l1': config['lambda_l1'],
            'lambda_l2': config['lambda_l2'],
            'min_gain_to_split': config['min_gain_to_split'],
        }

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # Add callbacks for early stopping and logging
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(verbose_eval),
        ]

        # Train the model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            valid_names=["valid"],   # Ensure validation set is named "valid"
            num_boost_round=num_boost_round,
            callbacks=callbacks
        )

        if dual_numbers > 0:
            _, auc = dual_label_one(
                train_data_in = X_train,
                train_label_in = y_train,
                test_data_in = X_test,
                test_label_in = y_test,
                initial_model = model,
                params = params,
                dual_numbers = dual_numbers,
                num_boost_round = num_boost_round,
                early_stopping_rounds = early_stopping_rounds,
                verbose_eval = verbose_eval
            )
            return {"auc": auc} 
        else:
            # Retrieve the best AUC score
            try:
                auc = model.best_score["valid"]["auc"]  # Extract AUC score under "valid" set
                return {"auc": auc}
            except KeyError:  # Catch if "auc" is not found
                print(f"Error: 'auc' was not found in best_score keys: {model.best_score.keys()}.")
                return {"auc": 0.0}

    # Define hyperparameter search space
    search_space = {
        'boosting_type': tune.choice(['gbdt']),
        'learning_rate': tune.loguniform(0.01, 0.3),
        'num_leaves': tune.randint(31, 512),
        'max_depth': tune.randint(3, 15),
        'feature_fraction': tune.uniform(0.4, 1.0),
        'bagging_fraction': tune.uniform(0.4, 1.0),
        'bagging_freq': tune.randint(1, 10),
        'lambda_l1': tune.loguniform(1e-5, 10.0),
        'lambda_l2': tune.loguniform(1e-5, 10.0),
        'min_gain_to_split': tune.uniform(0.0, 20.0)
    }

    # OptunaSearch for hyperparameter optimization
    optuna_search = OptunaSearch(
        metric="auc",  # Optimize validation AUC score
        mode="max"     # Maximize the AUC
    )

    # Run hyperparameter search using Ray Tune
    analysis = tune.run(
        objective_ray,
        config=search_space,
        search_alg=optuna_search,
        num_samples=n_trials,
        resources_per_trial={"cpu": cpus_per_trial,"gpu":0},
        time_budget_s=time_budget_minutes * 60  # Convert minutes to seconds
    )

    # Retrieve best trial's parameters and results
    best_params = analysis.get_best_config(metric="auc", mode="max")
    best_auc = analysis.get_best_trial(metric="auc", mode="max").last_result['auc']

    print(f"Best AUC: {best_auc:.4f}")
    print(f"Best parameters: {best_params}")


    # Train the final model with the best parameters
    final_params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        **best_params  # Add best hyperparameters
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds),lgb.log_evaluation(verbose_eval)]
    best_model = lgb.train(
        final_params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=num_boost_round,
        callbacks=callbacks
    )
    if dual_numbers > 0:
        print("trying to dual tring with best_params!")
        best_models, best_auc = dual_label_one(
            train_data_in = X_train,
            train_label_in = y_train,
            test_data_in = X_test,
            test_label_in = y_test,
            initial_model = best_model,
            params = final_params,
            dual_numbers = dual_numbers,
            num_boost_round = num_boost_round,
            early_stopping_rounds = early_stopping_rounds,
            verbose_eval = verbose_eval
        )
        return best_params, best_models, best_auc 
    else:
        return best_params, best_model, best_auc 


def train_folds(
    X: pl.DataFrame,
    y: pl.DataFrame,
    event_times: pl.DataFrame,
    args: Any,
    return_model: bool = True,
    num_class: int = None,
    dual_iterations: int = 5,
):
    """
    Train a LightGBM model using purged cross-validation with early stopping.

    Args:
        X (numpy.ndarray): Input features with shape [batch_size, feature_dim].
        y (numpy.ndarray): Returns or classification labels.
        event_times (pl.DataFrame): Event times for purged cross-validation as a Polars DataFrame.
        args (SimpleNamespace): A SimpleNamespace containing model configuration parameters.
    """
    # Ensure event_times is a Polars DataFrame
    if not isinstance(event_times, pl.DataFrame):
        event_times = pl.from_pandas(event_times)

    # Calculate num_class if not provided
    if num_class is None:
        num_class = len(y.unique())


    # Initialize PurgedKFold with specified parameters
    purged_kfold = PurgedKFold(
        n_splits=args.n_splits,
        embargo_pct=args.embargo_pct
    )
    models = [] 
    valid_score = 0 
    for train_indices, test_indices in purged_kfold.split(event_times):
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        X_train = X_train.to_pandas()
        X_test = X_test.to_pandas()
        y_train = y_train.to_pandas()
        y_test = y_test.to_pandas()

        # Train the model
        best_params, best_model,best_score = lgb_hyperopt_with_ray(
            X_train,
            y_train,
            X_test,
            y_test,
            dual_numbers = dual_iterations,
            n_trials=args.n_trials,
            time_budget_minutes=args.time_budget_minutes,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
            cpus_per_trial=args.cpus_per_trial,
            verbose_eval=args.verbose_eval
        )
        if dual_iterations > 0:
            print(f"the length of best_model is {len(best_model)}")
            models.extend(best_model)
        else:
            models.append(best_model)
        gc.collect()
        valid_score += best_score 

    valid_score /= len(models)
    print(f'The valid_score for the models are {valid_score}')
    if return_model:
        return models, valid_score
    else:
        return valid_score 
