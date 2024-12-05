from typing import List, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

def dual_label_one(
    train_data_in: pd.DataFrame,
    train_label_in: pd.DataFrame,
    test_data_in: pd.DataFrame,
    test_label_in: pd.DataFrame,
    initial_model: lgb.Booster,
    params: dict,
    dual_iterations: int,
    dual_ratio: float,
    num_boost_round: int,
    early_stopping_rounds: int = 100,
    verbose_eval: int = 100,
) -> Tuple[List[lgb.Booster], float]:
    """
    Perform dual label training for boosting models with iterative label updates and majority voting.

    Args:
        train_data_in: Training data (features).
        train_label_in: Training labels (DataFrame, should contain a "label" column).
        test_data_in: Test/validation data (features).
        test_label_in: Test/validation labels (DataFrame, should contain a "label" column).
        initial_model: Initial trained LightGBM Booster model.
        params: A dictionary of LightGBM parameters.
        dual_iterations: Number of iterations for dual label updates.
        dual_ratio: Ratio for splitting dual labels.
        num_boost_round: Number of boosting rounds for training.
        early_stopping_rounds: Number of early stopping rounds (default: 100).
        verbose_eval: Interval to log evaluation metrics (default: 100).

    Returns:
        Tuple containing:
            - A list of LightGBM boosters from each iteration.
            - The AUC score on the test/validation data.
    """
    # Copy inputs to avoid modifying original data
    train_data_new = train_data_in.copy()
    train_label_new = train_label_in.copy()
    models = []

    for iteration in range(dual_iterations):  # Perform dual training for 'dual_iterations' iterations
        # Predict probabilities for the current training data
        train_label_new['proba'] = initial_model.predict(train_data_new)

        # Initialize 'label1' and 'label2' columns
        train_label_new['label1'] = 0
        train_label_new['label2'] = 0

        # Define the splitting logic for dual labels
        sorted_indices = train_label_new['proba'].sort_values(ascending=False).index
        num1 = int(dual_ratio* len(sorted_indices))
        idx1 = sorted_indices[:num1]
        idx2 = sorted_indices[num1:]

        # Update temporary labels 'label1' and 'label2'
        train_label_new.loc[idx1, 'label1'] = 1
        train_label_new.loc[idx2, 'label2'] = 1

        # Update the main label column using dual logic
        train_label_new['label'] = (
            train_label_new['label'] & train_label_new['label1']) | (
            (1 - train_label_new['label']) & train_label_new['label2'])

        # Prepare LightGBM datasets
        lgb_train = lgb.Dataset(train_data_new, train_label_new["label"])
        lgb_valid = lgb.Dataset(test_data_in, test_label_in['label'], reference=lgb_train)

        # Train LightGBM model
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_valid],
            valid_names=["valid"],
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(verbose_eval)
            ]
        )

        # Append the trained model to the list and update the initial model
        models.append(model)
        initial_model = model

    # Convert predictions to int and apply majority voting
    predictions = (np.array([model.predict(test_data_in) for model in models]) > 0.5).astype(int)
    majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

    # Compute the AUC score
    auc_score = roc_auc_score(test_label_in['label'], majority_vote)

    return models, auc_score
