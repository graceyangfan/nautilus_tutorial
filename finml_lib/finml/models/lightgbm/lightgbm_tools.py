import os
import numpy as np
import lightgbm as lgb
from types import SimpleNamespace
from typing import List, Union

def define_args(**kwargs):
    """
    Define the arguments for the training process.

    Args:
        **kwargs: Additional keyword arguments for customizing the default parameters.

    Returns:
        SimpleNamespace: A namespace containing model configuration parameters.
    """
    args = SimpleNamespace(
        # Cross-validation settings
        n_splits=3,                # Number of splits for purged cross-validation
        embargo_pct=0.01,           # Percentage of embargo for purged cross-validation
        n_trials = 10,
        time_budget_minutes = 10,
        num_boost_round = 5000,
        early_stopping_rounds = 100,
        cpus_per_trial = 1,
        gpu_per_trail = 0,
        verbose_eval=100
    )

    # Update default arguments with user-defined values
    args.__dict__.update(kwargs)
    return args


def save_lightgbm_models(
    models: List[lgb.Booster],
    data_type: str,
    directory: str = "models"
) -> None:
    """
    Save LightGBM models to the specified directory.

    Args:
        models: List of LightGBM models to save.
        data_type: Identifier for the data type.
        directory: Directory to save the models.
    """
    for index, model in enumerate(models):
        file_path = os.path.join(directory, f"{data_type}_{index}.txt")
        model.save_model(file_path)

def load_lightgbm_models(
    data_type: str,
    num_models: int = 5,
    directory: str = "models",
) -> List[lgb.Booster]:
    """
    Load LightGBM models from the specified directory.

    Args:
        data_type: Identifier for the data type.
        num_models: Number of models to load.
        directory: Directory containing the models.

    Returns:
        List of loaded LightGBM models.
    """
    models = []
    for index in range(num_models):
        file_path = os.path.join(directory, f"{data_type}_{index}.txt")
        model = lgb.Booster(model_file=file_path)
        models.append(model)
    return models

def predict_average_probability(
    models: List[lgb.Booster],
    features: np.ndarray
) -> np.ndarray:
    """
    Predict average probability using ensemble of LightGBM models.

    Args:
        models: List of LightGBM models.
        features: Input features.

    Returns:
        Average predicted probability.
    """
    features = features.reshape((1, -1))
    predictions = [model.predict(features) for model in models]
    return np.mean(predictions, axis=0)