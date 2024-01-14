import os
import pickle
from types import SimpleNamespace

import torch 
import numpy as np
import pandas as pd
import polars as pl

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from finml.models.WaveNet.models import WaveNet
from finml.data.datamodule import ReturnBasedDataModule
from finml.evaluation.cross_validation import PurgedKFold
from finml.data.handler import StandardNorm


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
        n_splits=5,                # Number of splits for purged cross-validation
        embargo_pct=0.01,           # Percentage of embargo for purged cross-validation

        # Training control
        patience=3,                # Patience for early stopping
        max_epochs=20,             # Maximum number of epochs for training

        # Input sequence parameters
        sequence_len=30,           # Length of input sequences

        # Data preprocessing
        x_handler=StandardNorm(),  # Handler for preprocessing input features

        # Model checkpoint and saving
        save_path='model_checkpoints/',       # Path to save PyTorch Lightning model checkpoints
        save_prefix='model_checkpoints/scaler', # Prefix for saving preprocessing handler

        # Data loading settings
        batch_size=120,           # Batch size for training
        num_workers=4,            # Number of workers for data loading

        # Model params 
        input_dim = 16,           # Dimensionality of input features
        output_dim = 1,           # Desired output dimensionality of the model
        residual_dim = 32,        # Dimensionality of the residual blocks in the model
        skip_dim = 32,            # Dimensionality of the skip connections
        dilation_cycles = 1,      # Number of dilation cycles in the model
        dilation_depth = 4,       # Depth of dilation in each residual block

        # Optimizer settings
        learning_rate=1e-4                   # Learning rate for the optimizer
    )

    # Update default arguments with user-defined values
    args.__dict__.update(kwargs)

    return args



def train_folds(X, returns, event_times, args):
    """
    Train a WaveNet model using purged cross-validation with early stopping.

    Args:
        X (pl.DataFrame): Input features as a Polars DataFrame.
        returns (pl.DataFrame): Returns for asset allocation optimization as a Polars DataFrame.
        event_times (pl.DataFrame): Event times for purged cross-validation as a Polars DataFrame.
        args (SimpleNamespace): A SimpleNamespace containing model configuration parameters.

    Returns:
        None
    """
    # Convert DataFrames to pandas if needed
    if isinstance(X, pl.DataFrame):
        X = X.to_pandas()
    if isinstance(returns, pl.DataFrame):
        returns = returns.to_pandas()
    if not isinstance(event_times, pl.DataFrame):
        event_times = pl.from_pandas(event_times)

    seed_everything(42) 

    # Initialize PurgedKFold with specified parameters
    purged_kfold = PurgedKFold(
        n_splits=args.n_splits,
        embargo_pct=args.embargo_pct
    )

    # List to store validation losses for each fold
    val_losses = []

    # Iterate through folds
    for fold, (train_indices, val_indices) in enumerate(purged_kfold.split(event_times)):
        print(f"Fold {fold + 1}")

        # Create a new model instance for each fold
        fold_model = WaveNet(args)

        # Set up early stopping callback
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            verbose=True,
            mode='min'
        )

        # Set up model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath=args.save_path,
            filename=f'model_fold_{fold + 1}'  # You can customize the filename pattern
        )

        # Set up trainer with callbacks
        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator="auto", 
            devices="auto", 
            strategy="auto",
            callbacks=[early_stop_callback, checkpoint_callback]
        )

        # Use loc to obtain training and validation sets
        x_train, x_test = X.loc[train_indices], X.loc[val_indices]
        returns_train, returns_test = returns.loc[train_indices], returns.loc[val_indices]

        # Create and transform training dataset
        data_module = ReturnBasedDataModule(
            x_train=x_train,
            returns_train=returns_train,
            x_test=x_test,
            returns_test=returns_test,
            sequence_len=args.sequence_len,
            x_handler=args.x_handler,
            save_prefix=args.save_prefix,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # Train the model on the current fold
        trainer.fit(fold_model, data_module)

        best_score_tensor = early_stop_callback.best_score
        # Move the tensor to the CPU
        best_score_cpu = best_score_tensor.cpu()
        # Convert the CPU tensor to a NumPy array
        best_score_numpy = best_score_cpu.numpy()
        val_losses.append(best_score_numpy)

    mean_val_loss = np.mean(val_losses)
    print(f'The average val_loss on {args.n_splits} models is {mean_val_loss}')



def load_models(model_paths, args):
    """
    Load multiple trained models using PyTorch Lightning style.

    Args:
        model_paths (list): List of paths to the saved PyTorch Lightning models.
        args (SimpleNamespace): A SimpleNamespace containing model configuration parameters.

    Returns:
        list: List of loaded models.
    """
    loaded_models = []

    for model_path in model_paths:
        # Extract the module name from the checkpoint file
        module_name = os.path.splitext(os.path.basename(model_path))[0]

        # Load the model using PyTorch Lightning's load_from_checkpoint
        model = WaveNet.load_from_checkpoint(model_path, args=args)
        
        # Optionally, you can print the loaded module name
        print(f"Loaded {module_name} from {model_path}")

        loaded_models.append(model)

    return loaded_models


def load_x_handler(filename_prefix):
    """
    Load the x_handler from a saved file.

    Args:
        filename_prefix (str): Prefix for the filename where x_handler is saved.

    Returns:
        object: Loaded x_handler.
    """
    x_handler_path = filename_prefix + "_x.pkl"
    if os.path.exists(x_handler_path):
        x_handler = pickle.load(open(x_handler_path, "rb"))
        x_handler.fitted = True
        return x_handler
    else:
        raise FileNotFoundError(f"x_handler file not found: {x_handler_path}")

def predict_ensemble(models, input_data, x_handler, args):
    """
    Make predictions on a single input using an ensemble of trained WaveNet models.

    Args:
        models (list): List of trained WaveNet models.
        input_data (numpy.ndarray): Single input data as a NumPy array.
        x_handler (object): Loaded x_handler.
        args (SimpleNamespace): A SimpleNamespace containing model configuration parameters.

    Returns:
        numpy.ndarray: Ensemble predictions for the single input.
    """
    # Transform input data using the loaded x_handler
    if x_handler is not None:
        input_data = x_handler.transform(input_data)

    # Convert input_data to torch tensor and move to CPU
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to("cpu")

    # Set models in evaluation mode and move to CPU
    for model in models:
        model.eval()
        model.to("cpu")

    # Make predictions with each model
    ensemble_predictions = []
    with torch.no_grad():
        for model in models:
            predictions = model(input_tensor).numpy()
            ensemble_predictions.append(predictions)

    # Calculate the mean of predictions as the ensemble result
    ensemble_result = np.mean(ensemble_predictions, axis=0)

    return ensemble_result
