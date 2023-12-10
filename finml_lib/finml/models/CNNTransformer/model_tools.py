from finml.models.CNNTransformer.models import CNNTransformer
from finml.data.datamodule import ReturnBasedDataModule 
from finml.evaluation.cross_validation import PurgedKFold 
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything, Trainer
from finml.data.handler import StandardNorm
import numpy as np
import pandas as pd
import polars as pl 
import os 
import pickle 

def define_args():
    """
    Define the arguments for the training process.

    Returns:
        Namespace: A namespace containing model configuration parameters.
    """
    args = Namespace(
        n_splits=5,  # Example value, replace with your actual value
        embargo_pct=0.1,  # Example value, replace with your actual value
        patience=3,  # Example value, replace with your actual value
        max_epochs=10,  # Example value, replace with your actual value
        gpus=1,  # Example value, replace with your actual value
        tpus=0,  # Example value, replace with your actual value
        sequence_len=30,  # Example value, replace with your actual value
        x_handler=StandardNorm(),  # Example value, replace with your actual handler
        save_prefix='model_checkpoints/',  # Example value, replace with your actual value
        batch_size=64,  # Example value, replace with your actual value
        num_workers=4  # Example value, replace with your actual value
    )
    return args

def train_folds(X, returns, event_times, args):
    """
    Train a CNNTransformer model using purged cross-validation with early stopping.

    Args:
        X (pl.DataFrame): Input features as a Polars DataFrame.
        returns (pl.DataFrame): Returns for asset allocation optimization as a Polars DataFrame.
        event_times (pl.DataFrame): Event times for purged cross-validation as a Polars DataFrame.
        args (Namespace): A namespace containing model configuration parameters.

    Returns:
        None
    """
    # Convert DataFrames to pandas if needed
    if isinstance(X, pl.DataFrame):
        X = X.to_pandas()
    if isinstance(returns, pl.DataFrame):
        returns = returns.to_pandas()
    if not isinstance(event_times, pd.DataFrame):
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
        fold_model = CNNTransformer(args)

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
            dirpath=args.save_prefix,
            filename=f'model_fold_{fold + 1}'  # You can customize the filename pattern
        )

        # Set up trainer with callbacks
        trainer = Trainer(
            max_epochs=args.max_epochs,
            gpus=args.gpus,
            tpus=args.tpus,
            callbacks=[early_stop_callback, checkpoint_callback],
            progress_bar_refresh_rate=0
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

        # Append the validation loss for this fold to the list
        val_losses.append(early_stop_callback.best_score)

    mean_val_loss = np.mean(val_losses)
    print(f'The average val_loss on {args.n_splits} models is {mean_val_loss}')


def load_model(model_path, args):
    """
    Load a trained CNNTransformer model.

    Args:
        model_path (str): Path to the saved PyTorch Lightning model.
        args (Namespace): A namespace containing model configuration parameters.

    Returns:
        CNNTransformer: Loaded CNNTransformer model.
    """
    # Create a new model instance
    model = CNNTransformer(args)

    # Load the trained weights
    model.load_state_dict(torch.load(model_path))

    # Move the model to the CPU
    model.to("cpu")

    return model

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

def predict_single_input(model, input_data, x_handler, args):
    """
    Make predictions on a single input using a trained CNNTransformer model.

    Args:
        model (CNNTransformer): Trained CNNTransformer model.
        input_data (numpy.ndarray): Single input data as a NumPy array.
        x_handler (object): Loaded x_handler.
        args (Namespace): A namespace containing model configuration parameters.

    Returns:
        numpy.ndarray: Model predictions for the single input.
    """
    # Transform input data using the loaded x_handler
    if x_handler is not None:
        if not x_handler.fitted:
            input_data = x_handler.fit_transform(input_data)
        else:
            input_data = x_handler.transform(input_data)

    # Convert input_data to torch tensor and move to CPU
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to("cpu")

    # Set the model in evaluation mode and move to CPU
    model.eval()
    model.to("cpu")

    # Make predictions on the single input
    with torch.no_grad():
        predictions = model(input_tensor).numpy()

    return predictions

