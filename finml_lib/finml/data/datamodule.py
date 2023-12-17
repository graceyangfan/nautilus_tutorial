import pandas as pd
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from finml.data.dataset import SingleValueLabelDataset,ReturnBasedDataset


class SingleValueLabelDataModule(LightningDataModule):
    def __init__(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        x_handler = None,
        y_handler = None,
        is_classification = True,
        batch_size = 64,
        num_workers = 4
    ):
        super().__init__()
        self.batch_size = batch_size 
        self.num_workers = num_workers 
        self.train_dataset = SingleValueLabelDataset(
            x_train,
            y_train,
            x_handler,
            y_handler,
            is_classification
        )
        self.train_dataset.transform()
        self.valid_datset = SingleValueLabelDataset(
            x_test,
            y_test,
            x_handler,
            y_handler,
            is_classification
        )
        self.valid_datset.transform() 

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_datset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

class ReturnBasedDataModule(LightningDataModule):
    """
    Lightning DataModule for handling datasets with input features and returns.

    Args:
        x_train (array-like or DataFrame): Training input features.
        returns_train (array-like or DataFrame): Training returns for asset allocation optimization.
        x_test (array-like or DataFrame): Validation input features.
        returns_test (array-like or DataFrame): Validation returns for asset allocation optimization.
        sequence_len (int): Length of sequences to be used.
        x_handler (object): An optional handler for preprocessing input features.
        save_prefix (str): Prefix for saving the preprocessing handler.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.

    Attributes:
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.
        train_dataset (ReturnBasedDataset): Training dataset instance.
        valid_dataset (ReturnBasedDataset): Validation dataset instance.

    Note:
        If x_handler is provided, it will be used for preprocessing input features.
    """
    def __init__(
        self,
        x_train,
        returns_train,
        x_test,
        returns_test,
        sequence_len=30,
        x_handler=None,
        save_prefix="scaler",
        batch_size=64,
        num_workers=4
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Create and transform training dataset
        self.train_dataset = ReturnBasedDataset(
            X=x_train,
            returns=returns_train,
            sequence_len=sequence_len,
            x_handler=x_handler,
            save_prefix=save_prefix
        )

        # Create and transform validation dataset
        self.valid_dataset = ReturnBasedDataset(
            X=x_test,
            returns=returns_test,
            sequence_len=sequence_len,
            x_handler=x_handler,
            save_prefix=save_prefix
        )

    def train_dataloader(self):
        """
        Create DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Do not shuffle time series data
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        """
        Create DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Do not shuffle time series data
            num_workers=self.num_workers
        )
