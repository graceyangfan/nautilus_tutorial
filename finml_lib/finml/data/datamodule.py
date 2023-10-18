import pandas as pd
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from finml.data.dataset import SingleValueLabelDataset


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
            self.valid_datset
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

