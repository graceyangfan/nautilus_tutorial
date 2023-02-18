
import pandas as pd 
import numpy as np 
import torch 
from torch.utils.data import Dataset


class SequenceLabelDataset(Dataset):
    '''
    X : [max_sequence_length,feature_dim]
    y: (max_sequence_length,) => (max_sequence_length,1)
    '''
    def __init__(
        self, 
        X, 
        y, 
        predict_len=30,
        x_handler = None,
        y_handler = None 
    ):
        if isinstance(X,(pd.DataFrame,pd.Series)):
            X = X.values
        if y is not None and isinstance(y,(pd.DataFrame,pd.Series)):
            y = y.values
        if y is None:
            y_handler = None 

        self.X = X 
        self.y = y 
        self.predict_len = predict_len 
        self.x_handler = x_handler
        self.y_handler = y_handler
        self.transform()

    def transform(self):
        if self.x_handler:
            if not self.x_handler.is_fitted():
                self.X = self.x_handler.fit_transform(self.X)
            else:
                self.X = self.x_handler.transform(self.X)
        if self.y_handler:
            # reshape into (max_sequence_length,1)
            self.y = self.y.reshape(-1,1)
            if not self.y_handler.is_fitted():
                self.y = self.y_handler.fit_transform(self.y)
            else:
                self.y = self.y_handler.transform(self.y)
        
    def __len__(self):
        return self.X.shape[0] - self.predict_len + 1
    
    def __getitem__(self, idx):
        if self.y is not None:
            x = torch.as_tensor(self.X[idx:idx+self.predict_len],dtype=torch.float32)

            y = torch.as_tensor(self.y[idx:idx+self.predict_len],dtype=torch.float32).squeeze()
            return x, y
        else:
            x = torch.as_tensor(self.X[idx:idx+self.predict_len],dtype=torch.float32)
            return x