
import pandas as pd 
import numpy as np 
import torch 
import pickle 
from torch.utils.data import Dataset


class   SingleValueLabelDataset(Dataset):
    '''
     X:[batch_size,feature_dim]
     y:[batch_size]
    '''
    def __init__(
        self,
        X,
        y,
        x_handler = None,
        y_handler = None,
        is_classification = True,
        save_prefix = "scaler"
    ):
        super().__init__()
        self.is_classification = is_classification
        if isinstance(X,(pd.DataFrame,pd.Series)):
            X = X.values
        if y is not None and isinstance(y,(pd.DataFrame,pd.Series)):
            y = y.values
        if y is None:
            y_handler = None 

        self.X = X 
        self.y = y 
        self.x_handler = x_handler
        self.y_handler = y_handler
        self.transform()
    def transform(self):
        if self.x_handler:
            if not self.x_handler.is_fitted():
                self.X = self.x_handler.fit_transform(self.X)
                with open(self.save_prefix +"_x.pkl","wb") as f:
                    pickle.dump(self.x_handler, f)
            else:
                self.X = self.x_handler.transform(self.X)
        if self.y_handler:
            # reshape for scaler 
            self.y = self.y.reshape(-1,1)
            if not self.y_handler.is_fitted():
                self.y = self.y_handler.fit_transform(self.y)
                with open(self.save_prefix +"_y.pkl","wb") as f:
                    pickle.dump(self.y_handler, f)
            else:
                self.y = self.y_handler.transform(self.y)
            self.y = self.reshape(-1,)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        if self.y is not None:
            x = torch.as_tensor(self.X[idx],dtype=torch.float32)
            if self.is_classification:
                y = torch.as_tensor(self.y[idx],dtype=torch.LongTensor)
            else:
                y = torch.as_tensor(self.y[idx],dtype=torch.float32)
            return x, y
        else:
            x = torch.as_tensor(self.X[idx],dtype=torch.float32)
            return x

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
        y_handler = None,
        is_classification = True,
        save_prefix = "scaler"
    ):
        super().__init__()
        self.is_classification = is_classification 
        self.save_prefix = save_prefix 
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
                with open(self.save_prefix +"_x.pkl","wb") as f:
                    pickle.dump(self.x_handler, f)
            else:
                self.X = self.x_handler.transform(self.X)
        if self.y_handler:
            # reshape into (max_sequence_length,1)
            self.y = self.y.reshape(-1,1)
            if not self.y_handler.is_fitted():
                self.y = self.y_handler.fit_transform(self.y)
                with open(self.save_prefix +"_y.pkl","wb") as f:
                    pickle.dump(self.y_handler, f)
            else:
                self.y = self.y_handler.transform(self.y)
            self.y = self.reshape(-1,) 
        
    def __len__(self):
        return self.X.shape[0] - self.predict_len + 1
    
    def __getitem__(self, idx):
        if self.y is not None:
            x = torch.as_tensor(self.X[idx:idx+self.predict_len],dtype=torch.float32)
            if self.is_classification:
                y = torch.as_tensor(self.y[idx:idx+self.predict_len],dtype=torch.LongTensor)
            else:
                y = torch.as_tensor(self.y[idx:idx+self.predict_len],dtype=torch.float32)
            return x, y
        else:
            x = torch.as_tensor(self.X[idx:idx+self.predict_len],dtype=torch.float32)
            return x


class ReturnBasedDataset(Dataset):
    """
    A dataset designed for end-to-end optimization of indicators like the Sharpe ratio.

    Args:
        X (array-like or DataFrame): Input features.
        returns (array-like or DataFrame): Returns for asset allocation optimization.
        sequence_len (int): Length of sequences to be used.
        x_handler (object): An optional handler for preprocessing input features.
        save_prefix (str): Prefix for saving the preprocessing handler.

    Attributes:
        X (numpy.ndarray): Processed input features.
        returns (numpy.ndarray): Returns for asset allocation optimization.
        sequence_len (int): Length of sequences.
        x_handler (object): Handler for preprocessing input features.
        save_prefix (str): Prefix for saving the preprocessing handler.

    Note:
        If x_handler is provided, it will be used for preprocessing input features.
    """
    def __init__(
        self, 
        X, 
        returns, 
        sequence_len=30,
        x_handler=None,
        save_prefix="scaler"
    ):
        super().__init__()

        self.save_prefix = save_prefix
        
        # Convert DataFrame to numpy array if necessary
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if returns is not None and isinstance(returns, (pd.DataFrame, pd.Series)):
            returns = returns.values

        self.X = X
        self.returns = returns
        self.sequence_len = sequence_len
        self.x_handler = x_handler
        self.transform()

    def transform(self):
        """
        Apply preprocessing transformations to input features if a handler is provided.
        Save the handler if not already fitted.
        """
        if self.x_handler:
            if not self.x_handler.is_fitted():
                self.X = self.x_handler.fit_transform(self.X)
                # Save the handler for future use
                with open(self.save_prefix + "_x.pkl", "wb") as f:
                    pickle.dump(self.x_handler, f)
            else:
                self.X = self.x_handler.transform(self.X)

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.X.shape[0] - self.sequence_len + 1

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Tuple containing input sequence and returns for asset allocation optimization.
        """
        x = torch.as_tensor(self.X[idx:idx + self.sequence_len], dtype=torch.float32)
        #returns for asset allocation optimization
        returns = torch.as_tensor(self.returns[idx:idx + self.sequence_len], dtype=torch.float32)
        return x, returns
