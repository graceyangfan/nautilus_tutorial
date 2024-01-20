
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
    PyTorch Dataset for handling datasets with input features and returns labels.

    Args:
        X (list or array-like): Input features for the dataset.
        returns_label (list or array-like): Returns labels associated with each set of input features.
        x_handler (object, optional): An optional handler for preprocessing input features (default is None).
        is_classification (bool, optional): Indicates whether the task is a classification task (default is False).
        save_prefix (str, optional): Prefix for saving the preprocessing handler (default is "scaler").

    Attributes:
        save_prefix (str): Prefix for saving the preprocessing handler.
        X (list or array-like): Input features for the dataset.
        returns_label (list or array-like): Returns labels associated with each set of input features.
        x_handler (object): Handler for preprocessing input features.
        is_classification (bool): Indicates whether the task is a classification task.

    Note:
        If x_handler is provided, it will be used for preprocessing input features.
        X should shape like [Sequence_length,feature_dim].
    """
    def __init__(
        self,
        X,
        returns_label,
        x_handler=None,
        is_classification=False,
        save_prefix="scaler"
    ):
        super().__init__()

        self.save_prefix = save_prefix

        # Ensure the lengths of X and returns_label are the same
        assert len(X) == len(returns_label), "Lengths of X and returns_label must be the same."

        self.X = X
        self.returns_label = returns_label
        self.x_handler = x_handler
        self.is_classification = is_classification  # Added missing is_classification parameter
        self.fit()

    def fit(self):
        """
        Fit the input handler if provided and not already fitted.
        Save the handler for future use.
        """
        if self.x_handler:
            if not self.x_handler.is_fitted():
                self.x_handler.fit(np.vstack(self.X))
                # Save the handler for future use
                with open(self.save_prefix + "_x.pkl", "wb") as f:
                    pickle.dump(self.x_handler, f)

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Tuple containing input sequence and returns_label.
        """
        x = torch.as_tensor(self.x_handler.transform(self.X[idx]), dtype=torch.float32)
        # returns_label for asset allocation optimization
        if self.is_classification:
            return_label = torch.as_tensor(self.returns_label[idx],  dtype=torch.long)
        else:
            return_label = torch.as_tensor(self.returns_label[idx], dtype=torch.float32)

        return x,return_label