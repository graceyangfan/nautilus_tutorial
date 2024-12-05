import polars as pl
import pandas as pd 
import numpy as np 
from finml.models.lightgbm.lightgbm_binary_classification import train_folds 
from finml.models.lightgbm.lightgbm_tools import define_args

if __name__ == "__main__":
    # Example Data Preparation
    np.random.seed(42)
    train_data = pd.DataFrame(np.random.rand(1000, 10), columns=[f"feature_{i}" for i in range(10)])
    test_data = pd.DataFrame(np.random.rand(200, 10), columns=[f"feature_{i}" for i in range(10)])
    train_label = pd.DataFrame({"label": np.random.randint(0, 2, size=1000)})
    test_label = pd.DataFrame({"label": np.random.randint(0, 2, size=200)})
    event_times = pl.DataFrame({"event_starts":range(1000),"event_ends":range(20,1020,1)})
    args = define_args()
    models,score = train_folds(
        pl.from_pandas(train_data),
        pl.from_pandas(train_label),
        event_times,
        args,
        True,
        num_class = 2 ,
        dual_iterations = 2 
    )