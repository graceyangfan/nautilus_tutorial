import numpy as np
import pandas as pd
import copy
import pickle 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from finml.models.base import Model 
from finml.data.dataset import SequenceLabelDataset 


class GRUModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        '''
        X = [batch_size,sequence_length,feature_dim]
        out = [batch_size,sequence_length,hidden_size*D] D=2 if birdirection 
        return:
            [batch_size,sequence_length]
        '''
        out, _ = self.rnn(x)
        return self.fc_out(out).squeeze()


class GRU(Model):
    """GRU Model
    Parameters
    ----------
    d_feat : int
        input dimension for each time step
    metric: str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : str
        the GPU ID(s) used for training
    """

    def __init__(
        self,
        d_feat=6,
        predict_len=10,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        batch_size=2000,
        early_stop=20,
        optimizer="adam",
        n_jobs=10,
        GPU=0,
        seed=42,
        x_handler = None,
        y_handler = None,
        metric_fn = None,
        loss_fn = None,
        **kwargs
    ):
        # set hyper-parameters.
        self.d_feat = d_feat
        self.predict_len = predict_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.n_jobs = n_jobs
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.x_handler = x_handler 
        self.y_handler = y_handler 
        self.loss_fn = loss_fn 
        self.metric_fn = metric_fn


        self.GRU_model = GRUModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.GRU_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.GRU_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.GRU_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")


    def train_epoch(self, data_loader):

        self.GRU_model.train()

        for (feature, label) in data_loader:
            feature = feature.to(self.device)
            label = label.to(self.device)

            pred = self.GRU_model(feature.float())
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.GRU_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):

        self.GRU_model.eval()

        scores = []
        losses = []

        for (feature, label) in data_loader:
            feature = feature.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                pred = self.GRU_model(feature.float())
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        tune = None,
    ):
        train_loader = DataLoader(
            SequenceLabelDataset(
                X_train, y_train, self.predict_len, self.x_handler, self.y_handler,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_jobs,
            drop_last=True,
        )

        valid_loader = DataLoader(
            SequenceLabelDataset(
                X_test, y_test, self.predict_len, self.x_handler, self.y_handler,
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_jobs,
            drop_last=True,
        )
                                        
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_loss = np.inf 
        best_epoch = 0
        evals_result=dict()
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        print("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            print("Epoch%d:", step)
            print("training...")
            self.train_epoch(train_loader)
            print("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            print("train score %.6f, valid score %.6f" % (train_score, val_score))
            if tune is not None:
                tune.report(
                    train_score = train_score,
                    val_score = val_score
                )
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                best_loss = train_loss
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.GRU_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    print("early stop")
                    break

        print("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.GRU_model.load_state_dict(best_param)
        if self.use_gpu:
            torch.cuda.empty_cache()

        return best_loss,best_score 
        
    def predict(self, features):
        '''
        return 
        -------
            [sample_nums,sequence_length]
        '''
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        test_loader = DataLoader(
            SequenceLabelDataset(
                features , None, self.predict_len, self.x_handler, None,
            ),
            batch_size=self.batch_size, 
            num_workers=self.n_jobs,
            shuffle=False
        )
        self.GRU_model.eval()
        preds = []

        for data in test_loader:
            feature = data.to(self.device)

            with torch.no_grad():
                pred = self.GRU_model(feature.float()).detach().cpu().numpy()
            preds.append(pred)

        results = np.array(preds)

        if self.y_handler:
            results = self.y_handler.inverse_transform(results.reshape(-1,self.predict_len))

        return results 
    
    def save(self, save_prefix):
        torch.save(self.GRU_model, save_prefix + ".pth")
        if self.x_handler:
            with open(save_prefix +"_x.pkl","wb") as f:
                pickle.dump(self.x_handler, f)
        if self.y_handler:
            with open(save_prefix + "_y.pkl","wb") as f:
                pickle.dump(self.y_handler, f)


    def load(self, filename_prefix):
        self.GRU_model = torch.load(filename_prefix+".pth")
        if os.path.exists(filename_prefix+"_x.pkl"):
            self.x_handler = pickle.load(open(filename_prefix+"_x.pkl", "rb"))
            self.x_handler.fitted  = True 
        else:
            raise FileExistsError 
        if os.path.exists(filename_prefix+"_y.pkl"):
            self.y_handler = pickle.load(open(filename_prefix+"_y.pkl", "rb"))
            self.y_handler.fitted  = True 
        else:
            self.y_handler = None 

        self.fitted = True 
        self.GRU_model.to(self.device)

