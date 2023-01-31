# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2023 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

import gc 
import numpy as np
import polars as pl 
import xgboost as xgb 
#from finml.modelling.cross_validation import PurgedKFold 
from xgboostlss.gaussian import Gaussian 
from xgboostlss.xgblss import xgboostlss 
from xgboost_ray import RayDMatrix

def fit_xgboost(
    X_train,
    y_train,
    X_test,
    y_test,
    num_actors,
    cpus_per_actor,
    dist, 
):
    np.random.seed(123)

    # Specifies the parameters and their value range. The structure is as follows: "hyper-parameter": [lower_bound, upper_bound]. Currently, only the following hyper-parameters can be optimized:
    params = {"eta": [1e-5, 1],                   
            "max_depth": [1, 10],
            "gamma": [1e-8, 40],
            "subsample": [0.2, 1.0],
            "colsample_bytree": [0.2, 1.0],
            "min_child_weight": [0, 500]
            }

    dtrain = RayDMatrix(X_train,y_train)
    dtest = RayDMatrix(X_test, y_test)

    opt_params,best_score = xgboostlss.hyper_opt(
        params,
        dtrain=dtrain,
        dist=dist,
        evals = [(dtest, 'eval'),(dtrain, 'train'),],
        num_actors=num_actors,
        cpus_per_actor=cpus_per_actor,
        num_boost_round=500,       # Number of boosting iterations.
        verbose_eval = 100,
        max_minutes=120,           # Time budget in minutes, i.e., stop study after the given number of minutes.
        n_trials=30,             # The number of trials. If this argument is set to None, there is no limitation on the number of trials.
        silence=False             # Controls the verbosity of the trail, i.e., user can silence the outputs of the trail.
    )
    np.random.seed(123)

    n_rounds = opt_params["opt_rounds"]
    del opt_params["opt_rounds"]

    # Train Model with optimized hyper-parameters
    xgboostlss_model = xgboostlss.train(
        opt_params,
        dtrain,
        dist=dist,
        evals = [(dtest, 'eval'),(dtrain, 'train'),],
        num_actors=num_actors,
        cpus_per_actor=cpus_per_actor,
        num_boost_round=n_rounds,
    )
    return xgboostlss_model,best_score 

def objectivate(
    X, 
    y, 
    event_times, 
    num_actors = 10,
    cpus_per_actor = 1,
    n_splits=3,
    embargo_pct=0.01,
    distribution = Gaussian,
    return_info = False,
):
    cv = PurgedKFold(
        n_splits=n_splits,
        embargo_pct=embargo_pct
    )
    models = []
    valid_score = 0
    for train_indices, test_indices in cv.split(event_times):
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        X_train = X_train.to_pandas() 
        X_test = X_test.to_pandas()
        y_train = y_train.to_pandas()
        y_test = y_test.to_pandas()
        model, score = fit_xgboost(
            X_train,
            y_train,
            X_test,
            y_test,
            num_actors,
            cpus_per_actor,
            dist=distribution
        )
        models.append(model)
        gc.collect()
        valid_score += score 
    valid_score /= len(models)
    print(f'The valid_score for the models are {valid_score}')
    if  return_info:
        return models,valid_score
    else:
        return valid_score

def save_model(
    models,
    bar_type,
    base_dir = "models",
):
    import os
    for i in range(len(models)):
        file_path = os.path.join(base_dir,bar_type+"_"+str(i)+".txt")
        models[i].save_model(file_path)

def load_model(
    bar_type,
    n_spilt = 5,
    base_dir = "models",
    xgb_type = "regression",
):
    import os
    models = []  
    for i in range(n_spilt):
        if xgb_type == "regression":
            model = xgb.XGBRegressor() 
        else:
            model = xgb.XGBClassifier()
        file_path = os.path.join(base_dir,bar_type+"_"+str(i)+".txt")
        model.load_model(file_path)
        models.append(model)
    return model 


def predict_quantiles(
    xgboostlss_models,
    features,
    dist = Gaussian,
    quant_set = [0.05, 0.95], 
    seed = 123
):
    dtest =  xgb.DMatrix(features)
    predicted_values = [] 
    for model in xgboostlss_models:
        y_predict = xgboostlss.predict(
            model, 
            dtest, 
            dist=dist,
            pred_type="quantiles", 
            quantiles=quant_set, 
            seed=seed
        )
        y_predict_0 = y_predict.iloc[0,y_predict.columns[0]]
        #y_predict_1 = y_predict.iloc[0,y_predict.columns[1]]
        predicted_values.append(y_predict_0)
    return predicted_values 
                

def save_model(
    models,
    bar_type,
    base_dir = "models",
):
    import os
    for i in range(len(models)):
        file_path = os.path.join(base_dir,bar_type+"_"+str(i)+".txt")
        models[i].save_model(file_path)

def load_model(
    bar_type,
    n_spilt = 5,
    base_dir = "models",
    xgb_type = "regression",
):
    import os
    models = []  
    for i in range(n_spilt):
        if xgb_type == "regression":
            model = xgb.XGBRegressor() 
        else:
            model = xgb.XGBClassifier()
        file_path = os.path.join(base_dir,bar_type+"_"+str(i)+".txt")
        model.load_model(file_path)
        models.append(model)
    return models 

                

def predict(
    booster,
    dtest,
    dist = Gaussian,
    quantiles = [0.05, 0.95], 
    seed = 123
):

    dict_param = dist.param_dict()

    # Set base_margin as starting point for each distributional parameter. Requires base_score=0 in parameters.
    base_margin = (np.ones(shape=(dtest.num_row(), 1))) * dist.start_values
    dtest.set_base_margin(base_margin.flatten())

    predt = booster.predict(dtest, output_margin=True)

    dist_params_predts = []

    for i, (dist_param, response_fun) in enumerate(dict_param.items()):
        dist_params_predts.append(response_fun(predt[:, i]))

    dist_params_df = pd.DataFrame(dist_params_predts).T
    dist_params_df.columns = dict_param.keys()

    pred_quant_df = dist.pred_dist_quantile(
        quantiles = quantiles,
        pred_params = dist_params_df
    )

    pred_quant_df.columns = [str("quant_") + str(quantiles[i]) for i in range(len(quantiles))]
    return pred_quant_df


def predict_quantiles(
    models,
    features,
    dist = Gaussian,
    quant_set = [0.05, 0.95], 
    seed = 123
):
    dtest =  xgb.DMatrix(features)
    predicted_values = [] 
    for model in models:
        y_predict = predict(
            model, 
            dtest, 
            dist=dist, 
            quantiles=quant_set, 
            seed=seed
        )
        y_predict_0 = y_predict.iloc[0,y_predict.columns[0]]
        #y_predict_1 = y_predict.iloc[0,y_predict.columns[1]]
        predicted_values.append(y_predict_0)
    return np.mean(predicted_values)