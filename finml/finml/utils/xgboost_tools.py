import xgboost as xgb 
import numpy as np

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
    n_split = 5,
    base_dir = "models",
    xgb_type="classifier"
):
    import os
    models = []  
    for i in range(n_split):
        if xgb_type == "classifier":
            model = xgb.XGBClassifier()
        else:
            model = xgb.XGBRegressor()
        file_path = os.path.join(base_dir,bar_type+"_"+str(i)+".txt")
        model.load_model(file_path)
        models.append(model)
    return models 

def preidct_prob(
    models,
    feature
):
    feature = feature.reshape((1,-1))
    results = [] 
    for model in models:
        results.append(model.predict_proba(feature))
    return np.mean(results,axis=0)