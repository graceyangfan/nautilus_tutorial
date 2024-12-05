import lightgbm as lgb
import numpy as np
import os

def save_model(
    models,
    bar_type,
    base_dir = "models"
):
    """
    保存LightGBM模型

    Args:
        models: 模型列表
        bar_type: 数据类型标识
        base_dir: 保存目录
    """
    for i, model in enumerate(models):
        file_path = os.path.join(base_dir, f"{bar_type}_{i}.txt")
        model.save_model(file_path)

def load_model(
    bar_type,
    n_split = 5,
    base_dir = "models",
    lgb_type = "classifier"
):
    """
    加载LightGBM模型

    Args:
        bar_type: 数据类型标识
        n_split: 模型数量
        base_dir: 模型目录
        lgb_type: 模型类型("classifier"或"regressor")

    Returns:
        加载的模型列表
    """
    models = []
    for i in range(n_split):
        file_path = os.path.join(base_dir, f"{bar_type}_{i}.txt")
        model = lgb.Booster(model_file=file_path)
        models.append(model)
    return models

def predict_prob(
    models,
    feature
):
    """
    集成预测概率

    Args:
        models: 模型列表
        feature: 输入特征

    Returns:
        平均预测概率
    """
    feature = feature.reshape((1, -1))
    results = []
    for model in models:
        results.append(model.predict(feature))
    return np.mean(results, axis=0) 