import numpy as np
import polars as pl 
from sklearn.metrics import log_loss, accuracy_score
from .cross_validation import PurgedKFold 
from .metrics import Metrics 

def get_mean_decrease_impurity(model, feature_names):
    """
    Get the in-sample impurity decrease contributed by each feature in tree-based models.
    This measure is subject to substitution effects where a feature's importance is diluted by other
    interchangeable features.

    This modifies sklearn RF's feature_importances_ which considers a feature not chosen in a tree
    as having 0 importance to set them as np.nan instead.

    return: columns feature_names, row: mean and std 
    """
    #tree_importances = np.array([tree.feature_importances_ for _, tree in enumerate(model.estimators_)])
    tree_importances = pl.DataFrame(
        {
            feature_names[i]:model.feature_importances_[i]  for i in range(len(feature_names))
        }
    )
    tree_importances = tree_importances.select([
        pl.when(pl.col(item).abs() < 1e-10).then(np.nan).otherwise(pl.col(item)).alias(item) for item in feature_names
    ])
    importances = pl.concat([
        tree_importances.select(pl.all().mean().suffix("_mean")),
        tree_importances.select((pl.all().std()*tree_importances.shape[0]**-0.5).suffix("_std"))
    ],how="horizontal")
    importances = importances / tree_importances.mean().sum(axis=1)

    return importances


def get_mean_decrease_accuracy(
    clf,
    X, 
    y,
    cv, 
    sample_weight,
    event_times, 
    embargo_pct, 
    scoring=Metrics.NEG_LOG_LOSS
):
    """
    Get the out-of-sample (OOS) decrease of selected evaluation score contributed by each feature using purged kfold
    cross validation. This method is model agnostic, but still subject to substitution effects.
    return: MDA feature importances DataFrame, scores for each CV fold
    """
    if scoring not in [Metrics.NEG_LOG_LOSS, Metrics.ACCURACY]:
        raise NotImplementedError(scoring)
    sample_weight = sample_weight.select("values")

    cv_splitter = PurgedKFold(n_splits=cv, embargo_pct=embargo_pct)
    scores = pl.DataFrame({"fold": np.arange(cv),"scores":np.zeros(cv)})
    perm_scores = pl.DataFrame({item:np.zeros(cv) for item in X.columns})
    perm_scores = pl.concat([perm_scores,pl.DataFrame({"fold": np.arange(cv)})],how="horizontal")
    for fold, (train_indices, test_indices) in enumerate(cv_splitter.split(event_times)):
        X_train = X[train_indices,:].to_pandas()
        y_train = y[train_indices,:].to_pandas()
        w_train = sample_weight[train_indices,:].to_pandas()
        X_test = X[test_indices,:].to_pandas()
        y_test = y[test_indices,:].to_pandas()
        w_test = sample_weight[test_indices,:].to_pandas()

        clf = clf.fit(X_train, y_train, sample_weight=w_train.values)

        if scoring == Metrics.NEG_LOG_LOSS:
            prob = clf.predict_proba(X_test)
            labels = np.unique(y)
            score = -log_loss(y_test, prob, sample_weight=w_test.values.reshape(-1,), labels=labels)
            scores = scores.select([
                    pl.col("fold"),pl.when(pl.col("fold")==fold).then(score).otherwise(pl.col("scores")).alias("scores")
                ])
        else:
            pred = clf.predict(X_test)
            score = accuracy_score(y_test, pred, sample_weight=w_test.values.reshape(-1,))
            scores = scores.select([
                    pl.col("fold"),pl.when(pl.col("fold")==fold).then(score).otherwise(pl.col("scores")).alias("scores")
                ])

        for j in X.columns:
            X_test_perm = X_test.copy(deep=True)
            np.random.shuffle(X_test_perm[j].values)
            if scoring == Metrics.NEG_LOG_LOSS:
                labels = np.unique(y)
                prob = clf.predict_proba(X_test_perm)
                score = -log_loss(y_test, prob, sample_weight=w_test.values.reshape(-1,), labels=labels)
                perm_scores = perm_scores.select(
                    [
                        pl.col(item) if item!=j else pl.when(pl.col("fold") == fold).then(score).otherwise(pl.col(j)).alias(j)
                            for item in perm_scores.columns
                    ]
                )
            else:
                pred = clf.predict(X_test_perm)
                score = accuracy_score(y_test, pred, sample_weight=w_test.values.reshape(-1,))
                perm_scores = perm_scores.select(
                    [
                        pl.col(item) if item!=j else pl.when(pl.col("fold") == fold).then(score).otherwise(pl.col(j)).alias(j)
                            for item in perm_scores.columns
                    ]
                )
    importances = pl.concat([scores,perm_scores.select(pl.all().exclude("fold")*(-1))],how="horizontal")

    if scoring == Metrics.NEG_LOG_LOSS:
        importances = importances.select(
            [
                pl.col("fold"),
                (pl.all().exclude(scores.columns) + pl.col("scores"))/pl.all().exclude(scores.columns)
            ]
        )# Improvement relative to 0 log loss: (score - perm_score)/(0 - perm_score)
    else:
        importances = importances.select(
            [
                pl.col("fold"),
                (pl.all().exclude(scores.columns) + pl.col("scores"))/(1.0-pl.all().exclude(scores.columns))
            ]
        )# Improvement relative to 1. accuracy: (score - perm_score)/(1 - perm_score)

    importances = pl.concat([
        importances.select("fold"),
        importances.select(pl.all().exclude("fold").mean().suffix("_mean")),
        importances.select((pl.all().exclude("fold").std()*importances.shape[0]**-0.5).suffix("_std")),
    ],how="horizontal")

    return importances, scores.select([pl.col("fold"),pl.col("scores").mean()])


def timeseries_cv_score(
        clf,
        X, 
        y, 
        sample_weight, 
        scoring=Metrics.NEG_LOG_LOSS,
        event_times=None, 
        cv=None, 
        cv_splitter=None, 
        embargo_pct=None
):
    """
    Perform cross validation scoring on time series data. See also: sklearn cross_val_score

    return: an array of score for each CV split. Can be None if cv_splitter is provided
    """

    if scoring not in ['neg_log_loss', 'accuracy']:
        raise NotImplementedError(f'scoring: {scoring}')
    if cv_splitter is None:
        cv_splitter = PurgedKFold(n_splits=cv, embargo_pct=embargo_pct)

    scores = []
    for train_indices, test_indices in cv_splitter.split(event_times):
        X_train = X[train_indices, :].to_pandas()
        y_train = y[train_indices, :].to_pandas()
        train_sample_weight = sample_weight[train_indices, :].to_pandas()
        X_test = X[test_indices, :].to_pandas()
        y_test = y[test_indices, :].to_pandas()
        test_sample_weight = sample_weight[test_indices, :].to_pandas()

        fitted = clf.fit(
            X=X_train, 
            y=y_train, 
            sample_weight=train_sample_weight.values
        )

        if scoring == Metrics.NEG_LOG_LOSS:
            prob = fitted.predict_proba(X_test)
            split_score = -log_loss(y_test, prob, sample_weight=test_sample_weight.values.reshape(-1,))
        else:
            pred = fitted.predict(X_test)
            split_score = accuracy_score(y_test, pred, sample_weight=test_sample_weight.values.reshape(-1,))

        scores.append(split_score)

    return np.array(scores)



def get_single_feature_importance(
    features, 
    clf, 
    X, 
    y, 
    sample_weight, 
    scoring, 
    cv_splitter
):
    """
    Reference implementation of single feature importance (SFI) concept.
    The model OOS CV score is computed for each feature separately and hence remove the substitution effect

    return: DataFrame of mean and std of cv scores for each feature
    """
    importances = pl.DataFrame({item: np.zeros(2) for item in features})
    for feature in features:
        scores = timeseries_cv_score(
           clf, X=X[:,[feature]], y=y, sample_weight=sample_weight, scoring=scoring, cv_splitter=cv_splitter
        )
        importances.replace(feature,pl.Series([scores.mean(),scores.std()*scores.shape[0]**-0.5]))
    return importances