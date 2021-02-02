# from ff.modeling.prepare import Xy_split_list, Xy_split, data_scale
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# import pandas as pd
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score

# def fit_predict(est, X_train, y_train, X_test, pred_name='pred'):
#     """Fit model and predict test data

#     Args:
#         est (sklearn.model): Model object that has .fit() and .predict() methods
#         X_train (pandas.DataFrame): Training dataset to fit model with
#         y_train (numpy.arrary or pandas.Series): Training target dataset
#         X_test (pandas.DataFrame): Holdout dataset to predict with trained model

#     Returns:
#         pandas.Series: Dataset with predictions
#         sklearn.model: Trained model object
#     """    
#     # fit model based on training data
#     est.fit(X_train, y_train)

#     # predict test data and pass results into DataFrame
#     pred = pd.Series(est.predict(X_test), name=pred_name)
    
#     return pred, est


# def calc_metrics(y_act, y_pred, mets, s_weight=None, rnd=3, show=True):
#     """Calculate accuracy metrics for given predictions

#     Args:
#         y_act (numpy.array or pandas.DataFrame): Actual values
#         y_pred (numpy.array or pandas.DataFrame): Predicted values
#         mets (list): Metrics to be calculated.
#             Options: 'mse', 'r2', 'mae', 'matt', 'f1', 'prec', 'rec'
#         s_weight (numpy.array or pandas.DataFrame, optional):
#             Sample weights for weight scoring. Defaults to None.
#         rnd (int, optional): Precision for rounding. Defaults to 3.
#         show (bool, optional): Whether to print results. Defaults to True.

#     Returns:
#         [type]: [description]
#     """    
#     # create a dictionar of scoring algorithms to choose from
#     scorers = {
#         'mse': mean_squared_error,
#         'r2': r2_score,
#         'mae': mean_absolute_error,
#         'matt': matthews_corrcoef,
#         'f1': f1_score,
#         'prec': precision_score,
#         'rec': recall_score
#     }

#     # save the results for each metric into a dictionary
#     output = {}
#     for m in mets:
#         output[m] = scorers[m](y_act, y_pred, sample_weight=s_weight)
#         output[m] = round(output[m], rnd)
    
#     # if results are to be shown, print results
#     if show:
#         [print(f'{k}: {v}') for k, v in output.items()]
    
#     return output


# def get_coef(est, labels, cutoff=0., name='coefs'):
#     """Create Series with model variable importance

#     Args:
#         est (sklearn.model): Trained model object 
#         labels (list): Names of variable labels (e.g. df.columns)
#         cutoff (float, optional): Min value to remove importances. Defaults to 0.
#         name (str, optional): Name of return Series. Defaults to 'coefs'.

#     Returns:
#         [pandas.Series]: Series that contains labeled variable coefs
#     """    
#     # try to pull out variable weights based on model type
#     try: coef = est.coef_
#     except: coef = est.feature_importances_
    
#     # filter down given importance values
#     coef = coef[abs(coef) > cutoff]

#     # return a Series with the datset
#     return pd.Series(coef, index=labels, name=name)


# def fill_metrics(df, met, X_cols):
#     """Fill in null values of column by predicting with other columns

#     Args:
#         df (pandas.DataFrame): Dataframe containing various combine metrics
#         met (str): Metric to be predicted and filled
#         X_cols (list): Columns used to predict missing metric

#     Returns:
#         pandas.DataFrame: DataFrame containing predicted values
#     """    
#     print(f'============\n{met}\n------------')
    
#     # split train and predict based on whether the metric is missing
#     train = df[~df[met].isnull()]
#     predict = df[df[met].isnull()]
    
#     # split data into X and y for train / predict
#     X, y = Xy_split_list(train, met, X_cols)
#     X_test, _ = Xy_split_list(predict, 'DfTest', X_cols)

#     # standard scale the training data
#     X, X_test = data_scale(X, X_test)

#     # fit and predict null data, and add predictions back to df
#     lr = Ridge(alpha=10)
#     pred, lr = fit_predict(lr, X, y, X_test)
#     df.loc[df[met].isnull(), met] = pred.values

#     # print out relevant metrics
#     calc_metrics(y, lr.predict(X), mets=['r2'])
#     print(get_coef(lr, X.columns))

#     return df


# def reverse_metrics(train, predict, met, X_cols):
#     """Use modeling to calculate metrics from PlayerProfiler

#     Args:
#         train (pandas.DataFrame): PlayerProfiler data with unique metrics
#         predict (pandas.DataFrame): New Data without PlayerProfiler metrics
#         met (str): Metric to be fitted and predicted
#         X_cols (list): Columns to use during prediction

#     Returns:
#         pandas.DataFrame: Train DataFrame with new metric included
#     """    
#     print(f'============\n{met}\n------------')
    
#     X, y = Xy_split_list(train, met, X_cols)
#     X_test, _ = Xy_split_list(predict, 'DfTest', X_cols)

#     X, X_test = data_scale(X, X_test)

#     # fit and predict null data, and add predictions back to df
#     lr = Ridge(alpha=10)
#     pred, lr = fit_predict(lr, X, y, X_test)
#     predict[met] = pred.values

#     # print out relevant metrics
#     calc_metrics(y, lr.predict(X), mets=['r2'])
#     print(get_coef(lr, X.columns))
    
#     return predict
