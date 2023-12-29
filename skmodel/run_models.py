#%%
from asyncio import format_helpers
from skmodel.data_setup import DataSetup
from skmodel.pipe_setup import PipeSetup
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from functools import partial
import copy

import pandas as pd
import numpy as np

# model and search setup
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, StratifiedGroupKFold, GroupKFold

from hyperopt import fmin, hp, tpe, Trials, space_eval
from hyperopt.pyll import scope

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import matthews_corrcoef, f1_score, brier_score_loss
from sklearn.metrics import mean_pinball_loss
import gc

from statsmodels.stats.stattools import medcouple


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

# import pyspark as ps
# conf = ps.SparkConf()
# conf.set("spark.executor.heartbeatInterval","3600s")

class SciKitModel(PipeSetup):
    
    def __init__(self, data, model_obj='reg', set_seed=1234, **kwargs):
        """Class the automates model sklearn pipeline creation and model training

        Args:
            data (pandas.DataFrame): Data to be modeled with features and target
            model_obj (str, optional): Model objective for regression (reg) or 
                                       classification (class). Defaults to 'reg'.
            set_seed (int, optional): Numpy random seed to set state. Defaults to 1234.
        """        
        self.data = data
        self.model_obj = model_obj
        self.proba = False
        self.stacking = False
        self.alpha = 0.5
        self.randseed=set_seed
        self.num_k_folds = 1
        np.random.seed(self.randseed)

        self.r2_wt = kwargs.get('r2_wt', 0)
        self.sera_wt = kwargs.get('sera_wt', 1)
        self.mse_wt = kwargs.get('mse_wt', 0)
        self.matt_wt = kwargs.get('matt_wt', 0)
        self.brier_wt = kwargs.get('brier_wt', 1)
        self.mae_wt = kwargs.get('mae_wt', 0)

    def param_range(self, var_type, low, high, spacing, bayes_rand, label):

        if bayes_rand=='bayes':
            if var_type=='int': return scope.int(hp.quniform(label, low, high, 1))
            if var_type=='real': return hp.uniform(label, low, high)
            if var_type=='cat': return hp.choice(label, low)
            if var_type=='log': return hp.loguniform(label, low, high)

        elif bayes_rand=='rand':
            if var_type=='int': return range(low, high, spacing)
            if var_type=='real': return np.arange(low, high, spacing)
            if var_type=='cat': return low
            if var_type=='log': return 10**np.arange(low, high, spacing)


    def default_params(self, pipe, bayes_rand='rand', min_samples=20):
        """Function that returns default search parameters for pipe components

        Args:
            model_name (str): Abbreviation of model in pipe.
                              Model Name Options: 
                                       'ridge' = Ridge() params,
                                       'lasso' = Lasso() params,
                                       'enet' = ElasticNet() params,
                                       'rf' = RandomForestRegressor() or Classifier() params,
                                       'lgbm' = LGBMRegressor() or Classifier() params,
                                       'xgb' = XGBRegressor() or Classifier() params,
                                       'knn' = KNeighborsRegress() or Classifier() params,
                                       'svr' = LinearSVR() params

                              Numeric Param Options:
                                       'agglomeration' = FeatureAgglomeration() params
                                       'pca' = PCA() params,
                                       'k_best' = SelectKBest params,
                                        

            num_steps (dict, optional): [description]. Defaults to {}.

        Returns:
            [type]: [description]
        """        

        br = bayes_rand
        param_options = {

            # feature params
            'random_sample': {'frac': self.param_range('real', 0.25, 0.65, 0.02, br, 'frac'),
                              'seed': self.param_range('int', 0, 10000, 1000, br, 'seed')},
            'agglomeration': {'n_clusters': self.param_range('int', 2, 25, 2, br, 'n_clusters')},
            'pca': {'n_components': self.param_range('int', 2, 20, 2, br, 'n_components')},
            'k_best': {'k': self.param_range('int', 20, 125, 5, br, 'k')},
            'select_perc': {'percentile': self.param_range('int', 20, 55, 3, br, 'select_perc')},
            'k_best_c': {'k': self.param_range('int', 20, 125, 5, br, 'k_best_c')},
            'select_perc_c': {'percentile': self.param_range('int', 20, 55, 3, br, 'select_perc_c')},
            'select_from_model': {'estimator': [Ridge(alpha=0.1), Ridge(alpha=1), Ridge(alpha=10),
                                                Lasso(alpha=0.1), Lasso(alpha=1), Lasso(alpha=10),
                                                RandomForestRegressor(max_depth=5), 
                                                RandomForestRegressor(max_depth=10)]},
            'feature_drop': {'col': self.param_range('cat', ['avg_pick', None], None, None, br, 'feature_drop')},
            'feature_select': {'cols': self.param_range('cat', [['avg_pick'], ['avg_pick', 'year']], None, None, br, 'feature_select')},

            # model params
            'ridge': {
                        'alpha': self.param_range('log', 0, 3, 0.1, br, 'alpha')
                     },

            'lasso': {
                        'alpha': self.param_range('log', -2, 0.5, 0.05, br, 'alpha')
                    },

            'enet': {
                    'alpha': self.param_range('log', -2, 1, 0.1, br, 'alpha'),
                    'l1_ratio': self.param_range('real', 0.05, 0.5, 0.03, br, 'l1_ratio')
                    },

            'huber': {
                    'alpha': self.param_range('log', -1.5, 1.5, 0.05, br, 'alpha'),
                    'epsilon': self.param_range('real', 1.2, 1.5, 0.03, br, 'epsilon'),
                    'max_iter': self.param_range('int', 100, 200, 10, br, 'max_iter')
                    },

            'lr_c': {
                    'C': self.param_range('log', -4, 1, 0.1, br, 'C'),
                  #  'class_weight': [{0: i, 1: 1} for i in np.arange(0.2, 1, 0.1)]
                    },
            
            'qr_q': {
                        'alpha': self.param_range('log', -3, 0, 0.05, br, 'alpha'),
                        'solver': self.param_range('cat', ['highs-ds', 'highs-ipm', 'highs'], None, None, br, 'solver')
                    },

            'rf': {
                    'n_estimators': self.param_range('int', 50, 250, 10, br, 'n_estimators'),
                    'max_depth': self.param_range('int', 2, 30, 2, br, 'max_depth'),
                    'min_samples_leaf': self.param_range('int', 1, 10, 1, br, 'min_samples_leaf'),
                    'max_features': self.param_range('real', 0.1, 1, 0.2, br, 'max_features')
                    },

            'rf_c': {
                    'n_estimators': self.param_range('int', 50, 250, 20, br, 'n_estimators'),
                    'max_depth': self.param_range('int', 2, 20, 3, br, 'max_depth'),
                    'min_samples_leaf': self.param_range('int', 1, 20, 2, br, 'min_samples_leaf'),
                    'max_features': self.param_range('real', 0.1, 1, 0.2, br, 'max_features'),
                    # 'class_weight': [{0: i, 1: 1} for i in np.arange(0.01, 0.8, 0.05)],
                    # 'criterion': self.param_range('cat', ['gini', 'log_loss'], None, None, br, 'criterion'),
                    },

            'rf_q': {
                    'n_estimators': self.param_range('int', 50, 200, 10, br, 'n_estimators'),
                    'max_depth': self.param_range('int', 2, 20, 2, br, 'max_depth'),
                    'min_samples_leaf': self.param_range('int', 5, 25, 1, br, 'min_samples_leaf'),
                    'max_features': self.param_range('real', 0.1, 1, 0.1, br, 'max_features')
                    },

            'lgbm': {
                        'max_depth': self.param_range('int', 2, 15, 2, br, 'max_depth'),
                        'num_leaves': self.param_range('int', 20, 100, 5, br, 'num_leaves'),
                        'learning_rate': self.param_range('log', -3, 0, 0.1, br,  'learning_rate'),
                        'colsample_bytree': self.param_range('real', 0.2, 1, 0.05, br, 'colsample_bytree'),
                        'subsample':  self.param_range('real', 0.4, 1, 0.1, br, 'subsample'),
                        'min_child_samples': self.param_range('int', 1, 50, 2, br, 'min_child_samples')
                        
                    },

            'lgbm_c': {
                      'max_depth': self.param_range('int', 2, 15, 2, br, 'max_depth'),
                      'num_leaves': self.param_range('int', 20, 100, 5, br, 'num_leaves'),
                      'learning_rate': self.param_range('log', -3, 0, 0.1, br,  'learning_rate'),
                      'colsample_bytree': self.param_range('real', 0.2, 1, 0.05, br, 'colsample_bytree'),
                      'subsample':  self.param_range('real', 0.4, 1, 0.1, br, 'subsample'),
                      'min_child_samples': self.param_range('int', 1, 50, 2, br, 'min_child_samples')
                     },

            'lgbm_q': {
                        'max_depth': self.param_range('int', 2, 15, 2, br, 'max_depth'),
                        'num_leaves': self.param_range('int', 20, 100, 5, br, 'num_leaves'),
                        'learning_rate': self.param_range('log', -3, 0, 0.1, br,  'learning_rate'),
                        'colsample_bytree': self.param_range('real', 0.2, 1, 0.05, br, 'colsample_bytree'),
                        'subsample':  self.param_range('real', 0.4, 1, 0.1, br, 'subsample'),
                        'min_child_samples': self.param_range('int', 1, 50, 2, br, 'min_child_samples')
                    },

            'xgb': {
                    'n_estimators': self.param_range('int', 30, 250, 20, br, 'n_estimators'),
                     'max_depth': self.param_range('int', 2, 20, 2, br, 'max_depth'),
                     'colsample_bytree': self.param_range('real', 0.2, 0.9, 0.1, br,  'colsample_bytree'),
                     'subsample':  self.param_range('real', 0.4, 1, 0.1, br, 'subsample'),
                     'reg_lambda': self.param_range('log', 0, 2, 0.1, br,  'reg_lambda'),
                     'reg_alpha': self.param_range('int', 0, 50, 5, br,  'reg_alpha'),
                     'learning_rate': self.param_range('log', -3, -0.5, 0.1, br, 'learning_rate'),
                     },

            'xgb_c': {
                     'n_estimators': self.param_range('int', 30, 170, 10, br, 'n_estimators'),
                     'max_depth': self.param_range('int', 2, 20, 3, br, 'max_depth'),
                     'colsample_bytree': self.param_range('real', 0.2, 1, 0.25, br, 'colsample_bytree'),
                     'subsample':  self.param_range('real', 0.4, 1, 0.1, br, 'subsample'),
                     'reg_lambda': self.param_range('log', 0, 3, 0.1, br, 'reg_lambda'),
                     'learning_rate': self.param_range('log', -3, -0.5, 0.1, br, 'learning_rate'),
                    #  'scale_pos_weight': self.param_range('real', 1, 20, 1, br, 'scale_pos_weight')
                     },

            'gbm': {
                    'n_estimators': self.param_range('int', 20, 90, 3, br, 'n_estimators'),
                    'max_depth': self.param_range('int', 2, 18, 2, br,'max_depth'),
                    'min_samples_leaf': self.param_range('int', 5, 20, 2, br, 'min_samples_leaf'),
                    'max_features': self.param_range('real', 0.7, 1, 0.1, br, 'max_features'),
                    'subsample': self.param_range('real', 0.5, 1, 0.1, br, 'subsample'),
                    'learning_rate': self.param_range('log', -3, -0.5, 0.1, br, 'learning_rate'),
                    },

            'gbm_c': {
                    'n_estimators': self.param_range('int', 10, 80, 10, br, 'n_estimators'),
                    'max_depth': self.param_range('int', 2, 25, 3, br, 'max_depth'),
                    'min_samples_leaf': self.param_range('int', 5, 15, 1, br, 'min_samples_leaf'),
                    'max_features': self.param_range('real', 0.7, 1, 0.1, br, 'max_features'),
                    'learning_rate': self.param_range('log', -3, -0.5, 0.1, br, 'learning_rate'),
                    'subsample': self.param_range('real', 0.5, 1, 0.1, br, 'subsample')
                    },

            'gbm_q': {
                    'n_estimators': self.param_range('int', 10, 60, 5, br, 'n_estimators'),
                    'max_depth': self.param_range('int', 2, 18, 2, br,'max_depth'),
                    'min_samples_leaf': self.param_range('int', 4, 25, 3, br, 'min_samples_leaf'),
                    'max_features': self.param_range('real', 0.7, 1, 0.1, br, 'max_features'),
                    'subsample': self.param_range('real', 0.5, 1, 0.1, br, 'subsample')
                    },

            'gbmh': {
                    'max_iter': self.param_range('int', 30, 100, 10, br, 'max_iter'),
                    'max_depth': self.param_range('int', 4, 12, 1, br,'max_depth'),
                    'min_samples_leaf': self.param_range('int', 5, 25, 2, br, 'min_samples_leaf'),
                    'max_leaf_nodes': self.param_range('int', 20, 50, 3, br, 'max_leaf_nodes'),
                    'l2_regularization': self.param_range('real', 0, 10, 1, br, 'l2_regularization'),
                    'learning_rate': self.param_range('log', -2, -0.5, 0.1, br, 'learning_rate')
                    },

            'gbmh_c': {
                    'max_iter': self.param_range('int', 30, 100, 10, br, 'max_iter'),
                    'max_depth': self.param_range('int', 3, 12, 2, br,'max_depth'),
                    'min_samples_leaf': self.param_range('int', 5, 25, 2, br, 'min_samples_leaf'),
                    'max_leaf_nodes': self.param_range('int', 15, 50, 5, br, 'max_leaf_nodes'),
                    'l2_regularization': self.param_range('real', 0, 10, 1, br, 'l2_regularization'),
                    'learning_rate': self.param_range('log', -2, -0.5, 0.1, br, 'learning_rate')
                    },

            'gbmh_q': {
                    'max_iter': self.param_range('int', 30, 100, 10, br, 'max_iter'),
                    'max_depth': self.param_range('int', 3, 12, 2, br,'max_depth'),
                    'min_samples_leaf': self.param_range('int', 5, 25, 2, br, 'min_samples_leaf'),
                    'max_leaf_nodes': self.param_range('int', 15, 50, 5, br, 'max_leaf_nodes'),
                    'l2_regularization': self.param_range('real', 0, 10, 1, br, 'l2_regularization'),
                    'learning_rate': self.param_range('log', -2, -0.5, 0.1, br, 'learning_rate')
                    },

            'knn': {
                    'n_neighbors':  self.param_range('int',2, min_samples, 1, br, 'n_neighbors'),
                    'weights': self.param_range('cat',['distance', 'uniform'], None, None, br, 'weights'),
                    'algorithm': self.param_range('cat', ['auto', 'ball_tree', 'kd_tree', 'brute'], None, None, br, 'algorithm')
                    },

            'knn_c': {
                    'n_neighbors':  self.param_range('int',2, min_samples, 1, br, 'n_neighbors'),
                    'weights': self.param_range('cat',['distance', 'uniform'], None, None, br, 'weights'),
                    'algorithm': self.param_range('cat', ['auto', 'ball_tree', 'kd_tree', 'brute'], None, None, br, 'algorithm')
                    },

            'knn_q': {
                    'n_neighbors':  self.param_range('int',2, min_samples, 1, br, 'n_neighbors'),
                    'weights': self.param_range('cat', ['distance', 'uniform'], None, None, br, 'weights'),
                    'algorithm': self.param_range('cat', ['auto', 'ball_tree', 'kd_tree', 'brute'], None, None, br, 'algorithm')
                    },

            'svr': {
                    'C': self.param_range('log', -1, 2, 0.1, br, 'C')
                    },

            'svc': {
                    'C': self.param_range('log', -1, 2, 0.1, br, 'C'),
                    # 'class_weight': [{0: i, 1: 1} for i in np.arange(0.05, 1, 0.2)]
                    },

            'ada': {
                    'n_estimators': self.param_range('int', 25, 100, 20, br, 'n_estimators'),
                    'learning_rate': self.param_range('log', -2, 0, 0.1, br, 'learning_rate'),
                    'loss': self.param_range('cat', ['linear', 'square', 'exponential'], None, None, br, 'loss')
                    },

            'ada_c': {
                    'n_estimators': self.param_range('int', 25, 200, 20, br, 'n_estimators'),
                    'learning_rate': self.param_range('log', -2, 1, 0.1, br, 'learning_rate'),
                    'loss': self.param_range('cat', ['linear', 'square', 'exponential'], None, None, br, 'loss')
                    },
            
            'mlp': {
                    'hidden_layer_sizes': self.param_range('int', 20, 100, 10, br, 'hidden_layer_sizes'),
                    'alpha': self.param_range('log', -4, -1, 0.1, br, 'alpha'),
                    'learning_rate_init': self.param_range('log', -3, 0, 0.1, br, 'learning_rate_init'),
                    'max_iter': self.param_range('int', 100, 500, 50, br, 'max_iter'),
                    'learning_rate': self.param_range('cat', ['constant', 'invscaling', 'adaptive'], None, None, br, 'learning_rate')
                    },
            'mlp_c': {
                    'hidden_layer_sizes': self.param_range('int', 20, 100, 10, br, 'hidden_layer_sizes'),
                    'alpha': self.param_range('log', -5, -1, 0.1, br, 'alpha'),
                    'learning_rate_init': self.param_range('log', -3, 0, 0.1, br, 'learning_rate_init'),
                    'max_iter': self.param_range('int', 100, 500, 50, br, 'max_iter'),
                    'learning_rate': self.param_range('cat', ['constant', 'invscaling', 'adaptive'], None, None, br, 'learning_rate')
                    }
        }

        # initialize the parameter dictionary
        params = {}

        # get all the named steps in the pipe
        # if type(pipe)==Pipeline: 
        steps = pipe.named_steps
        pipe_type = 'normal'
        # elif type(pipe)==TransformedTargetRegressor: 
        #     pipe = pipe.regressor
        #     steps = pipe.named_steps
        #     pipe_type = 'y_transform'

        #-----------------
        # Create a dict using named pipe step prefixes + hyperparameters
        #-----------------

        # begin looping through each step
        for step, _ in steps.items():

            # if the step has default params, then loop through hyperparams and add to dict
            if step in param_options.keys():
                for hyper_param, value in param_options[step].items():
                    if pipe_type=='normal':
                        params[f'{step}__{hyper_param}'] = value
                    elif pipe_type=='y_transform':
                         params[f'regressor__{step}__{hyper_param}'] = value

            # if the step is feature union go inside and find steps within 
            if step == 'feature_union':
                outer_transform = pipe.named_steps[step].get_params()['transformer_list']

                # add each feature union step prefixed by feature_union
                for inner_step in outer_transform:
                    for hyper_param, value in param_options[inner_step[0]].items():
                        if pipe_type=='normal':
                            params[f'{step}__{inner_step[0]}__{hyper_param}'] = value
                        elif pipe_type=='y_transform':
                            params[f'regressor__{step}__{inner_step[0]}__{hyper_param}'] = value

            # if the step is feature union go inside and find steps within 
            if step == 'column_transform':
                num_outer_transform = steps[step].get_params()['transformers'][0][1:-1][0].named_steps
                cat_outer_transform = steps[step].get_params()['transformers'][1][1:-1][0].named_steps

                for num_cat, outer_transform in zip(['numeric', 'cat'], [num_outer_transform, cat_outer_transform]):

                  for inner_step, _ in outer_transform.items():
                    if inner_step in param_options.keys():
                      for hyper_param, value in param_options[inner_step].items():
                          params[f'{step}__{num_cat}__{inner_step}__{hyper_param}'] = value
                
        return params

    @staticmethod
    def get_quantile_stats(data):
        q_min, q_max = np.min(data), np.max(data)
        q1, q2, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1

        m = medcouple(data)

        if m >= 0: upper_med = q3 + 1.5*np.exp(3*m)*iqr
        else: upper_med = q3 + 1.5*np.exp(4*m)*iqr

        # if the upper value is higher than q_max, then 
        # modify values to make it lower than q_max
        if upper_med >= q_max: upper_med = q3 + 1.5*iqr
        if upper_med >= q_max: upper_med = q_max * 0.9
        if q2 >= upper_med: q2 = upper_med * 0.9
        if q_min >= q2: q_min = q2 * 0.9
        
        return q_min, q2, upper_med, q_max, len(data)

    @staticmethod
    def create_relevance_data(q_min, q2, upper_med, q_max, n):

        from scipy.interpolate import PchipInterpolator as pchip

        x = np.array([q_min, q2, upper_med, q_max])
        for i in range(3):
            if x[i+1] < x[i]: x[i+1] = x[i] + 0.000000001
        y = np.array([0, 0, 1, 1])

        interp = pchip(x, y)

        freq_data = np.linspace(q_min, q_max, n)
        freq_line = interp(freq_data)
        relevance = pd.concat([pd.Series(freq_line, name='phi'), 
                            pd.Series(freq_data, name='y_cut')], axis=1)

        return relevance

    @staticmethod
    def filter_relevance(relevance, q_min):
        min_phi_1 = relevance.loc[relevance.phi==1, 'y_cut'].min()
        relevance = relevance[~((relevance.phi==0) & (relevance.y_cut!=q_min)) & \
                            ~((relevance.phi==1) & (relevance.y_cut!=min_phi_1))]
        relevance = relevance.reset_index(drop=True)
        relevance['idx'] = 0

        return relevance

    def sera_loss(self, y, y_pred):
        q_min, q2, upper_med, q_max, n = self.get_quantile_stats(y)
        
        relevance = self.create_relevance_data(q_min, q2, upper_med, q_max, n)
        relevance = self.filter_relevance(relevance, q_min)

        ys = pd.concat([pd.Series(y, name='y'), pd.Series(y_pred, name='y_pred')], axis=1)
        ys['idx'] = 0

        sera = pd.merge(ys, relevance, on='idx')
        sera = sera[sera.y >= sera.y_cut]
        sera['error'] = (sera.y - sera.y_pred)**2

        return sera.error.sum() / np.sum((sera.y**2))


    def scorer(self, score_type, **kwargs):

        scorers = {
            'r2': make_scorer(r2_score, greater_is_better=True),
            'mse': make_scorer(mean_squared_error, greater_is_better=False),
            'mae': make_scorer(mean_absolute_error, greater_is_better=False),
            'sera': make_scorer(self.sera_loss, greater_is_better=False),
            'matt_coef': make_scorer(matthews_corrcoef, greater_is_better=True),
            'brier': make_scorer(brier_score_loss, greater_is_better=False, needs_proba=True),
            'f1': make_scorer(f1_score, greater_is_better=True),
            'pinball': make_scorer(mean_pinball_loss, greater_is_better=False, **kwargs),
        }

        return scorers[score_type]


    def custom_score(self, y, y_pred, sample_weight=None):
        
        if self.model_obj=='reg':
            r2 = r2_score(y, y_pred, sample_weight=sample_weight)
            sera = self.sera_loss(y, y_pred)
            mse = mean_squared_error(y, y_pred, sample_weight=sample_weight)
            mae = mean_absolute_error(y, y_pred, sample_weight=sample_weight)
            score = 100*(self.mae_wt*mae + self.mse_wt*mse + self.sera_wt*sera - self.r2_wt*r2)

        elif self.model_obj=='class':
            matt = matthews_corrcoef(np.where(np.array(y)>=0.5, 1, 0), np.where(np.array(y_pred)>=0.5, 1, 0), sample_weight=sample_weight)
            
            #sweights = np.array(y)*5 + 1
            brier = brier_score_loss(y, y_pred, sample_weight=sample_weight)
            score = 100*(self.brier_wt*brier - self.matt_wt*matt)
        
        elif self.model_obj=='quantile':
            score = mean_pinball_loss(y, y_pred, alpha=self.alpha)

        return score


    @staticmethod
    def param_select(params):
        param_select = {}
        for k in params.keys():
            v = params[k]
            select_idx = np.random.choice(len(v))
            param_select[k] = v[select_idx]
        return param_select


    def rand_objective(self, params):

        self.cur_model.set_params(**params) 

        try:
            if self.stacking:
            
                score = []
                for k in range(self.num_k_folds):
                    self.randseed = (2+k) + self.randseed + k * 4
                    val_predictions = self.cv_predict(self.cur_model)
                    y_val = self.y_vals
                    cur_score = self.custom_score(y_val, val_predictions, self.wts)
                    score.append(cur_score)
                score = np.mean(score)

            else:
                val_predictions, _ = self.cv_predict_time_holdout(self.cur_model)
                y_val = self.get_y_val()
                score = self.custom_score(y_val.values, val_predictions)
                
        except:
            print('Trial Failed')
            score=100000000

        return score




    def custom_rand_search(self, model, params, n_iters):

        from joblib import Parallel, delayed
        self.cur_model = model

        param_list = [self.param_select(params) for _ in range(n_iters)]

        scores = Parallel(n_jobs=-1, verbose=0)(delayed(self.rand_objective)(p) for p in param_list)
        param_output = pd.DataFrame(param_list, index=range(n_iters))
        param_output['scores'] = scores

        best_params = param_list[np.argmin(scores)]
        model.set_params(**best_params)
                    
        try: model.steps[-1][1].n_jobs=-1
        except: pass

        return model, param_output
    
    def get_bayes_params(self, num_past_runs, params):
        # store all scores and parameters from the trials object into dataframe
        param_output = pd.DataFrame()
        best_score = 100000000
        
        for t in self.trials.trials:
            trial_params = space_eval(params, {k:v[0] for k,v in t['misc']['vals'].items()})
            
            # find the best score from the most recent trials
            # and set the model parameters to the best parameters
            score = t['result']['loss']
            if t['tid'] > num_past_runs and score < best_score:
                best_score = score
                best_params = copy.deepcopy(trial_params)

            trial_params['score'] = score
            param_output = pd.concat([param_output, pd.Series(trial_params)], axis=1)
        param_output = param_output.T

        return param_output, best_params
    

    def custom_bayes_search(self, model, params, n_iters):

        try: model.steps[-1][1].n_jobs=-1
        except: pass

        self.cur_model = model

        if self.trials is None: 
            raise ValueError('Trials object must be provided for Bayesian Optimization')

        num_past_runs = len(self.trials.tids)
        max_evals = num_past_runs + n_iters
        n_startup_jobs = np.min([int(max_evals/2), 20])

        _ = fmin(self.rand_objective, space=params, algo=partial(tpe.suggest, n_startup_jobs=n_startup_jobs), 
                 trials=self.trials, max_evals=max_evals, rstate=np.random.default_rng(self.randseed))

        param_output, best_params = self.get_bayes_params(num_past_runs, params)
    
        model.set_params(**best_params)
        try: model.steps[-1][1].n_jobs=-1
        except: pass

        return model, param_output



    def cv_score(self, model, X, y, cv=5, scoring='neg_mean_squared_error', 
                 n_jobs=1, return_mean=True):

        score = cross_val_score(model, X, y, cv=cv, n_jobs=n_jobs, scoring=scoring)
        if return_mean:
            score = np.mean(score)
        return score


    def cv_predict(self, model, cv=5):
        
        from sklearn.model_selection import KFold

        X = self.X
        y = self.y

        predictions = []
        self.y_vals = []
        self.test_idx = []

        if self.wt_col is not None: self.wts = []
        else: self.wts = None

        if self.model_obj=='class' and self.grp is not None:
            kf = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=self.randseed)
            ksplt = 'kf.split(X,y,groups=self.grp)'
   
        elif self.model_obj=='class' and self.grp is None:
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.randseed)
            ksplt = 'kf.split(X,y)'
        
        else: 
            kf = KFold(n_splits=cv, random_state=self.randseed, shuffle=True)
            ksplt = 'kf.split(X)'
        
        for tr_idx, test_idx in eval(ksplt):
            
            X_train, X_test = X.loc[tr_idx, :], X.loc[test_idx, :]
            y_train, y_test = y[tr_idx], y[test_idx]
            self.test_idx.extend(test_idx)

            if self.wt_col is not None: fit_params = self.weight_params(model, X_train[self.wt_col].values, True)
            else: fit_params = {}

            model.fit(X_train, y_train, **fit_params)
            predictions = self.model_predict(model, X_test, predictions)
       
            self.y_vals.extend(y_test)
            if self.wt_col is not None: self.wts.extend(X_test[self.wt_col].values)

        return predictions


    def cv_predict_time_holdout(self, model, sample_weight=False):

        # set up list to store validation and holdout predictions
        val_predictions = []
        hold_predictions = []

        # iterate through both the training and holdout time series indices
        for (tr_train, te_train), (_, te_hold) in zip(self.cv_time_train, self.cv_time_hold):

            # extract out the training and validation datasets from the training folds
            X_train_cur, y_train_cur = self.X_train.loc[tr_train, :], self.y_train[tr_train]
            X_val = self.X_train.loc[te_train, :]
            X_hold = self.X_hold.loc[te_hold, :]

            # fit and predict the validation dataset
            fit_params = self.weight_params(model, y_train_cur, sample_weight)
            model.fit(X_train_cur, y_train_cur, **fit_params)
    
            val_predictions = self.model_predict(model, X_val, val_predictions)
            hold_predictions = self.model_predict(model, X_hold, hold_predictions)                

        return val_predictions, hold_predictions


    @staticmethod
    def cv_time_splits(X, col, val_start):
      
        ts = X[col].unique()
        ts = ts[ts>=val_start]

        cv_time = []
        for t in ts:
            train_idx = list(X[X[col] < t].index)
            test_idx = list(X[X[col] == t].index)
            cv_time.append((train_idx, test_idx))

        return cv_time

        
    def train_test_split_time(self, X, y, X_labels, col, time_split):

        X_train_only = X[X[col] < time_split]
        y_train_only = y[X_train_only.index].reset_index(drop=True)
        X_train_only.reset_index(drop=True, inplace=True)

        X_val = X[X[col] >= time_split]
        X_labels = X_labels.loc[X_labels.index.isin(X_val.index)]

        y_val = y[X_val.index].reset_index(drop=True)
        X_val.reset_index(drop=True, inplace=True)
        X_labels.reset_index(drop=True, inplace=True)

        return X_train_only, X_val, y_train_only, y_val, X_labels


    def get_fold_data(self, X, y, X_labels, time_col, val_cut, n_splits=5, shuffle=True, random_state=1234):
  
        # split the X and y data into purely Train vs Holdout / Validation
        X_train_only, X_val_hold, y_train_only, y_val_hold, X_labels = self.train_test_split_time(X, y, X_labels, time_col, val_cut)

        # stratify split the Holdout / Validation data and append to dictionary for each fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        folds = {}; i=-1
        for val_idx, hold_idx in skf.split(X_val_hold, X_val_hold[time_col]):

            i+=1; folds[i] = {}

            # split the val/hold dataset into random validation and holdout sets
            X_val, X_hold = X_val_hold.loc[val_idx,:], X_val_hold.loc[hold_idx,:]
            y_val, y_hold = y_val_hold.loc[val_idx], y_val_hold.loc[hold_idx]

            # concat the current training set using train and validation folds
            X_train = pd.concat([X_train_only, X_val], axis=0).reset_index(drop=True)
            y_train = pd.concat([y_train_only, y_val], axis=0).reset_index(drop=True)

            folds[i]['X_train'] = X_train
            folds[i]['y_train'] = y_train

            folds[i]['X_hold'] = X_hold
            folds[i]['y_hold'] = y_hold
            
            folds[i]['X_val_labels'] = X_labels.loc[val_idx].reset_index(drop=True)
            folds[i]['X_hold_labels'] = X_labels.loc[hold_idx].reset_index(drop=True)
            
        return folds


    @staticmethod
    def unpack_fold(folds, i):
        return folds[i]['X_train'], folds[i]['y_train'], folds[i]['X_hold'], folds[i]['y_hold'], folds[i]['X_val_labels'], folds[i]['X_hold_labels']


    def get_y_val(self):
        return self.y_train.loc[self.cv_time_train[0][1][0]:]

    @staticmethod
    def weight_params(model, wt_col, sample_weight=False):
        if sample_weight:
            sweight = f'{model.steps[-1][0]}__sample_weight'
            wts = np.where(wt_col > 0, wt_col, 0)
            fit_params={sweight: wts}
        else:
            fit_params = {}
        return fit_params

    @staticmethod
    def metrics_weights(val, hold, sample_weight):
        if sample_weight:
            val_wts = val
            hold_wts = hold 
        else:
            val_wts = None
            hold_wts = None
        return val_wts, hold_wts


    def model_predict(self, model, X, predictions_list):
        
        if self.proba: pred_val = model.predict_proba(X)[:,1]
        else: pred_val = model.predict(X)
        predictions_list.extend(pred_val)
        
        return predictions_list


    def time_series_cv(self, model, X, y, params, col_split, time_split, n_splits=5, n_iter=50,
                       bayes_rand='rand', proba=False, sample_weight=False, random_seed=1234, alpha=0.5,
                       scoring=None, trials=None):
   
        X_labels = self.data[['player', 'team', 'week', 'year', 'y_act']].copy()
        folds = self.get_fold_data(X, y, X_labels, time_col=col_split, val_cut=time_split, 
                                   n_splits=n_splits, random_state=random_seed)
        
        hold_results = pd.DataFrame()
        val_results = pd.DataFrame()
        param_scores = pd.DataFrame()
        best_models = []
        i_seed = 5
        for fold in range(n_splits):

            print('-----------------')

            # get the train and holdout data
            X_train, y_train, X_hold, y_hold, X_val_labels, X_hold_labels = self.unpack_fold(folds, fold)
            cv_time_train = self.cv_time_splits(X_train, 'game_date', time_split)
            cv_time_hold = self.cv_time_splits(X_hold, 'game_date', time_split)

            fit_params = self.weight_params(model, y_train, sample_weight)
            
            if scoring is None and self.model_obj=='reg': scoring = self.scorer('sera')
            elif scoring is None and self.model_obj=='class': scoring = self.scorer('brier')
            elif scoring is None and self.model_obj=='quantile': scoring = self.scorer('pinball', **{'alpha': alpha})

            self.X_train = X_train
            self.y_train = y_train
            self.X_hold = X_hold
            self.y_hold = y_hold
            self.cv_time_train = cv_time_train
            self.cv_time_hold = cv_time_hold
            self.proba=proba
            self.randseed = random_seed * i_seed
            self.alpha = alpha
            self.trials = trials
            np.random.seed(self.randseed)

            if bayes_rand == 'bayes':
                best_model, param_scores = self.custom_bayes_search(model, params, n_iter)
            
            elif bayes_rand == 'rand':
                best_model, ps = self.custom_rand_search(model, params, n_iters=n_iter)
                param_scores = pd.concat([param_scores, ps], axis=0)

            
            best_models.append(clone(best_model))
            val_pred, hold_pred = self.cv_predict_time_holdout(best_model, sample_weight)
            
            val_wts, hold_wts = self.metrics_weights(self.get_y_val(), y_hold, sample_weight)
            y_val = self.get_y_val()
          
            _ = self.test_scores(y_val, val_pred, val_wts, label='Val')
            print('---')
            _ = self.test_scores(y_hold, hold_pred, hold_wts)

            hold_results_cur = pd.Series(hold_pred, name='pred')
            hold_results_cur = pd.concat([X_hold_labels, hold_results_cur], axis=1)
            hold_results = pd.concat([hold_results, hold_results_cur], axis=0)

            val_results_cur = pd.Series(val_pred, name='pred')
            val_results_cur = pd.concat([X_val_labels, val_results_cur], axis=1)
            val_results = pd.concat([val_results, val_results_cur], axis=0)

            i_seed += 1

        print('\nOverall\n==============')
        val_wts, hold_wts = self.metrics_weights(val_results.y_act.values, hold_results.y_act.values, sample_weight)
        
        val_score = self.test_scores(val_results.y_act, val_results.pred, val_wts, label='Val')
        print('---')
        hold_score = self.test_scores(hold_results.y_act, hold_results.pred, hold_wts, label='Test')

        oof_data = {
            'scores': [np.round(val_score,3), np.round(hold_score,3)],
            'full_val': val_results, 
            'full_hold': hold_results, 
            'hold': hold_results.pred.values,
            'actual': hold_results.y_act.values
            }

        gc.collect()
        return best_models, oof_data, param_scores, self.trials

          
    def test_scores(self, y, pred, sample_weight=None, label='Test'):

        if self.model_obj == 'reg':
            mse = mean_squared_error(y, pred, sample_weight=sample_weight)
            mae = mean_absolute_error(y, pred, sample_weight=sample_weight)
            r2 = r2_score(y, pred, sample_weight=sample_weight)
            sera = self.sera_loss(y, pred)
            
            for v, m in zip([f'{label} MSE:', f'{label} R2:', f'{label} Sera'], [mse, r2, sera]):
                print(v, np.round(m, 3))

            return 100*(self.mae_wt*mae + self.mse_wt*mse + self.sera_wt*sera - self.r2_wt*r2)

        elif self.model_obj == 'class':
            
            bs = brier_score_loss(y, pred, sample_weight=sample_weight)
            if self.proba: pred = np.int32(np.round(pred))
            matt_coef = matthews_corrcoef(y, np.where(pred>0.5,1,0), sample_weight=sample_weight)
            
            for v, m in zip(['Test MC:', 'Test Brier:'], [matt_coef, bs]):
                print(v, np.round(m, 3))

            return 100*(self.brier_wt * bs - self.matt_wt * matt_coef)

        elif self.model_obj == 'quantile':
            
            pinball = mean_pinball_loss(y, pred, sample_weight=sample_weight, alpha=self.alpha)
            print('Test Pinball Loss:', np.round(pinball,2))

            return pinball

    def return_labels(self, cols, time_or_all='time'):
        
        if time_or_all=='time':
            return self.data.loc[self.test_indices, cols]

        elif time_or_all=='all':
            return self.data.loc[:, cols]

    
    def print_coef(self, model):
        
        cols = self.X.columns

        if len([i[0] for i in model.get_params()['steps'] if i[0]=='random_sample']) > 0:
            cols = model['random_sample'].columns
            X_cur = self.X[cols].copy()
        else:
            X_cur = self.X.copy()

        if len([i[0] for i in model.get_params()['steps'] if i[0]=='k_best']) > 0:
            cols = X_cur.columns[model['k_best'].get_support()] 

        if len([i[0] for i in model.get_params()['steps'] if i[0]=='k_best_c']) > 0:
            cols = X_cur.columns[model['k_best_c'].get_support()] 

        # get out the coefficients or feature importances from the model
        try: feat_imp = pd.Series(model[-1].coef_, index=cols)
        except: pass

        try: feat_imp = pd.Series(model[-1].coef_[0], index=cols)
        except: pass

        try: feat_imp = pd.Series(model[-1].feature_importances_, index=cols)
        except: pass
        
        # print out the oefficents of stacked model
        print('\nFeature Importances\n--------\n', feat_imp)
    

    def X_y_stack(self, met, pred, actual):

        X = pd.DataFrame([v for k,v in pred.items() if met in k]).T
        X.columns = [k for k,_ in pred.items() if met in k]
        y = pd.Series(actual[X.columns[0]], name='y_act')

        return X, y

        
    def best_stack(self, model, stack_params, X_stack, y_stack, n_iter=500, 
                   print_coef=True, run_adp=False, random_state=1234, bayes_rand='rand',
                   alpha=0.5, proba=False, num_k_folds=1,  grp=None, wt_col=None, trials=None):

        self.X = X_stack
        self.y = y_stack
        self.stacking = True
        self.randseed=random_state
        self.alpha = alpha
        self.proba = proba
        self.num_k_folds = num_k_folds
        self.grp = grp
        self.wt_col = wt_col
        self.trials = trials

        if bayes_rand == 'bayes':
            best_model, _ = self.custom_bayes_search(model, stack_params, n_iter)

        elif bayes_rand == 'rand':
            best_model, _ = self.custom_rand_search(model, stack_params, n_iters=n_iter)

        if run_adp:
            # print the OOS scores for ADP and model stack
            print('ADP Score\n--------')
            adp_col = [c for c in X_stack.columns if 'adp' in c or 'odds' in c]
            if wt_col is not None: adp_col.append(wt_col)
            self.X = X_stack[adp_col]

            if self.model_obj=='reg': adp_pipe = self.model_pipe([self.piece('lr')])
            elif self.model_obj=='class': adp_pipe = self.model_pipe([self.piece('lr_c')])
            
            adp_preds = self.cv_predict(adp_pipe, cv=5)
            adp_score = self.test_scores(self.y_vals, adp_preds, sample_weight=self.wts, label='ADP')
        
        else:
            adp_score = 0
            adp_preds = 0

        print('\nStack Score\n--------')
        
        full_preds = np.empty(shape=(len(self.y), self.num_k_folds))
        y_out = np.empty(shape=(len(self.y), self.num_k_folds))
        stack_score = []
    
        for k in range(num_k_folds):

            self.randseed=random_state*3+(k*2+7)
            self.X = X_stack

            full_preds_k = self.cv_predict(best_model, cv=5)
            print(f'\nIter {k}')
            stack_score_k = self.test_scores(self.y_vals, full_preds_k, sample_weight=self.wts)
            
            full_preds[:,k] = pd.Series(full_preds_k, index=self.test_idx).sort_index().values
            stack_score.append(stack_score_k)

        y_out = pd.Series(self.y_vals, index=self.test_idx).sort_index().values
        full_preds = np.mean(full_preds, axis=1)
        stack_score = np.mean(stack_score)

        if print_coef:
            self.print_coef(best_model)

        scores = {
                  'stack_score': round(stack_score, 3),
                  'adp_score': round(adp_score, 3)
                }

        predictions = {'adp': adp_preds,
                       'stack_pred': full_preds,
                       'y': y_out}

        return best_model, scores, predictions, trials


#%%

# from sklearn.datasets import make_regression, make_classification
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

# def get_dataset(model_obj, rs, weighting=False):
#     if model_obj=='reg':
#         X, y = make_regression(n_samples=1200, n_features=60, n_informative=20, n_targets=1, bias=2, effective_rank=5, tail_strength=0.5, noise=5, random_state=rs)
#     elif model_obj=='class':
#         X, y = make_classification(n_samples=1000, n_features=60, n_informative=15, weights=(0.8,0.2), 
#                                   n_redundant=3, flip_y=0.1, class_sep = 0.5, n_clusters_per_class=2, random_state=rs)
    

#     X = pd.DataFrame(X); 
#     X.columns = [str(c) for c in X.columns]
#     y = pd.Series(y, name='y')

#     if weighting:
#         X['wt_col'] = np.random.uniform(0,1,len(X))

#     if model_obj=='class': stratify=y
#     else: stratify=None
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=rs, shuffle=True, stratify=stratify)

#     skm = SciKitModel(pd.concat([X_train, y_train], axis=1), model_obj=model_obj, r2_wt=r2_wt, sera_wt=sera_wt, matt_wt=matt_wt, brier_wt=brier_wt)

#     return skm, X_train, X_test, y_train, y_test


# def get_models(model, skm, X, y, use_rs):

#     if skm.model_obj=='reg': kb = 'k_best'
#     elif skm.model_obj=='class': kb = 'k_best_c'
#     if use_rs:
#         pipe = skm.model_pipe([
#                                     skm.piece('random_sample'),
#                                     skm.piece('std_scale'), 
#                                     skm.piece(kb),
#                                     skm.piece(model)
#                             ])
#     else:
#         pipe = skm.model_pipe([
#                                     skm.piece('std_scale'), 
#                                     skm.piece(kb),
#                                     skm.piece(model)
#                             ])

#     params = skm.default_params(pipe, bayes_rand='bayes')
#     # if use_rs:
#     #     params['random_sample__frac'] = np.arange(0.3, 1, 0.05)
    
#     # params[f'{kb}__k'] = range(1, 60)

#     return pipe, params

# def show_calibration_curve(y_true, y_pred, n_bins=10, strategy='uniform'):

#     from sklearn.calibration import calibration_curve
    
#     x, y = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy=strategy)

#     # Plot perfectly calibrated
#     plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')
    
#     # Plot model's calibration curve
#     plt.plot(y, x, marker = '.', label = 'Model')
    
#     leg = plt.legend(loc = 'upper left')
#     plt.xlabel('Average Predicted Probability in each bin')
#     plt.ylabel('Ratio of positives')
#     plt.show()

# def show_results(skm, best_model, X_train, y_train, X_test, y_test, wts=None):
#     best_model.fit(X_train, y_train)

#     if model_obj=='reg': 
#         y_test_pred = best_model.predict(X_test)
#         _ = skm.test_scores(y_test, y_test_pred, sample_weight=wts)
#         plt.scatter(y_test_pred, y_test)
#     else: 
#         y_test_pred = best_model.predict_proba(X_test)[:,1]
#         _ = skm.test_scores(y_test, y_test_pred, sample_weight=wts)
#         show_calibration_curve(y_test, y_test_pred, n_bins=8)
        

# i = 1

# rs = 12901235
# num_k_folds = 1
# use_random_sample = True
# model = 'lr_c'
# model_obj = 'class'
# calibrate = False

# r2_wt=0; sera_wt=1
# matt_wt=1; brier_wt=5

# skm, X_train, X_test, y_train, y_test = get_dataset(model_obj, rs=rs, weighting=True)
# X_train = X_train.reset_index(drop=True)
# y_train = y_train.reset_index(drop=True)
# X_test = X_test.reset_index(drop=True)
# y_test = y_test.reset_index(drop=True)

# pipe, params = get_models(model, skm, X_train, y_train, use_random_sample)



# if model_obj=='class': proba=True 
# else: proba=False
# best_model, stack_scores, stack_pred = skm.best_stack(pipe, params,
#                                                       X_train, y_train, n_iter=25, 
#                                                       run_adp=False, print_coef=False,
#                                                     proba=proba,
#                                                       random_state=(i*12)+(i*17), num_k_folds=num_k_folds, 
#                                                       calibrate=calibrate,
#                                                       wt_col='wt_col')


# print(best_model)
# if model_obj=='reg': print('Sera Wt', skm.sera_wt, 'R2 Wt', skm.r2_wt)
# if model_obj=='class': print('Matt Wt', skm.matt_wt, 'Brier Wt', skm.brier_wt)
# print('Num K Folds:', skm.num_k_folds)
# print('Calibrate:', calibrate)
# print('Use Random Sample:', use_random_sample)
# print('\nOut of Sample Results\n--------------')

# # show_calibration_curve(stack_pred['y'], stack_pred['stack_pred'], n_bins=8, strategy='quantile')
# show_results(skm, best_model, X_train, y_train, X_test, y_test, wts=X_test['wt_col'])
# if model_obj=='class': show_calibration_curve(stack_pred['y'], stack_pred['stack_pred'], n_bins=8)

# %%

# # core packages
# import pandas as pd
# import numpy as np
# import os
# import gzip
# import pickle
# import datetime as dt
# import matplotlib.pyplot as plt
# import time
# from hyperopt import Trials
# from wakepy import keep

# from ff.db_operations import DataManage
# from ff import general as ffgeneral
# # from skmodel import SciKitModel
# import zModel_Functions as mf
# from joblib import Parallel, delayed

# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning) 
# warnings.filterwarnings("ignore", category=UserWarning) 

# #==========
# # General Setting
# #==========

# # set the root path and database management object
# root_path = ffgeneral.get_main_path('Daily_Fantasy')
# db_path = f'{root_path}/Data/Databases/'
# dm = DataManage(db_path)

# #---------------
# # Settings
# #---------------

# run_weeks = [17]
# verbosity = 50
# run_params = {
    
#     # set year and week to analyze
#     'set_year': 2022,

#     # set beginning of validation period
#     'val_year_min': 2020,
#     'val_week_min': 14,

#     # opt params
#     'n_iters': 20,
#     'n_splits': 5,

#     # other parameters
#     'use_sample_weight': False,
#     'opt_type': 'bayes',
#     'cuts': [33, 80, 95],
#     'met': 'y_act',

#     # set number of weeks back to begin validation
#     'back_weeks': {
#         'QB': 32,
#         'RB': 32,
#         'WR': 32,
#         'TE': 32,
#         'Defense': 32
#     },

#     'rush_pass': ''
# }

# n_splits = run_params['n_splits']

# # set position and model type
# set_pos = 'RB'
# model_type = 'full_model'

# # set weights for running model
# r2_wt = 0
# sera_wt = 1
# matt_wt = 0
# brier_wt = 1

# # set version and iterations
# vers = 'sera1_rsq0_brier1_matt0_bayes'

# #----------------
# # Data Loading
# #----------------

# def create_pkey_output_path(set_pos, run_params, model_type, vers):

#     pkey = f"{set_pos}_year{run_params['set_year']}_week{run_params['set_week']}_{model_type}{vers}"
#     db_output = {
#                 'pkey': [], 
#                 'set_pos': [], 
#                 'set_year': [], 
#                 'set_week':[], 
#                 'model_type':[],
#                 'model': [],
#                 'validation_score': [],
#                 'test_score': []
#                 }

#     model_output_path = f"{root_path}/Model_Outputs/{run_params['set_year']}/{pkey}/"
#     if not os.path.exists(model_output_path): os.makedirs(model_output_path)
    
#     return pkey, db_output, model_output_path

# def load_data(model_type, set_pos, run_params):

#     # load data and filter down
#     if model_type=='full_model': df = dm.read(f'''SELECT * 
#                                                   FROM {set_pos}_Data{run_params['rush_pass']}
#                                                 ''', 'Model_Features')
#     elif model_type=='backfill': df = dm.read(f'''SELECT * 
#                                                   FROM Backfill 
#                                                   WHERE pos='{set_pos}' ''', 'Model_Features')

#     if df.shape[1]==2000:
#         df2 = dm.read(f'''SELECT * 
#                           FROM {set_pos}_Data{run_params['rush_pass']}2
#                        ''', 'Model_Features')
#         df = pd.concat([df, df2], axis=1)

#     df = df.sort_values(by=['year', 'week']).reset_index(drop=True)
#     if set_pos == 'Defense':
#         df['team'] = df.player
    
#     df = df.dropna(axis=1)
#     drop_cols = list(df.dtypes[df.dtypes=='object'].index)
#     run_params['drop_cols'] = drop_cols
#     print(drop_cols)

#     return df, run_params

# def year_week_to_date(x):
#     return int(dt.datetime(x[0], 1, x[1]).strftime('%Y%m%d'))


# def get_cv_time_input(df, back_weeks):
#     df = df[(df.year < run_params['set_year']) | \
#             (
#               (df.year==run_params['set_year']) & \
#               (df.week<=run_params['set_week'])
#             )]
#     max_date = str(df.game_date.max())
#     year = int(max_date[:4])
#     week = int(max_date[-2:])

#     for i in range(back_weeks):
#         if week > 1:
#             week -= 1
#         else:
#             year -= 1
#             week = 17
#     cv_time_input = int(dt.datetime(year, 1, week).strftime('%Y%m%d'))
#     cv_time_input = np.max([cv_time_input, 20200114])
#     print(f'Begin Validation Using {cv_time_input}')
#     return cv_time_input


# def create_game_date(df, run_params):
    
#     # set up the date column for sorting
#     df['game_date'] = df[['year', 'week']].apply(year_week_to_date, axis=1)
#     cv_time_input = get_cv_time_input(df, run_params['back_weeks'][set_pos])
#     train_time_split = int(dt.datetime(run_params['set_year'], 1, run_params['set_week']).strftime('%Y%m%d'))

#     run_params['cv_time_input'] = cv_time_input
#     run_params['train_time_split'] = train_time_split

#     return df, run_params


# def train_predict_split(df, run_params):

#     # # get the train / predict dataframes and output dataframe
#     df_train = df[df.game_date < run_params['train_time_split']].reset_index(drop=True)
#     df_train = df_train.dropna(subset=['y_act']).reset_index(drop=True)

#     df_predict = df[df.game_date == run_params['train_time_split']].reset_index(drop=True)
#     output_start = df_predict[['player', 'dk_salary', 'fantasyPoints', 'projected_points', 'ProjPts']].copy().drop_duplicates()

#     # get the minimum number of training samples for the initial datasets
#     min_samples = int(df_train[df_train.game_date < run_params['cv_time_input']].shape[0])  
#     print('Shape of Train Set', df_train.shape)

#     return df_train, df_predict, output_start, min_samples


# def get_class_data(df, cut, run_params):

#     # set up the training and prediction datasets for the classification 
#     df_train_class = df[df.game_date < run_params['train_time_split']].reset_index(drop=True)
#     df_predict_class = df[df.game_date == run_params['train_time_split']].reset_index(drop=True)

#     # set up the target variable to be categorical based on Xth percentile
#     cut_perc = df_train_class.groupby('game_date')['y_act'].apply(lambda x: np.percentile(x, cut))
#     df_train_class = pd.merge(df_train_class, cut_perc.reset_index().rename(columns={'y_act': 'cut_perc'}), on=['game_date'])
    
#     if cut > 33:
#         df_train_class['y_act'] = np.where(df_train_class.y_act >= df_train_class.cut_perc, 1, 0)
#     else: 
#         df_train_class['y_act'] = np.where(df_train_class.y_act <= df_train_class.cut_perc, 1, 0)

#     df_train_class = df_train_class.drop('cut_perc', axis=1)

#     return df_train_class, df_predict_class
    

# #----------------
# # Modeling Functions
# #----------------

# def output_dict():
#     return {'pred': {}, 'actual': {}, 'scores': {}, 'models': {}, 'full_hold':{}, 'param_scores': {}, 'trials': {}}


# def get_trial_times(root_path, run_params, set_pos, model_type, vers):

#     newest_folder = get_newest_folder(f"{root_path}/Model_Outputs/")
#     recent_save = get_newest_folder_with_keywords(newest_folder, [set_pos, model_type, vers], [f"_week{run_params['set_week']}_"])

#     all_trials = load_pickle(recent_save, 'all_trials')

#     times = []
#     for k,v in all_trials.items():
#         if k!='reg_adp':
#             max_trial = len(v.trials) - 1
#             trial_times = []
#             for i in range(max_trial-100, max_trial):
#                 trial_times.append(v.trials[i]['refresh_time'] - v.trials[i]['book_time'])
#             trial_time = np.mean(trial_times).seconds
#             times.append([k, np.round(trial_time / 60, 2)])

#     time_per_trial = pd.DataFrame(times, columns=['model', 'time_per_trial']).sort_values(by='time_per_trial', ascending=False)
#     time_per_trial['total_time'] = time_per_trial.time_per_trial * 100
#     return time_per_trial

# def calc_num_trials(time_per_trial, run_params):

#     n_iters = run_params['n_iters']
#     time_per_trial['percentile_90_time'] = time_per_trial.time_per_trial.quantile(0.9)
#     time_per_trial['num_trials'] = n_iters * time_per_trial.percentile_90_time / time_per_trial.time_per_trial
#     time_per_trial['num_trials'] = time_per_trial.num_trials.apply(lambda x: np.min([n_iters, np.max([x, n_iters/2])])).astype('int')
    
#     return {k:v for k,v in zip(time_per_trial.model, time_per_trial.num_trials)}

# def reg_params(df_train, min_samples, num_trials, run_params):
#     model_list = ['adp', 'bridge', 'gbm', 'gbmh', 'rf', 'lgbm', 'ridge', 'svr', 'lasso', 'enet', 'knn','xgb']
#     label = 'reg'
#     func_params_reg = []
#     for i, m  in enumerate(model_list):
#         try: num_trial = num_trials[f'{label}_{m}']
#         except: num_trial = run_params['n_iters']
#         func_params_reg.append([m, label, df_train, 'reg', i, min_samples, '', num_trial])

#     return func_params_reg

# def class_params(df, min_samples, num_trials, run_params):
#     model_list = ['gbm_c', 'rf_c','gbmh_c', 'lgbm_c', 'lr_c', 'knn_c','xgb_c']
#     func_params_c = []
#     for cut in run_params['cuts']:
#         label = f'class_{cut}'
#         df_train_class, _ = get_class_data(df, cut, run_params) 
#         for i, m  in enumerate(model_list):
#             try: num_trial = num_trials[f'{label}_{m}']
#             except: num_trial = run_params['n_iters']
#             func_params_c.append([m, label, df_train_class, 'class', i, min_samples, '', num_trial])

#     return func_params_c

# def quant_params(df_train, alphas, min_samples, num_trials, run_params):
#     model_list =  ['rf_q','qr_q','lgbm_q','knn_q', 'gbm_q']
#     func_params_q = []
#     for alph in alphas:
#         label = f'quant_{alph}'
#         for i, m  in enumerate(model_list):
#             try: num_trial = num_trials[f'{label}_{m}']
#             except: num_trial = run_params['n_iters']
#             func_params_q.append([m, label, df_train, 'quantile', i, min_samples, alph, num_trial])

#     return func_params_q

# def million_params(df, num_trials, run_params):
#     model_list = ['gbm_c', 'rf_c', 'gbmh_c', 'lgbm_c', 'lr_c', 'knn_c','xgb_c' ]
#     label = 'million'
#     df_train_mil, _, min_samples_mil = predict_million_df(df, run_params)

#     func_params_mil = []
#     for i, m  in enumerate(model_list):
#         try: num_trial = num_trials[f'{label}_{m}']
#         except: num_trial = run_params['n_iters']
#         func_params_mil.append([m, label, df_train_mil, 'class', i, min_samples_mil, '', num_trial])

#     return func_params_mil

# def order_func_params(func_params, trial_times):
    
#     trial_order = list(trial_times.model.values) + ['reg_adp']
#     func_params = sorted(func_params, key=lambda x: trial_order.index(f'{x[1]}_{x[0]}'))
#     return func_params

# def get_skm(skm_df, model_obj, to_drop):
    
#     skm_options = {
#         'reg': SciKitModel(skm_df, model_obj='reg', r2_wt=r2_wt, sera_wt=sera_wt),
#         'class': SciKitModel(skm_df, model_obj='class', brier_wt=brier_wt, matt_wt=matt_wt),
#         'quantile': SciKitModel(skm_df, model_obj='quantile')
#     }
    
#     skm = skm_options[model_obj]
#     X, y = skm.Xy_split(y_metric='y_act', to_drop=to_drop)

#     return skm, X, y


# def get_full_pipe(skm, m, alpha=None, stack_model=False, min_samples=10, bayes_rand='rand'):

#     if m == 'adp':
        
#         # set up the ADP model pipe
#         pipe = skm.model_pipe([skm.piece('feature_select'), 
#                                skm.piece('std_scale'), 
#                                skm.piece('k_best'),
#                                skm.piece('lr')])

#     elif stack_model:
#         pipe = skm.model_pipe([
#                             skm.piece('std_scale'),
#                             skm.piece('k_best'), 
#                             skm.piece(m)
#                         ])

#     elif skm.model_obj == 'reg':
#         pipe = skm.model_pipe([skm.piece('random_sample'),
#                                 skm.piece('std_scale'), 
#                                 skm.piece('select_perc'),
#                                 skm.feature_union([
#                                                 skm.piece('agglomeration'), 
#                                                 skm.piece('k_best'),
#                                                 skm.piece('pca')
#                                                 ]),
#                                 skm.piece('k_best'),
#                                 skm.piece(m)])

#     elif skm.model_obj == 'class':
#         pipe = skm.model_pipe([skm.piece('random_sample'),
#                                skm.piece('std_scale'), 
#                                skm.piece('select_perc_c'),
#                                skm.feature_union([
#                                                 skm.piece('agglomeration'), 
#                                                 skm.piece('k_best_c'),
#                                                 ]),
#                                skm.piece('k_best_c'),
#                                skm.piece(m)])
    
#     elif skm.model_obj == 'quantile':
#         pipe = skm.model_pipe([
#                                 skm.piece('random_sample'),
#                                 skm.piece('std_scale'), 
#                                 skm.piece('k_best'), 
#                                 skm.piece(m)
#                                 ])

#         if m == 'qr_q': pipe.steps[-1][-1].quantile = alpha
#         elif m in ('rf_q', 'knn_q'): pipe.steps[-1][-1].q = alpha
#         else: pipe.steps[-1][-1].alpha = alpha
    


#     # get the params for the current pipe and adjust if needed
#     params = skm.default_params(pipe, bayes_rand, min_samples=min_samples)
#     if m=='adp': 
#         params['feature_select__cols'] = [
#                                             ['game_date', 'year', 'week', 'ProjPts', 'dk_salary', 'fd_salary', 'projected_points', 'fantasyPoints', 'ffa_points', 'avg_proj_points', 'fc_proj_fantasy_pts_fc', 'log_fp_rank', 'log_avg_proj_rank'],
#                                             ['year', 'week',  'ProjPts', 'dk_salary', 'fd_salary', 'projected_points', 'fantasyPoints', 'ffa_points', 'avg_proj_points', 'fc_proj_fantasy_pts_fc', 'log_fp_rank', 'log_avg_proj_rank'],
#                                             [ 'ProjPts', 'dk_salary', 'fd_salary', 'projected_points', 'fantasyPoints', 'ffa_points', 'avg_proj_points', 'fc_proj_fantasy_pts_fc', 'log_fp_rank', 'log_avg_proj_rank']
#                                         ]
#         params['k_best__k'] = range(1, 14)
    
#     return pipe, params

# def update_trials_params(trials, m, params, pipe):

#     m_params = [p.split('__')[1] for p in params.keys() if m in p]

#     hyper_params = {
#         p: pipe.steps[-1][1].get_params()[p] for p in m_params
#     }

#     for trial in trials:
#         # Update each trial with the new hyperparameters
#         for k,v in hyper_params.items():
#             if k not in trial['misc']['vals']:
#                 trial['misc']['vals'][k] = [v]
#                 trial['misc']['idxs'][k] = [trial['tid']]

#     return trials

# def get_proba(model_obj):
#     if model_obj == 'class': proba = True
#     else: proba = False
#     return proba

# def adjust_adp(model_name):
#     if model_name == 'adp': bayes_rand = 'rand'
#     else: bayes_rand = run_params['opt_type']
#     return bayes_rand

# def get_newest_folder(path):
#     folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
#     newest_folder = max(folders, key=lambda f: os.path.getctime(os.path.join(path, f)))
#     return os.path.join(path, newest_folder)

# def get_newest_folder_with_keywords(path, keywords, ignore_keywords=None):
#     folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    
#     # Apply ignore_keywords if provided
#     if ignore_keywords:
#         folders = [f for f in folders if not any(ignore_keyword in f for ignore_keyword in ignore_keywords)]
    
#     matching_folders = [f for f in folders if all(keyword in f for keyword in keywords)]
    
#     if not matching_folders:
#         return None
    
#     newest_folder = max(matching_folders, key=lambda f: os.path.getctime(os.path.join(path, f)))
#     return os.path.join(path, newest_folder)


# def get_trials(label, m, bayes_rand):

#     newest_folder = get_newest_folder(f"{root_path}/Model_Outputs/")
#     recent_save = get_newest_folder_with_keywords(newest_folder, [set_pos, model_type, vers], [f"_week{run_params['set_week']}_"])

#     if recent_save is not None and bayes_rand=='bayes': 
#         try:
#             trials = load_pickle(recent_save, 'all_trials')
#             trials = trials[f'{label}_{m}']
#             print('Loading previous trials')
#         except:
#             print('No Previous Trials Exist')
#             trials = Trials()

#     elif bayes_rand=='bayes':
#         print('Creating new Trials object')
#         trials = Trials()

#     else:
#         trials = None

#     return trials
    

# def get_model_output(model_name, label, cur_df, model_obj, run_params, i, min_samples=10, alpha='', n_iter=20):

#     print(f'\n{model_name}\n============\n')
    
#     bayes_rand = adjust_adp(model_name)
#     proba = get_proba(model_obj)
#     trials = get_trials(label, model_name, bayes_rand)

#     skm, X, y = get_skm(cur_df, model_obj, to_drop=run_params['drop_cols'])
#     pipe, params = get_full_pipe(skm, model_name, alpha, min_samples=min_samples, bayes_rand=bayes_rand)
#     if trials is not None: trials = update_trials_params(trials, model_name, params, pipe)

#     # fit and append the ADP model
#     start = time.time()
#     best_models, oof_data, param_scores, trials = skm.time_series_cv(pipe, X, y, params, n_iter=n_iter, 
#                                                                      n_splits=run_params['n_splits'], col_split='game_date', 
#                                                                      time_split=run_params['cv_time_input'],
#                                                                      bayes_rand=bayes_rand, proba=proba, trials=trials,
#                                                                      random_seed=(i+7)*19+(i*12)+6, alpha=alpha)

#     print('Time Elapsed:', np.round((time.time()-start)/60,1), 'Minutes')
    
#     return best_models, oof_data, param_scores, trials


# #------------------
# # Million Predict
# #------------------

# def select_main_slate_teams(df):

#     import datetime as dt

#     good_teams = dm.read(f'''
#                     SELECT away_team team, gametime, week, year 
#                     FROM Gambling_Lines 
#                     WHERE year >= 2020
#                     UNION
#                     SELECT home_team team, gametime, week, year 
#                     FROM Gambling_Lines
#                     WHERE year >= 2020
#                 ''', 'Pre_TeamData')


#     good_teams.gametime = pd.to_datetime(good_teams.gametime)
#     good_teams['day_of_week'] = good_teams.gametime.apply(lambda x: x.weekday())
#     good_teams['hour_in_day'] = good_teams.gametime.apply(lambda x: x.hour)

#     good_teams = good_teams[(good_teams.day_of_week==6) & (good_teams.hour_in_day <= 17) & (good_teams.hour_in_day > 10)]
#     good_teams = good_teams[['team', 'week', 'year']]

#     if set_pos == 'Defense': 
#         good_teams = good_teams.rename(columns={'team': 'player'})
#         df = pd.merge(df, good_teams, on=['player', 'week', 'year'])

#     else:
#         df = pd.merge(df, good_teams, on=['team', 'week', 'year'])

#     return df


# def add_sal_columns(df):
    
#     for c in df.columns:
#         if 'expert' in c: df[c+'_salary'] = df[c] * df.dk_salary
#         if 'rank' in c: df[c+'_salary'] = df[c] * df.dk_salary
#         if 'proj' in c: df[c+'_salary'] = df[c] / df.dk_salary
#         if 'ffa' in c: df[c+'_salary'] = df[c] / df.dk_salary
    
#     return df

# def predict_million_df(df, run_params):

#     df = select_main_slate_teams(df)

#     df = df.drop('y_act', axis=1)
#     top_players = dm.read("SELECT player, week, year, y_act FROM Top_Players", "DK_Results")

#     df = pd.merge(df, top_players, on=['player', 'week', 'year'], how='left')
#     df = df.fillna({'y_act': 0})

#     df_train_mil, df_predict_mil, _, min_samples_mil = train_predict_split(df, run_params)

#     return df_train_mil, df_predict_mil, min_samples_mil


# def predict_million_df_roi(df, run_params):

#     df = select_main_slate_teams(df)

#     df = df.drop('y_act', axis=1)
#     top_players = dm.read('''SELECT player, week, year, prize_return_delta y_act
#                              FROM Top_Players_ROI
#                              WHERE total_lineups > 250
#                           ''', "DK_Results")

#     df = pd.merge(df, top_players, on=['player', 'week', 'year'], how='inner')

#     df_train_mil, df_predict_mil, _, min_samples_mil = train_predict_split(df, run_params)

#     return df_train_mil, df_predict_mil, min_samples_mil, run_params


# #-----------------
# # Saving Data / Handling
# #-----------------

# def update_output_dict(out_dict, label, m, result):

#     best_models, oof_data, param_scores, trials = result

#     # append all of the metric outputs
#     lbl = f'{label}_{m}'
#     out_dict['pred'][lbl] = oof_data['hold']
#     out_dict['actual'][lbl] = oof_data['actual']
#     out_dict['scores'][lbl] = oof_data['scores']
#     out_dict['models'][lbl] = best_models
#     out_dict['full_hold'][lbl] = oof_data['full_hold']
#     out_dict['param_scores'][lbl] = param_scores
#     out_dict['trials'][lbl] = trials

#     return out_dict


# def unpack_results(out_dict, func_params, results):
#     for fp, result in zip(func_params, results):
#         model_name, label, _, _, _, _, _, _ = fp
#         out_dict = update_output_dict(out_dict, label, model_name, result)
#     return out_dict


# def save_pickle(obj, path, fname, protocol=-1):
#     with gzip.open(f"{path}/{fname}.p", 'wb') as f:
#         pickle.dump(obj, f, protocol)

#     print(f'Saved {fname} to path {path}')


# def load_pickle(path, fname):
#     with gzip.open(f"{path}/{fname}.p", 'rb') as f:
#         loaded_object = pickle.load(f)
#         return loaded_object


# def save_output_dict(out_dict, label, model_output_path):

#     label = label.split('_')[0]
#     save_pickle(out_dict['pred'], model_output_path, f'{label}_pred')
#     save_pickle(out_dict['actual'], model_output_path, f'{label}_actual')
#     save_pickle(out_dict['models'], model_output_path, f'{label}_models')
#     save_pickle(out_dict['scores'], model_output_path, f'{label}_scores')
#     save_pickle(out_dict['full_hold'], model_output_path, f'{label}_full_hold')
#     save_pickle(out_dict['param_scores'], model_output_path, f'{label}_param_scores')
#     save_pickle(out_dict['trials'], model_output_path, f'{label}_trials')



#  #%%
# with keep.running() as m:
#     if not m.success:
#         print('Fell Asleep')

#     run_list = [
#                 # ['QB', '', 'full_model'],
#                 # ['RB', '', 'full_model'],
#                 # ['WR', '', 'full_model'],
#                 # ['TE', '', 'full_model'],
#                 # ['Defense', '', 'full_model'],
#                 # ['QB', '', 'backfill'],
#                 ['RB', '', 'backfill'],
#                 # ['WR', '', 'backfill'],
#                 # ['TE', '', 'backfill'],
#     ]

#     for w in run_weeks:
#         run_params['set_week'] = w

#         for set_pos, rush_pass, model_type in run_list:

#             run_params['rush_pass'] = rush_pass
#             run_params['n_splits'] = n_splits
#             print(f"\n==================\n{set_pos} {model_type} {rush_pass} {run_params['set_year']} {run_params['set_week']} {vers}\n====================")

#             #==========
#             # Pull and clean compiled data
#             #==========

#             # load data and filter down
#             pkey, db_output, model_output_path = create_pkey_output_path(set_pos, run_params, model_type, vers)
#             df, run_params = load_data(model_type, set_pos, run_params)
#             df, run_params = create_game_date(df, run_params)

#             df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)

#             try:
#                 trial_times = get_trial_times(root_path, run_params, set_pos, model_type, vers)
#                 num_trials = calc_num_trials(trial_times, run_params)
#                 print('Lower trials:', {k:v for k,v in num_trials.items() if v < run_params['n_iters']})
#             except:
#                 num_trials = None

#             func_params = []
#             func_params.extend(quant_params(df_train, [0.8, 0.95], min_samples,  num_trials, run_params))
#             # func_params.extend(reg_params(df_train, min_samples, num_trials, run_params))
#             # func_params.extend(class_params(df, min_samples, num_trials, run_params))
#             # func_params.extend(million_params(df, num_trials, run_params))
#             func_params = order_func_params(func_params, trial_times)
            
#             # # run all models in parallel
#             # results = Parallel(n_jobs=-1, verbose=verbosity)(
#             #                 delayed(get_model_output)
#             #                 (m, label, df, model_obj, run_params, i, min_samples, alpha, n_iter) \
#             #                     for m, label, df, model_obj, i, min_samples, alpha, n_iter in func_params
#             #                 )
            
#             # # save output for all models
#             # out_dict = output_dict()
#             # out_dict = unpack_results(out_dict, func_params, results)
#             # save_output_dict(out_dict, 'all', model_output_path)

# # %%
# run_params['opt_type'] = 'rand'
# for m, label, df, model_obj, i, min_samples, alpha, n_iter in reversed(func_params):
#     print(m)
#     get_model_output(m, label, df, model_obj, run_params, i, min_samples, alpha, n_iter)
# # %%
