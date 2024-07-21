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
import warnings
np.warnings = warnings

# model and search setup
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, StratifiedGroupKFold, GroupKFold
from sklearn.feature_selection import f_regression, f_classif, r_regression, chi2, mutual_info_regression, mutual_info_classif
from hyperopt import fmin, hp, atpe, tpe, Trials, space_eval
from hyperopt.pyll import scope
import optuna

from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor, LGBMClassifier

from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import matthews_corrcoef, f1_score, brier_score_loss, log_loss
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
        self.sera_wt = kwargs.get('sera_wt', 0)
        self.mse_wt = kwargs.get('mse_wt', 1)
        self.matt_wt = kwargs.get('matt_wt', 0)
        self.brier_wt = kwargs.get('brier_wt', 1)
        self.mae_wt = kwargs.get('mae_wt', 0)
        self.logloss_wt = kwargs.get('logloss_wt', 0)

        self.hp_algo = kwargs.get('hp_algo', 'tpe')
        

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

        elif bayes_rand=='optuna':
            if var_type=='int': return [var_type, low, high]
            if var_type=='real': return [var_type, low, high]
            if var_type=='cat': return [var_type, low]
            if var_type=='log': return [var_type, low, high]


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
            'random_sample': {
                              'frac': self.param_range('real', 0.25, 0.65, 0.02, br, 'frac'),
                              'seed': self.param_range('int', 0, 10, 1, br, 'seed'),
                              },

            'agglomeration': {
                              'n_clusters': self.param_range('int', 2, 25, 2, br, 'n_clusters')
                              },

            'pca': {
                'n_components': self.param_range('int', 2, 20, 2, br, 'n_components')
                },

            'k_best': {
                'k': self.param_range('int', 20, 125, 5, br, 'k'),
            #    'score_func': self.param_range('cat', [f_regression, r_regression], None, None, br, 'kb_score_func')
                },

            'k_best_fu': {
                'k': self.param_range('int', 20, 125, 5, br, 'k_fu'),
                # 'score_func': self.param_range('cat', [f_regression, r_regression], None, None, br, 'kb_score_func_fu')
                },

            'select_perc': {
               'percentile': self.param_range('int', 20, 55, 3, br, 'select_perc'),
            #    'score_func': self.param_range('cat', [f_regression, r_regression], None, None, br, 'sp_score_func')
                },

            'k_best_c': {
                'k': self.param_range('int', 20, 125, 5, br, 'k_best_c'),
                # 'score_func': self.param_range('cat', [f_classif], None, None, br, 'kb_score_func')
                },

            'k_best_c_fu': {
                'k': self.param_range('int', 20, 125, 5, br, 'k_best_c_fu'),
                # 'score_func': self.param_range('cat', [f_classif], None, None, br, 'kb_score_func')
                },

            'select_perc_c': {
               'percentile': self.param_range('int', 20, 55, 3, br, 'select_perc_c'),
              # 'score_func': self.param_range('cat', [f_classif, mutual_info_classif], None, None, br, 'sp_score_func')
                },

            'select_from_model': {
                'estimator': self.param_range('cat', [Ridge(alpha=0.1), Ridge(alpha=10), Lasso(alpha=0.1), Lasso(alpha=10), 
                                                      LGBMRegressor(), LGBMRegressor(learning_rate=0.3, num_leaves=50, subsample=0.8)],
                                              None, None, br, 'estimator'),
                'threshold': self.param_range('cat', ['mean', '0.5*mean', '1.5*mean'],None, None, br, 'threshold'),
                'max_features': self.param_range('int', 20, 125, 5, br, 'max_features')
                              },

            'select_from_model_c': {
                'estimator': self.param_range('cat', [LogisticRegression(C=0.1), LogisticRegression(C=1), LogisticRegression(C=10),
                                                      LGBMRegressor(), LGBMRegressor(learning_rate=0.3, num_leaves=50, subsample=0.8)],
                                               None, None, br, 'estimator'),
                'threshold': self.param_range('cat', ['mean', '0.5*mean', '1.5*mean'],None, None, br, 'threshold'),
                'max_features': self.param_range('int', 20, 125, 5, br, 'max_features')
                              },

            'feature_drop': {
                'col': self.param_range('cat', ['avg_pick', None], None, None, br, 'feature_drop')
                },

            'feature_select': {
                'cols': self.param_range('cat', [['avg_pick'], ['avg_pick', 'year']], None, None, br, 'feature_select')
                },

            'corr_collinear': {
                'corr_percentile': self.param_range('real', 0, 75, 5, br, 'corr_percentile'),
                'collinear_threshold': self.param_range('real', 0.75, 0.95, 0.02, br, 'collinear_threshold'),
                'corr_type': self.param_range('cat', ['pearson', 'spearman'], None, None, br, 'corr_type'),
                'abs_collinear': self.param_range('cat', [True, False], None, None, br, 'abs_collinear')
            },

            # model params
            'ridge': {
                        'alpha': self.param_range('log', -2, 6, 0.1, br, 'alpha')
                     },

            'lasso': {
                        'alpha': self.param_range('log', -4, 4, 0.05, br, 'alpha')
                    },

            'enet': {
                    'alpha': self.param_range('log', -4, 4, 0.1, br, 'alpha'),
                    'l1_ratio': self.param_range('real', 0.1, 0.9, 0.03, br, 'l1_ratio')
                    },

            'huber': {
                    'alpha': self.param_range('log', -3, 3, 0.05, br, 'alpha'),
                    'epsilon': self.param_range('real', 1.2, 1.5, 0.03, br, 'epsilon'),
                    'max_iter': self.param_range('int', 50, 200, 10, br, 'max_iter')
                    },

            'lr_c': {
                    'C': self.param_range('log', -4, 4, 0.1, br, 'C'),
                  #  'class_weight': [{0: i, 1: 1} for i in np.arange(0.2, 1, 0.1)]
                    },
            
            'qr_q': {
                        'alpha': self.param_range('log', -3, 3, 0.05, br, 'alpha'),
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
                        'learning_rate': self.param_range('log', -4, 0, 0.1, br,  'learning_rate'),
                        'colsample_bytree': self.param_range('real', 0.2, 1, 0.05, br, 'colsample_bytree'),
                        'subsample':  self.param_range('real', 0.4, 1, 0.1, br, 'subsample'),
                        'min_child_samples': self.param_range('int', 1, 50, 2, br, 'min_child_samples')
                        
                    },

            'lgbm_c': {
                      'max_depth': self.param_range('int', 2, 15, 2, br, 'max_depth'),
                      'num_leaves': self.param_range('int', 20, 100, 5, br, 'num_leaves'),
                      'learning_rate': self.param_range('log', -4, 0, 0.1, br,  'learning_rate'),
                      'colsample_bytree': self.param_range('real', 0.2, 1, 0.05, br, 'colsample_bytree'),
                      'subsample':  self.param_range('real', 0.4, 1, 0.1, br, 'subsample'),
                      'min_child_samples': self.param_range('int', 1, 50, 2, br, 'min_child_samples')
                     },

            'lgbm_q': {
                        'max_depth': self.param_range('int', 2, 15, 2, br, 'max_depth'),
                        'num_leaves': self.param_range('int', 20, 100, 5, br, 'num_leaves'),
                        'learning_rate': self.param_range('log', -4, 0, 0.1, br,  'learning_rate'),
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
                     'learning_rate': self.param_range('log', -4, -0.5, 0.1, br, 'learning_rate'),
                     },

            'xgb_c': {
                     'n_estimators': self.param_range('int', 30, 170, 10, br, 'n_estimators'),
                     'max_depth': self.param_range('int', 2, 20, 3, br, 'max_depth'),
                     'colsample_bytree': self.param_range('real', 0.2, 1, 0.25, br, 'colsample_bytree'),
                     'subsample':  self.param_range('real', 0.4, 1, 0.1, br, 'subsample'),
                     'reg_lambda': self.param_range('log', 0, 3, 0.1, br, 'reg_lambda'),
                     'learning_rate': self.param_range('log', -4, -0.5, 0.1, br, 'learning_rate'),
                    #  'scale_pos_weight': self.param_range('real', 1, 20, 1, br, 'scale_pos_weight')
                     },

            'gbm': {
                    'n_estimators': self.param_range('int', 20, 90, 3, br, 'n_estimators'),
                    'max_depth': self.param_range('int', 2, 18, 2, br,'max_depth'),
                    'min_samples_leaf': self.param_range('int', 5, 20, 2, br, 'min_samples_leaf'),
                    'max_features': self.param_range('real', 0.7, 1, 0.1, br, 'max_features'),
                    'subsample': self.param_range('real', 0.5, 1, 0.1, br, 'subsample'),
                    'learning_rate': self.param_range('log', -4, -0.5, 0.1, br, 'learning_rate'),
                    },

            'gbm_c': {
                    'n_estimators': self.param_range('int', 10, 80, 10, br, 'n_estimators'),
                    'max_depth': self.param_range('int', 2, 18, 2, br, 'max_depth'),
                    'min_samples_leaf': self.param_range('int', 5, 15, 1, br, 'min_samples_leaf'),
                    'max_features': self.param_range('real', 0.7, 1, 0.1, br, 'max_features'),
                    'learning_rate': self.param_range('log', -4, -0.5, 0.1, br, 'learning_rate'),
                    'subsample': self.param_range('real', 0.5, 1, 0.1, br, 'subsample')
                    },

            'gbm_q': {
                    'n_estimators': self.param_range('int', 10, 60, 5, br, 'n_estimators'),
                    'max_depth': self.param_range('int', 2, 18, 2, br,'max_depth'),
                    'min_samples_leaf': self.param_range('int', 4, 25, 3, br, 'min_samples_leaf'),
                    'max_features': self.param_range('real', 0.7, 1, 0.1, br, 'max_features'),
                    'subsample': self.param_range('real', 0.5, 1, 0.1, br, 'subsample'),
                    'learning_rate': self.param_range('log', -4, -0.5, 0.1, br, 'learning_rate'),
                    },

            'gbmh': {
                    'max_iter': self.param_range('int', 30, 100, 10, br, 'max_iter'),
                    'max_depth': self.param_range('int', 4, 12, 1, br,'max_depth'),
                    'min_samples_leaf': self.param_range('int', 5, 25, 2, br, 'min_samples_leaf'),
                    'max_leaf_nodes': self.param_range('int', 20, 50, 3, br, 'max_leaf_nodes'),
                    'l2_regularization': self.param_range('real', 0, 10, 1, br, 'l2_regularization'),
                    'learning_rate': self.param_range('log', -4, -0.5, 0.1, br, 'learning_rate')
                    },

            'gbmh_c': {
                    'max_iter': self.param_range('int', 30, 100, 10, br, 'max_iter'),
                    'max_depth': self.param_range('int', 3, 12, 2, br,'max_depth'),
                    'min_samples_leaf': self.param_range('int', 5, 25, 2, br, 'min_samples_leaf'),
                    'max_leaf_nodes': self.param_range('int', 15, 50, 5, br, 'max_leaf_nodes'),
                    'l2_regularization': self.param_range('real', 0, 10, 1, br, 'l2_regularization'),
                    'learning_rate': self.param_range('log', -4, -0.5, 0.1, br, 'learning_rate')
                    },

            'gbmh_q': {
                    'max_iter': self.param_range('int', 30, 100, 10, br, 'max_iter'),
                    'max_depth': self.param_range('int', 3, 12, 2, br,'max_depth'),
                    'min_samples_leaf': self.param_range('int', 5, 25, 2, br, 'min_samples_leaf'),
                    'max_leaf_nodes': self.param_range('int', 15, 50, 5, br, 'max_leaf_nodes'),
                    'l2_regularization': self.param_range('real', 0, 10, 1, br, 'l2_regularization'),
                    'learning_rate': self.param_range('log', -4, -0.5, 0.1, br, 'learning_rate')
                    },
            
            'cb': {
                    'iterations': self.param_range('int', 50, 175, 10, br, 'iterations'),
                    'depth': self.param_range('int', 1, 8, 1, br, 'depth'),
                    'learning_rate': self.param_range('log', -5, 0, 0.1, br, 'learning_rate'),
                    'l2_leaf_reg': self.param_range('real', 0, 20, 1, br, 'l2_leaf_reg'),
                    'random_strength': self.param_range('real', 0, 20, 1, br, 'random_strength')
                    },

            'cb_c': {
                    'iterations': self.param_range('int', 50, 175, 10, br, 'iterations'),
                    'depth': self.param_range('int', 1, 8, 1, br, 'depth'),
                    'learning_rate': self.param_range('log', -5, 0, 0.1, br, 'learning_rate'),
                    'l2_leaf_reg': self.param_range('real', 0, 20, 1, br, 'l2_leaf_reg'),
                    'random_strength': self.param_range('real', 0, 20, 1, br, 'random_strength')
                    },

            'cb_q': {
                    'iterations': self.param_range('int', 50, 175, 10, br, 'iterations'),
                    'depth': self.param_range('int', 1, 8, 1, br, 'depth'),
                    'learning_rate': self.param_range('log', -5, 0, 0.1, br, 'learning_rate'),
                    'l2_leaf_reg': self.param_range('real', 0, 20, 1, br, 'l2_leaf_reg'),
                    'random_strength': self.param_range('real', 0, 20, 1, br, 'random_strength')
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
                    'alpha': self.param_range('log', -3, 0, 0.1, br, 'alpha'),
                    'learning_rate_init': self.param_range('log', -4, 0.5, 0.1, br, 'learning_rate_init'),
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
            brier = brier_score_loss(y, y_pred, sample_weight=sample_weight)
            lloss = log_loss(y, y_pred, sample_weight=sample_weight)
            score = 100*(self.logloss_wt*lloss + self.brier_wt*brier - self.matt_wt*matt)
        
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


    def rand_objective(self, model, params):

        print(params)
        self.cur_model = copy.deepcopy(model)
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
                
        except Exception as e:
            print('Trial Failed')
            print(e)
            y = self.y.values
            score = self.custom_score(y, np.repeat(np.mean(y), repeats=len(y)))

        return score

    def custom_rand_search(self, model, params, n_iters):

        from joblib import Parallel, delayed

        param_list = [self.param_select(params) for _ in range(n_iters)]

        scores = Parallel(n_jobs=-1, verbose=0)(delayed(self.rand_objective)(model, p) for p in param_list)
        param_output = pd.DataFrame(param_list, index=range(n_iters))
        param_output['scores'] = scores

        best_params = param_list[np.argmin(scores)]
        best_model = copy.deepcopy(model)
        best_model.set_params(**best_params)
                    
        try: best_model.steps[-1][1].n_jobs=-1
        except: pass

        return best_model, param_output
    
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

        if self.trials is None: 
            raise ValueError('Trials object must be provided for Bayesian Optimization')

        num_past_runs = len(self.trials.tids)
        max_evals = num_past_runs + n_iters
        n_startup_jobs = np.min([int(max_evals/2), 20])
        if self.hp_algo=='tpe': algo = partial(tpe.suggest, n_startup_jobs=n_startup_jobs)
        elif self.hp_algo=='atpe': algo = atpe.suggest

        _ = fmin(partial(self.rand_objective, copy.deepcopy(model)), space=params, algo=algo, 
                 trials=self.trials, max_evals=max_evals, rstate=np.random.default_rng(self.randseed))

        param_output, best_params = self.get_bayes_params(num_past_runs, params)
    
        best_model = copy.deepcopy(model)
        best_model.set_params(**best_params)
        try: best_model.steps[-1][1].n_jobs=-1
        except: pass

        return best_model, param_output
    

    def optuna_objective(self, trial, model, params):

        params_opt = {}
        for k, v in params.items():
            if v[0] == 'real': params_opt[k] = trial.suggest_float(k, v[1], v[2], log=False)
            elif v[0] == 'log': params_opt[k] = trial.suggest_float(k, np.exp(v[1]), np.exp(v[2]), log=True)
            elif v[0] == 'int': params_opt[k] = trial.suggest_int(k, v[1], v[2])
            elif v[0] == 'categorical': params_opt[k] = trial.suggest_categorical(k, v[1])

        return self.rand_objective(model, params_opt)
    
    def get_optuna_params(self, num_past_runs):
        min_val = 10000000
        for t in self.trials.get_trials()[num_past_runs:]:
            if t.values[0] < min_val:
                min_val = t.values[0]
                min_trial = t

        return min_trial.params


    def custom_optuna_search(self, model, params, n_iters):

        try: model.steps[-1][1].n_jobs=-1
        except: pass

        num_past_runs = len(self.trials.trials)
        self.trials.optimize(partial(self.optuna_objective, model=copy.deepcopy(model), params=params), 
                             n_trials=n_iters, timeout=self.optuna_timeout)
        
        best_params = self.get_optuna_params(num_past_runs)

        best_model = copy.deepcopy(model)
        best_model.set_params(**best_params)
        try: best_model.steps[-1][1].n_jobs=-1
        except: pass

        return best_model, pd.DataFrame()

    


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
                       scoring=None, trials=None, optuna_timeout=60*10):
   
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
            self.optuna_timeout = optuna_timeout

            np.random.seed(self.randseed)

            if bayes_rand == 'bayes':
                best_model, param_scores = self.custom_bayes_search(model, params, n_iter)
            
            elif bayes_rand == 'rand':
                best_model, ps = self.custom_rand_search(model, params, n_iters=n_iter)
                param_scores = pd.concat([param_scores, ps], axis=0)

            elif bayes_rand == 'optuna':
                best_model, ps = self.custom_optuna_search(model, params, n_iters=n_iter)
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

# # set to this year for the projections
# year = 2024
# from ff.db_operations import DataManage
# from ff import general
# import pandas as pd
# import numpy as np
# import warnings
# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
# pd.set_option('display.max_columns', 999)

# # set the root path and database management object
# root_path = general.get_main_path('Fantasy_Football')
# db_path = f'{root_path}/Data/Databases/'
# dm = DataManage(db_path)
# DB_NAME = 'Season_Stats_New'

# pos = 'WR'


# # from skmodel import SciKitModel
# from hyperopt import Trials

# alpha = 0.8
# proba = False
# model_obj = 'reg'

# # Xy = dm.read(f"SELECT * FROM {pos}_{year}_ProjOnly WHERE pos='{pos}' ", 'Model_Inputs')
# Xy = dm.read(f"SELECT * FROM {pos}_{year}_Stats WHERE pos='{pos}' ", 'Model_Inputs')
# # Xy = dm.read(f"SELECT * FROM {pos}_{year}_Rookie ", 'Model_Inputs')

# Xy = Xy.sort_values(by='year').reset_index(drop=True)
# Xy['team'] = 'team'
# Xy['week'] = 1
# Xy['game_date'] = Xy.year

# if proba: Xy = Xy.drop('y_act', axis=1).rename(columns={'y_act_class': 'y_act'})
# else: Xy = Xy.drop('y_act_class', axis=1)

# pred = Xy[Xy.year==2024].copy().reset_index(drop=True)
# train = Xy[Xy.year<2024].copy().reset_index(drop=True)
# print(Xy.shape)

# preds = []
# actuals = []

# skm = SciKitModel(train, model_obj=model_obj, alpha=alpha, hp_algo='atpe')
# to_drop = list(train.dtypes[train.dtypes=='object'].index)
# to_drop.extend(['games_next'])
# X, y = skm.Xy_split('y_act', to_drop = to_drop)

# pipe = skm.model_pipe([skm.piece('random_sample'),
#                         skm.piece('std_scale'), 
#                         skm.piece('select_perc'),
#                         # skm.feature_union([
#                         #                skm.piece('agglomeration'), 
#                         #                 skm.piece('k_best'),
#                         #                 skm.piece('pca')
#                         #                 ]),
#                         skm.piece('k_best'),
#                         skm.piece('cb')
#                     ])

# params = skm.default_params(pipe, 'optuna')

# #%%

# def rename_existing(study_db, study_name):
#     import datetime as dt
#     new_study_name = study_name + '_' + dt.datetime.now().strftime('%Y%m%d%H%M%S')
#     optuna.copy_study(from_study_name=study_name, from_storage=study_db, to_storage=study_db, to_study_name=new_study_name)
#     optuna.delete_study(study_name=study_name, storage=study_db)


# def get_new_study(db, old_name, new_name, num_trials):
    
#     if old_name is not None:
#         old_study = optuna.create_study(
#             study_name=old_name,
#             storage=db,
#             load_if_exists=True
#         )
    
#     try:
#         next_study = optuna.create_study(
#             study_name=new_name, 
#             storage=db, 
#             load_if_exists=False
#         )

#     except:
#         rename_existing(db, new_name)
#         next_study = optuna.create_study(
#             study_name=new_name, 
#             storage=db, 
#             load_if_exists=False
#         )
    
#     if old_name is not None:
#         print(f"Loaded {new_name} study with {old_name} {len(old_study.trials)} trials")
#         next_study.add_trials(old_study.trials[-num_trials:])

#     return next_study
        


# study_db = "sqlite:///optuna/reuse_test.sqlite3"
# old_name = 'v16'
# new_name = 'v17'
# num_trials = 25
# next_study = get_new_study(study_db, old_name, new_name, num_trials)
# next_study.trials_dataframe(attrs=("number", "value", "params", "state"))
                            
# #%%
# # pipe.steps[-1][-1].set_params(**{'loss_function': f'Quantile:alpha={alpha}'})
# best_models, oof_data, param_scores, trials = skm.time_series_cv(pipe, X, y, params, n_iter=5,
#                                                                 col_split='year',n_splits=5,
#                                                                 time_split=2016, alpha=alpha,
#                                                                 bayes_rand='optuna', proba=proba,
#                                                                 sample_weight=False, trials=next_study,
#                                                                 random_seed=12345, optuna_timeout=10)
# # %%

# trials.trials_dataframe().duration.mean().seconds

# # %%
# best_models
# # %%
