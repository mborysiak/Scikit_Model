from skmodel.data_setup import DataSetup
from skmodel.pipe_setup import PipeSetup
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np

# model and search setup
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV, StratifiedKFold

from hyperopt import fmin, hp, tpe
from hyperopt import SparkTrials, STATUS_OK
from hyperopt.pyll import scope

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import matthews_corrcoef, f1_score

class SciKitModel(PipeSetup):
    
    def __init__(self, data, model_obj='reg', set_seed=1234):
        """Class the automates model sklearn pipeline creation and model training

        Args:
            data (pandas.DataFrame): Data to be modeled with features and target
            model_obj (str, optional): Model objective for regression (reg) or 
                                       classification (class). Defaults to 'reg'.
            set_seed (int, optional): Numpy random seed to set state. Defaults to 1234.
        """        
        self.data = data
        self.model_obj = model_obj
        np.random.seed(set_seed)


    def param_range(self, var_type, low, high, spacing, bayes_rand, label):

        if bayes_rand=='bayes':
            if var_type=='int': return scope.int(hp.quniform(label, low, high, 1))
            if var_type=='real': return hp.uniform(label, low, high)
            if var_type=='cat': return hp.choice(label, low)
        elif bayes_rand=='rand':
            if var_type=='int': return range(low, high, spacing)
            if var_type=='real': return np.arange(low, high, spacing)
            if var_type=='cat': return low


    def default_params(self, pipe, bayes_rand='rand'):
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
            'agglomeration': {'n_clusters': self.param_range('int', 2, 30, 4, br, 'n_clusters')},
            'pca': {'n_components': self.param_range('int', 2, 30, 4, br, 'n_components')},
            'k_best': {'k': self.param_range('int', 5, 40, 5, br, 'k')},
            'select_perc': {'percentile': self.param_range('int', 20, 80, 4, br, 'select_perc')},
            'k_best_c': {'k': self.param_range('int', 5, 40, 4, br, 'k_best_c')},
            'select_perc_c': {'percentile': self.param_range('int', 20, 80, 4, br, 'select_perc_c')},
            'select_from_model': {'estimator': [Ridge(alpha=0.1), Ridge(alpha=1), Ridge(alpha=10),
                                                Lasso(alpha=0.1), Lasso(alpha=1), Lasso(alpha=10),
                                                RandomForestRegressor(max_depth=5), 
                                                RandomForestRegressor(max_depth=10)]},
            'feature_drop': {'col': self.param_range('cat', ['avg_pick', None], None, None, br, 'feature_drop')},
            'feature_select': {'cols': self.param_range('cat', [['avg_pick'], ['avg_pick', 'year']], None, None, br, 'feature_select')},

            # model params
            'ridge': {'alpha': self.param_range('int', 1, 1000, 1, br, 'alpha')},
            'lasso': {'alpha': self.param_range('real', 0.01, 25, 0.1, br, 'alpha')},
            'enet': {'alpha': self.param_range('real', 0.01, 50, 0.1, br, 'alpha'),
                    'l1_ratio': self.param_range('real', 0.05, 0.95, 0.05, br, 'l1_ratio')},
            'rf': {'n_estimators': self.param_range('int', 50, 250, 10, br, 'n_estimators'),
                    'max_depth': self.param_range('int', 2, 30, 2, br, 'max_depth'),
                    'min_samples_leaf': self.param_range('int', 1, 10, 1, br, 'min_samples_leaf'),
                    'max_features': self.param_range('real', 0.1, 1, 0.2, br, 'max_features')},
            'lgbm': {'n_estimators': self.param_range('int', 25, 300, 25, br, 'n_estimators'),
                     'max_depth': self.param_range('int', 2, 50, 5, br, 'max_depth'),
                     'colsample_bytree': self.param_range('real', 0.1, 1, 0.2, br, 'colsample_bytree'),
                     'subsample':  self.param_range('real', 0.1, 1, 0.2, br, 'subsample'),
                     'reg_lambda': self.param_range('int', 0, 1000, 100, br, 'reg_lambda'),
                     'reg_alpha': self.param_range('int', 0, 1000, 100, br, 'reg_alpha'),
                    #  'learning_rate': self.param_range('real', 0.0001, 0.1, 0.001, br, 'learning_rate'),
                     'min_data_in_leaf': self.param_range('int', 1, 25, 5, br, 'min_data_in_leaf')},
            'xgb': {'n_estimators': self.param_range('int', 50, 250, 25, br, 'n_estimators'),
                     'max_depth': self.param_range('int', 2, 20, 2, br, 'max_depth'),
                     'colsample_bytree': self.param_range('real', 0.2, 1, 0.2, br,  'colsample_bytree'),
                     'subsample':  self.param_range('real', 0.2, 1, 0.2, br, 'subsample'),
                     'reg_lambda': self.param_range('int', 0, 1000, 100, br,  'reg_lambda')},
            'gbm': {'n_estimators': self.param_range('int', 10, 100, 10, br, 'n_estimators'),
                    'max_depth': self.param_range('int', 2, 30, 3, br,'max_depth'),
                    'min_samples_leaf': self.param_range('int', 4, 15, 2, br, 'min_samples_leaf'),
                    'max_features': self.param_range('real', 0.7, 1, 0.1, br, 'max_features'),
                    'subsample': self.param_range('real', 0.5, 1, 0.1, br, 'subsample')},
            'knn': {'n_neighbors':  self.param_range('int',1, 30, 1, br, 'n_neighbors'),
                    'weights': self.param_range('cat',['distance', 'uniform'], None, None, br, 'weights'),
                    'algorithm': self.param_range('cat', ['auto', 'ball_tree', 'kd_tree', 'brute'], None, None, br, 'algorithm')},
            'svr': {'C': self.param_range('int', 1, 100, 1, br, 'C')},

            # classification params
            'lr_c': {'C': self.param_range('real', 0.01, 25, 0.1, br, 'C'),
                     'class_weight': [{0: i, 1: 1} for i in np.arange(0.05, 1, 0.1)]},
            'rf_c': {'n_estimators': self.param_range('int', 50, 250, 25, br, 'n_estimators'),
                    'max_depth': self.param_range('int', 2, 20, 3, br, 'max_depth'),
                    'min_samples_leaf': self.param_range('int', 1, 10, 1, br, 'min_samples_leaf'),
                    'max_features': self.param_range('real', 0.1, 1, 0.2, br, 'max_features'),
                    'class_weight': [{0: i, 1: 1} for i in np.arange(0.05, 1, 0.2)]},
            'lgbm_c': {'n_estimators': self.param_range('int', 50, 250, 30, br, 'n_estimators'),
                     'max_depth': self.param_range('int', 2, 20, 3, br, 'max_depth'),
                     'colsample_bytree': self.param_range('real', 0.2, 1, 0.25, br, 'colsample_bytree'),
                     'subsample':  self.param_range('real', 0.2, 1, 0.25, br, 'subsample'),
                     'reg_lambda': self.param_range('int', 0, 1000, 100, br, 'reg_lambda'),
                     'class_weight': [{0: i, 1: 1} for i in np.arange(0.05, 1, 0.2)]},
            'xgb_c': {'n_estimators': self.param_range('int', 50, 250, 30, br, 'n_estimators'),
                     'max_depth': self.param_range('int', 2, 20, 3, br, 'max_depth'),
                     'colsample_bytree': self.param_range('real', 0.2, 1, 0.25, br, 'colsample_bytree'),
                     'subsample':  self.param_range('real', 0.2, 1, 0.25, br, 'subsample'),
                     'reg_lambda': self.param_range('int', 0, 1000, 100, br, 'reg_lambda'),
                     'scale_pos_weight': self.param_range('real', 1, 10, 1, br, 'scale_pos_weight')},
            'gbm_c': {'n_estimators': self.param_range('int', 10, 100, 10, br, 'n_estimators'),
                    'max_depth': self.param_range('int', 2, 30, 3, br, 'max_depth'),
                    'min_samples_leaf': self.param_range('int', 3, 10, 1, br, 'min_samples_leaf'),
                    'max_features': self.param_range('real', 0.7, 1, 0.1, br, 'max_features'),
                    'subsample': self.param_range('real', 0.5, 1, 0.1, br, 'subsample')},
            'knn_c': {'n_neighbors':  self.param_range('int',1, 30, 1, br, 'n_neighbors'),
                    'weights': self.param_range('cat',['distance', 'uniform'], None, None, br, 'weights'),
                    'algorithm': self.param_range('cat', ['auto', 'ball_tree', 'kd_tree', 'brute'], None, None, br, 'algorithm')},
            'svc': {'C': self.param_range('int', 1, 100, 1, br, 'C'),
                    'class_weight': [{0: i, 1: 1} for i in np.arange(0.05, 1, 0.2)]},
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


    def scorer(self, score_type):

        scorers = {
            'r2': make_scorer(r2_score, greater_is_better=True),
            'mse': make_scorer(mean_squared_error, greater_is_better=False),
            'mae': make_scorer(mean_absolute_error, greater_is_better=False),
            'matt_coef': make_scorer(matthews_corrcoef, greater_is_better=True),
            'f1': make_scorer(f1_score, greater_is_better=True)
        }

        return scorers[score_type]


    def fit_opt(self, opt_model):
        
        opt_model.fit(self.X, self.y)
        return opt_model.best_estimator_


    def grid_search(self, pipe_to_fit, X, y, params, cv=5, scoring='neg_mean_squared_error'):

        search = GridSearchCV(pipe_to_fit, params, cv=cv, scoring=scoring, refit=True)
        best_model = search.fit(X, y)

        return best_model.best_estimator_


    def random_search(self, pipe_to_fit, X, y, params, cv=5, n_iter=50, n_jobs=-1, scoring='neg_mean_squared_error'):

        search = RandomizedSearchCV(pipe_to_fit, params, n_iter=n_iter, cv=cv, 
                                    scoring=scoring, n_jobs=n_jobs, refit=True)
        best_model = search.fit(X, y)

        return best_model.best_estimator_


    def cv_score(self, model, X, y, cv=5, scoring='neg_mean_squared_error', 
                 n_jobs=1, return_mean=True):

        score = cross_val_score(model, X, y, cv=cv, n_jobs=n_jobs, scoring=scoring)
        if return_mean:
            score = np.mean(score)
        return score


    def cv_predict(self, model, X, y, cv=5, n_jobs=-1, method='predict'):
        pred = cross_val_predict(model, X, y, cv=cv, n_jobs=n_jobs, method=method)
        return pred


    def cv_time_splits(self, col, X, val_start):

        X_sort = X.sort_values(by=col).reset_index(drop=True)

        ts = X_sort[col].unique()
        ts = ts[ts>=val_start]

        cv_time = []
        for t in ts:
            train_idx = list(X_sort[X_sort[col] < t].index)
            test_idx = list(X_sort[X_sort[col] == t].index)
            cv_time.append((train_idx, test_idx))

        return cv_time

    
    def cv_predict_time(self, model, X, y, cv_time):

        predictions = []
        self.test_indices = []
        for tr, te in cv_time:
            
            X_train, y_train = X.iloc[tr, :], y[tr]
            X_test, _ = X.iloc[te, :], y[te]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            
            predictions.extend(pred)
            self.test_indices.extend(te)

        return predictions

        
    def train_test_split_time(self, X, y, col, time_split):

        X_train_only = X[X[col] < time_split]
        y_train_only = y[X_train_only.index].reset_index(drop=True)
        X_train_only.reset_index(drop=True, inplace=True)

        X_val = X[X[col] >= time_split]
        y_val = y[X_val.index].reset_index(drop=True)
        X_val.reset_index(drop=True, inplace=True)

        return X_train_only, X_val, y_train_only, y_val


    def get_y_val(self):
        return self.y_train.loc[self.cv_time_train[0][1][0]:]


    def bayes_search(self, model, params, n_iters=64):
        
        self.cur_model = model
        try: 
            self.cur_model.steps[-1][1].n_jobs=2
            parallelism=8
        except: 
            parallelism=16

        spark_trials = SparkTrials(parallelism=parallelism)
        best_hyperparameters = fmin(
                                    fn=self.bayes_objective,
                                    space=params,
                                    algo=tpe.suggest,
                                    trials=spark_trials,
                                    max_evals=n_iters
                                    )
        best_params = {}
        for k, v in best_hyperparameters.items():
            if (v).is_integer():
                v = int(v)
            for k_full, _ in params.items():
                if k in k_full:
                    best_params[k_full] = v
        
        model.set_params(**best_params)
        try: model.steps[-1][1].n_jobs=-1
        except: pass

        return model

    def bayes_objective(self, params):

        self.cur_model.set_params(**params) 

        val_predictions, _ = self.cv_predict_time_holdout(self.cur_model)
        y_val = self.get_y_val()
        rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        r2 = r2_score(y_val, val_predictions)
        mae = mean_absolute_error(y_val, val_predictions)
        score = mae + rmse - 100*r2

        return {'loss': score, 'status': STATUS_OK}

    def cv_predict_time_holdout(self, model):
  
      """Perform a rolling time-series prediction for both validation data in a training set
         plus predictions on a holdout test set

         For example, train on slices 1-5 and predict slice 6a for the validation data that will
         be added to training and also predict 6b for holdout test data. Next train on slices
         1-6a and predict 7a + 7b, while storing 6b, 7b, etc. separately.

          Args:
              model (sklearn.Model or Pipe): Model or pipeline to be trained
              X_train (pandas.DataFrame or numpy.array): Set of samples and features for training data with time component
              y_train (pandas.Series or numpy.array): Target to be predicted
              X_hold (pandas.Series or numpy.array): Subset of X data held-out to only be predicted
              cv_time_train (tuple): Indices of data to be trained / validated on the X_train/y_trian datasets for rolling time
              cv_time_hold (tuple): Indices of data to be predicted on holdout dataset for rolling time

          Returns:
              list: In fold predictions for validation data
              list: Out of fold predictions for holdout data
          """ 

      # set up list to store validation and holdout predictions + dataframe indices
      val_predictions = []
      hold_predictions = []

      # iterate through both the training and holdout time series indices
      for (tr_train, te_train), (_, te_hold) in zip(self.cv_time_train, self.cv_time_hold):

          # extract out the training and validation datasets from the training folds
          X_train_cur, y_train_cur = self.X_train.iloc[tr_train, :], self.y_train[tr_train]
          X_val = self.X_train.iloc[te_train, :]

          # fit and predict the validation dataset
          model.fit(X_train_cur, y_train_cur)
          pred_val = model.predict(X_val)
          val_predictions.extend(pred_val)

          # predict the holdout dataset for the current time period
          X_hold_test = self.X_hold.iloc[te_hold, :]
          pred_hold = model.predict(X_hold_test)
          hold_predictions.extend(pred_hold)

      return val_predictions, hold_predictions


    def time_series_cv(self, model, X, y, params, col_split, time_split, n_splits=5, n_iter=50, bayes_rand='rand'):
        """Train a time series model using rolling cross-validation combined
           with K-Fold holdouts for a complete holdout prediction set.

           E.g. Train on 1-4 time-stratified folds using rolling cross-validation
                and predict Fold 5. Train on Folds 1,2,3,5 and predict Fold 4.

        Args:
            model (sklearn.Model or Pipe): Model or pipeline to be trained
            X (pandas.DataFrame or numpy.array): Set of samples and features for training data with time component
            y (pandas.Series or numpy.array): Target to be predicted
            params (dict): Dictionary containing hyperparameters to optimize
            col_split (str): Time column to be used for splitting up dataset
            time_split (int): Point in time to split into training (early) or validation (late)
            n_splits (int, optional): Number of folds to split validation. Defaults to 5.
            n_iter (int, optional): Random search iterations for optimization. Defaults to 50.

        Returns:
            sklearn.Model or Pipe: Best performing model with hyperparameters
            list: Validation score metrics
            dict: Out of fold predictions for each holdout set and actual target data
        """        
        # split into the train only and val/holdout datasets
        X_train_only, X_val_hold, y_train_only, y_val_hold = self.train_test_split_time(X, y, col_split, time_split)

        #--------------
        # Set up place holders for metrics
        #--------------
        
        # list to store accuracy metrics
        mean_val_sc = []
        mean_hold_sc = []
        
        # # arrays to hold all predictions and actuals
        # hold_predictions = np.array([])
        # hold_actuals = np.array([])
        val_predictions = np.array([])
        hold_results = pd.DataFrame()

        # list to hold the best models
        best_models = []

        #----------------
        # Run the KFold train-prediction loop
        #----------------
        X_val_hold = X_val_hold.sample(frac=1, random_state=1234)
        y_val_hold = y_val_hold.sample(frac=1, random_state=1234)
        skf = StratifiedKFold(n_splits=n_splits)
        for val_idx, hold_idx in skf.split(X_val_hold, X_val_hold[col_split]):
            
            print('-------')

            # split the val/hold dataset into random validation and holdout sets
            X_val, X_hold = X_val_hold.iloc[val_idx,:], X_val_hold.iloc[hold_idx,:]
            y_val, y_hold = y_val_hold.iloc[val_idx], y_val_hold.iloc[hold_idx]

            # concat the current training set using train and validation folds
            X_train = pd.concat([X_train_only, X_val], axis=0).reset_index(drop=True)
            y_train = pd.concat([y_train_only, y_val], axis=0).reset_index(drop=True)

            
            # get the CV time splits and find the best model
            cv_time = self.cv_time_splits(col_split, X_train, time_split)
            
            # score the best model on validation and holdout sets
            cv_time_hold = self.cv_time_splits(col_split, X_hold, time_split)

            self.X_train = X_train
            self.y_train=y_train
            self.X_hold=X_hold
            self.y_hold=y_hold
            self.cv_time_train = cv_time
            self.cv_time_hold = cv_time_hold

            if self.model_obj=='class':
                best_model = self.random_search(model, X_train, y_train, params, cv=cv_time, 
                                                n_iter=n_iter, scoring=self.scorer('matt_coef'))
            elif self.model_obj=='reg':
                if bayes_rand=='rand':
                    best_model = self.random_search(model, X_train, y_train, params, cv=cv_time, n_iter=n_iter)
                elif bayes_rand=='bayes':
                    best_model = self.bayes_search(model, params, n_iters=n_iter)

            print(best_model)
            val_pred_cur, hold_pred = self.cv_predict_time_holdout(best_model)
            _, val_sc = self.test_scores(y_train[cv_time[0][1][0]:], val_pred_cur)
            _, hold_sc = self.test_scores(y_hold, hold_pred)
            
            # append the scores and best model
            mean_val_sc.append(val_sc); mean_hold_sc.append(hold_sc)
            best_models.append(best_model)

            # # get the holdout and validation predictions and store
            # hold_predictions = np.concatenate([hold_predictions, np.array(hold_pred)])
            # hold_actuals = np.concatenate([hold_actuals, y_hold])

            hold_results_cur = pd.DataFrame([hold_idx, y_hold, hold_pred]).T
            hold_results = pd.concat([hold_results, hold_results_cur], axis=0)
            
            # cv_time = self.cv_time_splits(col_split, X, time_split)
            # val_pred = self.cv_predict_time(best_model, X, y, cv_time)
            # val_predictions = np.append(val_predictions, np.array(val_pred))

        # calculate the mean scores
        mean_scores = [np.round(np.mean(mean_val_sc), 3), np.round(np.mean(mean_hold_sc), 3)]
        print('Mean Scores:', mean_scores)
        
        # # aggregate all the prediction for val, holds, and combined val/hold
        # val_predictions = np.mean(val_predictions.reshape(n_splits, len(val_pred)), axis=0)

        oof_data = {
            'val': val_predictions, 
            'hold': hold_results.iloc[:, 2].values,
            # 'combined': np.mean([val_predictions, hold_predictions], axis=0),
            'actual': hold_results.iloc[:, 1].values
            }

        return best_models, mean_scores, oof_data


    def val_scores(self, model, X, y, cv):

        if self.model_obj == 'reg':
            mse = self.cv_score(model, X, y, cv=cv, scoring=self.scorer('mse'))
            r2 = self.cv_score(model, X, y, cv=cv, scoring=self.scorer('r2'))
            r2_fit = model.fit(X, y).score(X, y)

            for v, m in zip(['Val MSE:', 'Val R2:', 'Fit R2:'], [mse, r2, r2_fit]):
                print(v, np.round(m, 3))

            return mse, r2

        elif self.model_obj == 'class':
            matt_coef = self.cv_score(model, X, y, cv=cv, scoring=self.scorer('matt_coef'))
            f1 = self.cv_score(model, X, y, cv=cv, scoring=self.scorer('f1'))

            for v, m in zip(['Val MC:', 'Val F1:'], [matt_coef, f1]):
                print(v, np.round(m, 3))

            return f1, matt_coef 

          
    def test_scores(self, y, pred):

        if self.model_obj == 'reg':
            mse = mean_squared_error(y, pred)
            r2 = r2_score(y, pred)
            
            for v, m in zip(['Test MSE:', 'Test R2:'], [mse, r2]):
                print(v, np.round(m, 3))

            return mse, r2

        elif self.model_obj == 'class':
            matt_coef = matthews_corrcoef(y, pred)
            f1 = f1_score(y, pred)

            for v, m in zip(['Test MC:', 'Test F1:'], [matt_coef, f1]):
                print(v, np.round(m, 3))

            return f1, matt_coef

    def return_labels(self, cols, time_or_all='time'):
        
        if time_or_all=='time':
            return self.data.loc[self.test_indices, cols]

        elif time_or_all=='all':
            return self.data.loc[:, cols]

    
    def print_coef(self, model, cols):

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

        
    def best_stack(self, est, stack_params, X_stack, y_stack, n_iter=500, print_coef=True, run_adp=False):

        X_stack_shuf = X_stack.sample(frac=1, random_state=1234).reset_index(drop=True)
        y_stack_shuf = y_stack.sample(frac=1, random_state=1234).reset_index(drop=True)

        if self.model_obj=='class':
            best_model = self.random_search(est, X_stack_shuf, y_stack_shuf, stack_params, cv=5, 
                                            n_iter=n_iter, scoring=self.scorer('matt_coef'))
        elif self.model_obj=='reg':
            best_model = self.random_search(est, X_stack_shuf, y_stack_shuf, stack_params, cv=5, 
                                            n_iter=n_iter)

        if run_adp:
            # print the OOS scores for ADP and model stack
            print('ADP Score\n--------')
            adp_col = [c for c in X_stack.columns if 'adp' in c]
            adp_preds = self.cv_predict(self.piece('lr')[1], X_stack_shuf[adp_col], y_stack_shuf, cv=5)
            adp_score = r2_score(y_stack_shuf, adp_preds)
            print(f'ADP R2: {round(adp_score,3)}')
        
        else:
            adp_score = 0

        print('\nStack Score\n--------')
        full_preds = self.cv_predict(best_model, X_stack_shuf, y_stack_shuf, cv=5)
        stack_score = r2_score(y_stack_shuf, full_preds)
        print(f'Val R2: {round(stack_score,3)}')

        if print_coef:
            try:
                imp_cols = X_stack_shuf.columns[best_model['k_best'].get_support()]
            except:
                imp_cols = X_stack_shuf.columns
            self.print_coef(best_model, imp_cols)

        return best_model, round(stack_score,3), round(adp_score,3)



