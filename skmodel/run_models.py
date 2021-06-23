from skmodel.data_setup import DataSetup
from skmodel.pipe_setup import PipeSetup
import pandas as pd
import numpy as np

# model and search setup
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV, KFold

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
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


    def param_range(self, var_type, low, high, spacing, bayes_rand):

        if bayes_rand=='bayes':
            if var_type=='int': return Integer(low, high)
            if var_type=='real': return Real(low, high)
            if var_type=='cat': return Categorical(low)
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
            'agglomeration': {'n_clusters': self.param_range('int', 2, 40, 4, br)},
            'pca': {'n_components': self.param_range('int', 2, 40, 4, br)},
            'k_best': {'k': self.param_range('int', 2, 40, 4, br)},
            'select_perc': {'percentile': self.param_range('int', 20, 80, 4, br)},
            'k_best_c': {'k': self.param_range('int', 2, 40, 4, br)},
            'select_perc_c': {'percentile': self.param_range('int', 20, 80, 4, br)},
            'select_from_model': {'estimator': [Ridge(alpha=0.1), Ridge(alpha=1), Ridge(alpha=10),
                                                Lasso(alpha=0.1), Lasso(alpha=1), Lasso(alpha=10),
                                                RandomForestRegressor(max_depth=5), 
                                                RandomForestRegressor(max_depth=10)]},
            'feature_drop': {'col': self.param_range('cat', ['avg_pick', None], None, None, br)},
            'feature_select': {'cols': self.param_range('cat', [['avg_pick'], ['avg_pick', 'year']], None, None, br)},

            # model params
            'ridge': {'alpha': self.param_range('int', 1, 1000, 1, br)},
            'lasso': {'alpha': self.param_range('real', 0.01, 25, 0.1, br)},
            'enet': {'alpha': self.param_range('real', 0.01, 50, 0.1, br),
                    'l1_ratio': self.param_range('real', 0.05, 0.95, 0.05, br)},
            'rf': {'n_estimators': self.param_range('int', 50, 250, 10, br),
                    'max_depth': self.param_range('int', 2, 30, 2, br),
                    'min_samples_leaf': self.param_range('int', 1, 10, 1, br),
                    'max_features': self.param_range('real', 0.1, 1, 0.1, br)},
            'lgbm': {'n_estimators': self.param_range('int', 50, 250, 25, br),
                     'max_depth': self.param_range('int', 2, 30, 3, br),
                     'colsample_bytree': self.param_range('real', 0.2, 1, 0.2, br),
                     'subsample':  self.param_range('real', 0.2, 1, 0.2, br),
                    # 'min_child_weight':  self.param_range('int', 1, 100, 20, br),
                     'reg_lambda': self.param_range('int', 0, 1000, 100, br)},
            'xgb': {'n_estimators': self.param_range('int', 50, 250, 25, br),
                     'max_depth': self.param_range('int', 2, 30, 2, br),
                     'colsample_bytree': self.param_range('real', 0.2, 1, 0.2, br),
                     'subsample':  self.param_range('real', 0.2, 1, 0.2, br),
                    # 'min_child_weight':  self.param_range('int', 1, 100, 20, br),
                     'reg_lambda': self.param_range('int', 0, 1000, 50, br)},
            'gbm': {'n_estimators': self.param_range('int', 50, 250, 20, br),
                    'max_depth': self.param_range('int', 2, 30, 3, br),
                    'min_samples_leaf': self.param_range('int', 1, 10, 2, br),
                    'max_features': self.param_range('real', 0.1, 1, 0.2, br),
                    'subsample': self.param_range('real', 0.1, 1, 0.2, br)},
            'knn': {'n_neighbors':  self.param_range('int',1, 30, 1, br),
                    'weights': self.param_range('cat',['distance', 'uniform'], None, None, br),
                    'algorithm': self.param_range('cat', ['auto', 'ball_tree', 'kd_tree', 'brute'], None, None, br)},
            'svr': {'C': self.param_range('int', 1, 100, 1, br)},

            # classification params
            'lr_c': {'C': self.param_range('real', 0.01, 25, 0.1, br),
                     'class_weight': [{0: i, 1: 1} for i in np.arange(0.1, 1, 0.1)]},
            'rf_c': {'n_estimators': self.param_range('int', 50, 250, 10, br),
                    'max_depth': self.param_range('int', 2, 30, 2, br),
                    'min_samples_leaf': self.param_range('int', 1, 10, 1, br),
                    'max_features': self.param_range('real', 0.1, 1, 0.1, br),
                    'class_weight': [{0: i, 1: 1} for i in np.arange(0.1, 1, 0.1)]},
            'lgbm_c': {'n_estimators': self.param_range('int', 50, 250, 25, br),
                     'max_depth': self.param_range('int', 2, 30, 3, br),
                     'colsample_bytree': self.param_range('real', 0.2, 1, 0.2, br),
                     'subsample':  self.param_range('real', 0.2, 1, 0.2, br),
                     'reg_lambda': self.param_range('int', 0, 1000, 100, br),
                     'class_weight': [{0: i, 1: 1} for i in np.arange(0.1, 1, 0.1)]},
            'xgb_c': {'n_estimators': self.param_range('int', 50, 250, 25, br),
                     'max_depth': self.param_range('int', 2, 30, 2, br),
                     'colsample_bytree': self.param_range('real', 0.2, 1, 0.2, br),
                     'subsample':  self.param_range('real', 0.2, 1, 0.2, br),
                     'reg_lambda': self.param_range('int', 0, 1000, 50, br),
                     'scale_pos_weight': self.param_range('real', 1, 10, 0.5, br)},
            'gbm_c': {'n_estimators': self.param_range('int', 50, 250, 20, br),
                    'max_depth': self.param_range('int', 2, 30, 3, br),
                    'min_samples_leaf': self.param_range('int', 1, 10, 2, br),
                    'max_features': self.param_range('real', 0.1, 1, 0.2, br),
                    'subsample': self.param_range('real', 0.1, 1, 0.2, br)},
            'knn_c': {'n_neighbors':  self.param_range('int',1, 30, 1, br),
                    'weights': self.param_range('cat',['distance', 'uniform'], None, None, br),
                    'algorithm': self.param_range('cat', ['auto', 'ball_tree', 'kd_tree', 'brute'], None, None, br)},
        }

        # initialize the parameter dictionary
        params = {}

        # get all the named steps in the pipe
        steps = pipe.named_steps

        #-----------------
        # Create a dict using named pipe step prefixes + hyperparameters
        #-----------------

        # begin looping through each step
        for step, _ in steps.items():

            # if the step has default params, then loop through hyperparams and add to dict
            if step in param_options.keys():
                for hyper_param, value in param_options[step].items():
                    params[f'{step}__{hyper_param}'] = value

            # if the step is feature union go inside and find steps within 
            if step == 'feature_union':
                outer_transform = pipe.named_steps[step].get_params()['transformer_list']

                # add each feature union step prefixed by feature_union
                for inner_step in outer_transform:
                    for hyper_param, value in param_options[inner_step[0]].items():
                        params[f'{step}__{inner_step[0]}__{hyper_param}'] = value
                
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


    def random_search(self, pipe_to_fit, X, y, params, cv=5, n_iter=50, scoring='neg_mean_squared_error'):

        search = RandomizedSearchCV(pipe_to_fit, params, n_iter=n_iter, cv=cv, scoring=scoring, refit=True)
        best_model = search.fit(X, y)

        return best_model.best_estimator_


    def bayes_search(self, pipe_to_fit, X, y, params, cv=5, n_iter=50, random_starts=25, 
                     scoring='neg_mean_squared_error', verbose=0):

        search = BayesSearchCV(pipe_to_fit, params, n_iter=n_iter, scoring=scoring, refit=True,
                                cv=cv, optimizer_kwargs={'n_initial_points': random_starts}, verbose=verbose)
        best_model = search.fit(X, y)

        return best_model.best_estimator_


    def cv_score(self, model, X, y, cv=5, scoring='neg_mean_squared_error', 
                 n_jobs=1, return_mean=True):

        score = cross_val_score(model, X, y, cv=cv, n_jobs=n_jobs, scoring=scoring)
        if return_mean:
            score = np.mean(score)
        return score


    def cv_predict(self, model, X, y, cv=5, n_jobs=-1):
        pred = cross_val_predict(model, X, y, cv=cv, n_jobs=n_jobs)
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


    def time_series_cv(self, model, X, y, params, col_split, time_split, n_splits=5, n_iter=50):
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
        
        # arrays to hold all predictions and actuals
        hold_predictions = np.array([])
        hold_actuals = np.array([])
        val_predictions = np.array([])
        
        # list to hold the best models
        best_models = []

        #----------------
        # Run the KFold train-prediction loop
        #----------------
        kf = KFold(n_splits=n_splits)
        for val_idx, hold_idx in kf.split(X_val_hold):
            
            print('-------')

            # split the val/hold dataset into random validation and holdout sets
            X_val, X_hold = X_val_hold.iloc[val_idx,:], X_val_hold.iloc[hold_idx,:]
            y_val, y_hold = y_val_hold.iloc[val_idx], y_val_hold.iloc[hold_idx]

            # concat the current training set using train and validation folds
            X_train = pd.concat([X_train_only, X_val], axis=0).reset_index(drop=True)
            y_train = pd.concat([y_train_only, y_val], axis=0).reset_index(drop=True)
            
            # get the CV time splits and find the best model
            cv_time = self.cv_time_splits(col_split, X_train, time_split)

            if self.model_obj=='class':
                best_model = self.random_search(model, X_train, y_train, params, cv=cv_time, 
                                                n_iter=n_iter, scoring=self.scorer('matt_coef'))
            elif self.model_obj=='reg':
                best_model = self.random_search(model, X_train, y_train, params, cv=cv_time, n_iter=n_iter)

            # score the best model on validation and holdout sets
            _, val_sc = self.val_scores(best_model, X_train, y_train, cv=cv_time)
            _, hold_sc = self.test_scores(best_model, X_hold, y_hold)
            mean_val_sc.append(val_sc); mean_hold_sc.append(hold_sc)
            best_models.append(best_model)

            # get the holdout and validation predictions and store
            hold_predictions = np.concatenate([hold_predictions, best_model.predict(X_hold)])
            hold_actuals = np.concatenate([hold_actuals, y_hold])

            cv_time = self.cv_time_splits(col_split, X, time_split)
            val_pred_cur = self.cv_predict_time(best_model, X, y, cv_time)
            val_predictions = np.append(val_predictions, np.array(val_pred_cur))

        # calculate the mean scores
        mean_scores = [round(np.mean(mean_val_sc), 3), round(np.mean(mean_hold_sc), 3)]
        print('Mean Scores:', mean_scores)

        # aggregate all the prediction for val, holds, and combined val/hold
        val_predictions = np.mean(val_predictions.reshape(n_splits, len(val_pred_cur)), axis=0)
        oof_data = {
            'val': val_predictions, 
            'hold': hold_predictions,
            'combined': np.mean([val_predictions, hold_predictions], axis=0),
            'actual': hold_actuals
            }

        return best_models, mean_scores, oof_data


    def val_scores(self, model, X, y, cv):

        if self.model_obj == 'reg':
            mse = self.cv_score(model, X, y, cv=cv, scoring=self.scorer('mse'))
            r2 = self.cv_score(model, X, y, cv=cv, scoring=self.scorer('r2'))
            r2_fit = model.fit(X, y).score(X, y)

            for v, m in zip(['Val MSE:', 'Val R2:', 'Fit R2:'], [mse, r2, r2_fit]):
                print(v, round(m, 3))

            return mse, r2

        elif self.model_obj == 'class':
            matt_coef = self.cv_score(model, X, y, cv=cv, scoring=self.scorer('matt_coef'))
            f1 = self.cv_score(model, X, y, cv=cv, scoring=self.scorer('f1'))

            for v, m in zip(['Val MC:', 'Val F1:'], [matt_coef, f1]):
                print(v, round(m, 3))

            return f1, matt_coef 

    def test_scores(self, model, X, y):

        if self.model_obj == 'reg':
            mse = mean_squared_error(y, model.predict(X))
            r2 = r2_score(y, model.predict(X))
            
            for v, m in zip(['Test MSE:', 'Test R2:'], [mse, r2]):
                print(v, round(m, 3))

            return mse, r2

        elif self.model_obj == 'class':
            matt_coef = matthews_corrcoef(y, model.predict(X))
            f1 = f1_score(y, model.predict(X))

            for v, m in zip(['Test MC:', 'Test F1:'], [matt_coef, f1]):
                print(v, round(m, 3))

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

        
    def best_stack(self, est, X_stack, y_stack, n_iter=500, print_coef=True, run_adp=False):

        X_stack_shuf = X_stack.sample(frac=1, random_state=1234).reset_index(drop=True)
        y_stack_shuf = y_stack.sample(frac=1, random_state=1234).reset_index(drop=True)

        stack_params = self.default_params(est)
        stack_params['k_best__k'] = range(1, X_stack_shuf.shape[1])

        if self.model_obj=='class':
            best_model = self.random_search(est, X_stack, y_stack, stack_params, cv=5, 
                                                n_iter=n_iter, scoring=self.scorer('matt_coef'))
        elif self.model_obj=='reg':
            best_model = self.random_search(est, X_stack, y_stack, stack_params, cv=5, n_iter=n_iter)

        if run_adp:
            # print the OOS scores for ADP and model stack
            print('ADP Score\n--------')
            adp_col = [c for c in X_stack.columns if 'adp' in c]
            _, _ = self.val_scores(self.piece('lr')[1], X_stack_shuf[adp_col], y_stack_shuf, cv=5)

        print('\nStack Score\n--------')
        _, _ = self.val_scores(best_model, X_stack_shuf, y_stack_shuf, cv=5)

        if print_coef:
            imp_cols = X_stack_shuf.columns[best_model['k_best'].get_support()]
            self.print_coef(best_model, imp_cols)

        return best_model


    



