from skmodel.data_setup import DataSetup
from sklearn.pipeline import FeatureUnion#,Pipeline
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import numpy as np

# feature selection
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel, SelectKBest,SelectPercentile, mutual_info_regression, f_regression, f_classif
from sklearn.cluster import FeatureAgglomeration

# import all various models
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression, BayesianRidge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import VotingRegressor, VotingClassifier, StackingRegressor, StackingClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from ngboost import NGBRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import LinearSVR, LinearSVC

class FeatureExtractionSwitcher(BaseEstimator):

    def __init__(self, estimator=FeatureAgglomeration()):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 
        self.estimator = estimator


    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self


    def transform(self, X, y=None):
        return self.estimator.transform(X)


class PercentileCut(BaseEstimator, TransformerMixin):

    def __init__(self, cutoff=50, greater_or_less='greater_equal', verbose=False):
        self.cutoff = cutoff
        self.greater_or_less = greater_or_less
        self.verbose = verbose

    def calc_perc(self, X):
        if self.greater_or_less == 'greater_equal':
            return  np.where(X >= np.percentile(X, self.cutoff), 1, 0)
       
        elif self.greater_or_less == 'less_equal':
            return  np.where(X <= np.percentile(X, self.cutoff), 1, 0)
        
        else:
            print("Must specify greater_or_less = ('greater_equal', 'less_equal'). X not transformed.")
            return X

    def fit(self, X):
        return self

    def transform(self, X):
        return self.calc_perc(X)

    def inverse_transform(self, X):
        return X


class FeatureDrop(BaseEstimator,TransformerMixin):

    def __init__(self, col=[]):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 
        self.col = col


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        if self.col is None:
            return X
        else:
            return X.drop(self.col, axis=1)


class FeatureSelect(BaseEstimator, TransformerMixin):

    def __init__(self, cols=[]):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 
        self.cols = cols
        self.n_features_in_ = len(cols)


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        return X.loc[:, self.cols]


class RandomSample(BaseEstimator, TransformerMixin):

    def __init__(self, frac, seed=1234):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 
        self.frac = frac
        self.seed = seed

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.sample(frac=self.frac, axis=1, random_state=self.seed)


class SelectAtMostKBest(SelectKBest):

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            # set k to "all" (skip feature selection), if less than k features are available
            self.k = "all"


class PipeSetup(DataSetup):

    def __init__(self, data, model_obj):
        self.data = data
        self.model_obj = model_obj

    
    def model_pipe(self, pipe_steps):
        """Internal function that accepts a dictionary and creates Pipeline

        Args:
            pipe_steps (dict, optional): Dictionary with key-value for steps in Pipeline.

        Returns:
            sklearn.Pipeline: Pipeline object with specified steps embedded.
        """
        return Pipeline(steps=[ps for ps in pipe_steps])


    def feature_union(self, pieces):
        """Create FeatureUnion object of multiple pieces

        Args:
            pieces (list): List of pieces to combine features together with

        Returns:
            tuple: ('feature_union', FeatureUnion object with specified pieces)
        """        
        return ('feature_union', FeatureUnion(pieces))


    def piece(self, label, label_rename=None):
        """Helper function to return piece or component of pipeline.

        Args:
            label (str): Label of desired component to return. To return all
                         choices use function return_piece_options().
                         
                         Options: Preprocessing
                                  ----
                                  'one_hot' = OneHotEncoder()
                                  'impute' = SimpleImputer()
                                  'std_scale' = OneHotEncoder() 
                                  'min_max_scale' = MinMaxScaler()
                                  'smote' = SMOTE()
                                  'y_perc' = PercentileCut()

                                  Feature Selection
                                  ----
                                  'feature_select' = FeatureSelect()
                                  'feature_drop' = FeatureDrop()
                                  'select_perc' = SelectPercentile(score_func=f_regression)
                                  'select_perc_c' = SelectPercentile(score_func=f_classif)
                                  'k_best' = SelectKBest(score_func=f_regression, k=10)
                                  'k_best_c' = SelectKBest(score_func=f_classif, k=10)
                                  'select_from_model' = SelectFromModel()
                                  'feature_switcher' = FeatureExtractionSwitcher()
                                   
                                  Feature Transformation
                                  ----
                                  'pca' = PCA()
                                  'agglomeration' = FeatureAgglomeration()
                                  
                                  Regression Models
                                  ----
                                  'lr' = LinearRegression()
                                  'ridge' = Ridge()
                                  'lasso' = Lasso()
                                  'enet' = ElasticNet()
                                  'rf' = RandomForestRegressor()
                                  'xgb' = XGBRegressor()
                                  'lgbm' = LGBMRegressor()
                                  'knn' = KNeighborsRegressor()
                                  'gbm' = GradientBoostingRegressor()
                                  'svr' = LinearSVR()
                                  'ngb' = NGBRegressor()
                                  'bridge' = BayesianRidge()

                                  Classification Models
                                  ----
                                  'lr_c' = LogisticRegression()
                                  'rf_c' = RandomForestClassifier()
                                  'xgb_c' = XGBClassifier()
                                  'lgbm_c' = LGBMClassifier()
                                  'knn_c' = KNeighborsClassifier()
                                  'gbm_c' = GradientBoostingClassifier()
                                  'svc' = LinearSVC()

        Returns:
            tuple: (Label, Object) tuple of specified component
        """      
        piece_options = {

                # data transformation
                'one_hot': OneHotEncoder(handle_unknown='ignore'),
                'impute': SimpleImputer(),
                'std_scale': StandardScaler(),
                'min_max_scale': MinMaxScaler(),
                'smote': SMOTE(),
                'y_perc': PercentileCut(),

                #feature selection
                'random_sample': RandomSample(frac=1),
                'feature_select': FeatureSelect(['avg_pick']),
                'feature_drop': FeatureDrop(),
                'select_perc': SelectPercentile(score_func=f_regression, percentile=10),
                'select_perc_c': SelectPercentile(score_func=f_classif, percentile=10),
                'select_from_model': SelectFromModel(estimator=Ridge()),
                'k_best': SelectAtMostKBest(score_func=f_regression, k=10),
                'k_best_c': SelectAtMostKBest(score_func=f_classif, k=10),
                'feature_switcher': FeatureExtractionSwitcher(),

                # feature transformation
                'pca': PCA(),
                'agglomeration': FeatureAgglomeration(),

                # regression algorithms
                'lr': LinearRegression(),
                'ridge': Ridge(),
                'lasso': Lasso(),
                'enet': ElasticNet(),
                'rf': RandomForestRegressor(n_jobs=1),
                'xgb': XGBRegressor(n_jobs=1),
                'lgbm': LGBMRegressor(n_jobs=1, verbose=-1),
                'knn': KNeighborsRegressor(n_jobs=1),
                'gbm': GradientBoostingRegressor(),
                'svr': LinearSVR(),
                'ngb': NGBRegressor(),
                'bridge': BayesianRidge(),

                # quantile regression
                'gbm_q': GradientBoostingRegressor(loss='quantile'),
                'lgbm_q': LGBMRegressor(objective='quantile', verbose=0, n_jobs=1),

                # classification algorithms
                'lr_c': LogisticRegression(),
                'rf_c': RandomForestClassifier(n_jobs=1),
                'xgb_c': XGBClassifier(n_jobs=1),
                'lgbm_c': LGBMClassifier(n_jobs=1, verbose=0),
                'knn_c': KNeighborsClassifier(n_jobs=1),
                'gbm_c': GradientBoostingClassifier(),
                'svc': LinearSVC()
            }

        piece = (label, piece_options[label])

        if label_rename:
            piece = self.piece_rename(piece, label_rename)

        return piece

    def piece_rename(self, piece, label_rename):
        """Rename the label of each piece

        Args:
            piece (tuple): Tuple of (label, object) to rename
            label_rename (str): Name to rename the tuple label with

        Returns:
            tuple: Renamed (label, object) tuple for given piece
        """
        piece = list(piece)
        piece[0] = label_rename

        return tuple(piece)

    
    def column_transform(self, num_pipe, cat_pipe):
        """Function that combines numeric and categorical pipeline transformers

        Args:
            num_pipe (sklearn.Pipeline): Numeric transformation pipeline
            cat_pipe (sklearn.Pipeline): Categorical transformation pipeline

        Returns:
            sklear.ColumnTransformer: ColumnTransformer object with both numeric 
                                      and categorical trasnformation steps
        """        
        return ('column_transform', ColumnTransformer(
                    transformers=[
                        ('numeric', Pipeline(num_pipe), self.num_features),
                        ('cat', Pipeline(cat_pipe), self.cat_features)
                    ])
               )


    def target_transform(self, pipe, y_transform):
        """Function that combines numeric and categorical pipeline transformers

        Args:
            pipe (sklearn.Pipeline): Full model pipe to train
            y_transform (sklearn.Transform): Transformer object to modify y-variable

        Returns:
            sklearn.TransformedTargetRegressor: Regressor object that can be trained
            with a modified target variable
        """        
        return TransformedTargetRegressor(regressor=pipe, transformer=y_transform)
    

    def ensemble_pipe(self, pipes):
        """Create a mean ensemble pipe where individual pipes feed into 
           a mean voting ensemble model.

        Args:
            pipes (list): List of pipes that will have their outputs averaged

        Returns:
            Pipeline: Pipeline object that has multiple multiple feeding Voting object
        """
        ests = []
        for i, p in enumerate(pipes):
            ests.append((f'p{i}', p))

        if self.model_obj in ('reg', 'quantile'):
            ensemble = VotingRegressor(estimators=ests, n_jobs=-1)
        elif self.model_obj == 'class':
            ensemble = VotingClassifier(estimators=ests, voting='soft', n_jobs=-1)
        
        return Pipeline([('ensemble', ensemble)])

    
    def ensemble_params(self, pipe_params):
        """Create the dictionary of all ensemble params for list of pipes

        Args:
            pipe_params (list): List of dictionary parameters for each pipe

        Returns:
            dict: Large dictionary containing all parameters within ensemble
        """        
        stack_params = {}
        for i, p in enumerate(pipe_params):
            for k, v in p.items():
                stack_params[f'ensemble__p{i}__{k}'] = v

        return stack_params


    def stack_pipe(self, pipes, final_estimator):
        """Create a stacking ensemble pipe where individual pipes feed into
           a final stacking estimator model.

        Args:
            pipes (list): List of pipes that will have their outputs averaged
            final_estimator (sklearn.estimator): Estimator that will fit on model input predictions

        Returns:
            skklearn.StackingEstimator: Stacked estimator that will train on other model inputs
        """
        ests = []
        for i, p in enumerate(pipes):
            ests.append((f'stack_p{i}', p))

        if self.model_obj == 'reg':
            return StackingRegressor(pipes, final_estimator=final_estimator, n_jobs=-1)
        if self.model_obj == 'class':
            return StackingClassifier(pipes, final_estimator=final_estimator, n_jobs=-1)