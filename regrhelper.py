import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer
try:
    from sklearn.preprocessing import OrdinalEncoder # just to raise an ImportError if Scikit-Learn < 0.20
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    from future_encoders import OneHotEncoder # Scikit-Learn < 0.20
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    from future_encoders import ColumnTransformer # Scikit-Learn < 0.20

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import os

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression

import regr_prepare_data as rpd

class NNRegressor:
    # Читает обучающий набор из data_url. Если метки лежат в отдельном файле,
    # то они читаются из label_url
    
    def __init__(self, poly_features = False, poly_degree = 2, interaction_only = False):
        self.poly_features = poly_features
        self.poly_degree = poly_degree
        self.interaction_only = interaction_only
      
    def read_train_data(self, data_url, label_url = 'nan', threshold = 0):
        self.train_data = pd.read_csv(data_url)
        if os.path.isfile('features.csv'):
          used_columns = pd.read_csv('features.csv').iloc[0:, 0]
          self.train_data = self.train_data.filter(items = used_columns, axis = 1)

        dropped_items, self.train_data = rpd.varianceThreshold(self.train_data, threshold)

        if label_url != 'nan':
            self.train_data_lables = pd.read_csv(label_url).iloc[0:, 0]

    def read_test_data(self, data_test_url):
        self.test_data = pd.read_csv(data_test_url)
        self.test_data = self.test_data.filter(items = self.train_data.columns, axis = 1)

    def build_full_pipeline(self):
        steps = [
                ('imputer', SimpleImputer(strategy="median")),
                ('minmax_scaler', MinMaxScaler()),
        ]
        
        if self.poly_features == True:
          steps.insert(0, ('poly_features', PolynomialFeatures(self.poly_degree, self.interaction_only)))
          
        num_pipeline = Pipeline(steps)
        
        cat_attribs = [c for c in self.train_data.columns if self.train_data[c].dtype.name == 'object']
        num_attribs = [c for c in self.train_data.columns if self.train_data[c].dtype.name != 'object']
        
        self.full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),
        ])

    def prepare_train_data(self):
        self.build_full_pipeline()
        self.train_data_prepared = self.full_pipeline.fit_transform(self.train_data)

    def prepare_regressor(self, regressor, n_estimators=10, n_neighbors=5, max_depth=None, min_samples_leaf=1, bootstrap=True):
        if regressor == 'LinearRegression':
            self.regressor = LinearRegression()
        elif regressor =='DecisionTreeRegressor':
            self.regressor = DecisionTreeRegressor(random_state=42, max_depth = max_depth,
                                                   min_samples_leaf = min_samples_leaf)
        elif regressor == 'RandomForestRegressor':
            self.regressor = RandomForestRegressor(n_estimators, random_state=42, n_jobs = -1,
                                                   min_samples_leaf = min_samples_leaf, bootstrap = bootstrap)
            self.params_dict = [
                {'n_estimators': [100, 300, 500], 'max_features': [2, 4, 6, 8]},
            ]
        elif regressor == 'ExtraTreesRegressor':
            self.regressor = ExtraTreesRegressor(n_estimators, bootstrap = bootstrap, random_state=42)
            self.params_dict = [
                {'n_estimators': [100, 300, 500], 'max_features': [2, 4, 6, 8]},
            ]
        elif regressor == 'KNeighborsRegressor':
            self.regressor = KNeighborsRegressor(n_neighbors)
            self.params_dict = [{'n_neighbors':[2,3,4,5,6,7,8,9]}]
        elif regressor == 'svr':
            self.regressor = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
        elif regressor == 'AdaBoostRegressor':
            self.regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators, random_state=42)
        else:
            print("Unsupported regressor ", regressor)

    def train_predict(self):
        y_predicted = self.final_model.predict(self.train_data_prepared)
        mape = np.mean(np.abs((self.train_data_lables - y_predicted)/self.train_data_lables))
        print('mape for train data', mape)

    def no_tune_model(self):
        self.final_model = self.regressor
        self.final_model.fit(self.train_data_prepared, self.train_data_lables)
        
        print('best estimator is \n', self.final_model)
        self.train_predict()

    def tune_model(self, searchCV, params_cv = 'nan'):
        if params_cv != 'nan':
          self.params_dict = params_cv
            
        if searchCV == 'GridSearchCV':
          self.searchCV = GridSearchCV(self.regressor, self.params_dict, cv=5, scoring='neg_mean_squared_error',
                                       return_train_score=True)
        elif searchCV == 'RandomizedSearchCV':
          self.searchCV = RandomizedSearchCV(self.regressor, param_distributions=self.params_dict, n_iter=10, cv=5,
                                             scoring='neg_mean_squared_error', random_state=42)
        else:
            print('Unsupported tune method', searchCV)
            return

        self.searchCV.fit(self.train_data_prepared, self.train_data_lables)
        self.final_model = self.searchCV.best_estimator_
        print('best estimator is \n', self.final_model)
        self.train_predict()

    def predict(self, csv_file_to_save):
        self.test_data_prepared = self.full_pipeline.transform(self.test_data)
        pd.DataFrame(self.final_model.predict(self.test_data_prepared)).to_csv(
           csv_file_to_save, header=None, index=False)
