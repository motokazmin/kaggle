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

from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import os
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

import regr_prepare_data as rpd

class NNRegressor:
    # Читает обучающий набор из data_url. Если метки лежат в отдельном файле,
    # то они читаются из label_url
    
    def __init__(self, poly_features = False, poly_degree = 2, interaction_only = False):
        self.poly_features = poly_features
        self.poly_degree = poly_degree
        self.interaction_only = interaction_only

    # Читает данные для тренировки. Если label = 'None', тогда предполагается, что
    # метки располагаются в том же файле, что и данные. Иначе метки читаются из
    # отдельного файла label. Таргет указывается параметром target.
    # После чтения данных происходит их предобработка:
    #  удаление категориальных признаков|, если параметр drop_cat = True
    #  удаление признаков, важность которых меньше threshold
    #  данные разбиваются на наборы для тренировки и тестирования согласно параметру test_size
    # Также возможно указать, какие колонки из исходных данных должны быть использованы при работе.
    # Это используется, если существует файл features.csv
    def read_train_data(self, data_url, label_url = 'None', target = 'None', threshold = 0):
        if target == 'None':
          print('No target name')
          return

        self.target = target
        self.rtrain_data = pd.read_csv(data_url)

        if os.path.isfile('features.csv'):
          used_columns = pd.read_csv('features.csv').iloc[0:, 0]
          self.rtrain_data = self.rtrain_data.filter(items = used_columns, axis = 1)

        dropped_items, self.rtrain_data = rpd.varianceThreshold(self.rtrain_data, threshold)

        self.train_data_lables_url = label_url

    def build_full_pipeline(self):
        steps = [
                ('imputer', SimpleImputer(strategy="median")),
                ('minmax_scaler', MinMaxScaler()),
        ]
        
        if self.use_pca == True:
          try:
            pca = nnr.full_pipeline.named_transformers_['pca']
          except:
            steps.insert(1, ('pca', PCA(n_components=0.991)))
          
        num_pipeline = Pipeline(steps)
        
        cat_attribs = [c for c in self.train_data.columns if self.train_data[c].dtype.name == 'object']
        num_attribs = [c for c in self.train_data.columns if self.train_data[c].dtype.name != 'object']
        
        self.full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),
        ])

    def prepare_train_data(self, use_pca, drop_cat = False, test_size=0.2):
      if self.train_data_lables_url != 'None':
        self.train_data_lables = pd.read_csv(self.train_data_lables_url).iloc[0:, 0]

      self.use_pca = use_pca
      self.train_data = self.rtrain_data.copy()
      
      if drop_cat == True:
        self.train_data = rpd.drop_category_attrs(self.train_data)

      self.full_data = pd.concat((self.train_data, self.train_data_lables), axis=1)

      full_train_data, full_test_data = train_test_split(self.full_data, test_size=test_size, random_state=42)
      
      self.train_data_lables = full_train_data.filter([self.target], axis=1)
      self.train_data = full_train_data.drop([self.target], axis=1)

      self.test_data_lables = full_test_data.filter([self.target], axis=1)
      self.test_data = full_test_data.drop([self.target], axis=1)

      self.build_full_pipeline()
      self.train_data_prepared = self.full_pipeline.fit_transform(self.train_data)

    def prepare_regressor(self, regressor, n_estimators=10, n_neighbors=5, max_depth=None, min_samples_leaf=1,
                          min_samples_split = 2, criterion = 'mse'):
        if regressor == 'LinearRegression':
            self.regressor = LinearRegression()
        elif regressor =='DecisionTreeRegressor':
            self.regressor = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                   criterion=criterion, splitter='best', random_state=42)
        elif regressor == 'RandomForestRegressor':
            self.regressor = RandomForestRegressor(n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                   criterion=criterion, n_jobs = -1, random_state=42)
            self.params_dict = [
                {'n_estimators': [500], 'max_features': [2, 4, 6, 8]},
            ]
        elif regressor == 'ExtraTreesRegressor':
            self.regressor = ExtraTreesRegressor(n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                 criterion=criterion, n_jobs = -1, random_state=42)
            self.params_dict = [
                {'criterion': ['mse'], 'n_estimators' : [500],
                 'min_samples_leaf': [1], 'min_samples_split' : [2]},
            ]
        elif regressor == 'KNeighborsRegressor':
            self.regressor = KNeighborsRegressor(n_neighbors)
            self.params_dict = [{'n_neighbors':[2,3,4,5,6,7,8,9]}]
        elif regressor == 'SVR':
            self.params_dict = [
                {'kernel':['poly'], 'degree' : [3], 'gamma': [5.0, 1.0, 'scale'], 'C' : [100, 200, 300], 'cache_size' : [500]},
            ]
            self.regressor = LinearSVR(loss='squared_epsilon_insensitive', dual=False, tol=1e-5, C=100)
        elif regressor == 'AdaBoostRegressor':
            self.regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=None), n_estimators,
                                               random_state=42, loss='exponential', learning_rate =0.05)
        elif regressor == 'XGBRegressor':
            self.regressor = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                          colsample_bytree=0.4, gamma=0, importance_type='gain',
                                          learning_rate=0.07, max_delta_step=0, max_depth=3,
                                          min_child_weight=1.5, missing=None, n_estimators=10000, n_jobs=1,
                                          nthread=None, objective='reg:linear', random_state=0,
                                          reg_alpha=0.75, reg_lambda=0.45, scale_pos_weight=1, seed=42,
                                          silent=True, subsample=0.6)
        elif regressor == 'BaggingRegressor':
            self.regressor = BaggingRegressor(base_estimator=ExtraTreesRegressor(n_estimators, max_depth=max_depth,
                                              min_samples_leaf=min_samples_leaf, criterion=criterion, n_jobs = -1, random_state=42))
        else:
            print("Unsupported regressor ", regressor)

    def no_tune_model(self):
        self.final_model = self.regressor
        self.final_model.fit(self.train_data_prepared, self.train_data_lables.values.ravel())
        
        print('best estimator is \n', self.final_model)
        self.train_predict()

    def predict(self, url, csv_file_to_save='results.csv'):
        validate_data = pd.read_csv(url)
        validate_data = validate_data.filter(items = self.train_data.columns, axis = 1)
        validate_data_prepared = self.full_pipeline.transform(validate_data)
        pd.DataFrame(self.final_model.predict(validate_data_prepared)).to_csv(
           csv_file_to_save, header=None, index=False)

    def tune_model(self, searchCV, params_cv = 'nan'):
        if params_cv != 'nan':
          self.params_dict = params_cv
        if searchCV == 'GridSearchCV':
          self.searchCV = GridSearchCV(self.regressor, self.params_dict, n_jobs = -1, verbose = 51)
        elif searchCV == 'RandomizedSearchCV':
          self.searchCV = RandomizedSearchCV(self.regressor, param_distributions=self.params_dict, n_iter=10, cv=5,
                                             scoring='neg_mean_squared_error', random_state=42, n_jobs = -1)
        else:
            print('Unsupported tune method', searchCV)
            return

        self.searchCV.fit(self.train_data_prepared, self.train_data_lables.values.ravel())
        self.final_model = self.searchCV.best_estimator_
        print('best estimator is \n', self.final_model)
        self.train_predict()

    def train_predict(self):
        self.test_data_prepared = self.full_pipeline.transform(self.test_data)
        y_predicted = self.final_model.predict(self.test_data_prepared)
        mape = np.mean(np.abs((self.test_data_lables[self.target] - y_predicted)/self.test_data_lables[self.target]))
        print('mape for validate set', mape)

        y_predicted = self.final_model.predict(self.train_data_prepared)
        mape = np.mean(np.abs((self.train_data_lables[self.target] - y_predicted)/self.train_data_lables[self.target]))
        print('mape for train    set', mape)
