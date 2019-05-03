import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

import regr_prepare_data as rpd


def drop_numeric_onlyone_unique_params(data):
  num_attribs   = [c for c in data.columns if data[c].dtype.name == 'int64']
  dummy_columns = [c for c in num_attribs if data[c].std() == 0]
  return data.drop(dummy_columns, axis=1)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
   
class NNRegressor:
    # Читает обучающий набор из data_url. Если метки лежат в отдельном файле,
    # то они читаются из label_url
    def read_train_data(self, data_url, label_url = 'nan', threshold = 0.1):
        self.train_data = pd.read_csv(data_url)
        self.train_data = rpd.varianceThreshold(self.train_data, threshold)

        if label_url != 'nan':
            self.train_data_lables = pd.read_csv(label_url).iloc[0:, 0]

    def drop_category_attrs(self):
        cat_attribs = [c for c in self.train_data.columns if self.train_data[c].dtype.name == 'object']
        self.train_data = self.train_data.drop(cat_attribs, axis=1)
        
    def read_test_data(self, data_test_url):
        self.test_data = pd.read_csv(data_test_url)

    def build_full_pipeline(self):
        num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('std_scaler', StandardScaler()),
        ])
        
        cat_attribs = [c for c in self.train_data.columns if self.train_data[c].dtype.name == 'object']
        num_attribs = [c for c in self.train_data.columns if self.train_data[c].dtype.name != 'object']
        
        self.full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),
        ])

    def prepare_train_data(self, pick_data_size = 0):
        if pick_data_size != 0:
            self.pick_data  = self.train_data.iloc[:pick_data_size]
            self.train_data = self.train_data.iloc[pick_data_size + 1:]
            self.pick_data_lables  = self.train_data_lables.iloc[:pick_data_size]
            self.train_data_lables = self.train_data_lables.iloc[pick_data_size + 1:]

        self.build_full_pipeline()
        self.train_data_prepared = self.full_pipeline.fit_transform(self.train_data)

    def prepare_regressor(self, regressor):
        if regressor == 'LinearRegression':
            self.regressor = LinearRegression()
        elif regressor =='DecisionTreeRegressor':
            self.regressor = DecisionTreeRegressor(random_state=42)
        elif regressor == 'RandomForestRegressor':
            self.regressor = RandomForestRegressor(n_estimators=10, random_state=42)
        else:
            print("Unsupported regressor ", regressor)

    def no_tune_model(self):
        self.final_model = self.regressor
        self.final_model.fit(self.train_data_prepared, self.train_data_lables)

    def tune_model(self, searchCV):
        if searchCV == 'GridSearchCV':
            param_grid = [
                {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
            ]
            self.searchCV = GridSearchCV(self.regressor, param_grid, cv=5,
                                         scoring='neg_mean_squared_error', return_train_score=True)
        elif searchCV == 'RandomizedSearchCV':
            param_distribs = {
                'n_estimators': randint(low=1, high=200),
                'max_features': randint(low=1, high=8),
            }
            self.searchCV = RandomizedSearchCV(self.regressor, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
        else:
            print('Unsupported tune method', searchCV)
            return

        self.searchCV.fit(self.train_data_prepared, self.train_data_lables)
        self.final_model = self.searchCV.best_estimator_

    def predict(self, csv_file_to_save = 'nan'):
        self.test_data_prepared = self.full_pipeline.transform(self.test_data)
        if csv_file_to_save == 'nan':
            print("Predictions:\t", self.final_model.predict(self.test_data_prepared))
        else:
            pd.DataFrame(self.final_model.predict(self.test_data_prepared)).to_csv(
               csv_file_to_save, header=None, index=False)

    def predict_pick(self):
        pick_data_prepared = self.full_pipeline.transform(self.pick_data)
        print("Predictions:\t", self.final_model.predict(pick_data_prepared))
        print("lables     :\t", list(self.pick_data_lables))
