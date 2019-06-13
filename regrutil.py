from regrhelper import  NNRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn.decomposition import PCA


# def validate(nnr) - Выводит геометрию и NAN значения для тренировочного и испытательного набора
# def show_pipiline_info(pipeline)
# def show_hist(data)
# def display_scores(scores)
# def show_features_importances(nnr, num_to_plot = 10, save_to_file = False) - Печатает/рисует сортированный список важности признака согласно feature_importances_
# def show_searchCV_results(nnr) - Печатает результаты тюнинга с помощью searchCV
# def show_pca_explained_variances(nnr) - Печатает объясненную дисперсию для PCA


# Выводит геометрию и NAN значения для тренировочного и испытательного набора
def validate(nnr):
  print("NNRegressor, train data size is              : ", nnr.train_data.shape)
  if hasattr(nnr, 'test_data'):
    print("NNRegressor, test data size is             : ", nnr.test_data.shape)

  print("NNRegressor, train lables size is            : ", nnr.train_data_lables.shape)
  if hasattr(nnr, 'test_data_lables'):
    print("NNRegressor, test lables size is           : ", nnr.test_data_lables.shape)

  if hasattr(nnr, 'train_data_prepared'):
    print("NNRegressor, train prepared data size is   : ", nnr.train_data_prepared.shape)
  if hasattr(nnr, 'test_data_prepared'):
    print("NNRegressor, test prepared data size is    : ", nnr.test_data_prepared.shape, "\n")

  print("Train data has NAN values                    : ", nnr.train_data.isnull().values.any())
  if hasattr(nnr, 'test_data'):
    print("Test data has NAN values                   : ", nnr.test_data.isnull().values.any())

def show_pipiline_info(pipeline):
  print("Pipeline transformers:\n", pipeline.transformers_)
  
def show_hist(data):
  data.hist(bins=20, figsize=(10,7))
  plt.show()

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# Печатает сортированный список важности признака согласно feature_importances_
# Рисует графическое представление этой информации в виде баров. Может сохранять в файл
def show_features_importances(nnr, num_to_plot = 10, save_to_file = False):
  features = pd.Index([c for c in nnr.train_data.columns if nnr.train_data[c].dtype.name != 'object'])
  cat_attribs = nnr.train_data.drop(features, axis = 1)

  try:
    onehotencoder = nnr.full_pipeline.named_transformers_['cat']
    cat_indices = pd.Index(onehotencoder.get_feature_names())
    features = features.append(cat_indices)
  except:
    print('CAT doesnt present')

  try:
    importances = nnr.regressor.feature_importances_
  except:
    print(type(nnr.regressor).__name__, "has no attribute feature_importances_")
    return

  indices = np.argsort(importances)[::-1]
  feature_indices = [ind for ind in indices[:num_to_plot]]

  features_save_to_file = []

  print(type(nnr.regressor).__name__, "feature ranking:")
  for f in range(num_to_plot):
    print("%d. %s %f " % (f + 1, features[indices[f]], importances[indices[f]]))
    if re.match(r'x[0-9]+_', features[indices[f]]):
      feature = cat_attribs.columns[int(features[indices[f]][1])]
      if feature not in features_save_to_file:
        features_save_to_file.append(feature)
    else:
      features_save_to_file.append(features[indices[f]])

  if save_to_file:
    pd.DataFrame(features_save_to_file).to_csv("features.csv", index=False)
  
  plt.figure(figsize=(15,5))
  plt.title(u"Значимость признаков")
  bars = plt.bar(range(num_to_plot), 
               importances[indices[:num_to_plot]],
               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),
               align="center")
  plt.xticks(range(num_to_plot), feature_indices)

  plt.xlim([-1, num_to_plot])
  plt.legend(bars, [u''.join(features[i]) for i in feature_indices])

# Печатает результаты тюнинга с помощью searchCV
def show_searchCV_results(nnr):
  try:
    cvres = nnr.searchCV.cv_results_
    print('CV:', type(nnr.searchCV).__name__)
    for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
      print(np.sqrt(-mean_score), params)
  except:
    print('NNRegressor does not use SearchCV')

# Печатает объясненную дисперсию для PCA
def show_pca_explained_variances(nnr):
  for s in nnr.full_pipeline.named_transformers_['num'].steps:
    if s[0] == 'pca':
      print('PCA explained variance ratio is\n', s[1].explained_variance_ratio_)
      return
