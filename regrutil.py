from regrhelper import  NNRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

def validate(nnr):
  print("NNRegressor, train data size is          : ", nnr.train_data.shape)
  print("NNRegressor, train lables size is        : ", nnr.train_data_lables.shape)

  if hasattr(nnr, 'test_data'):
    print("NNRegressor, test data size is          : ", nnr.train_data.shape)
  if hasattr(nnr, 'train_data_prepared'):
    print("NNRegressor, train prepared data size is : ", nnr.train_data_prepared.shape)
  if hasattr(nnr, 'test_data_prepared'):
    print("NNRegressor, test prepared data size is : ", nnr.test_data_prepared.shape)

    print("Train data has NAN values                           : ", nnr.train_data.isnull().values.any())
  if hasattr(nnr, 'test_data'):
    print("Test data has NAN values                           : ", nnr.test_data.isnull().values.any())

def validate_model(pipeline):
  print("Pipeline transformers:\n", pipeline.transformers_)
  
def hist(data):
  data.hist(bins=20, figsize=(10,7))
  plt.show()

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

def show_features_importances(nnr, num_to_plot = 10, save_to_file = False, debug = False):
  onehotencoder = nnr.full_pipeline.named_transformers_['cat']

  features = pd.Index([c for c in nnr.train_data.columns if nnr.train_data[c].dtype.name != 'object'])
  cat_attribs = nnr.train_data.drop(features, axis = 1)
  cat_indices = pd.Index(onehotencoder.get_feature_names())
  features = features.append(cat_indices)
   
  importances = nnr.regressor.feature_importances_
  indices = np.argsort(importances)[::-1]
  feature_indices = [ind for ind in indices[:num_to_plot]]

  features_save_to_file = []

  print("Feature ranking:\n")
  for f in range(num_to_plot):
    if debug == True:
      print("%d. %s %f " % (f + 1, features[indices[f]], importances[indices[f]]))
    if re.match(r'x[0-9]+_', features[indices[f]]):
      feature = cat_attribs.columns[int(features[indices[f]][1])]
      if feature not in features_save_to_file:
        features_save_to_file.append(feature)
    else:
      features_save_to_file.append(features[indices[f]])
  
  if save_to_file:
    pd.DataFrame(features_save_to_file).to_csv("features.csv", index=False)
  
  if debug == False:
    return
  
  plt.figure(figsize=(15,5))
  plt.title(u"Значимость признаков")
  bars = plt.bar(range(num_to_plot), 
               importances[indices[:num_to_plot]],
               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),
               align="center")
  plt.xticks(range(num_to_plot), feature_indices)

  plt.xlim([-1, num_to_plot])
  plt.legend(bars, [u''.join(features[i]) for i in feature_indices])