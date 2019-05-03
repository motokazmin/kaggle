from regrhelper import  NNRegressor
import matplotlib.pyplot as plt

def validate(nnr):
  print("NNRegressor, train data size is          : ", nnr.train_data.shape)
  print("NNRegressor, train lables size is        : ", nnr.train_data_lables.shape)
  if hasattr(nnr, 'train_data_prepared'):
    print("NNRegressor, train prepared data size is : ", nnr.train_data_prepared.shape)
  print("Has NAN values                           : ", nnr.train_data.isnull().values.any())

def hist(data):
  data.hist(bins=20, figsize=(10,7))
  plt.show()