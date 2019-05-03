import pandas as pd

from sklearn.feature_selection import VarianceThreshold

def varianceThreshold(data, threshold):
  cat_attribs = data.filter(items = [c for c in data.columns if data[c].dtype.name != 'int64' and data[c].dtype.name != 'float64'])
  num_attribs = data.drop(cat_attribs, axis=1)
  
  vt = VarianceThreshold(threshold)
  vt.fit(num_attribs)
  
  threshold_attribs = [num_attribs.columns[x] for x in vt.get_support(indices=True)]
  num_attribs = data.filter(items=threshold_attribs, axis=1)
  return pd.concat((num_attribs, cat_attribs), axis=1)