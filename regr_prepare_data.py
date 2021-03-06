import pandas as pd

# def varianceThreshold(data, threshold) - Удаляет признаки с дисперсией, меньшей чем threshold
# def drop_category_attrs(data) - Удаляет все категориальные признаки
# def cat_features_to_int_features(data, dtype ='int64', target = 'None') - Трансформирует категориальные признаки в числовые - каждое уникальное значение категориального признака дает новый числовой
# def drop_features_min_value(df, val = 0) - Удаляет из данных признаки, у которых минимальное значение меньше или равняется val
# def drop_feature_unique_count(df, count = 2) - Удаляет из данных признаки, у которых колличество уникальных значений меньше count

from sklearn.feature_selection import VarianceThreshold

try:
    from sklearn.preprocessing import OrdinalEncoder # just to raise an ImportError if Scikit-Learn < 0.20
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    from future_encoders import OneHotEncoder # Scikit-Learn < 0.20

# Удаляет признаки с дисперсией, меньшей чем threshold
def varianceThreshold(data, threshold):
  cat_attribs = data.filter(items = [c for c in data.columns if data[c].dtype.name != 'int64' and data[c].dtype.name != 'float64'])
  num_attribs = data.drop(cat_attribs, axis=1)

  vt = VarianceThreshold(threshold)
  vt.fit(num_attribs)

  threshold_attribs = [num_attribs.columns[x] for x in vt.get_support(indices=True)]

  dropped_attribs = num_attribs.drop(threshold_attribs, axis=1)
  num_attribs = data.filter(items=threshold_attribs, axis=1)
  return dropped_attribs, pd.concat((num_attribs, cat_attribs), axis=1)

# Удаляет все категориальные признаки
def drop_category_attrs(data):
  cat_attribs = [c for c in data.columns if data[c].dtype.name == 'object']
  return data.drop(cat_attribs, axis=1)

# Удаляет из данных признаки, у которых минимальное значение меньше или равняется val
def drop_features_min_value(df, val = 0):
  return df.loc[:, (df.min(axis=0) > 0)]

# Удаляет из данных признаки, у которых колличество уникальных значений меньше или равняется count
def drop_feature_unique_count(df, count = 2):
  return df.loc[:, (df.nunique() > count)]


# Трансформирует категориальные признаки в OneHot - каждое уникальное
# значение категориального признака дает новый числовой
def cat_features_to_int_features(data, dtype ='int64', target = 'None'):
  if target != 'None':
    target_attrib = data.filter(target, axis=1)
    data = data.drop(target, axis=1)

  cat_columns = [c for c in data.columns if data[c].dtype.name == 'object']
  enc = OneHotEncoder(sparse=False, dtype = dtype)
  cat_attribs = data.filter(cat_columns, axis = 1)
  enc.fit(cat_attribs)
  cat_attribs_modified = enc.transform(cat_attribs)

  num_attribs = data.drop(cat_columns, axis = 1)

# Задает имена новых признаков, именуя их последовательностью целых чисел
  ln = len(cat_attribs_modified)
  cat_columns = [str(i) for i in range(ln)]

  modified_data = pd.concat((num_attribs, pd.DataFrame(cat_attribs_modified, columns = cat_columns)), axis=1)
  if target != 'None':
    modified_data = pd.concat((modified_data, target_attrib), axis=1)

  # Возвращает модифицированный дата фрейм в виде: числовые, модифицированные категориальные
  # аттрибуты и таргет(если присутствовал)
  return modified_data
