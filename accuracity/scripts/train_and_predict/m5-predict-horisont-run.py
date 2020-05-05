# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from m5_simple_train_and_predict import *

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

def get_rounding_type(good, horisont):
    good_last_month = good[-horisont - 30: -horisont]
    month_sparse_level = len(good_last_month[good_last_month > 0])/len(good_last_month)
    return month_sparse_level < 0.5

def m5_predict_horisont(pivot_tbl, good_types, methods, horisont, Debug = False):
  i = 0
  retry_counter = 0
  last_sales_counter = 0
  last_month_counter = 0

  predicts = pd.DataFrame(columns = ['id', *(['F' + str(x + 1) for x in range(28)])])

  good_types = good_types.drop(['first_sale', 'last_sale', 'sparse level', 'pvalue', 'last_long_nosales_interval_length', 'last_long_sales_interval_length', 'use_floor'], axis = 1)

  for i, row in good_types.iterrows():
    good = row.good
    df = pivot_tbl[pivot_tbl.index >= row.start_last_long_sales]
    use_floor = get_rounding_type(df[good], horisont = 30)

    method_counter = 0
    for mtype in row[2:].sort_values().index:
      isLastSales = True
      if '_month' in mtype:
        isLastSales = False
        mtype = mtype[:-6]
        df = df[-60:]
      
      method = methods[methods.type == mtype]
      trend = method.trend.iloc[0]
      seasonal = method.seasonal.iloc[0]
      remove_bias = method.remove_bias.iloc[0]
      use_boxcox = method.use_boxcox.iloc[0]

      if trend == 'None':
        trend = None
      if seasonal == 'None':
        seasonal = None

      try:
        triple_preds = m5_simple_train_and_predict(df[good], use_floor, horisont, seasonal_periods = 7,
                                          trend = trend, seasonal = seasonal, remove_bias = remove_bias, use_boxcox = use_boxcox)
        if triple_preds.isnull().values.any() == False:
          triple_preds[triple_preds < 0] = 0
          predicts.loc[len(predicts)] = [good, *triple_preds]
          if isLastSales == True:
            last_sales_counter += 1
          else:
            last_month_counter += 1
          break
      except:
        if ((method_counter == len(methods) - 1) | (Debug == True)):
          print('Нет метода для прогноза {}, method_counter = {}, retry_counter = {}'.format(good, method_counter, retry_counter))

        method_counter += 1
        retry_counter += 1

        continue

    i += 1
    if i % 1000 == 0:
      print('iteration {}/{}'.format(i, len(good_types)))

  print('\npredict_horisont stat: last_sales_counter = {}, last_month_counter = {}, retry_counter = {}'.format(last_sales_counter, last_month_counter, retry_counter))

  return predicts