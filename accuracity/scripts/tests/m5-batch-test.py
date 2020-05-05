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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from m5_validation_utils import *
from m5_simple_train_and_predict import *

def get_rounding_type(good, horisont):
    good_last_month = good[-horisont - 30: -horisont]
    month_sparse_level = len(good_last_month[good_last_month > 0])/len(good_last_month)
    return month_sparse_level < 0.5

# Предсказывает для товаров, у которых есть история последних продаж больше месяца
def predict_more_one_month_goods(pivot_tbl, good_types, methods, last_month_use):
    good_types = good_types[(good_types.last_long_sales_interval_length > 30)]
    
    for mrow in methods.itertuples():
        mtype = mrow.type
        trend = mrow.trend
        seasonal = mrow.seasonal
        remove_bias = mrow.remove_bias
        use_boxcox = mrow.use_boxcox
        
        if trend == 'None':
            trend = None
        if seasonal == 'None':
            seasonal = None
        
        print('type {}, used params: trend = {}, seasonal = {}, remove_bias = {}, use_boxcox = {}'.format(mtype, trend, seasonal, remove_bias, use_boxcox))

        i = 0
        for row in good_types.itertuples():
            good = row.good
            cname = mtype
            df = pivot_tbl[pivot_tbl.index >= row.start_last_long_sales]
            if last_month_use == True:
                df = df[-60:]
                cname += '_month'
                
            try:
                use_floor = get_rounding_type(pivot_tbl[good], horisont = 30)
                good_types.loc[row.Index, 'use_floor'] = use_floor
                triple_preds, rmmse = m5_simple_train_and_predict(df[good], use_floor, horisont = 30, seasonal_periods = 7, trend = trend,
                                                                        seasonal = seasonal, remove_bias = remove_bias, use_boxcox = use_boxcox)
                good_types.loc[row.Index, cname] = rmmse
            except:
                i = i
                #print('Cannot process {}'.format(good))

            i += 1
            if i % 1000 == 0:
                print('iteration {}/{}'.format(i, len(good_types)))
                
    return good_types