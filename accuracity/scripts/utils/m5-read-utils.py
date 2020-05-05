# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def m5_read():
    calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
    sales_train = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

    sales_by_cat = sales_train.drop(['item_id', 'dept_id', 'cat_id', 'state_id', 'store_id'], axis = 1).set_index(['id']).T.reset_index()
    pivot_tbl = sales_by_cat.merge(calendar[['date', 'd', 'wday', 'snap_CA']], left_on = 'index', right_on = 'd')
    pivot_tbl = pivot_tbl.drop(['d', 'index'], axis = 1).set_index('date')
    pivot_tbl.index = pd.to_datetime(pivot_tbl.index)
    
    good_types = pd.read_csv('/kaggle/input/all-types/good_types_stat_full.csv', index_col = 0)
    methods = pd.read_csv('/kaggle/input/methods/method.csv', index_col = 0)
    
    return pivot_tbl, good_types, methods
