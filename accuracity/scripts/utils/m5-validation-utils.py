# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return (np.mean(np.abs((y_true - y_pred) / y_true)) * 100).fillna(1.0)

def RMSSE(train, valid, predict, horisont): 
    assert type(train)   == type(np.array([]))
    assert type(valid)   == type(np.array([]))
    assert type(predict) == type(np.array([]))
    assert len(valid) == len(predict)

    numerator = np.sum((valid - predict)**2)
    denominator  = 1/(len(train) - 1)*np.sum((train[1:] - train[: -1])**2)
    return (1/horisont*numerator/denominator)**0.5
