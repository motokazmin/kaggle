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

from statsmodels.tsa.api import ExponentialSmoothing

from m5_validation_utils import *

def m5_simple_train_and_predict(df, use_floor, horisont = 30, seasonal_periods = 7, trend = 'additive', seasonal = 'additive', remove_bias = False, use_boxcox = None, Debug = False):
    shift = 0
    if (df.min() == 0) & ((trend == 'multiplicative')|(seasonal == 'multiplicative')|(use_boxcox == True)):
        shift = 2 # Используется логарифм
        
    train = df[:-horisont] + shift
    test = df[-horisont:]

    model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods, freq='D')
    triple = model.fit(optimized=True, use_boxcox = use_boxcox, remove_bias = remove_bias)
    triple_preds = triple.forecast(len(test))
    
    if use_floor == True:
       triple_preds = triple_preds.apply(np.floor) 
    else:
       triple_preds = triple_preds.apply(np.ceil) 
    
    train -= shift
    triple_preds -= shift
    triple_mse = RMSSE(train.to_numpy(), test.to_numpy(), triple_preds.to_numpy(), horisont)

    if Debug == True:
        print('trend {}, seasonal {}, remove_bias {}, use_boxcox {}, triple MSE: {}'.format(trend, seasonal, remove_bias, use_boxcox, triple_mse))
    return triple_preds, triple_mse

def m5_simple_predict(df, use_floor, horisont = 28, seasonal_periods = 7, trend = 'additive', seasonal = 'additive', remove_bias = False, use_boxcox = None, Debug = False):
    shift = 0
    if (df.min() == 0) & ((trend == 'multiplicative')|(seasonal == 'multiplicative')|(use_boxcox == True)):
        shift = 2 # Используется логарифм
        
    train = df + shift

    model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods, freq='D')
    triple = model.fit(optimized=True, use_boxcox = use_boxcox, remove_bias = remove_bias)
    triple_preds = triple.forecast(horisont)
    
    if use_floor == True:
       triple_preds = triple_preds.apply(np.floor) 
    else:
       triple_preds = triple_preds.apply(np.ceil) 
    
    triple_preds -= shift

    return triple_preds
