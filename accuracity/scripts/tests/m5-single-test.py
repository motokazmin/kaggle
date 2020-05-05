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

from m5_validation_utils import *
from m5_simple_train_and_predict import *
from m5_draw_utils import *

def m5_stat_by_good_method(pivot_tbl, good_types, methods, good_name, method_name, horisont = 30, Plot_Train_Data = False, Plot_Validation_Data = False):
    idx = good_types[good_types.good == good_name].index[0]
    method = methods[methods.type == method_name]

    start = good_types.loc[idx].start_last_long_sales
    use_floor = good_types.loc[idx].use_floor

    trend = method.iloc[0].trend
    seasonal = method.iloc[0].seasonal
    remove_bias = method.iloc[0].remove_bias
    use_boxcox = method.iloc[0].use_boxcox

    if trend == 'None':
        trend = None
    if seasonal == 'None':
        seasonal = None

    print('used model parameters : train start {}, use_floor {}'.format(start, use_floor))
    print('method : {}, trend {}, seasonal {}, remove_bias {}, use_boxcox {}'.format(method_name, trend, seasonal, remove_bias, use_boxcox))

    df = pivot_tbl[[good_name, 'wday']]
    df = df[(df.index >= start)]

    triple_preds, triple_mse = m5_simple_train_and_predict(df[good_name], use_floor = use_floor, horisont = horisont, seasonal_periods = 7, trend = trend,
                                          seasonal = seasonal, remove_bias = remove_bias, use_boxcox = use_boxcox)

    plt.figure(figsize = (20, 4))
    run_sequence_plot(df[-horisont:].index, df[-horisont:][good_name], label = 'Real', style = 'ro-')
    run_sequence_plot(triple_preds.index, triple_preds, label = 'Predict', title="Sales", style = 'go-.')
    
    if Plot_Train_Data == True:
        run_sequence_plot(df.index, df[good_name], label="Train", style = 'y-.')
        
    if Plot_Validation_Data == True:
        triple_preds = m5_simple_predict(df[good_name], use_floor = use_floor, horisont = 28, seasonal_periods = 7, trend = trend,
                                              seasonal = seasonal, remove_bias = remove_bias, use_boxcox = use_boxcox)
        run_sequence_plot(triple_preds.index, triple_preds, label="Validation", style = 'b-.')
        
    print(good_name + '  , triple_mse = ' + str(triple_mse) + ' , train len = ' + str(len(df) - horisont))

#def find_best_period(df, horisont = 30):
#    mse = pd.DataFrame(columns = ['i', 'mse'])

#    plt.figure(figsize = (35, 4))
#    for i in range(20, len(df) - horisont, 2):
#        dfi = df[-horisont - i:]
#        triple_preds, triple_mse = m5_simple_train_and_predict(dfi[good], use_floor = Fhorisont = horisont, seasonal_periods = 7, trend = None, seasonal = 'additive', remove_bias = True, use_boxcox = False)
#        mse.loc[i - 20] = [i, triple_mse]
        
#    print(mse.sort_values('mse'))
#    run_sequence_plot(mse.i, mse.mse)
