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

import inspect
import os
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

def run_sequence_plot(x, y, title = None, label=None , xlabel="time", ylabel="series", style = 'k-'):
    plt.plot(x, y, style, label = label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3);
    plt.legend(loc='upper left')
    
# function to write the definition of our function to the file
def write_function_to_file(function, file):
    if os.path.exists(file):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not
    with open(file, append_write) as file:
        function_definition = inspect.getsource(function)
        file.write(function_definition)

write_function_to_file(run_sequence_plot, "m5_draw.py")
