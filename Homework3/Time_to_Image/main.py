#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:00:00 2020

@author: ebunchalit
"""


import numpy as np
from series2gaf import GenerateGAF
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import datetime as dt
import yfinance as yf
import pandas as pd
from pandas import DataFrame


from sklearn.preprocessing import normalize

stocks = ["MSFT"]
start = dt.datetime.today()-dt.timedelta(360)
end = dt.datetime.today()
cl_price: DataFrame = pd.DataFrame() # empty dataframe which will be filled with closing prices of each stock

cl_price["MSFT"] = yf.download("MSFT",start,end)["Adj Close"]
cl_price = cl_price.values

clp = cl_price.reshape((cl_price.shape[0],))
clp_n = normalize(clp[:,np.newaxis], axis=0).ravel()



# set parameters
timeSeries = list(clp_n)
windowSize = 50
rollingLength = 10
#fileName = 'demo_%02d_%02d'%(windowSize, rollingLength)
fileName = 'msft_test'

# generate GAF pickle file (output by Numpy.dump)
GenerateGAF(all_ts = timeSeries,
            window_size = windowSize,
            rolling_length = rollingLength,
            fname = fileName)

from series2gaf import PlotHeatmap

gaf = np.load('msft_test_gaf.pkl',allow_pickle=True)

x = PlotHeatmap(gaf)
plt.plot(clp_n)
plt.ylabel('MSFT price')


