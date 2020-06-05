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

# create a random sequence with 200 numbers
# all numbers are in the range of 50.0 to 150.0
#random_series = np.random.uniform(low=50.0, high=150.0, size=(200,))
stocks = ["MSFT"]
start = dt.datetime.today()-dt.timedelta(360)
end = dt.datetime.today()
cl_price: DataFrame = pd.DataFrame() # empty dataframe which will be filled with closing prices of each stock

cl_price[ticker] = yf.download(ticker,start,end)["Adj Close"]
clprice_v = cl_price.values
clp = clprice_v.reshape((clprice_v.shape[0],))

# set parameters
timeSeries = list(clp)
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
plt.plot(clp)
plt.ylabel('MSFT price')


