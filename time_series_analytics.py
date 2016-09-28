from __future__ import print_function
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import statsmodels.api as sm
from pandas import DataFrame
from statsmodels.graphics.api import qqplot
# dta = sm.datasets.sunspots.load_pandas().data
# dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
# del dta["YEAR"]

data_file = file("data/speed.csv")
names=['segment', 'speed', 'count', 'timestamp']
dta = pd.read_csv(data_file, header=0,
                  names=names,
                  index_col='timestamp', parse_dates=['timestamp'])
# dta = pd.read_csv(data_file, header=0, names=names)

print(dta.head(10))
# dta.index = dta.index.map(lambda tstr: datetime.strptime(tstr, '%y%m%d%H%M'))
dta = dta.sort_index(ascending=True)
dta = dta.resample('3H', how='mean', fill_method='ffill')
# dta.index = range(len(dta))
# dta = DataFrame(dta.values.copy(), columns=['speed', 'count'])

print(dta.head(10))
# print(dta.loc['1473476400000'])

if __name__ == '__main__':

    fig = plt.figure(figsize=(12,20))
    ax = fig.add_subplot(411)
    dta.plot(y=['speed'], ax=ax)
    pd.options.display.float_format = '{:,.3f}'.format
    # print(dta.describe())

    ax1 = fig.add_subplot(412)
    tau = len(dta)/4
    fig = sm.graphics.tsa.plot_acf(dta['speed'].values, lags=tau, ax=ax1)
    ax1.set_ylim([-1, 2])
    for i in range(tau/8):
        ax1.annotate('day', xy=(i*8, 1),
                     xytext=(i*8, 1.5),
                    arrowprops=dict(facecolor='black'))
    ax2 = fig.add_subplot(413)
    fig = sm.graphics.tsa.plot_pacf(dta['speed'], lags=tau, ax=ax2)
    del dta['count']
    arma_mod22 = sm.tsa.ARMA(dta, (8,1)).fit()

    print(arma_mod22.summary())


    arma_mod22.plot_predict('201609160900', '201609190900', dynamic=True, ax=ax, plot_insample=False)
    #
    # print(arma_mod22.summary())
    # ax3 = fig.add_subplot(414)
    # arma_mod22.plot_predict(40, 60, dynamic=True, ax=ax3, plot_insample=False)
    plt.subplots_adjust(hspace=0.5,wspace=0.5)
    plt.show()