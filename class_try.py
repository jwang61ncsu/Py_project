import csv
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

import warnings
warnings.filterwarnings('ignore')

class garch(object):
    ''' Class to model garch model.

    Parameters
    ==========
    return : array
        daily stock return
    sd : array
        daily stock standard deviation
    p: integer(1-5)
       parameter for garch model garch terms
    q: integer(1-5)
       parameter for garch model arch terms

    Methods
    =======
    get_discount_factors :
        returns discount factors for given list/array
        of dates/times (as year fractions)
    '''

    def __init__(self, dayreturn, daysigma, p, q):
        self.dayreturn = dayreturn
        self.daysigma = daysigma
        self.p = p
        self.q = q

        # value of pacf
        pacf = statsmodels.tsa.stattools.pacf(dayreturn, nlags=19)
        # get the last value of pacf whose absolute value is larger than 0.5
        for i in range(0, 19):
            if abs(pacf[19 - i]) > 0.05:
                self.r = 19 - i
                break

        # Build rank r AR Model
        order = (self.r, 0)
        model = sm.tsa.ARMA(self.dayreturn, order).fit(disp = 0)

        # Residuals of AR model
        self.at = model.resid

        #p = subprocess.Popen(self.model.resid, shell=True, stdout=subprocess.PIPE)
        #self.at = b''.join(p.stdout).decode('utf-8')

    def returnplot(self):
        x = np.arange(0, len(self.dayreturn), 1)
        y = np.zeros_like(x)
        y = self.dayreturn

        # Calculate the simple average of the data
        y_mean = [np.mean(self.dayreturn) for i in x]

        fig, ax = plt.subplots()
        # Plot the data
        data_line = ax.plot(x, y, label='Data', marker='o')
        # Plot the average line
        mean_line = ax.plot(x, y_mean, label='Mean', linestyle='--')

        legend = ax.legend(loc='upper right')
        plt.xlabel('Time')
        plt.ylabel('Daily Return')
        plt.title('Daily Return chart for stationary')
        return plt.show(mean_line)
    # p value of ADF Test,
    # ? what if fail?
    def adf(self):
        t = sm.tsa.stattools.adfuller(self.dayreturn)
        return t[1]

    def pacf(self):
        # Before building Autocorrelation model :AR(p), we need to decide rank of AR model
        fig = plt.figure(figsize=(20, 5))  # 20 x-axis, 5 y-axis
        ax1 = fig.add_subplot(111)
        fig = sm.graphics.tsa.plot_pacf(self.dayreturn, lags=20, ax=ax1)


        return plt.show(fig), self.r

    def ar(self):
        return print(self.model.summary())

    def archlm(self):
        # ARCH-LM Test: small p-value shows ARCH effect.
        p_arch = statsmodels.stats.diagnostic.het_arch(self.at)[1]
        return p_arch

    def model(self):

        N = len(self.at)

        # Var
        X = np.zeros((5, N - max(self.p, self.q)))

        for i in range(0, self.p):
            X[i] = np.square(self.daysigma[i:N - max(self.p, self.q) + i])

        # Residuals
        Y = np.zeros((5, N - max(self.p, self.q)))

        for i in range(0, self.q):
            Y[i] = np.square(self.at[i:N - max(self.p, self.q) + i])

        # t sigma^ data
        Z = np.square(self.daysigma[max(self.p, self.q):N])

        data = pd.DataFrame(
            {'x1': X[0], 'x2': X[1], 'x3': X[2], 'x4': X[3], 'x5': X[4], 'y1': Y[0], 'y2': Y[1], 'y3': Y[2], 'y4': Y[3],
             'y5': Y[4], 'z': Z})

        # Fit the model
        if self.p == 1 and self.q == 1:
            self.model_garch = ols("z ~ x1 + y1 ", data).fit()
        elif self.p == 1 and self.q == 2:
            self.model_garch = ols("z ~ x1 + y1 + y2", data).fit()
        elif self.p == 1 and self.q == 3:
            self.model_garch = ols("z ~ x1 + y1 + y2 +y3", data).fit()
        elif self.p == 2 and self.q == 1:
            self.model_garch = ols("z ~ x1 + x2 + y1", data).fit()
        elif self.p == 2 and self.q == 2:
            self.model_garch = ols("z ~ x1 + x2 + y1 + y2", data).fit()
        elif self.p == 2 and self.q == 3:
            self.model_garch = ols("z ~ x1 + x2 + y1 + y2 +y3", data).fit()
        elif self.p == 3 and self.q == 1:
            self.model_garch = ols("z ~ x1 + x2 + x3 + y1", data).fit()
        elif self.p == 3 and self.q == 2:
            self.model_garch = ols("z ~ x1 + x2 + x3 + y1 + y2", data).fit()
        elif self.p == 3 and self.q == 3:
            self.model_garch = ols("z ~ x1 + x2 + x3 + y1 + y2 + y3", data).fit()

        return self.model_garch


#------------------------------------------------#
#                  1. import data                #
#------------------------------------------------#
# have two columns: dates and price
with open('price.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    dates = np.array([])
    price = np.array([])
    for row in readCSV:
        date1 = row[0]
        price1 = np.float(row[1])  # change string into float
        dates = np.append(dates, date1)
        price = np.append(price, price1)

# change string into number
dateint = np.zeros((len(dates), 9))
for i in range(len(dates)):
    dateint[i, :] = time.strptime(dates[i], '%Y/%m/%d %H:%M')

returnmin = np.zeros((len(dates), 1))  # returnmin is the 5-minutes return
square = np.array([])  # square is the sum of the square of return-mean
num = np.array([])  # num is the number of samples each day.(48 each day)
dayreturn = np.array([])  # dayreturn is the daily return
num = np.append(num, 0)
sumsquare = 0.0
a = 1
returnd = 0.0
index = 0
for ii in range(1, len(dates)):
    returnmin[ii] = np.log(price[ii] / price[ii - 1])
    if dateint[ii, 2] == dateint[ii - 1, 2]:
        mean = np.mean(returnmin[np.int(np.sum(num[:index + 1])):ii])
        sumsquare = np.sum((returnmin[np.int(np.sum(num[:index + 1])):ii] - mean) ** 2)
        returnd = np.log(price[ii] / price[np.int(np.sum(num[:index + 1]))])
        a += 1
    else:
        square = np.append(square, sumsquare)
        num = np.append(num, a)
        dayreturn = np.append(dayreturn, returnd)
        sumsquare = 0.0
        a = 1
        index += 1

num = np.delete(num, 0)
daysigma = np.sqrt(square / num)  # daysigma is the daily sigma

# run the class
c = garch(dayreturn, daysigma, 1, 2)
#c.returnplot()
#c.adf()
#c.pacf()
#c.ar()
#c.archlm()
print(c.model().summary())




