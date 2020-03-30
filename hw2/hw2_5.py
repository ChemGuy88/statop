#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
hw2_5

Prompt:

Implement gardient descent with a universal stepsize for:
    1. Regression
    2. Logistic regression
Try to increase your universal stepsize to see if it is tight.
'''

import csv, json, re, requests, string, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.linalg as la
import pandas as pd
import scipy as sp
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, train_test_split

# For running IPython magic commands (e.g., %matplotlib)
# ipython = get_ipython()

# Display plots inline and change default figure size
# ipython.magic("matplotlib")

'''#####################################################################
###### Universal Variables #############################################
########################################################################'''

userDir = '/Users/herman'
workDir = f'{userDir}/Documents/statop/hw2'

randomState = None
np.random.seed(1)

numIterations = 100

'''#####################################################################
###### Functions #############################################
########################################################################'''

'''
'batchGradientDescent' and 'costFunction' from https://medium.com/@pytholabs/multivariate-linear-regression-from-scratch-in-python-5c4f219be6a
'''

def costFunction(X, y, beta):
    '''
    '''
    n, p = X.shape
    yhat = (X.dot(beta)).reshape((n,1))
    cost = np.average( (yhat - y) ** 2, axis=0)/2
    return cost

def batchGradientDescent(X, y, alpha, numIterations, betaHat):
    '''
    '''
    n, p = X.shape
    y = y.reshape((n,1))
    costHistory = np.zeros((numIterations,1))
    betaHistory = np.zeros((numIterations,p))

    if not betaHat:
        betaHat = np.random.uniform(size=p, low=-1, high=1)
        betaHat = betaHat.reshape((p,1))

    for i in range(numIterations):
        yhat = X.dot(betaHat)
        loss = yhat - y
        gradient = X.T.dot(loss) / n

        betaHat = betaHat - alpha * gradient
        betaHistory[i,:] = betaHat.ravel()

        cost = costFunction(X, y, betaHat)
        costHistory[i] = cost.ravel()

    return betaHistory, costHistory

'''#####################################################################
###### Workspace #######################################################
########################################################################'''

plt.close('all')

'''#################### Generate data ##########################'''

X, y, beta = make_regression(n_samples=100, n_features=2, n_informative=2,\
                             n_targets=1, bias=0.0, effective_rank=None,\
                             tail_strength=0.5, noise=0.0, shuffle=True,\
                             coef=True, random_state=randomState)

n, p = X.shape
y = y.reshape((n,1))

'''#################### Train ##########################'''

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=randomState)

scaler = StandardScaler(with_std=True, with_mean=True)
X_train_scaled = scaler.fit_transform(X_train, y_train)

alphas = [0.01, 0.25, 0.5, 0.75, 0.9, 0.99, 2, 10, 1000]
N = len(alphas)
betas = []
costs = []
for i in range(N):
    betaHistory, costHistory = batchGradientDescent(X_train_scaled, y_train,
                                                    alpha=0.1, numIterations=numIterations,
                                                    betaHat=None)
    betas.append(betaHistory)
    costs.append(costHistory)

'''#################### Plot descent ##########################'''

numIterationsShort = 10

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
ax1 = ax.ravel()[0]
ax2 = ax.ravel()[1]
for i in range(0,N):
    label = r'$\alpha$ = {0:0.2f}'.format(alphas[i]) # syntax {0th variable:two decimal place float}
    ax1.plot(costs[i][:numIterations], label=label, alpha=0.5)
    ax2.plot(costs[i][:numIterationsShort], label=label, alpha=0.5)

fig.legend(loc='lower center', shadow=True, fontsize='small', ncol=N)

# Plot cost of true beta as horizontal line.
trueCost = costFunction(X_train_scaled, y_train, beta)
ax1.plot([trueCost] * numIterations, c='k', alpha=0.5)
ax2.plot([trueCost] * numIterationsShort, c='k', alpha=0.5)

# Design
ax1.set_title(f'Gradient descent with {numIterations} iterations')
ax2.set_title(f'Zoomed-in view of gradient descent')

fname = f'{workDir}/hw2_5_fig01.png'
plt.savefig(fname)
plt.show()
