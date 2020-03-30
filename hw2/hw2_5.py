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
workDir = f'{userDir}/Documents/statop'

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

def batchGradientDescent(X, y, alpha, numIterations, betaHat=None):
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
                             coef=True, random_state=None)

n, p = X.shape
y = y.reshape((n,1))

'''#################### Train ##########################'''

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

scaler = StandardScaler(with_std=True, with_mean=True)
X_train_scaled = scaler.fit_transform(X_train, y_train)

betaHistory, costHistory = batchGradientDescent(X_train_scaled, y_train,
                                                alpha=0.1, numIterations=100,
                                                betaHat=None)

'''#################### Plot descent ##########################'''
# https://scipython.com/blog/visualizing-the-gradient-descent-method/

# Select representative betaHats
betaHistoryIndices = [0,10,25,50,99]
samplesBetaHats = betaHistory[betaHistoryIndices]
M = len(samplesBetaHats)

# Data and fit
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6.15))
plt.show(block = False)
a = np.linspace(-1, 1, 51)

# Plot fit of representative betahats
colors = ['r', 'orange', 'y', 'g', 'b']
for i in range(1, M):
    betaHat = samplesBetaHats[i,:]
    b0 = betaHat[0]
    c = a * b0
    label = r'$\beta_0 = {:.3f}$'.format(b0)
    ax[0].scatter(a, c, marker='.', facecolor='None', edgecolor=colors[i], lw=1,
                  label=label)

# Plot fit of actual beta
b0 = beta[0]
c = a * b0
ax[0].scatter(a, c, marker='x', s=40, color='k', alpha=0.5)

# Add labels, title, and legend for data and fit
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$y$')
ax[0].set_title('Data and fit')
ax[0].legend(loc='upper left', fontsize='small')

# Construct cost function grid
b0absmax = np.max(np.abs(betaHistory[:,0] - beta[0])) * 2
b1absmax = np.max(np.abs(betaHistory[:,1] - beta[1])) * 2
beta0_axis = np.linspace(beta[0] - b0absmax, beta[0] + b0absmax, 101)
beta1_axis = np.linspace(beta[1] - b1absmax, beta[1] + b1absmax, 101)

len_beta0_axis = len(beta0_axis)
len_beta1_axis = len(beta1_axis)
costGrid = np.zeros( ( len_beta1_axis, len_beta0_axis))
X_train_scaled_2d = X_train_scaled[:,:2]

for i in range(len_beta0_axis):
    for j in range(len_beta1_axis):
        smallBeta = np.array([ beta0_axis[i] , beta1_axis[j] ])
        costGrid[i,j] = costFunction(X_train_scaled_2d, y_train, smallBeta)

beta0_grid, beta1_grid = np.meshgrid(beta0_axis, beta1_axis)
contours = ax[1].contour(beta0_grid, beta1_grid, costGrid, 15)
ax[1].clabel(contours)

# Plot value of true beta on costGrid
ax[1].scatter([beta[0]]*2, [beta[1]]*2, s=[50,10], color=['k', 'w'])

# Plot value of representative betahats on costGrid
betaHistoryIndices = [0,10,25,50,99]
samplesBetaHats = betaHistory[betaHistoryIndices]
M = len(samplesBetaHats)
for i in range(1, M):
    oldBetaHat = samplesBetaHats[i-1,:]
    betaHat = samplesBetaHats[i,:]
    ax[1].annotate('', xy=betaHat, xytext=oldBetaHat,
                   arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                   va='center', ha='center')
ax[1].scatter(*zip(*samplesBetaHats), c=colors, s=40, lw=0, alpha=0.5)

# Labels, titles, and legend
ax[1].set_xlabel(r'$\theta_0$')
ax[1].set_ylabel(r'$\theta_1$')
ax[1].set_title('Cost function')

plt.show()

'''#################### Test ##########################'''

X_test_scaled = scaler.transform(X_test)
betaHat = betaHistory[-1,:]
