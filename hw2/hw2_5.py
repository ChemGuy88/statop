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
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, classification_report, precision_recall_curve, roc_curve
from sklearn.model_selection import ShuffleSplit, train_test_split

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

def logisticCostFunction(X, y, beta):
    '''
    An alternative to https://stackoverflow.com/questions/35956902/how-to-evaluate-cost-function-for-scikit-learn-logisticregression
    '''
    n, p = X.shape
    cost = np.zeros((n))
    cost0 = 1 - np.log( 1 - sigmoid( X, beta) )
    cost1 = - np.log( 1 - sigmoid( X, beta) )
    cost[y == 0] =  cost0[y==0]
    cost[y == 1] =  cost1[y==1]

    return cost.sum()

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

def sigmoid(X, beta):
    '''
    '''
    score = X.dot(beta)
    z = ( 1 / ( 1 + np.e ** - score ) )

    return z

def batchLogisticGradientDescent(X, y, alpha, numIterations, betaHat):
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
        score = X.dot(betaHat)
        # yhat = np.round(1/(1+np.e ** - score), decimals=0)
        loss = np.sum( np.log( 1 + np.e ** (- y * score) ) )
        gradient = score - y

        betaHat = betaHat - alpha * gradient
        betaHistory[i,:] = betaHat.ravel()

        cost = costFunction(X, y, betaHat)
        costHistory[i] = cost.ravel()

    return betaHistory, costHistory

'''#####################################################################
###### Workspace #######################################################
########################################################################'''

plt.close('all')
figd = {}

regressionTypes = ['ols', 'logistic']
for regType in regressionTypes:

    '''#################### Generate data ##########################'''

    if regType == 'ols':
        X, y, beta = make_regression(n_samples=100, n_features=2, n_informative=2,
                                     n_targets=1, bias=0.0, effective_rank=None,
                                     tail_strength=0.5, noise=0.0, shuffle=True,
                                     coef=True, random_state=randomState)
    elif regType == 'logistic':
        X, y = make_classification(n_samples=100, n_features=20, n_informative=2,
                                   n_classes=2, n_clusters_per_class=2,
                                   weights=None, flip_y=0.01, class_sep=1.0,
                                   hypercube=True, shift=0.0, scale=1.0,
                                   shuffle=True, random_state=randomState)

    n, p = X.shape
    if regType == 'ols':
        y = y.reshape((n,1))
    elif regType == 'logistic':
        pass

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

    fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
    figd[fig1.number] = {'regType' : regType, 'fig' : fig1}
    ax1 = ax.ravel()[0]
    ax2 = ax.ravel()[1]
    for i in range(0,N):
        label = r'$\alpha$ = {0:0.2f}'.format(alphas[i]) # syntax {0th variable:two decimal place float}
        ax1.plot(costs[i][:numIterations], label=label, alpha=0.5)
        ax2.plot(costs[i][:numIterationsShort], label=label, alpha=0.5)

    fig1.legend(loc='lower center', shadow=True, fontsize='small', ncol=N)

    # Plot cost of true beta as horizontal line.
    if regType == 'ols':
        trueCost = costFunction(X_train_scaled, y_train, beta)
        ax1.plot([trueCost] * numIterations, c='k', alpha=0.5)
        ax2.plot([trueCost] * numIterationsShort, c='k', alpha=0.5)
    elif regType == 'logistic':
        pass
        # betaHat = betaHistory[-1,:]
        # trueCost = logisticCostFunction(X_train_scaled, y_train, betaHat)
        # ax1.plot([trueCost] * numIterations, c='k', alpha=0.5)
        # ax2.plot([trueCost] * numIterationsShort, c='k', alpha=0.5)

    # Plot aesthetics
    ax1.set_title(f'Gradient descent with {numIterations} iterations')
    ax2.set_title(f'Zoomed-in view of gradient descent')
    fig1.suptitle(f'{regType} Regression')

    '''#################### Test ##########################'''

    if regType == 'ols':
        trueCost = costFunction(X_train_scaled, y_train, beta)
        ax1.plot([trueCost] * numIterations, c='k', alpha=0.5)
        ax2.plot([trueCost] * numIterationsShort, c='k', alpha=0.5)
    elif regType == 'logistic':
        betaHat = betaHistory[-1,:]
        z = sigmoid(X_train, betaHat)
        yhat = np.round( z, decimals = 0)

        report = classification_report(y_train, yhat)
        print(report)

        fpr, tpr, _ = roc_curve(y_train, z)
        roc_auc = auc(fpr, tpr)

        fig2, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
        figd[fig2.number] = {'regType' : regType, 'fig' : fig2}

        ax.plot(fpr, tpr, label=f'ROC curve (area = {np.round(roc_auc,4)})', color='orange', lw=4)
        plt.plot([0, 1], [0, 1], color='navy', lw=4, linestyle='--')
        plt.xlim([-0.005, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        fig2.suptitle(f'{regType} Regression')

# Save images
for fignum in plt.get_fignums():
    regType = figd[fignum]['regType']
    fig = figd[fignum]['fig']
    fname = f'{workDir}/hw2_5_fig{fignum}_{regType}.png'
    fig.savefig(fname)

plt.show()
