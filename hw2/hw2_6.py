#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Implement gradient descent with line search. As mentioned in class, backtracking can be used to halve the stepsize or double the inverse stepsize parameter ρt till the new iterate βt+1 satisfies either the relaxed majorization criterion

g(βt+1, βt) + (1 − c) ρ_t D_2(βt, βt+1) ≥ f(βt+1), (1)

or the majorization criterion

g(βt+1, βt) ≥ f(βt+1), (2)

or Amijo’s rule

f(β )≤f(β )+0.5c⟨∇f(β ),β −β ⟩ (3)

where D2(α, β) = ∥α − β∥2/2, and c is a positive small constant. (Here, I used 0.5c instead of c in Amijo’s rule to match the constant in (1).) Apply your algorithm(s) to a Poisson log-linear model, and make a comparison of the line search criteria (1)–(3).

REFERENCS:

    1. https://thatdatatho.com/2019/07/01/introduction-gradient-descent-line-search/
'''

import csv, json, os, re, requests, string, sys
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
workFile = os.path.basename(__file__)
workFile = re.search('(\w+).*', workFile)
workFile = workFile.groups()[0]

randomState = None
np.random.seed(1)

numIterations = 100

verbose = 0

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

def batchGradientDescentLineSearch(X, y, betaHat, alpha, rho, betaHistory, costHistory, lineSearch=False):
    '''
    '''


    return gradient, betaHistory, costHistory

def batchGradientDescent(X, y, betaHat, alpha, rho, numIterations, lineSearch):
    '''
    '''
    n, p = X.shape
    y = y.reshape((n,1))
    # costHistory = np.zeros((numIterations,1))
    # betaHistory = np.zeros((numIterations,p))
    costHistory = []
    betaHistory = []

    if not betaHat:
        betaHat = np.random.uniform(size=p, low=-1, high=1)
        betaHat = betaHat.reshape((p,1))

    if lineSearch == 'Armijo':
        c = 0.9
        rho = rho
        numIterations = 0
        gradient = 1
        while la.norm(gradient) > 1e-4:
            # gradient, betaHistory, costHistory = batchGradientDescentLineSearch(X=X, y=y, betaHat=betaHat, alpha=alpha, rho=rho, betaHistory=betaHistory, costHistory=costHistory, lineSearch=False)
            yhat = X.dot(betaHat)
            loss = yhat - y
            gradient = X.T.dot(loss) / n

            if lineSearch == 'Armijo':
                alpha = 1
                p = -gradient
                betaHatNew = betaHat + alpha * p
                LHS = costFunction(X, y, betaHatNew)
                RHS = costFunction(X, y, betaHat) + c * gradient.T.dot((betaHatNew - betaHat))

                if verbose > 0:
                    i = 0

                while LHS > RHS:
                    alpha = rho * alpha
                    betaHatNew = betaHat + alpha * p
                    LHS = costFunction(X, y, betaHatNew)
                    RHS = costFunction(X, y, betaHat) + c * gradient.T.dot((betaHatNew - betaHat))

                    if verbose > 0:
                        i += 1
                        if i % 5 == 0:
                            text = f'\nIteration {i}\t|\tLHS: {LHS}\tRHS: {RHS}'
                            print(text)

            betaHat = betaHat - alpha * gradient
            betaHistory.append(betaHat.ravel())

            cost = costFunction(X, y, betaHat)
            costHistory.append(cost.ravel())

            numIterations += 1
            if verbose > 0:
                if numIterations % 5 == 0:
                    text = f'\nIteration {numIterations} | cost: {costHistory[-1]}\tgradient: {la.norm(gradient)}'
                    print(text)
                if numIterations == 100:
                    break
    elif lineSearch == False:
        for i in range(numIterations):
            # gradient, betaHistory, costHistory = batchGradientDescentLineSearch(X=X, y=y, betaHat=betaHat, alpha=alpha, rho=rho, betaHistory=betaHistory, costHistory=costHistory, lineSearch=False)
            pass

    return betaHistory, costHistory, numIterations

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

regressionTypes = ['ols', 'logistic']
for regType in regressionTypes[:1]:

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
    N = len(alphas[:1])
    betas = []
    costs = []
    for i in range(N):
        betaHistory, costHistory, numIterations = batchGradientDescent(X=X_train_scaled, y=y_train,
                                                        alpha=0.1, rho=0.95, numIterations=numIterations,
                                                        betaHat=None, lineSearch='Armijo')
        betas.append(betaHistory)
        costs.append(costHistory)

    '''#################### Plot descent ##########################'''

    numIterationsShort = 10

    fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
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
    fname = f'{workDir}/{workFile}_fig{fignum}_{regType}.png'
    plt.savefig(fname)

plt.show()
