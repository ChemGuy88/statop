#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    hw1_3
'''
import csv, json, re, requests, string, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.linalg as la
import pandas as pd
import scipy as sp
import scipy.spatial as spt

'''
Dr. She mentions in Topic 1:

Let X be a n-by-n real matrix. Then the spectral norm is

    ||X||_2 = the greatest singular value of X
            = d_1

eng.hmc.edu says (http://fourier.eng.hmc.edu/e161/lectures/algebra/node12.html)

    ||X||_2 = the greatest singular value of X
            = sq(greatest eigenvalue of X^TX), X^T = transpose of X

Wolfram says:

    ||X||_2 = sq(greatest eigenvalue of A^HA), A^H = conjugate transpose of A

Upenn says (Topic 2_supp):
                                            , when A is real, A^H = A^T
                                            , when A is Hermitian, A^H = A

This spectral decomposition is only applicable to square DIAGONALIZABLE (AKA nondefective) MATRICES. An n x n matrix A is nondefective if there exists an invertible matrix P such that P^{-1}AP is a diagonal matrix.
https://en.wikipedia.org/wiki/Diagonalizable_matrix
'''

def make_sym_matrix(n,vals):
  m = np.zeros([n,n], dtype=np.double)
  xs,ys = np.triu_indices(n,k=1)
  m[xs,ys] = vals
  m[ys,xs] = vals
  m[ np.diag_indices(n) ] = 0 - np.sum(m, 0)
  return m

n = 9; nn = int((n*(n-1))/2)
nn =
p = 3
m = 3
r = 1
locX = 0; scaleX = 1;
locY = 3; scaleY = 0.5;
X = np.random.normal(loc=locX, scale=scaleX, size=n*p)
# X = X.reshape((n,p))
X = make_sym_matrix(nn, X)
Y = np.random.normal(loc=locY, scale=scaleY, size=n*p)
# Y = Y.reshape((n,m))
Y = make_sym_matrix(nn, Y)

P_x = X.dot(X.T)
P_y = Y.dot(Y.T)

A = P_x-P_y
A = A.T.dot(A)
l, e = la.eig(A)
specnorm = np.sqrt(np.max(l))
print(specnorm)

# Plot original data
plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
graphData = [('o', X, 'X'),\
             ('^', Y, 'Y')]
for marker, a, label in graphData:
    x = a[:,0]
    y = a[:,1]
    if a.shape[1] == 3:
        z = a[:,2]
    elif len(a.shape) == 1:
        z = np.array([0]*n)
    ax.scatter3D(x,y,z, marker=marker, label=label)
ax.legend()
plt.show()

if True:
    # Graph projections
    px = P_x.dot(X)
    py = P_y.dot(Y)

    fig2 = plt.figure()
    graphData = [('o', px, 'X'),\
                 ('^', py, 'Y')]
    for marker, a, label in graphData:
        x = a[:,0]
        y = a[:,1]
        plt.scatter(x, y, marker=marker, label=label)
    ax.legend()
    plt.show()
